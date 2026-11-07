"""Seven matched closed-loop cases for the nonzero robot-history residual only."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import ROOT, identity, now, require, save
from delivery import read
from event_review import rollout
from gpu_preview_worker import GpuPolicyProcess
from interim_review import GIB, TRAINING, fixed_rows, load_physical_robot, media
from paused_preview import checkpoint, gpu_command, preflight
from paused_runtime import own_gpu
from preview_review import environment, require_plant, runtime_paths
from runtime_guard import allocated_tree, guarded_run
from stream_events import EventSpec, EventTimeline
from student_causal_review import CausalReviewBudget, baseline_cases, review_cases, validate_causal_timing
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_proprio_review_worker import checkpoint_contract
from student_proprio_training import LEDGER, OUTPUT as TRAINED
from student_same_state import unique

DATA = ROOT/"data/supermimic_for_q1"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_REVIEW_20261108"
PLAN = HERE/"student_proprio_review_plan_20261108.json"
STAGE = "F0_PROPRIO_PAIR01_MATCHED_CLOSED_LOOP"
FILES = ("student_proprio_review.py", "student_proprio_review_worker.py", "test_student_proprio_review.py")


def zero_identity(current, previous):
    import torch
    from student_proprio_models import zero_adapter
    require(current.get("schema") == "q1_f0_proprio_residual_v1"
        and current.get("arm") == "zero_history" and current.get("selected_update") == 0
        and current.get("actual_updates") == 3000 and current.get("actual_adam_steps") == [3000]*4
        and previous.get("schema") == "q1_low_latency_f0_history_encoder_v1"
        and previous.get("selected_update") == 500 and zero_adapter(current["adapter"]),
        "Only the exact zero selected residual may reuse the original f0 outcome")
    keys = ("interface", "current_statistics", "frozen_teacher_policy_sha256", "decoder_sha256",
        "human_real_future_frames", "history_frames", "history_frame_dim")
    require(all(current.get(k) == previous.get(k) for k in keys)
        and current["provenance"]["teacher"] == previous["provenance"]["teacher"]
        and current["base_encoder"].keys() == previous["encoder"].keys()
        and all(torch.equal(v, previous["encoder"][k]) for k,v in current["base_encoder"].items()),
        "A zero residual has another base encoder, normalization, decoder or timing")
    return dict(status="EXACT_ORIGINAL_F0_ZERO_RESIDUAL", selected_update=0,
        base_tensors_exact=len(current["base_encoder"]), maximum_abs_difference=0.,
        zero_last_layer=True, equal_policy_metadata=list(keys), cuda_initialized=torch.cuda.is_initialized())


def selected_audit(settings):
    import torch
    from upper_mps_forecast import audit_cpu_tensors
    values = [torch.load(settings[k]["path"], map_location="cpu", weights_only=True)
        for k in ("zero_checkpoint", "original_checkpoint", "selected_checkpoint")]
    for value in values:
        audit_cpu_tensors(torch, value)
    zero, previous, selected = values
    proof = zero_identity(zero, previous)
    checkpoint_contract(selected, previous["provenance"]["teacher"],
        previous["frozen_teacher_policy_sha256"], previous["decoder_sha256"], "causal_history")
    from student_proprio_models import zero_adapter
    require(selected["selected_update"] == settings["selected_update"] == 3000
        and not zero_adapter(selected["adapter"])
        and selected["current_statistics"] == previous["current_statistics"]
        and selected["base_encoder"].keys() == previous["encoder"].keys()
        and all(torch.equal(v, previous["encoder"][k]) for k,v in selected["base_encoder"].items())
        and selected["provenance"]["initial_student"] == settings["original_checkpoint"]
        and zero["provenance"]["initial_student"] == settings["original_checkpoint"]
        and proof["base_tensors_exact"] == 8 and not torch.cuda.is_initialized(),
        "Require the nonzero3000 causal arm with the same eight frozen human tensors")
    return dict(status="SELECTED_NONZERO_CAUSAL_RESIDUAL", selected_update=3000,
        arm="causal_history", zero_history_reuse_proof=proof, base_tensors_exact=8,
        cuda_initialized=False, original_causal_action_path=True, cpu_state_converter="original")


def trained_evidence(settings):
    result, fit, guard, auth = (read(TRAINED/n) for n in
        ("result.json", "fit_result.json", "fit.guard.json", "authorization.json"))
    require(result.get("stage") == "F0_PROPRIO_RESIDUAL_PAIR"
        and result.get("status") == fit.get("status") == "PAIRED_RESIDUAL_TRAINED_PENDING_CLOSED_LOOP"
        and result.get("source_integrity") == "PASSED" and result.get("actual_optimizer_updates") == 6000
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and result["fit_result"] == identity(TRAINED/"fit_result.json")
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists()
        and fit.get("matched_batch_order_and_exposure") is True
        and fit.get("test_sources_read") is False and fit.get("human_real_future_frames") == 0,
        "Wait for paired training, frozen sources, settled cost and owned cleanup")
    require(settings.get("schema") == "q1_f0_proprio_review_v1" and settings["training_root"] == str(TRAINED)
        and settings["maximum_new_cases"] == 7 and settings["maximum_wall_seconds"] == 1800
        and settings["maximum_output_bytes"] == 3*GIB and settings["new_optimizer_updates"] == 0
        and settings["heldout_events_opened"] is False and settings["cpu_conversion_candidate_used"] is False
        and settings["exploratory_events"] == dict(first_frame=50, duration_frames=10, transport_frames=0),
        "Use the immutable seven-case exploratory plan without fresh-event or CPU changes")
    records = [identity(TRAINED/n) for n in ("result.json", "fit_result.json", "fit.guard.json",
        "authorization.json", "checkpoint_audit.json", "reused_dataset.json")]
    for arm, update, is_zero in (("zero_history", 0, True), ("causal_history", 3000, False)):
        fit_arm = read(TRAINED/arm/"fit_result.json")
        trace = read(TRAINED/arm/"training_trace.json")
        eligible = [r for r in trace if r["fixed_anchor_non_regression"]]
        selected = min(eligible, key=lambda r:r["validation_risk"])
        require(fit_arm["actual_updates"] == 3000 and fit_arm["actual_adam_steps"] == [3000]*4
            and fit_arm["selected_update"] == selected["update"] == update
            and fit_arm["selected_adapter_is_zero"] is is_zero
            and trace[0]["update"] == 0 and trace[-1]["update"] == 3000,
            "Paired selection, optimizer counters or fixed validation anchors changed")
        records += [identity(TRAINED/arm/n) for n in ("selected.pt", "fit_result.json", "training_trace.json")]
    records += auth["protected_inputs"]+[auth["residual_plan"]]
    settings_records = [settings[k] for k in
        ("training_result", "fit_result", "selected_checkpoint", "zero_checkpoint", "original_checkpoint")]
    require(all(identity(r["path"]) == r for r in settings_records), "Registered selected artifacts changed")
    return records+settings_records


def admission(args):
    sample = resources(True)
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    require(args.output == OUTPUT and args.review_plan == PLAN
        and identity(PLAN)["sha256"] == args.review_plan_sha256, "Use the one published residual review root")
    settings = read(PLAN)
    protected = trained_evidence(settings)
    selected = selected_audit(settings)
    records = settings["prior_comparison"]
    require(len(records) == 2 and all(identity(r["path"]) == r for r in records), "Prior saved comparison changed")
    summary, gate = (read(r["path"]) for r in records)
    require(summary.get("total_comparison_cases") == 28 and gate.get("status") == "PASSED"
        and gate.get("source_inputs_unchanged") is True, "Reuse the actual preserved28 comparison")
    controls = baseline_cases(summary)
    old = [r for r in summary["rows"] if r.get("variant") == "original_f0"]
    require(len(old) == 7 and {(r["event"], r["clip"]) for r in old} == set(review_cases())
        and all(not r["complete"] for r in old), "Reuse the seven original f0 failures without identical replay")
    protected += summary["inputs"]+records+sources+code
    protected += [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt,
        args.code_manifest, PLAN, TRAINING/"e1024_train_plan.json")]
    protected = unique(protected)
    required = {str(HERE/n) for n in FILES}
    require(required <= {r["path"] for r in code} and not required & {r["path"] for r in sources}
        and str(LEDGER) not in {r["path"] for r in protected}
        and all(identity(r["path"]) == r for r in protected), "Review code, preserved source or ledger scope changed")
    ledger = checked(read(LEDGER))
    by_stage = {s:charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == s])) for s in STAGES}
    require(charged(ledger) == settings["expected_cumulative_seconds"] == 3671
        and by_stage == settings["expected_by_stage"]
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r2"]-by_stage["r2"] >= 1800 and charged(ledger)+1800 <= LIMIT,
        "Preserve the settled3671 ledger, r2 limit and cumulative four-hour budget")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+3*GIB < 500_000_000_000,
        "Project500GB bound reached")
    with own_gpu() as owner:
        require(owner is not None, "Review idle original-lock admission failed")
    return config, runtime, receipt, audit, protected, controls, old, selected, sample


def folder(args):
    return args.output/f"{args.event_kind}_f0_student_h0_{args.clip}"


def replay(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    selected, baseline = checkpoint(args, config)
    output = folder(args)
    output.mkdir(parents=True, exist_ok=False)
    item = next(r for r in fixed_rows(frozen) if Path(r["q1_primebot"]["path"]).stem == args.clip)
    command = gpu_command(args, config, runtime, plan, selected, baseline, 0, "masked")
    command[2] = str(HERE/"student_proprio_review_worker.py")
    student = TRAINED/"causal_history/selected.pt"
    command.extend(("--student-checkpoint", str(student), "--event-kind", args.event_kind,
        "--residual-arm", "causal_history"))
    result = dict(status="RUNNING", checkpoint=selected, cases=[], policy_device="cuda", final_acceptance=False)
    actor, timeline = None, None
    telemetry = (output/"stream_timing.jsonl").open("x")

    class Recorder(GpuPolicyProcess):
        def start(self, item, seed, interface):
            nonlocal timeline
            value = super().start(item, seed, interface)
            timeline = EventTimeline(value["paired_reference"]["frames"], EventSpec(args.event_kind, 50, 10, 0))
            return value

        def action(self, frame, state):
            started = time.perf_counter_ns()
            reply = super().action(frame, state)
            rpc_ns = time.perf_counter_ns()-started
            timing = validate_causal_timing(reply["stream"], frame, timeline)
            telemetry.write(json.dumps(dict(timing, local_pipe_rpc_ns=rpc_ns),
                allow_nan=False)+"\n")
            if frame % 50 == 0:
                telemetry.flush()
            return reply

    try:
        actor = Recorder(command, output/"policy_worker.log", config["expected_gpu"])
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == 32000
            and actor.ready.get("human_real_future_frames") == 0 and actor.ready.get("variant") == "f0_student"
            and actor.ready["student_checkpoint"] == identity(student)
            and actor.ready.get("selected_update") == 3000 and actor.ready.get("residual_arm") == "causal_history"
            and actor.ready.get("original_causal_action_path") is True
            and actor.ready.get("cpu_state_converter") == "original", "Residual worker loaded another policy or action path")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and Path(settings["robot_xml"]) == Path(runtime["xml"]),
            "Residual physical configuration changed")
        row = rollout(robot, actor, settings, item, args, output/"review"/(args.clip+"_s0"))
        controls = [r for r in read(args.output/"baseline_cases.json")
            if (r["event"], r["clip"]) == (args.event_kind, args.clip)]
        require(len(controls) == 2 and all(row["derived_reference_sha256"] == r["derived_reference_sha256"]
            for r in controls), "Residual and reused H4 controls have different physical targets")
        row.update(variant="f0_student", residual_arm="causal_history", student_checkpoint=identity(student),
            human_real_future_frames=0, input_interface="f0_plus_history", full_clip_quality_accepted=False)
        save(Path(row["trace"]).parent/"result.json", row)
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "Residual runtime isolation failed")
        result.update(status="PHYSICS_PASSED", cases=[row], render_xml=identity(settings["robot_xml"]),
            policy_worker_reaped=True, policy_worker_return_code=0, cuda_initialized=False,
            asset_sha256=robot.asset_sha256, termination=robot.reward_identity())
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if actor is not None:
            actor.close(graceful=False)
        telemetry.close()
        result.update(ended_utc=now(), stream_telemetry=identity(output/"stream_timing.jsonl"))
        save(output/"review/result.json", result)


def child_command(args, runtime, owner, case, phase):
    require(phase in ("replay", "media") and case in review_cases(), "Unexpected residual case or phase")
    return [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child", phase,
        "--config", str(args.config), "--runtime-config", str(args.runtime_config), "--output", str(args.output),
        "--event-kind", case[0], "--clip", case[1], "--gpu-owner-pid", str(owner["pid"]),
        "--gpu-owner-ticks", owner["start_ticks"]]


def execute(args):
    if args.child:
        config, runtime = read(args.config), read(args.runtime_config)
        if args.child == "media":
            runtime_paths(runtime)
            return media(SimpleNamespace(output=folder(args)), config, runtime)
        return replay(args, config, runtime, read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"]))
    config, runtime, receipt, audit, protected, controls, old, selected, sample = admission(args)
    if args.admit_only:
        print(json.dumps(dict(status="F0_PROPRIO_REVIEW_ADMITTED", resources=sample,
            protected_files=len(protected), new_cases=7, reused_cases=21, selected_audit=selected)))
        return
    with own_gpu() as lock:
        owner = dict(lock, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        for name, value in (("checkpoint_audit.json", audit), ("baseline_cases.json", controls),
                ("original_f0_cases.json", old), ("selected_residual_audit.json", selected)):
            save(args.output/name, value)
        protected += [identity(args.output/n) for n in
            ("baseline_cases.json", "original_f0_cases.json", "selected_residual_audit.json")]
        save(args.output/"authorization.json", dict(stage=STAGE, evaluation_commit=args.evaluation_commit,
            protected_inputs=protected, cases=review_cases(), new_physical_cases=7, reused_baseline_cases=21,
            maximum_wall_seconds=1800, phase_wall_seconds=1800, maximum_output_bytes=3*GIB,
            gpu_lock_owner=owner, selected_update=3000, human_real_future_frames=0, new_optimizer_updates=0,
            original_causal_action_path=True, cpu_conversion_candidate_used=False, heldout_events_opened=False,
            user_accepted_checkpoint=None))
        budget = CausalReviewBudget(config, receipt, args.output, owner)
        previous, requested, reservation, last_guard = dict(os.environ), False, None, None
        result = dict(stage=STAGE, status="RUNNING", controller=owner, phases=[], cases=[], started_utc=now(),
            reused_baseline_cases=21, user_accepted_checkpoint=None, natural_teleoperation_qualified=False)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage="r2", owner=owner,
                gpu=config["expected_gpu"], commit=args.evaluation_commit)
            result["cost_reservation"] = reservation
            os.environ.update(environment(runtime))
            for case in review_cases():
                for phase in ("replay", "media"):
                    require(all(identity(r["path"]) == r for r in protected), "Residual review source changed")
                    name = f"{case[0]}_f0_student_h0_{case[1]}_{phase}"
                    result["phase"] = name
                    save(args.output/"result.json", result)
                    budget.phase_started = time.monotonic()
                    last_guard = args.output/(name+".guard.json")
                    guard = guarded_run(child_command(args, runtime, owner, case, phase), ROOT, budget, last_guard,
                        args.output/(name+".log"), check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name, guard=identity(last_guard)))
                    require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                        and guard["cleanup"].get("worker_return_code") == 0, "Residual review phase failed: "+name)
                result["cases"].append(identity(args.output/f"{case[0]}_f0_student_h0_{case[1]}"/"review/result.json"))
            require(all(identity(r["path"]) == r for r in protected), "Residual final source gate failed")
            result.update(status="COMPLETED", source_integrity="PASSED", new_physical_cases=7,
                total_comparison_cases=28, new_optimizer_updates=0, quality_conclusion="PENDING_SAVED_ANALYSIS")
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if reservation:
                cleanup = (read(last_guard).get("cleanup", {}) if last_guard and last_guard.is_file()
                    else dict(worker_started=False, worker_reaped=True))
                if cleanup.get("worker_reaped") is True:
                    result["cost_settlement"] = settle(LEDGER, run_id=str(args.output), owner=owner, cleanup=cleanup)
            result.update(ended_utc=now(), cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json", result)
            os.environ.clear()
            os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    defaults = dict(config=ROOT/"script/q1_five_stage/conditional_shared_v04_20261104.json",
        runtime_config=ROOT/"data/manifests/primebot_q1_monitored_config_20261103.json",
        pause_receipt=ROOT/"logs/session_records/q1_shared_user_pause_20261106.json",
        audit_root=DATA/"Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106")
    for name, value in defaults.items():
        p.add_argument("--"+name.replace("_", "-"), type=Path, default=value)
    for name in ("output", "code-manifest", "review-plan"):
        p.add_argument("--"+name, type=Path)
    for name in ("evaluation-commit", "manifest-sha256", "review-plan-sha256", "event-kind", "clip", "gpu-owner-ticks"):
        p.add_argument("--"+name)
    p.add_argument("--gpu-owner-pid", type=int)
    p.add_argument("--child", choices=("replay", "media"))
    p.add_argument("--admit-only", action="store_true")
    p.set_defaults(horizon=0, tail_method="masked", variant="student")
    return p


if __name__ == "__main__":
    execute(parser().parse_args())
