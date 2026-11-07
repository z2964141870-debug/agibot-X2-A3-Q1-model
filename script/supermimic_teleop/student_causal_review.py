"""Seven new f0 replays, reusing fourteen completed R1 comparison cases."""

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
from common import FIXED, ROOT, identity, now, require, save
from delivery import read
from event_review import rollout
from forecast_data import source_support
from gpu_preview_worker import GpuPolicyProcess
from interim_review import GIB, fixed_rows, load_physical_robot, media
from paused_preview import Budget, checkpoint, gpu_command, preflight
from paused_runtime import own_gpu
from preview_review import environment, require_plant, runtime_paths, validate_code_manifest
from runtime_guard import allocated_tree, guarded_run
from stream_contract import DT_NS
from stream_events import EventSpec, EventTimeline
from student_causal_training import saved_analysis_gate
from student_review import training_evidence

WALL_SECONDS = 1800


def review_cases():
    return [("nominal", clip) for clip in FIXED]+[(event, FIXED[0])
        for event in ("stop", "reverse", "dropout", "late_burst")]


def validate_causal_timing(value, frame, timeline):
    control = frame*DT_NS
    available = []
    for packet in timeline.packets:
        if packet.capture_ns <= control:
            dependency = max(packet.capture_ns, source_support(packet.source_index)["dependency_end_ns"])
            if max(packet.arrival_ns, dependency) <= control:
                available.append((packet, dependency))
    samples = []
    for target in range(frame-9, frame+1):
        eligible = [p for p, dependency in available if p.capture_ns <= target*DT_NS and dependency <= target*DT_NS]
        packet = max(eligible, key=lambda p: p.capture_ns) if eligible else None
        if packet is not None and target*DT_NS-packet.capture_ns > 200_000_000:
            packet = None
        samples.append(packet)
    ready = samples[-1] is not None
    last_support = next((p for p in reversed(samples) if p is not None), None)
    status = "READY" if ready else "NO_CURRENT_SUPPORT"
    fallback = None if ready else ("initial_zero_delta" if frame == 0 else "previous_action_hold")
    expected = dict(frame=frame, horizon_frames=0, human_real_future_frames=0, future_preview_ns=0,
        variant="f0_student", event=timeline.spec.kind, anchor_ns=control, scheduled_anchor_to_control_ns=0,
        interface="f0_plus_history_oldest_first", requested_history_ns=[i*DT_NS for i in range(frame-9, frame+1)],
        used_sequences=[None if p is None else p.sequence for p in samples],
        used_capture_ns=[None if p is None else p.capture_ns for p in samples],
        used_arrival_ns=[None if p is None else max(p.arrival_ns, p.capture_ns,
            source_support(p.source_index)["dependency_end_ns"]) for p in samples],
        current_capture_age_ns=None if last_support is None else control-last_support.capture_ns,
        max_used_dependency_ns=max((max(p.capture_ns, source_support(p.source_index)["dependency_end_ns"])
            for p in samples if p is not None), default=None),
        input_ready=ready, input_status=status, status=status, fallback=fallback,
        upstream_preprocessing="offline_unverified", measured_end_to_end_latency_ns=None)
    require(all(value.get(k) == v for k, v in expected.items()),
            "f0 timing crosses current/history dependency, arrival or capture-age boundaries")
    require(value.get("policy_device") == "cuda" and all(type(value.get(k)) is int and value[k] >= 0
        for k in ("gpu_policy_wall_compute_ns", "worker_action_wall_ns")),
        "Missing synchronized f0 GPU and complete worker software timing")
    return value


def trained_causal_evidence(root, parent):
    result, fit, guard, auth = (read(root/name) for name in
        ("result.json", "fit_result.json", "fit.guard.json", "authorization.json"))
    require(result.get("stage") == auth.get("stage") == "R2_F0_HISTORY_ENCODER_DISTILLATION"
        and result.get("status") == fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP"
        and result.get("source_integrity") == "PASSED" and fit.get("actual_updates", 0) > 0
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and result.get("fit_result") == identity(root/"fit_result.json")
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists()
        and fit.get("human_real_future_frames") == 0
        and fit.get("provenance", {}).get("original_r1_fit") == identity(parent/"fit_result.json")
        and fit.get("current_statistics") == read(root/"current_statistics.json"),
        "Wait for actual causal training, settled cost, source gate and cleanup")
    records = [identity(root/name) for name in ("result.json", "fit_result.json", "fit.guard.json",
        "authorization.json", "selected.pt", "current_statistics.json", "training_trace.json", "reused_dataset.json")]
    records += auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
    require(all(identity(r["path"]) == r for r in records), "Trained causal evidence or protected sources changed")
    return records


def baseline_cases(summary):
    selected = [r for r in summary["rows"] if r.get("horizon_frames") == 4
        and r.get("variant") in ("teacher", "student") and (r.get("event"), r.get("clip")) in review_cases()]
    require(len(selected) == 14 and {(r["event"], r["clip"], r["variant"]) for r in selected}
        == {(e, c, v) for e, c in review_cases() for v in ("teacher", "student")},
        "Reuse fourteen unique actual R1 baselines on the same seven tasks")
    for event, clip in review_cases():
        pair = [r for r in selected if (r["event"], r["clip"]) == (event, clip)]
        require(pair[0]["derived_reference_sha256"] == pair[1]["derived_reference_sha256"],
                "Saved comparison targets differ")
    return selected


def folder(args):
    return args.output/f"{args.event_kind}_f0_student_h0_{args.clip}"


def replay(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    selected, baseline = checkpoint(args, config)
    output = folder(args)
    output.mkdir(parents=True, exist_ok=False)
    item = next(r for r in fixed_rows(frozen) if Path(r["q1_primebot"]["path"]).stem == args.clip)
    command = gpu_command(args, config, runtime, plan, selected, baseline, 0, "masked")
    command[2] = str(HERE/"student_causal_review_worker.py")
    command.extend(("--student-checkpoint", str(args.causal_training_root/"selected.pt"),
                    "--event-kind", args.event_kind))
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
            telemetry.write(json.dumps(dict(timing, local_pipe_rpc_ns=rpc_ns), allow_nan=False)+"\n")
            if frame % 50 == 0:
                telemetry.flush()
            return reply

    try:
        actor = Recorder(command, output/"policy_worker.log", config["expected_gpu"])
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == 32000
            and actor.ready.get("human_real_future_frames") == 0 and actor.ready.get("variant") == "f0_student"
            and actor.ready["student_checkpoint"] == identity(args.causal_training_root/"selected.pt"),
            "f0 worker loaded another teacher, encoder or input interface")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and Path(settings["robot_xml"]) == Path(runtime["xml"]),
                "Causal review physical configuration changed")
        row = rollout(robot, actor, settings, item, args, output/"review"/(args.clip+"_s0"))
        controls = [r for r in read(args.output/"baseline_cases.json")
                    if (r["event"], r["clip"]) == (args.event_kind, args.clip)]
        require(len(controls) == 2 and all(row["derived_reference_sha256"] == r["derived_reference_sha256"]
                for r in controls), "f0 and reused H4 controls have different physical targets")
        row.update(variant="f0_student", student_checkpoint=identity(args.causal_training_root/"selected.pt"),
            human_real_future_frames=0, input_interface="f0_plus_history", full_clip_quality_accepted=False)
        save(Path(row["trace"]).parent/"result.json", row)
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "f0 review runtime isolation failed")
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


class CausalReviewBudget(Budget):
    def __init__(self, *args):
        super().__init__(*args)
        self.project_time, self.project = 0., None

    def check(self):
        require(time.monotonic()-self.started < WALL_SECONDS, "Causal review1800s total bound reached")
        if self.project is None or time.monotonic()-self.project_time >= 60:
            self.project, self.project_time = allocated_tree(ROOT), time.monotonic()
        require(self.project["unique_allocated_bytes"] < 500_000_000_000, "Project500GB bound reached")
        return super().check()


def child_command(args, runtime, owner, case, phase):
    event, clip = case
    require(phase in ("replay", "media"), "Unknown causal review phase")
    return [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child", phase,
        "--config", str(args.config), "--runtime-config", str(args.runtime_config), "--output", str(args.output),
        "--causal-training-root", str(args.causal_training_root), "--evaluation-commit", args.evaluation_commit,
        "--event-kind", event, "--clip", clip, "--gpu-owner-pid", str(owner["pid"]),
        "--gpu-owner-ticks", owner["start_ticks"]]


def execute(args):
    if args.child:
        config, runtime = read(args.config), read(args.runtime_config)
        if args.child == "media":
            runtime_paths(runtime)
            return media(SimpleNamespace(output=folder(args)), config, runtime)
        from interim_review import TRAINING
        return replay(args, config, runtime, read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"]))
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    parent = training_evidence(args.training_root)
    protected = parent+saved_analysis_gate(args.r1_analysis_root, parent)
    protected += trained_causal_evidence(args.causal_training_root, args.training_root)
    baselines = baseline_cases(read(args.r1_analysis_root/"summary.json"))
    required = {str(HERE/name) for name in ("student_causal_review.py", "student_causal_review_worker.py")}
    require(required <= {r["path"] for r in code} and not required & {r["path"] for r in sources},
            "Published causal review sources are missing or overlap frozen inputs")
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt, args.code_manifest)]
    with own_gpu() as owner:
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"baseline_cases.json", baselines)
        controls.append(identity(args.output/"baseline_cases.json"))
        save(args.output/"authorization.json", dict(stage="R2_F0_MATCHED_CLOSED_LOOP",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code, training_source_inputs=sources,
            controller_inputs=controls, preserved_research_inputs=protected, cases=review_cases(),
            new_physical_cases=7, reused_baseline_cases=14, maximum_wall_seconds=WALL_SECONDS,
            phase_wall_seconds=1800, maximum_output_bytes=3*GIB, gpu_lock_owner=owner,
            new_optimizer_updates=0, human_real_future_frames=0, user_accepted_checkpoint=None))
        budget, previous, requested = CausalReviewBudget(config, receipt, args.output, owner), dict(os.environ), False
        result = dict(status="RUNNING", stage="R2_F0_MATCHED_CLOSED_LOOP", controller=owner,
            phases=[], cases=[], started_utc=now(), reused_baseline_cases=14,
            natural_teleoperation_qualified=False, user_accepted_checkpoint=None)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            for case in review_cases():
                args.event_kind, args.clip = case
                for phase in ("replay", "media"):
                    require(all(identity(r["path"]) == r for r in protected+controls), "Causal review evidence changed")
                    validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                    name = f"{case[0]}_f0_student_h0_{case[1]}_{phase}"
                    result["phase"] = name
                    save(args.output/"result.json", result)
                    budget.phase_started = time.monotonic()
                    guard = guarded_run(child_command(args, runtime, owner, case, phase), ROOT, budget,
                        args.output/(name+".guard.json"), args.output/(name+".log"),
                        check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name, guard=identity(args.output/(name+".guard.json"))))
                    require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                        and guard["cleanup"].get("worker_return_code") == 0, "Causal review phase failed: "+name)
                result["cases"].append(identity(folder(args)/"review/result.json"))
            require(all(identity(r["path"]) == r for r in protected+controls+sources+code), "Causal review source changed")
            result.update(status="COMPLETED", source_integrity="PASSED", new_physical_cases=7,
                total_comparison_cases=21, new_optimizer_updates=0, quality_conclusion="PENDING_SAVED_ANALYSIS")
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result["ended_utc"] = now()
            save(args.output/"result.json", result)
            os.environ.clear()
            os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "causal-training-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("training-root", "r1-analysis-root", "pause-receipt", "audit-root", "code-manifest"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--child", choices=("replay", "media"))
    parser.add_argument("--event-kind", choices=("nominal", "stop", "reverse", "dropout", "late_burst"))
    parser.add_argument("--clip", choices=FIXED)
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    parser.set_defaults(horizon=0, tail_method="masked")
    args = parser.parse_args()
    required = (args.training_root, args.r1_analysis_root, args.pause_receipt, args.audit_root,
                args.code_manifest, args.manifest_sha256)
    if not args.child and not all(required):
        parser.error("Controller requires completed R1 evidence, published source and original paused admission")
    if args.child and not all((args.event_kind, args.clip, args.gpu_owner_pid, args.gpu_owner_ticks)):
        parser.error("Child needs its exact case and original GPU owner")
    execute(args)
