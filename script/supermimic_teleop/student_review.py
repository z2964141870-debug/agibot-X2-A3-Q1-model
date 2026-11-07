"""Matched R1 closed-loop research evaluation; the original final gate is separate."""

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
from preview_review import environment, require_plant, validate_code_manifest
from runtime_guard import allocated_tree, guarded_run
from stream_contract import DT_NS
from stream_events import EventSpec, EventTimeline


def review_cases():
    nominal = [("nominal", variant, horizon, clip) for variant, horizon in
        (("teacher", 10), ("teacher", 4), ("student", 4), ("teacher", 1), ("student", 1)) for clip in FIXED]
    events = [(event, variant, 4, FIXED[0]) for event in ("stop", "reverse", "dropout", "late_burst")
              for variant in ("teacher", "student")]
    return nominal+events


def validate_timing(value, frame, horizon, variant, timeline):
    control = (frame+horizon)*DT_NS
    visible = {p.sequence: p for p in timeline.packets if p.capture_ns <= control
        and max(p.arrival_ns, p.capture_ns, source_support(p.source_index)["dependency_end_ns"]) <= control}
    expected = []
    for sequence in range(frame+1, min(frame+horizon+1, timeline.frames)):
        if sequence not in visible:
            break
        expected.append(sequence)
    ready = bool(expected)
    dependency = max((max(visible[i].capture_ns,
        source_support(visible[i].source_index)["dependency_end_ns"]) for i in expected), default=None)
    fallback = None if ready else ("initial_zero_delta" if frame == 0 else "previous_action_hold")
    require(value.get("frame") == frame and value.get("horizon_frames") == horizon
            and value.get("variant") == variant and value.get("event") == timeline.spec.kind
            and value.get("requested_preview_ns") == value.get("scheduled_origin_to_control_ns") == horizon*DT_NS
            and value.get("used_sequences") == expected and value.get("input_ready") is ready
            and value.get("observed_prefix_frames") == value.get("policy_prefix_frames") == len(expected)
            and value.get("max_used_dependency_ns") == dependency and value.get("fallback") == fallback
            and value.get("source_preprocessing") == "offline_unverified"
            and value.get("measured_end_to_end_latency_ns") is None,
            "Review target crosses the declared interpolation/arrival/preview boundary")
    require(value.get("policy_device") == "cuda" and type(value.get("gpu_policy_wall_compute_ns")) is int
            and value["gpu_policy_wall_compute_ns"] >= 0
            and type(value.get("worker_action_wall_ns")) is int and value["worker_action_wall_ns"] >= 0,
            "Missing synchronized GPU and complete worker software timing")
    if variant == "teacher" and ready:
        require(type(value.get("packing_maximum_action_difference")) in (int, float)
                and 0 <= value["packing_maximum_action_difference"] <= 2e-5,
                "Teacher native encoder/decoder packing is not equivalent")
    return value


def training_evidence(root):
    result, fit, guard, dataset = (read(root/name) for name in
        ("result.json", "fit_result.json", "collection_fit.guard.json", "dataset_receipt.json"))
    require(result.get("status") == fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP"
            and result.get("source_integrity") == "PASSED" and fit.get("actual_updates", 0) > 0
            and result.get("cost_settlement", {}).get("state") == "SETTLED"
            and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == 0
            and not Path("/proc", str(result["controller"]["pid"])).exists()
            and dataset.get("status") == "PACKING_ACTION_PARITY_PASSED",
            "Wait for actual trained R1, source integrity, settled cost and owned cleanup")
    require(fit.get("selection_split") == "validation" and fit.get("test_used_for_selection") is False
            and fit.get("test_sources_read") is False, "Student model selection provenance differs")
    auth = read(root/"authorization.json")
    records = [identity(root/name) for name in ("result.json", "fit_result.json", "collection_fit.guard.json",
        "dataset_receipt.json", "collection_result.json", "authorization.json", "selected.pt", "training_trace.json")]
    records += auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
    require(all(identity(r["path"]) == r for r in records), "Completed R1 artifacts or protected sources changed")
    return records


def folder(args):
    return args.output/f"{args.event_kind}_{args.variant}_h{args.horizon}_{args.clip}"


def replay(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    selected, baseline = checkpoint(args, config)
    output = folder(args)
    output.mkdir(parents=True, exist_ok=False)
    item = next(r for r in fixed_rows(frozen) if Path(r["q1_primebot"]["path"]).stem == args.clip)
    command = gpu_command(args, config, runtime, plan, selected, baseline, args.horizon, "masked")
    command[2] = str(HERE/"student_review_worker.py")
    command.extend(("--student-checkpoint", str(args.training_root/"selected.pt"), "--variant", args.variant,
                    "--event-kind", args.event_kind))
    result = dict(status="RUNNING", checkpoint=selected, cases=[], policy_device="cuda", final_acceptance=False)
    actor = None
    telemetry = (output/"stream_timing.jsonl").open("x")
    timeline = None

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
            timing = validate_timing(reply["stream"], frame, args.horizon, args.variant, timeline)
            telemetry.write(json.dumps(dict(timing, local_pipe_rpc_ns=rpc_ns), allow_nan=False)+"\n")
            if frame % 50 == 0:
                telemetry.flush()
            return reply

    try:
        actor = Recorder(command, output/"policy_worker.log", config["expected_gpu"])
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == 32000
                and actor.ready["variant"] == args.variant
                and actor.ready["student_checkpoint"] == identity(args.training_root/"selected.pt"),
                "Closed-loop worker loaded another teacher or student")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and Path(settings["robot_xml"]) == Path(runtime["xml"]),
                "Closed-loop physical configuration changed")
        row = rollout(robot, actor, settings, item, args, output/"review"/(args.clip+"_s0"))
        row.update(variant=args.variant, student_checkpoint=identity(args.training_root/"selected.pt"),
            receiver="120Hz dependency-aware short preview; no future-labelled encoder input",
            full_clip_quality_accepted=False)
        save(Path(row["trace"]).parent/"result.json", row)
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "Review runtime isolation failed")
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


class ReviewBudget(Budget):
    def __init__(self, *args):
        super().__init__(*args)
        self.project_time, self.project = 0., None

    def check(self):
        require(time.monotonic()-self.started < 2700, "R1 evaluation2700s total bound reached")
        if self.project is None or time.monotonic()-self.project_time >= 60:
            self.project, self.project_time = allocated_tree(ROOT), time.monotonic()
        require(self.project["unique_allocated_bytes"] < 500_000_000_000, "Project500GB bound reached")
        return super().check()


def execute(args):
    if args.child:
        config, runtime = read(args.config), read(args.runtime_config)
        from interim_review import TRAINING
        plan, frozen = read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"])
        if args.child == "replay":
            replay(args, config, runtime, plan, frozen)
        else:
            media(SimpleNamespace(output=folder(args)), config, runtime)
        return
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    required = {str(HERE/name) for name in ("student_review.py", "student_review_worker.py")}
    require(required <= {row["path"] for row in code} and not required & {row["path"] for row in sources},
            "Independent evaluation code missing or overlaps frozen training")
    parent = training_evidence(args.training_root)
    controls = [identity(path) for path in (args.config, args.runtime_config, args.pause_receipt, args.code_manifest)]
    with own_gpu() as owner:
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"authorization.json", dict(stage="R1_MATCHED_CLOSED_LOOP", evaluation_commit=args.evaluation_commit,
            evaluation_sources=code, training_source_inputs=sources, controller_inputs=controls,
            trained_student_inputs=parent, cases=review_cases(), maximum_wall_seconds=2700,
            phase_wall_seconds=1800, maximum_output_bytes=3*GIB, gpu_lock_owner=owner,
            new_optimizer_updates=0, accepted_checkpoint=None, hardware_end_to_end_measured=False))
        budget, previous, requested = ReviewBudget(config, receipt, args.output, owner), dict(os.environ), False
        result = dict(status="RUNNING", stage="R1_MATCHED_CLOSED_LOOP", controller=owner, phases=[], cases=[],
            started_utc=now(), natural_teleoperation_qualified=False, user_accepted_checkpoint=None)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            for event, variant, horizon, clip in review_cases():
                args.event_kind, args.variant, args.horizon, args.clip = event, variant, horizon, clip
                phases = ["replay"]
                if clip == FIXED[0] and ((event == "nominal" and (variant, horizon) in
                        (("teacher", 10), ("teacher", 4), ("student", 4)))
                        or (event in ("stop", "reverse", "dropout", "late_burst") and variant == "student")):
                    phases.append("media")
                for phase in phases:
                    require(all(identity(r["path"]) == r for r in parent+controls), "R1 evidence or controls changed")
                    validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                    name = f"{event}_{variant}_h{horizon}_{clip}_{phase}"
                    result["phase"] = name
                    save(args.output/"result.json", result)
                    python = runtime["primebot_python"] if phase == "replay" else config["python"]
                    command = [python, "-u", str(Path(__file__).resolve()), "--child", phase,
                        "--config", str(args.config), "--runtime-config", str(args.runtime_config),
                        "--output", str(args.output), "--training-root", str(args.training_root),
                        "--evaluation-commit", args.evaluation_commit, "--event-kind", event,
                        "--variant", variant, "--horizon", str(horizon), "--clip", clip,
                        "--gpu-owner-pid", str(owner["pid"]), "--gpu-owner-ticks", owner["start_ticks"]]
                    budget.phase_started = time.monotonic()
                    guard = guarded_run(command, ROOT, budget, args.output/(name+".guard.json"),
                        args.output/(name+".log"), check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name, guard=identity(args.output/(name+".guard.json"))))
                    require(guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"] is True
                            and guard["cleanup"]["worker_return_code"] == 0, "R1 review phase failed: "+name)
                result["cases"].append(identity(folder(args)/"review/result.json"))
            require(all(identity(r["path"]) == r for r in parent+sources+code+controls), "R1 review source changed")
            result.update(status="COMPLETED", source_integrity="PASSED", physical_cases=len(review_cases()),
                new_optimizer_updates=0, quality_conclusion="PENDING_SAVED_ANALYSIS")
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
    for name in ("config", "runtime-config", "output", "training-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("pause-receipt", "audit-root", "code-manifest"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--child", choices=("replay", "media"))
    parser.add_argument("--event-kind", choices=("nominal", "stop", "reverse", "dropout", "late_burst"))
    parser.add_argument("--variant", choices=("teacher", "student"))
    parser.add_argument("--horizon", type=int, choices=(1, 4, 10))
    parser.add_argument("--clip", choices=FIXED)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    args = parser.parse_args()
    if not args.child and not all((args.pause_receipt, args.audit_root, args.code_manifest, args.manifest_sha256)):
        parser.error("Controller needs published source manifest and paused32k admission")
    if args.child and not all((args.event_kind, args.variant, args.horizon, args.clip,
                             args.gpu_owner_pid, args.gpu_owner_ticks)):
        parser.error("Child needs the exact evaluation case and GPU owner identity")
    execute(args)
