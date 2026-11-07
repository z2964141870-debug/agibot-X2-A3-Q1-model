"""One bounded R2 student-state aggregation round after its real failed review."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import ROOT, identity, now, require, save
from delivery import read
from interim_review import TRAINING, load_physical_robot
from paused_preview import checkpoint, gpu_command
from paused_runtime import assert_owner, own_gpu
from preview_review import environment, require_plant
from runtime_guard import guarded_run
from stream_events import EventSpec, EventTimeline
from student_causal_review import trained_causal_evidence, validate_causal_timing
from student_causal_training import CausalBudget, train_preflight
from student_cost import reserve, settle
from student_review import training_evidence


PLAN_SHA = "f671c45946fa0663e422d87735d5d61ceb83ab3f1316ae407e314a45702ce612"
NEW_CODE = ("student_aggregate.py", "student_aggregate_worker.py", "student_aggregate_fit.py")


def validate_records(data, split, jobs, traces):
    required = {"f0", "history", "teacher_action", "clip_id", "frame", "actual_state",
                "behavior_action", "student_action"}
    require(set(data) == required and len(data["frame"]) > 0, "Missing aggregate record arrays")
    count = len(data["frame"])
    shapes = dict(f0=(count, 10, 226), history=(count, 930), teacher_action=(count, 29),
        clip_id=(count,), frame=(count,), actual_state=(count, 79), behavior_action=(count, 29),
        student_action=(count, 29))
    require(all(data[key].shape == shape and np.isfinite(data[key]).all() for key, shape in shapes.items()),
            "Invalid aggregate record shape or finite-value contract")
    require(all(np.issubdtype(data[key].dtype, np.integer) for key in ("clip_id", "frame")),
            "Keep integer aggregate sample identities")
    keys = set()
    for index, (clip_id, frame) in enumerate(zip(data["clip_id"], data["frame"])):
        clip_id, frame = int(clip_id), int(frame)
        require(0 <= clip_id < len(jobs) and (clip_id, frame) not in keys, "Duplicate or unknown aggregate row")
        keys.add((clip_id, frame))
        job, trace = jobs[clip_id], traces[clip_id]
        require(job["split"] == split and 0 <= frame < job["maximum_steps"] and frame % 2 == 0
            and frame < len(trace["canonical_action"]), "Aggregate row crosses its source or step boundary")
        for key, trace_key in (("actual_state", "actual"), ("behavior_action", "canonical_action"),
                ("teacher_action", "teacher_action"), ("student_action", "student_action")):
            require(np.array_equal(data[key][index], trace[trace_key][frame]),
                    "Aggregate state, behavior or queried teacher label differs from physical trace")
        require(np.allclose(data["behavior_action"][index],
            .5*(data["teacher_action"][index]+data["student_action"][index]), atol=2e-6, rtol=2e-6),
            "Collection executed another teacher/student mixture")
    expected = {i for i, job in enumerate(jobs) if job["split"] == split}
    require(set(map(int, data["clip_id"])) == expected, "Aggregation lost an original source group")
    first_frames = {}
    for clip_id in expected:
        ready = np.flatnonzero(traces[clip_id]["input_ready"])
        eligible = ready[ready % 2 == 0]
        require(len(eligible) > 0 and (clip_id, int(eligible[0])) in keys,
                "Keep the first causally available sampled history for every source")
        first_frames[str(clip_id)] = int(eligible[0])
    from student_causal_fit import causal_records
    causal_records(data)
    return dict(records=count, source_clips=len(expected), state_and_behavior_match=True,
        labels_match_saved_queries=True, teacher_labels_are_executed_actions=False,
        first_causally_available_sampled_frames=first_frames,
        frame_zero_included_where_available=True)


def collect(args):
    from gpu_preview_worker import GpuPolicyProcess
    config, runtime = read(args.config), read(args.runtime_config)
    torch = require_plant(runtime)
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator
    plan, frozen = read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"])
    jobs = read(args.output/"collection_plan.json")["jobs"]
    selected, baseline = checkpoint(args, config)
    robot, settings = load_physical_robot(baseline["path"])
    rows = {Path(row["q1_primebot"]["path"]).stem:row for row in frozen["rows"]}
    command = gpu_command(args, config, runtime, plan, selected, baseline, 0, "masked")
    command[2] = str(HERE/"student_aggregate_worker.py")
    for name, value in (("student-checkpoint", args.initial_r2_root/"selected.pt"),
            ("output", args.output), ("aggregate-plan", args.aggregate_plan),
            ("r1-training-root", args.training_root), ("deadline-ns", args.deadline_ns)):
        command.extend(("--"+name, str(value)))
    actor = GpuPolicyProcess(command, args.output/"native_worker.log", config["expected_gpu"])
    clips, traces = [], {}
    try:
        require(actor.ready["student_checkpoint"] == identity(args.initial_r2_root/"selected.pt")
            and actor.ready["checkpoint"] == selected, "Aggregation loaded another teacher or student")
        for clip_id, job in enumerate(jobs):
            item = rows[job["clip"]]
            require(all(item["human"][k] == job["human"][k] for k in ("bytes", "sha256")),
                    "Aggregation source differs from original paired human")
            reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
            started = actor.request(dict(op="start", clip=job["clip"], seed=0, interface="human", clip_id=clip_id))
            timeline = EventTimeline(started["paired_reference"]["frames"], EventSpec(**job["packet_event"]))
            simulator = Q1Simulator(robot, 1, "cpu", settings["action_scale"],
                use_tn=settings.get("use_tn", False), physics_timestep=settings.get("physics_timestep"))
            simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
            state = simulator.state()
            actual, actions, labels, students, ready, failure = [state[0].numpy().copy()], [], [], [], [], None
            with (args.output/f"collection_{clip_id:02d}_timing.jsonl").open("x") as telemetry:
                for frame in range(job["maximum_steps"]):
                    require(time.monotonic_ns() < args.deadline_ns-60_000_000_000, "Aggregate collection deadline")
                    reply = actor.action(frame, state[0])
                    timing = validate_causal_timing(reply["stream"], frame, timeline)
                    ready.append(timing["input_ready"])
                    require(reply["behavior_teacher_fraction"] == .5 and reply["causal_student_input"] is True
                        and reply["collection_behavior_is_causal"] is False,
                        "Do not label privileged collection behavior as causal inference")
                    telemetry.write(json.dumps(timing, allow_nan=False)+"\n")
                    canonical = np.asarray(reply["canonical_action"], dtype=np.float32)
                    label = np.asarray(reply["teacher_action"], dtype=np.float32)
                    student = np.asarray(reply["student_action"], dtype=np.float32)
                    action = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
                    require(action.shape == (1, 22) and all(v.shape == (29,) and np.isfinite(v).all()
                        for v in (canonical, label, student)), "Invalid aggregate action or teacher label")
                    state = simulator.step(action)
                    require(torch.isfinite(state).all().item(), "Nonfinite aggregate physical state")
                    actual.append(state[0].numpy().copy())
                    actions.append(canonical)
                    labels.append(label)
                    students.append(student)
                    if robot.terminated(state, reference[frame+1:frame+2]).item():
                        failure = frame+1
                        break
            traces[clip_id] = dict(actual=np.asarray(actual), canonical_action=np.asarray(actions),
                teacher_action=np.asarray(labels), student_action=np.asarray(students), input_ready=np.asarray(ready))
            path = args.output/f"collection_{clip_id:02d}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **traces[clip_id])
            clips.append(dict(clip_id=clip_id, split=job["split"], clip=job["clip"], actual_steps=len(actions),
                requested_steps=job["maximum_steps"], failure_frame=failure,
                prefix_complete=failure is None, trace=identity(path),
                timing=identity(args.output/f"collection_{clip_id:02d}_timing.jsonl")))
            save(args.output/"collection_progress.json", dict(completed_clips=len(clips),
                physical_steps=sum(row["actual_steps"] for row in clips), latest=clips[-1]))
        receipt = actor.request(dict(op="finish_data"))
        checked = {}
        for split in ("train", "validation"):
            with np.load(args.output/(split+"_data.npz"), allow_pickle=False) as values:
                checked[split] = validate_records(dict(values), split, jobs, traces)
        save(args.output/"collection_result.json", dict(status="PASSED", stage="ACTUAL_MIXED_STUDENT_STATE_LABELS",
            clips=clips, dataset_receipt=receipt, data_validation=checked, test_sources_read=False,
            teacher_behavior_fraction=.5, collection_behavior_is_causal=False,
            actual_physical_steps=sum(row["actual_steps"] for row in clips)))
        actor.timeout = max(1, (args.deadline_ns-time.monotonic_ns())/1e9)+5
        result = actor.request(dict(op="train"))
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "Aggregation runtime isolation failed")
        return result
    finally:
        if not actor.closed:
            actor.close(graceful=False)


def admission(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = train_preflight(args)
    require(args.aggregate_plan == HERE/"student_aggregate_plan_20261107.json"
        and identity(args.aggregate_plan)["sha256"] == PLAN_SHA,
        "Publish and retain the independent aggregation plan")
    required = {str(HERE/name) for name in NEW_CODE}
    require(required <= {row["path"] for row in code} and not required & {row["path"] for row in sources},
            "Aggregation source is missing or overlaps frozen inputs")
    parent = trained_causal_evidence(args.initial_r2_root, args.training_root)
    summary, saved_audit = read(args.r2_analysis_root/"summary.json"), read(args.r2_analysis_root/"audit.json")
    require(summary.get("stage") == "R2_F0_MATCHED_SAVED_REVIEW_ANALYSIS"
        and summary.get("new_saved_cases") == 7 and saved_audit.get("status") == "PASSED"
        and saved_audit.get("source_inputs_unchanged") is True
        and saved_audit.get("source_files") == len(summary["inputs"])
        and all(identity(row["path"]) == row for row in summary["inputs"]),
        "Wait for the actual initial f0 closed-loop saved analysis")
    records = controls+parent+[identity(args.aggregate_plan)]
    records += [identity(args.r2_analysis_root/name) for name in ("summary.json", "audit.json")]+summary["inputs"]
    records.append(identity(args.initial_r2_root/"cache_boundary.json"))
    unique = {}
    for row in records:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting protected aggregation input")
        unique[row["path"]] = row
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


def execute(args):
    if args.child:
        assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
        return collect(args)
    config, runtime, receipt, sources, audit, code, controls, dataset = admission(args)
    jobs = read(args.training_root/"collection_plan.json")["jobs"]
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"reused_dataset.json", dataset)
        save(args.output/"collection_plan.json", dict(jobs=jobs, test_sources_read=False))
        save(args.output/"cache_boundary_reuse.json", dict(reused=identity(args.initial_r2_root/"cache_boundary.json"),
            new_reclaim_attempted=False, reason="Single bounded R2 boundary already completed; do not repeat"))
        budget = CausalBudget(config, receipt, args.output, owner)
        budget.check()
        started_ns = time.monotonic_ns()
        reservation = reserve(args.cost_ledger, run_id=str(args.output), stage="r2", owner=owner,
            gpu=config["expected_gpu"], commit=args.evaluation_commit, started_ns=started_ns)
        budget.maximum_wall = min(1740, max(0, reservation["reserved_seconds"]-5))
        args.deadline_ns = started_ns+max(0, budget.maximum_wall-5)*1_000_000_000
        save(args.output/"authorization.json", dict(stage="R2_F0_HISTORY_ENCODER_DISTILLATION",
            method="one_round_student_state_aggregation", evaluation_commit=args.evaluation_commit,
            evaluation_sources=code, training_source_inputs=sources, controller_inputs=controls,
            gpu_lock_owner=owner, cost_reservation=reservation, maximum_wall_seconds=budget.maximum_wall,
            original_phase_protection_seconds=1800, maximum_output_bytes=1073741824,
            maximum_physical_steps=sum(row["maximum_steps"] for row in jobs), human_real_future_frames=0,
            teacher_behavior_fraction=.5, collection_behavior_is_causal=False,
            original_training_restarted=False, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R2_F0_HISTORY_ENCODER_DISTILLATION", method="one_round_student_state_aggregation",
            status="RUNNING", controller=owner, started_utc=now(), user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {value:signal.signal(value, stop) for value in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child"]
            for key in ("config", "runtime_config", "output", "training_root", "initial_r2_root", "aggregate_plan"):
                command.extend(("--"+key.replace("_", "-"), str(getattr(args, key))))
            command += ["--gpu-owner-pid", str(owner["pid"]), "--gpu-owner-ticks", owner["start_ticks"],
                        "--deadline-ns", str(args.deadline_ns)]
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"collection_fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"]["worker_return_code"] == 0, "Bounded aggregation worker did not complete")
            require(all(identity(row["path"]) == row for row in controls+sources+code), "Protected aggregation source changed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP" and fit.get("actual_updates", 0) > 0,
                    "No completed aggregate fit")
            result.update(status="TRAINED_PENDING_CLOSED_LOOP", source_integrity="PASSED",
                actual_optimizer_updates=fit["actual_updates"], fit_result=identity(args.output/"fit_result.json"),
                new_physical_transitions=read(args.output/"collection_result.json")["actual_physical_steps"],
                human_real_future_frames=0, natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if guard_path.exists() and read(guard_path).get("cleanup", {}).get("worker_reaped") is True:
                    result["cost_settlement"] = settle(args.cost_ledger, run_id=str(args.output), owner=owner,
                                                       cleanup=read(guard_path)["cleanup"])
                result["ended_utc"] = now()
                save(args.output/"result.json", result)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "training-root", "initial-r2-root", "aggregate-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("pause-receipt", "audit-root", "code-manifest", "causal-plan", "source-split", "cost-ledger",
                 "r1-analysis-root", "r2-analysis-root"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--evaluation-commit")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    parser.add_argument("--deadline-ns", type=int)
    args = parser.parse_args()
    if not args.child and not all((args.pause_receipt, args.audit_root, args.code_manifest, args.causal_plan,
            args.source_split, args.cost_ledger, args.r1_analysis_root, args.r2_analysis_root,
            args.manifest_sha256, args.evaluation_commit)):
        parser.error("Controller needs the original paused admission and actual R1/R2 evidence")
    if args.child and not all((args.gpu_owner_pid, args.gpu_owner_ticks, args.deadline_ns)):
        parser.error("Child needs its actual owning controller and bounded deadline")
    execute(args)
