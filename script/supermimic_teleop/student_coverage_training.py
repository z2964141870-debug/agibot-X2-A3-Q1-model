"""Collect only registered new sources and fit under the original GPU ledger."""

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
from student_auxiliary_training import aggregate_evidence
from student_causal_review import trained_causal_evidence
from student_causal_training import CausalBudget, train_preflight
from student_cost import reserve, settle
from student_coverage_data import validate_records
from student_coverage_plan import build
from student_rollouts import collection_jobs, paired_source

NEW_CODE = ("student_coverage_training.py", "student_coverage_worker.py", "student_coverage_fit.py",
            "student_coverage_data.py", "student_coverage_plan.py", "student_input_audit.py")


def coverage_preflight(args):
    config, runtime, receipt, sources, audit, code, controls, reused = train_preflight(args)
    settings, candidate = read(args.coverage_plan), read(args.candidate_plan)
    base = ROOT/"data/supermimic_for_q1"
    require(args.coverage_plan == HERE/"student_coverage_experiment_20261107.json"
        and identity(args.coverage_plan)["sha256"] == args.coverage_plan_sha256
        and args.candidate_plan == ROOT/"data/manifests/q1_student_coverage_candidate_20261107.json"
        and identity(args.candidate_plan)["sha256"] == settings["candidate_sha256"]
        and candidate == build(read(args.source_split), collection_jobs(read(args.source_split)))
        and settings.get("schema") == "q1_owned_source_coverage_experiment_v1"
        and settings.get("maximum_new_physical_steps") == candidate["maximum_new_physical_steps"] == 9585
        and settings.get("maximum_new_records") == candidate["maximum_planned_new_records"] == 4795
        and len(candidate["new_collection_jobs"]) == 30
        and settings.get("normalization_refitted") is False and settings.get("test_sources_read") is False
        and settings.get("collection_behavior") == "privileged_full_teacher_only"
        and settings["fit"]["seed"] == 2026110704 and settings["fit"]["maximum_updates"] == 3000
        and settings["fit"]["learning_rate"] == .0001 and settings["fit"]["batch_size"] == 128,
        "Use the exact registered coverage candidate and fitting plan")
    require(args.initial_r2_root == base/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107"
        and args.aggregate_root == base/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_AGGREGATE01_20261107"
        and args.input_audit_root == base/"Q1_LOW_LATENCY_CKPT32000_INPUT_AUDIT01_20261107",
        "Coverage must reuse the actual audited original datasets")
    path = args.input_audit_root/"input_audit.json"
    checked = read(path)
    require(identity(path)["sha256"] == settings["input_audit_sha256"]
        and checked.get("status") == checked.get("source_integrity") == "PASSED"
        and checked.get("new_model_inference_calls") == checked.get("new_physical_transitions") == 0
        and checked.get("normalization_refitted") is False,
        "Require the real completed saved-input audit, not its plan")
    required = {str(HERE/n) for n in NEW_CODE}
    require(required <= {r["path"] for r in code} and not required & {r["path"] for r in sources},
        "Independent coverage sources missing or overlap frozen training")
    controls += trained_causal_evidence(args.initial_r2_root, args.training_root)+aggregate_evidence(args.aggregate_root)
    controls += [identity(p) for p in (args.coverage_plan, args.candidate_plan, path)]+checked["source_inputs"]
    unique = {}
    for row in controls:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting coverage source identities")
        unique[row["path"]] = row
    require(all(identity(r["path"]) == r for r in unique.values()), "Audited coverage inputs changed")
    return config, runtime, receipt, sources, audit, code, list(unique.values()), reused


def validate_visibility(timing, frame):
    from stream_contract import DT_NS
    control = frame*DT_NS
    targets = [i*DT_NS for i in range(frame-9, frame+1)]
    require(timing.get("frame") == frame and timing.get("anchor_ns") == control
        and timing.get("human_real_future_frames") == timing.get("future_preview_ns") == 0
        and timing.get("requested_history_ns") == targets
        and timing.get("collection_behavior_is_causal") is False
        and timing.get("measured_end_to_end_latency_ns") is None,
        "Coverage labels and current/history visibility scopes changed")
    captures, arrivals = timing.get("used_capture_ns", []), timing.get("used_arrival_ns", [])
    require(len(captures) == len(arrivals) == 10 and all(
        (capture is None and arrival is None) or
        (type(capture) is int and type(arrival) is int and capture <= target and arrival <= control)
        for capture, arrival, target in zip(captures, arrivals, targets)), "Future packet entered coverage history")
    require(timing.get("max_used_dependency_ns") is None or timing["max_used_dependency_ns"] <= control,
        "Future resampling dependency entered coverage history")


def collect(args):
    from gpu_preview_worker import GpuPolicyProcess
    config, runtime, candidate = read(args.config), read(args.runtime_config), read(args.candidate_plan)
    torch = require_plant(runtime)
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator
    plan, frozen = read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"])
    selected, baseline = checkpoint(args, config)
    robot, physical = load_physical_robot(baseline["path"])
    rows = {Path(r["q1_primebot"]["path"]).stem:r for r in frozen["rows"]}
    command = gpu_command(args, config, runtime, plan, selected, baseline, 10, "masked")
    command[2] = str(HERE/"student_coverage_worker.py")
    for key in ("output", "coverage_plan", "candidate_plan", "training_root", "aggregate_root", "deadline_ns"):
        command.extend(("--"+key.replace("_", "-"), str(getattr(args, key))))
    command.extend(("--student-checkpoint", str(args.initial_r2_root/"selected.pt")))
    actor = GpuPolicyProcess(command, args.output/"native_worker.log", config["expected_gpu"])
    traces, clips = {}, []
    jobs = candidate["new_collection_jobs"]
    try:
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == 32000,
            "Coverage loaded another teacher")
        for clip_id, job in enumerate(jobs):
            item = rows[job["clip"]]
            require(paired_source(item, job), "New coverage source differs from its frozen paired human")
            reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
            actor.request(dict(op="start", clip=job["clip"], seed=0, interface="human", clip_id=clip_id))
            simulator = Q1Simulator(robot, 1, "cpu", physical["action_scale"],
                use_tn=physical.get("use_tn", False), physics_timestep=physical.get("physics_timestep"))
            simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
            state = simulator.state()
            actual, actions, ready, failure = [state[0].numpy().copy()], [], [], None
            with (args.output/f"collection_{clip_id:02d}_timing.jsonl").open("x") as telemetry:
                for frame in range(job["maximum_steps"]):
                    require(time.monotonic_ns() < args.deadline_ns-120_000_000_000, "Coverage reserved fit deadline")
                    reply = actor.action(frame, state[0])
                    validate_visibility(reply["stream"], frame)
                    telemetry.write(json.dumps(reply["stream"], allow_nan=False)+"\n")
                    ready.append(reply["stream"]["input_ready"])
                    canonical = np.asarray(reply["canonical_action"], dtype=np.float32)
                    action = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
                    require(action.shape == (1, 22) and canonical.shape == (29,) and np.isfinite(canonical).all(),
                        "Invalid new coverage action")
                    state = simulator.step(action)
                    require(torch.isfinite(state).all().item(), "Nonfinite new physical coverage state")
                    actual.append(state[0].numpy().copy())
                    actions.append(canonical)
                    if robot.terminated(state, reference[frame+1:frame+2]).item():
                        failure = frame+1
                        break
            traces[clip_id] = dict(actual=np.asarray(actual), canonical_action=np.asarray(actions), input_ready=np.asarray(ready))
            path = args.output/f"collection_{clip_id:02d}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **traces[clip_id])
            clips.append(dict(clip_id=clip_id, clip=job["clip"], split=job["split"], actual_steps=len(actions),
                requested_steps=job["maximum_steps"], failure_frame=failure, prefix_complete=failure is None,
                trace=identity(path), timing=identity(args.output/f"collection_{clip_id:02d}_timing.jsonl")))
            save(args.output/"collection_progress.json", dict(completed_clips=len(clips),
                physical_steps=sum(r["actual_steps"] for r in clips), latest=clips[-1]))
        receipt = actor.request(dict(op="finish_data"))
        checked = {}
        for split in ("train", "validation"):
            with np.load(args.output/(split+"_data.npz"), allow_pickle=False) as arrays:
                checked[split] = validate_records(dict(arrays), split, jobs, traces)
        save(args.output/"collection_result.json", dict(status="PASSED", clips=clips,
            dataset_receipt=receipt, data_validation=checked, test_sources_read=False,
            collection_behavior_is_causal=False, collection_behavior="privileged_full_teacher_only",
            actual_physical_steps=sum(r["actual_steps"] for r in clips), original_prefixes_replayed=False))
        actor.timeout = max(1, (args.deadline_ns-time.monotonic_ns())/1e9)+5
        result = actor.request(dict(op="train"))
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "Coverage runtime isolation failed")
        return result
    finally:
        if not actor.closed:
            actor.close(graceful=False)


def execute(args):
    if args.child:
        assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
        return collect(args)
    config, runtime, receipt, sources, audit, code, controls, reused = coverage_preflight(args)
    candidate = read(args.candidate_plan)
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"reused_dataset.json", reused)
        save(args.output/"collection_plan.json", dict(jobs=candidate["new_collection_jobs"], test_sources_read=False))
        budget = CausalBudget(config, receipt, args.output, owner)
        budget.check()
        started = time.monotonic_ns()
        reservation = reserve(args.cost_ledger, run_id=str(args.output), stage="r3", owner=owner,
            gpu=config["expected_gpu"], commit=args.evaluation_commit, started_ns=started)
        budget.maximum_wall = min(1740, max(0, reservation["reserved_seconds"]-5))
        args.deadline_ns = started+max(0, budget.maximum_wall-5)*1_000_000_000
        save(args.output/"authorization.json", dict(stage="R3_OWNED_SOURCE_COVERAGE",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code, training_source_inputs=sources,
            controller_inputs=controls, gpu_lock_owner=owner, cost_reservation=reservation,
            maximum_wall_seconds=budget.maximum_wall, original_phase_protection_seconds=1800,
            maximum_output_bytes=1073741824, maximum_new_physical_steps=9585, maximum_new_records=4795,
            human_real_future_frames=0, collection_behavior_is_causal=False, original_prefixes_replayed=False,
            original_training_restarted=False, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R3_OWNED_SOURCE_COVERAGE", status="RUNNING", controller=owner,
            started_utc=now(), user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {v:signal.signal(v, stop) for v in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child"]
            for key in ("config", "runtime_config", "output", "training_root", "initial_r2_root", "aggregate_root",
                    "coverage_plan", "candidate_plan", "deadline_ns"):
                command.extend(("--"+key.replace("_", "-"), str(getattr(args, key))))
            command += ["--gpu-owner-pid", str(owner["pid"]), "--gpu-owner-ticks", owner["start_ticks"]]
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"collection_fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"]["worker_return_code"] == 0, "Bounded coverage worker did not complete")
            require(all(identity(r["path"]) == r for r in controls+sources+code), "Protected coverage input changed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP" and 0 < fit.get("actual_updates", 0) <= 3000,
                "No completed bounded coverage fit")
            result.update(status=fit["status"], source_integrity="PASSED", fit_result=identity(args.output/"fit_result.json"),
                actual_optimizer_updates=fit["actual_updates"],
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
    for name in ("config", "runtime-config", "output", "training-root", "initial-r2-root", "aggregate-root",
            "coverage-plan", "candidate-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("pause-receipt", "audit-root", "code-manifest", "causal-plan", "source-split", "cost-ledger",
            "r1-analysis-root", "input-audit-root"):
        parser.add_argument("--"+name, type=Path)
    for name in ("manifest-sha256", "evaluation-commit", "coverage-plan-sha256", "gpu-owner-ticks"):
        parser.add_argument("--"+name)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--deadline-ns", type=int)
    args = parser.parse_args()
    if not args.child and not all((args.pause_receipt, args.audit_root, args.code_manifest, args.causal_plan,
            args.source_split, args.cost_ledger, args.r1_analysis_root, args.input_audit_root,
            args.manifest_sha256, args.evaluation_commit, args.coverage_plan_sha256)):
        parser.error("Controller requires original pause, sources, cost ledger and actual input-audit evidence")
    if args.child and not all((args.gpu_owner_pid, args.gpu_owner_ticks, args.deadline_ns)):
        parser.error("Child requires owning controller identity and a bounded deadline")
    execute(args)
