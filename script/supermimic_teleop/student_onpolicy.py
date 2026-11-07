"""Bounded pure-student collection before choosing a recovery-training objective."""

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
from paused_runtime import assert_owner, own_gpu, paused_state
from preview_review import environment, require_plant
from runtime_guard import allocated_tree, guarded_run
from stream_events import EventSpec, EventTimeline
from student_causal_review import validate_causal_timing
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_onpolicy_data import collection_jobs, validate_records
from student_proprio_training import LEDGER
from student_same_state import unique
from student_training import TrainingBudget

DATA = ROOT/"data/supermimic_for_q1"
REVIEW = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_REVIEW_20261108"
ANALYSIS = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_REVIEW_ANALYSIS_20261108"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_LABELS01_20261108"
PLAN = HERE/"student_onpolicy_plan_20261108.json"
FILES = ("student_onpolicy.py", "student_onpolicy_worker.py", "student_onpolicy_data.py",
    "test_student_onpolicy.py", "student_onpolicy_plan_20261108.json")


def admission(args):
    sample = resources(True)
    config, receipt = read(args.config), read(args.pause_receipt)
    paused_state(config, receipt)
    require(args.output == OUTPUT and not OUTPUT.exists(), "Single-use on-policy root already exists")
    require(identity(args.manifest)["sha256"] == args.manifest_sha256, "New code manifest changed")
    manifest, plan = read(args.manifest), read(PLAN)
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set("0123456789abcdef")
        and manifest.get("schema") == "q1_f0_onpolicy_labels_code_v1"
        and manifest.get("evaluation_commit") == args.evaluation_commit
        and {r["path"] for r in manifest["files"]} == {str(HERE/n) for n in FILES},
        "Use only independently published on-policy sources")
    summary, audit, prior = read(ANALYSIS/"summary.json"), read(ANALYSIS/"audit.json"), read(REVIEW/"result.json")
    require(prior.get("status") == "COMPLETED" and prior.get("source_integrity") == "PASSED"
        and prior.get("cost_settlement", {}).get("state") == "SETTLED"
        and not Path("/proc", str(prior["controller"]["pid"])).exists()
        and summary.get("complete_new_cases") == 0 and summary.get("total_comparison_cases") == 28
        and audit.get("status") == "PASSED" and audit.get("source_inputs_unchanged") is True,
        "Wait for actual seven-case failure, cleanup and saved analysis")
    candidate = read(plan["candidate"]["path"])
    jobs = collection_jobs(candidate)
    require(plan.get("schema") == "q1_pure_student_shadow_labels_v1" and plan["jobs"] == jobs
        and plan["teacher_behavior_fraction"] == 0. and plan["maximum_physical_steps"] == 1699
        and plan["maximum_teacher_queries"] == 850 and plan["maximum_policy_forwards"] == 2549
        and plan["maximum_wall_seconds"] == 600 and plan["maximum_output_bytes"] == 128*(1 << 20)
        and plan["new_optimizer_updates"] == 0 and plan["heldout_events_opened"] is False,
        "Collection schedule, behavior or bounds changed")
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt, args.manifest,
        PLAN, ANALYSIS/"summary.json", ANALYSIS/"audit.json", REVIEW/"checkpoint_audit.json")]
    protected = unique(summary["inputs"]+controls+manifest["files"]+[plan["candidate"], plan["selected_checkpoint"]])
    require(not {str(HERE/n) for n in FILES}&{r["path"] for r in summary["inputs"]}
        and str(LEDGER) not in {r["path"] for r in protected}
        and all(identity(r["path"]) == r for r in protected), "Frozen or new sources changed")
    ledger = checked(read(LEDGER))
    r2 = charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == "r2"]))
    require(charged(ledger) == 4256 and r2 == 1271
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r2"]-r2 >= 600 and charged(ledger)+600 <= LIMIT,
        "Preserve the settled4256 ledger and remaining r2 budget")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+128*(1 << 20) < 500_000_000_000,
        "Project500GB bound reached")
    return sample, config, receipt, plan, protected


class CollectionBudget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)["unique_allocated_bytes"] < 128*(1 << 20), "Output128MiB bound")
        return super().check()


def collect(args):
    from gpu_preview_worker import GpuPolicyProcess
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    config, runtime, experiment = read(args.config), read(args.runtime_config), read(PLAN)
    auth = read(args.output/"authorization.json")
    require(all(identity(r["path"]) == r for r in auth["protected_inputs"]), "Collection sources changed")
    torch = require_plant(runtime)
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator
    selected, baseline = checkpoint(args, config)
    robot, settings = load_physical_robot(baseline["path"])
    original, frozen = read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"])
    jobs = experiment["jobs"]
    rows = {Path(r["q1_primebot"]["path"]).stem:r for r in frozen["rows"]}
    command = gpu_command(args, config, runtime, original, selected, baseline, 0, "masked")
    command[2] = str(HERE/"student_onpolicy_worker.py")
    command += ["--student-checkpoint", experiment["selected_checkpoint"]["path"], "--output", str(args.output),
        "--deadline-ns", str(args.deadline_ns)]
    actor = GpuPolicyProcess(command, args.output/"native_worker.log", config["expected_gpu"])
    clips, traces, queries, policy_calls = [], {}, 0, 0
    try:
        require(actor.ready["student_checkpoint"] == experiment["selected_checkpoint"]
            and actor.ready["selected_update"] == 3000 and actor.ready["checkpoint"] == selected,
            "Collector loaded a different actual student or teacher")
        for clip_id, job in enumerate(jobs):
            item = rows[job["clip"]]
            require(all(item["human"][k] == job["human"][k] for k in ("bytes", "sha256")), "Source pairing differs")
            reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
            started = actor.request(dict(op="start", clip=job["clip"], seed=0, interface="human", clip_id=clip_id))
            timeline = EventTimeline(started["paired_reference"]["frames"], EventSpec(**job["packet_event"]))
            simulator = Q1Simulator(robot, 1, "cpu", settings["action_scale"], use_tn=settings.get("use_tn", False),
                physics_timestep=settings.get("physics_timestep"))
            simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
            state = simulator.state()
            actual, actions, labels, ready, queried, failure = [state[0].numpy().copy()], [], [], [], [], None
            with (args.output/f"collection_{clip_id:02d}_timing.jsonl").open("x") as telemetry:
                for frame in range(job["maximum_steps"]):
                    require(time.monotonic_ns() < args.deadline_ns-30_000_000_000, "Collection wall bound")
                    reply = actor.action(frame, state[0])
                    timing = validate_causal_timing(reply["stream"], frame, timeline)
                    require(reply["teacher_behavior_fraction"] == 0. and reply["causal_student_input"] is True
                        and reply["teacher_queried"] == (frame % 2 == 0 and timing["input_ready"]),
                        "Shadow teacher affected behavior or query schedule")
                    telemetry.write(json.dumps(timing, allow_nan=False)+"\n")
                    action = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
                    canonical = np.asarray(reply["canonical_action"], dtype=np.float32)
                    label = np.asarray(reply["teacher_action"] if reply["teacher_queried"] else np.zeros(29), dtype=np.float32)
                    require(action.shape == (1, 22) and canonical.shape == label.shape == (29,)
                        and torch.isfinite(action).all().item() and np.isfinite(canonical).all()
                        and np.isfinite(label).all(), "Invalid pure-student action or teacher query")
                    state = simulator.step(action)
                    require(torch.isfinite(state).all().item(), "Nonfinite on-policy state")
                    actual.append(state[0].numpy().copy()); actions.append(canonical); labels.append(label)
                    ready.append(timing["input_ready"]); queried.append(reply["teacher_queried"])
                    policy_calls += int(timing["input_ready"]); queries += int(reply["teacher_queried"])
                    if robot.terminated(state, reference[frame+1:frame+2]).item():
                        failure = frame+1; break
            traces[clip_id] = dict(actual=np.asarray(actual), canonical_action=np.asarray(actions),
                teacher_action=np.asarray(labels), input_ready=np.asarray(ready), teacher_queried=np.asarray(queried),
                reference_frames=np.int64(timeline.frames), failure_frame=np.int64(-1 if failure is None else failure))
            path = args.output/f"collection_{clip_id:02d}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **traces[clip_id])
            clips.append(dict(clip_id=clip_id, split=job["split"], clip=job["clip"], actual_steps=len(actions),
                requested_steps=job["maximum_steps"], failure_frame=failure, prefix_complete=failure is None,
                trace=identity(path), timing=identity(args.output/f"collection_{clip_id:02d}_timing.jsonl")))
            save(args.output/"collection_progress.json", dict(completed_clips=len(clips), latest=clips[-1],
                physical_steps=sum(r["actual_steps"] for r in clips), student_policy_forwards=policy_calls,
                teacher_queries=queries))
        actor.request(dict(op="finish_data"))
        checks, transitions = {}, {}
        for split in ("train", "validation"):
            with np.load(args.output/(split+"_data.npz"), allow_pickle=False) as source:
                values = dict(source)
            next_states, checks[split] = validate_records(values, split, jobs, traces)
            path = args.output/(split+"_transitions.npz")
            with path.open("xb") as stream:
                terminal = np.asarray([traces[int(c)]["failure_frame"] == int(f)+1
                    for c, f in zip(values["clip_id"], values["frame"])], dtype=bool)
                np.savez(stream, **values, next_state=next_states, terminal_after_action=terminal)
            transitions[split] = identity(path)
        require(queries <= 850 and policy_calls+queries <= 2549, "New forward budget exceeded")
        save(args.output/"collection_result.json", dict(status="PURE_STUDENT_SHADOW_LABELS_PASSED", clips=clips,
            data_validation=checks, transitions=transitions, student_policy_forwards=policy_calls,
            teacher_queries=queries, actual_physical_steps=sum(r["actual_steps"] for r in clips),
            teacher_behavior_fraction=0., new_optimizer_updates=0, test_sources_read=False,
            teacher_recovery_ability_verified=False, natural_teleoperation_qualified=False))
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "Worker cleanup or runtime isolation failed")
    finally:
        actor.close(graceful=False)


def execute(args):
    if args.child:
        return collect(args)
    sample, config, receipt, plan, protected = admission(args)
    if args.admit_only:
        print(json.dumps(dict(status="PURE_STUDENT_COLLECTION_ADMITTED", protected_files=len(protected), resources=sample)))
        return
    runtime = read(args.runtime_config)
    with own_gpu() as lock:
        owner = dict(lock, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", read(REVIEW/"checkpoint_audit.json"))
        save(args.output/"collection_plan.json", dict(jobs=plan["jobs"], test_sources_read=False))
        budget = CollectionBudget(config, receipt, args.output, owner); budget.maximum_wall = 600
        previous, requested, reservation = dict(os.environ), False, None
        result = dict(stage="F0_PURE_STUDENT_SHADOW_LABEL_COLLECTION", status="RUNNING", controller=owner,
            started_utc=now(), new_optimizer_updates=0, user_accepted_checkpoint=None)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"collection.guard.json"
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage="r2", owner=owner,
                gpu=config["expected_gpu"], commit=args.evaluation_commit)
            save(args.output/"authorization.json", dict(stage=result["stage"], evaluation_commit=args.evaluation_commit,
                protected_inputs=protected, selected_checkpoint=plan["selected_checkpoint"],
                gpu_lock_owner=owner, maximum_wall_seconds=600, maximum_output_bytes=128*(1 << 20),
                maximum_physical_steps=1699, maximum_policy_forwards=2549, teacher_behavior_fraction=0.,
                new_optimizer_updates=0, heldout_events_opened=False, user_accepted_checkpoint=None))
            command = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child",
                "--config", str(args.config), "--runtime-config", str(args.runtime_config), "--output", str(args.output),
                "--gpu-owner-pid", str(owner["pid"]), "--gpu-owner-ticks", owner["start_ticks"],
                "--deadline-ns", str(time.monotonic_ns()+590_000_000_000)]
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"collection.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"]["worker_return_code"] == 0, "Pure-student collection failed")
            collection = read(args.output/"collection_result.json")
            require(collection["status"] == "PURE_STUDENT_SHADOW_LABELS_PASSED"
                and len(collection["clips"]) == 9 and all(identity(r["path"]) == r for r in protected),
                "Collection terminal or source gate failed")
            result.update(status="COMPLETED", source_integrity="PASSED", collection=identity(args.output/"collection_result.json"),
                new_physical_transitions=collection["actual_physical_steps"], new_rendered_frames=0,
                student_policy_forwards=collection["student_policy_forwards"], teacher_queries=collection["teacher_queries"])
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}"); raise
        finally:
            if reservation and guard_path.exists() and read(guard_path).get("cleanup", {}).get("worker_reaped") is True:
                result["cost_settlement"] = settle(LEDGER, run_id=str(args.output), owner=owner, cleanup=read(guard_path)["cleanup"])
            result.update(ended_utc=now(), cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json", result)
            os.environ.clear(); os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, default in dict(config=ROOT/"script/q1_five_stage/conditional_shared_v04_20261104.json",
            runtime_config=ROOT/"data/manifests/primebot_q1_monitored_config_20261103.json",
            pause_receipt=ROOT/"logs/session_records/q1_shared_user_pause_20261106.json", output=OUTPUT).items():
        p.add_argument("--"+name.replace("_", "-"), type=Path, default=default)
    for name in ("manifest",):
        p.add_argument("--"+name, type=Path)
    for name in ("manifest-sha256", "evaluation-commit", "gpu-owner-ticks"):
        p.add_argument("--"+name)
    for name in ("gpu-owner-pid", "deadline-ns"):
        p.add_argument("--"+name, type=int)
    p.add_argument("--child", action="store_true")
    p.add_argument("--admit-only", action="store_true")
    args = p.parse_args()
    if args.child and not all((args.gpu_owner_pid, args.gpu_owner_ticks, args.deadline_ns)):
        p.error("Child requires its actual owner and deadline")
    if not args.child and not all((args.manifest, args.manifest_sha256, args.evaluation_commit)):
        p.error("Controller requires published source identity")
    execute(args)
