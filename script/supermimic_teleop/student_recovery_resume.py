"""Independent continuation after the preserved CPU package-import failure."""

import argparse
import ast
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
from student_causal_review import validate_causal_timing
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_recovery_state import checked_state, fixed_jobs, restore, snapshot
from student_same_state import unique
from student_training import TrainingBudget
from stream_events import EventSpec, EventTimeline

DATA = ROOT/"data/supermimic_for_q1"
COLLECTION = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_LABELS01_20261108"
PAIR = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_PAIR01_20261108"
FAILED = DATA/"Q1_LOW_LATENCY_CKPT32000_TEACHER_RECOVERY01_20261108"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_TEACHER_RECOVERY01_RESUME1_20261108"
FAILURE_RECORD = ROOT/"logs/session_records/q1_teacher_recovery_import_failed_20261108.json"
PLAN = HERE/"student_recovery_plan_20261108.json"
LEDGER = ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json"
FILES = ("student_recovery.py", "student_recovery_worker.py", "student_recovery_state.py",
    "student_recovery_plan_20261108.json", "test_student_recovery.py",
    "student_recovery_resume.py", "test_student_recovery_resume.py")
WALL, OUTPUT_BYTES = 886, 128*(1 << 20)


def canonical_joint_names(text):
    tree = ast.parse(text)
    values = {n.target.id:n.value for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    require("_JOINT_STEMS" in values and "JOINT_NAMES" in values, "Missing frozen joint declarations")
    expected = ast.parse('tuple(f"{j}_joint" for j in _JOINT_STEMS)', mode="eval").body
    require(ast.dump(values["JOINT_NAMES"]) == ast.dump(expected), "Frozen canonical-name expression changed")
    stems = ast.literal_eval(values["_JOINT_STEMS"])
    require(type(stems) is tuple and len(stems) == len(set(stems)) == 29
        and all(isinstance(v, str) and v.isidentifier() for v in stems), "Invalid literal canonical stems")
    return tuple(v+"_joint" for v in stems)


def admission(args):
    sample = resources(True)
    config, receipt = read(args.config), read(args.pause_receipt)
    paused_state(config, receipt)
    runtime = read(args.runtime_config)
    names = canonical_joint_names((Path(runtime["morph_source"])/"src/morphmimic/constants.py").read_text())
    require(len(names) == 29 and "morphmimic" not in sys.modules, "CPU admission must not initialize the Morph package")
    require(args.output == OUTPUT and not OUTPUT.exists(), "Use the single-use independent recovery root")
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set("0123456789abcdef")
        and identity(args.manifest)["sha256"] == args.manifest_sha256, "Require independently published sources")
    manifest, plan = read(args.manifest), read(PLAN)
    require(manifest.get("schema") == "q1_teacher_recovery_resume_code_v1"
        and manifest.get("evaluation_commit") == args.evaluation_commit
        and len(manifest["files"]) == len(FILES)
        and {r["path"] for r in manifest["files"]} == {str(HERE/n) for n in FILES}, "Recovery source scope differs")
    failure = read(FAILURE_RECORD)
    require(identity(FAILURE_RECORD) == manifest["preserved_failure"]
        and failure.get("status") == "FAILED_PRESERVED"
        and failure.get("conservative_spent_seconds") == 14 and failure.get("remaining_original_wall_seconds") == WALL
        and failure.get("actual_new_q1_physical_steps") == failure.get("actual_new_model_forwards") == 0
        and all(identity(r["path"]) == r for r in failure["files"]),
        "Preserve the diagnosed failure, zero physical work and original900s remainder")
    previous, auth = read(PAIR/"result.json"), read(PAIR/"authorization.json")
    guard, fit = read(PAIR/"fit.guard.json"), read(PAIR/"fit_result.json")
    collection, original = read(COLLECTION/"collection_result.json"), read(HERE/"student_onpolicy_plan_20261108.json")
    require(previous.get("status") == "ONPOLICY_PAIR_TRAINED_PENDING_CLOSED_LOOP"
        and previous.get("source_integrity") == "PASSED" and previous.get("cost_settlement", {}).get("state") == "SETTLED"
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"]["worker_return_code"] == 0
        and all(not Path("/proc", str(pid)).exists() for pid in
            (previous["controller"]["pid"], guard["cleanup"]["worker_pid"]))
        and all(a["selected_update"] == 0 and a["selected_adapter_unchanged"] for a in fit["arms"]),
        "Require completed rejected paired training and actual cleanup")
    require(plan.get("schema") == "q1_teacher_recovery_plan_v1"
        and plan["collection_root"] == str(COLLECTION) and plan["jobs"] == fixed_jobs(original["jobs"])
        and plan["selected_checkpoint"] == original["selected_checkpoint"]
        and plan["maximum_physical_steps"] == 1800 and plan["maximum_policy_forwards"] == 1280
        and plan["maximum_wall_seconds"] == 900 and plan["maximum_output_bytes"] == OUTPUT_BYTES
        and plan["new_optimizer_updates"] == 0 and plan["test_sources_read"] is False,
        "Fixed anchors, original student, bounds or source ownership changed")
    generated = [identity(PAIR/n) for n in ("result.json", "fit.guard.json", "fit_result.json", "authorization.json")]
    generated += [identity(COLLECTION/n) for n in ("result.json", "collection_result.json", "collection.guard.json")]
    generated += [r[k] for r in collection["clips"] for k in ("trace", "timing")]
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt, args.manifest, PLAN)]
    protected = unique(auth["protected_inputs"]+read(FAILED/"authorization.json")["protected_inputs"]
        +failure["files"]+[identity(FAILURE_RECORD)]+generated+controls+manifest["files"]+[plan["selected_checkpoint"]])
    require(not {str(HERE/n) for n in FILES}&{r["path"] for r in auth["protected_inputs"]}
        and str(LEDGER) not in {r["path"] for r in protected}
        and all(identity(r["path"]) == r for r in protected), "Old protected or new execution source changed")
    ledger = checked(read(LEDGER))
    by_stage = {s:charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == s])) for s in STAGES}
    require(charged(ledger) == 4507 and by_stage == dict(r1=1614, r2=1522, r3=1371)
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r2"]-by_stage["r2"] >= WALL and charged(ledger)+WALL <= LIMIT, "Preserve the settled4507 ledger and14s failure allowance")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+OUTPUT_BYTES < 500_000_000_000, "Project500GB bound")
    with own_gpu() as lock:
        require(lock is not None, "Original nonblocking GPU lock unavailable")
    return sample, protected, plan


def probe(args):
    from gpu_preview_worker import GpuPolicyProcess
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    config, runtime, plan = read(args.config), read(args.runtime_config), read(PLAN)
    auth = read(args.output/"authorization.json")
    require(all(identity(r["path"]) == r for r in auth["protected_inputs"]), "Fork inputs changed")
    torch = require_plant(runtime)
    sys.path[:0] = [str(Path(runtime["morph_source"])/"src"), str(HERE.parent/"morphmimic_q1")]
    JOINT_NAMES = canonical_joint_names((Path(runtime["morph_source"])/"src/morphmimic/constants.py").read_text())
    from canonical import SparseJointMap
    from evaluate import diagnostics, loads
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator
    selected, baseline = checkpoint(args, config)
    robot, settings = load_physical_robot(baseline["path"])
    mapping = SparseJointMap.q1(JOINT_NAMES, robot.joint_names)
    original, frozen = read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"])
    rows = {Path(r["q1_primebot"]["path"]).stem:r for r in frozen["rows"]}
    jobs = read(HERE/"student_onpolicy_plan_20261108.json")["jobs"]
    def plant(reference):
        value = Q1Simulator(robot, 1, "cpu", settings["action_scale"], use_tn=settings.get("use_tn", False),
            physics_timestep=settings.get("physics_timestep"))
        value.reset(torch.tensor([0]), reference[:1], randomize=False)
        return value
    steps, calls, cases, gates = 0, 0, [], []
    def step(simulator, action):
        nonlocal steps
        require(steps < 1800 and time.monotonic_ns() < args.deadline_ns-30_000_000_000, "New physics or wall bound")
        steps += 1
        return simulator.step(action)
    command = gpu_command(args, config, runtime, original, selected, baseline, 0, "masked")
    command[2] = str(HERE/"student_recovery_worker.py")
    command += ["--student-checkpoint", plan["selected_checkpoint"]["path"], "--recovery-plan", str(PLAN),
        "--deadline-ns", str(args.deadline_ns)]
    actor = GpuPolicyProcess(command, args.output/"native_worker.log", config["expected_gpu"])
    try:
        require(actor.ready["student_checkpoint"] == plan["selected_checkpoint"] and actor.ready["selected_update"] == 3000,
            "Recovery loaded a different selected policy")
        for clip_id in (0, 2, 6, 8):
            source = jobs[clip_id]
            with np.load(COLLECTION/f"collection_{clip_id:02d}.npz", allow_pickle=False) as data:
                saved = {k:data[k].copy() for k in ("actual", "canonical_action", "input_ready")}
            reference = Q1Motions(robot, [Path(rows[source["clip"]]["q1_primebot"]["path"])], device="cpu").load(0)
            simulator = plant(reference)
            prefix_actions = mapping.native(torch.tensor(saved["canonical_action"]), delta=True)
            present = {job["anchor"] for job in plan["jobs"] if job["clip_id"] == clip_id
                and job["anchor"] < len(prefix_actions) and bool(saved["input_ready"][job["anchor"]])}
            captures, following, maximum = {}, {}, checked_state(simulator.state()[0].numpy(), saved["actual"][0])
            for frame in range(max(present, default=-1)+1):
                if frame in present:
                    captures[frame] = snapshot(simulator)
                state = step(simulator, prefix_actions[frame:frame+1])
                maximum = max(maximum, checked_state(state[0].numpy(), saved["actual"][frame+1]))
                if frame in present:
                    following[frame] = state.clone()
            for index, job in enumerate(plan["jobs"]):
                if job["clip_id"] != clip_id:
                    continue
                anchor = job["anchor"]
                if anchor not in captures:
                    gates.append(dict(job_index=index, status="ABSENT_OR_UNAVAILABLE_FIXED_ANCHOR", substituted=False))
                    continue
                snap = captures[anchor]
                restore(simulator, snap)
                replayed = step(simulator, prefix_actions[anchor:anchor+1])
                require(torch.equal(replayed, following[anchor]), "Full snapshot same-action next state is not bitwise identical")
                reset_errors = {}
                for mode in ("reset_only", "reset_and_action_memory"):
                    reset = plant(reference)
                    reset.reset(torch.tensor([0]), torch.tensor(saved["actual"][anchor:anchor+1]), randomize=False)
                    if mode == "reset_and_action_memory":
                        reset.last_action.copy_(torch.tensor(saved["actual"][anchor:anchor+1, 57:79]))
                        if robot.modeled_action_delay:
                            reset.context[:, reset.layout.previous_executed_action].copy_(reset.last_action)
                    before = float(np.max(np.abs(reset.state()[0].numpy()-saved["actual"][anchor])))
                    after = step(reset, prefix_actions[anchor:anchor+1])[0].numpy()
                    reset_errors[mode] = dict(observation_max_abs_difference=before,
                        next_state_max_abs_difference=float(np.max(np.abs(after-saved["actual"][anchor+1]))))
                gate = dict(job_index=index, clip=job["clip"], split=job["split"], anchor=anchor,
                    status="PREFIX_AND_FULL_SNAPSHOT_PASSED", prefix_max_abs_difference=maximum,
                    full_snapshot_next_state_bitwise_equal=True, state_only_diagnostic=reset_errors)
                gates.append(gate)
                save(args.output/"restoration_gates.json", gates)
                priming = []
                for arm in job["arm_order"]:
                    state = restore(simulator, snap)
                    started = actor.request(dict(op="prepare", job_index=index, arm=arm))
                    priming.append(started["history_sha256"])
                    require(started["priming_model_forwards"] == 0, "Priming must not execute a new policy")
                    timeline = EventTimeline(started["paired_reference"]["frames"], EventSpec(**source["packet_event"]))
                    cap = min(80, len(reference)-10-anchor)
                    require(cap > 0, "No same-clip privileged future support")
                    actual, force, actions, streams = [state[0].numpy().copy()], [loads(simulator)], [], []
                    failure, first_difference = None, None
                    for frame in range(anchor, anchor+cap):
                        reply = actor.action(frame, state[0])
                        if arm == "student_f0":
                            validate_causal_timing(reply["stream"], frame, timeline)
                            require(reply["training_only_privileged_reference"] is False, "Student received privileged teacher input")
                        else:
                            require(reply["stream"]["human_real_future_frames"] == 10
                                and reply["training_only_privileged_reference"] is True, "Teacher privilege must be explicit")
                        if frame == anchor:
                            first_difference = reply["anchor_action_max_abs_difference"]
                        action = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
                        canonical = np.asarray(reply["canonical_action"], dtype=np.float32)
                        require(np.array_equal(mapping.native(torch.tensor(canonical), delta=True).numpy(), action[0].numpy()),
                            "Canonical to physical action map differs")
                        state = step(simulator, action)
                        actual.append(state[0].numpy().copy()); force.append(loads(simulator))
                        actions.append(canonical); streams.append(reply["stream"])
                        calls = reply["cumulative_model_forwards"]
                        if robot.terminated(state, reference[frame+1:frame+2]).item():
                            failure = frame+1; break
                    actual, force, actions = np.asarray(actual), np.asarray(force), np.asarray(actions)
                    target = reference[anchor:anchor+len(actual)].numpy()
                    body, body_ref, height, slip, tilt, target_height, diagnosis = diagnostics(robot, actual, target, force)
                    directory = args.output/f"fork_{index:02d}_{arm}"
                    directory.mkdir(exist_ok=False)
                    with (directory/"rollout.npz").open("xb") as output:
                        np.savez(output, actual=actual, reference=target, load=force, canonical_action=actions,
                            body_actual=body, body_reference=body_ref, sole_height=height,
                            material_point_slip=slip, tilt=tilt, reference_sole_height=target_height)
                    save(directory/"timing.json", streams)
                    row = dict(job_index=index, clip=job["clip"], split=job["split"], anchor=anchor, arm=arm,
                        maximum_steps=80, supported_steps=cap, attempted_steps=len(actions), failure_frame=failure,
                        complete_short_fork=failure is None and len(actions) == cap,
                        full80_steps_supported=cap == 80, diagnostics=diagnosis,
                        diagnostics_scope="attempted short fork including termination; never a full-clip metric",
                        anchor_action_max_abs_difference=first_difference,
                        action_delta_l2_p95=float(np.percentile(np.linalg.norm(np.diff(np.vstack((saved["canonical_action"][anchor-1], actions)), axis=0), axis=1), 95)),
                        trace=identity(directory/"rollout.npz"), timing=identity(directory/"timing.json"),
                        privileged_teacher=arm == "privileged_teacher_h10", new_optimizer_updates=0)
                    valid = len(actions) if failure is None else max(0, len(actions)-1)
                    row["valid_prefix_steps"] = valid
                    if valid:
                        row["valid_prefix_diagnostics"] = diagnostics(robot, actual[:valid+1], target[:valid+1], force[:valid+1])[-1]
                    save(directory/"result.json", row); cases.append(row)
                    save(args.output/"progress.json", dict(completed_forks=len(cases), physical_steps=steps,
                        policy_forwards=calls, latest=row))
                require(len(set(priming)) == 1, "Forks did not share exactly the same robot observation history")
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized() and calls <= 1280 and steps <= 1800,
            "Fork worker cleanup, isolation or total-call bound failed")
        save(args.output/"recovery_result.json", dict(status="RESTORATION_GATED_RECOVERY_FORKS_COMPLETED", cases=cases,
            restoration_gates=gates, actual_physical_steps=steps, actual_model_forwards=calls,
            new_optimizer_updates=0, new_rendered_frames=0, teacher_is_deployable=False,
            test_sources_read=False, natural_teleoperation_qualified=False, user_accepted_checkpoint=None))
    finally:
        actor.close(graceful=False)


class RecoveryBudget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)["unique_allocated_bytes"] < OUTPUT_BYTES, "Recovery128MiB output bound")
        return super().check()


def execute(args):
    if args.child:
        return probe(args)
    sample, protected, plan = admission(args)
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    with own_gpu() as lock:
        owner = dict(lock, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", read(COLLECTION/"checkpoint_audit.json"))
        budget = RecoveryBudget(config, receipt, args.output, owner); budget.maximum_wall = WALL
        previous, requested, reservation = dict(os.environ), False, None
        result = dict(stage="RESTORATION_GATED_TEACHER_RECOVERY_RESUME1", status="RUNNING", controller=owner,
            preserved_failed_root=str(FAILED), original_wall_seconds=900, conservative_prior_spent_seconds=14,
            started_utc=now(), new_optimizer_updates=0, user_accepted_checkpoint=None)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"recovery.guard.json"
        try:
            budget.check()
            reservation = reserve(LEDGER, run_id=str(args.output), stage="r2", owner=owner,
                gpu=config["expected_gpu"], commit=args.evaluation_commit)
            budget.maximum_wall = min(WALL, reservation["reserved_seconds"]-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/"authorization.json", dict(stage=result["stage"], evaluation_commit=args.evaluation_commit,
                protected_inputs=protected, gpu_lock_owner=owner, recovery_plan=identity(PLAN),
                maximum_wall_seconds=budget.maximum_wall, maximum_physical_steps=1800, maximum_policy_forwards=1280,
                maximum_output_bytes=OUTPUT_BYTES, new_optimizer_updates=0, original_training_restarted=False,
                cache_reclaim_attempted=False, final_acceptance=False, user_accepted_checkpoint=None))
            command = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--child",
                "--gpu-owner-pid", str(owner["pid"]), "--gpu-owner-ticks", owner["start_ticks"], "--deadline-ns", str(deadline)]
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES="")
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"recovery.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Restoration or fork worker failed")
            value = read(args.output/"recovery_result.json")
            require(value.get("status") == "RESTORATION_GATED_RECOVERY_FORKS_COMPLETED"
                and all(identity(r["path"]) == r for r in protected), "Recovery terminal result or protected inputs changed")
            result.update(status="COMPLETED", source_integrity="PASSED", recovery_result=identity(args.output/"recovery_result.json"),
                completed_forks=len(value["cases"]), actual_physical_steps=value["actual_physical_steps"],
                actual_model_forwards=value["actual_model_forwards"], natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}"); raise
        finally:
            if reservation and guard_path.exists() and read(guard_path).get("cleanup", {}).get("worker_reaped") is True:
                result["cost_settlement"] = settle(LEDGER, run_id=str(args.output), owner=owner,
                    cleanup=read(guard_path)["cleanup"])
            result.update(ended_utc=now(), cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json", result)
            os.environ.clear(); os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, default in dict(config=ROOT/"script/q1_five_stage/conditional_shared_v04_20261104.json",
            runtime_config=ROOT/"data/manifests/primebot_q1_monitored_config_20261103.json",
            pause_receipt=ROOT/"logs/session_records/q1_shared_user_pause_20261106.json").items():
        p.add_argument("--"+name.replace("_", "-"), type=Path, default=default)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--evaluation-commit", required=True)
    p.add_argument("--admit-only", action="store_true")
    p.add_argument("--child", action="store_true")
    p.add_argument("--gpu-owner-pid", type=int)
    p.add_argument("--gpu-owner-ticks")
    p.add_argument("--deadline-ns", type=int)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.admit_only:
        sample, protected, plan = admission(args)
        print(json.dumps(dict(status="TEACHER_RECOVERY_RESUME_ADMITTED", protected_files=len(protected), resources=sample)))
    else:
        execute(args)
