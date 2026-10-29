"""Prepare or run one separately authorized, zero-optimizer substep observation."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_ppo")]
from resources import identity, save, stamp

WALKS = {"walk_ff_stop_180_R_002__A095": 320, "walk_ff_start_180_R_002__A071": 210}
SEEDS = (32000, 32001, 32002)
BASE = ROOT / "data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029"
PPO = ROOT / "data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029"
SITE = ROOT / "data/environments/supermimic_q1_hp_20261026/lib/python3.11/site-packages"
BACKEND_FILES = ("unisim/backend/mjwarp/backend.py", "unilab/base/torch_env.py",
    "mujoco_warp/_src/forward.py", "mujoco_warp/_src/constraint.py", "mujoco_warp/_src/types.py")
SCRIPT_DIRS = ("supermimic_q1_gait_comp", "supermimic_q1_balance", "supermimic_q1_ppo",
    "supermimic_q1_contact_comp", "supermimic_q1_sole", "supermimic_q1_gait_goal",
    "supermimic_q1_target_probe", "supermimic_q1_hp", "supermimic_for_q1")


def verify(item):
    if identity(item["path"]) != item:
        raise ValueError("Frozen observation input changed: " + item["path"])


def option(command, name):
    if command.count("--" + name) != 1:
        raise ValueError("Expected one original command option: " + name)
    return command[command.index("--" + name) + 1]


def case_spec(line, checkpoints):
    if line == "base":
        conditions = [("initial", checkpoints["initial"]), ("trained301", checkpoints["stage_01"])]
    elif line == "ppo":
        conditions = [("zero_head", checkpoints["stage_02"]), ("trained600", checkpoints["stage_02"])]
    else:
        raise ValueError("Unknown independent line")
    return [dict(condition=condition, checkpoint=checkpoint, clip=clip, repeat=repeat, seed=seed,
                 maximum_controls=maximum, maximum_physical_substeps=4 * maximum)
            for condition, checkpoint in conditions for clip, maximum in WALKS.items()
            for repeat, seed in enumerate(SEEDS)]


def authorize(plan, plan_path, record, line):
    if (record.get("status") != "USER_AUTHORIZED_ZERO_OPTIMIZER_SUBSTEP_OBSERVATION"
            or record.get("plan_sha256") != identity(plan_path)["sha256"]
            or line not in record.get("allowed_lines", [])
            or record.get("maximum_controls_per_line") != 3180
            or record.get("optimizer_updates") != 0):
        raise ValueError("A new explicit decision for this exact observation plan is required")
    if (plan["maximum_controls_per_line"] != 3180 or plan["optimizer_updates"] != 0
            or plan["maximum_physical_substeps_per_line"] != 12720
            or plan["maximum_policy_batches_per_line"] != 6360
            or plan["maximum_worlds_at_once"] != 1 or plan["control_dt_s"] != .02
            or plan["physics_dt_s"] != .005 or plan["automatic_retry"] is not False
            or plan["maximum_whole_wall_seconds_per_line"] != 1800 or plan["grace_seconds"] != 10):
        raise ValueError("Observation budget changed")
    cases = plan["lines"][line]["cases"]
    conditions = ("initial", "trained301") if line == "base" else ("zero_head", "trained600")
    expected_cases = {(condition, clip, repeat) for condition in conditions for clip in WALKS for repeat in range(3)}
    if (len(cases) != 12 or sum(case["maximum_controls"] for case in cases) != 3180
            or {(case["condition"], case["clip"], case["repeat"]) for case in cases} != expected_cases
            or any(case["clip"] not in WALKS or case["maximum_controls"] != WALKS[case["clip"]]
                   or case["maximum_physical_substeps"] != 4 * case["maximum_controls"]
                   or case["repeat"] not in range(3) or case["seed"] != SEEDS[case["repeat"]] for case in cases)):
        raise ValueError("Observation cases differ from the reviewed bound")


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Observation preparation cannot overwrite an existing root")
    lines, frozen = {}, {}
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        original_plan = trial / "plan.json"
        selection = trial / "run/checkpoint_selection.json"
        plan = json.loads(original_plan.read_text())
        command = plan["phases"][0]["command"]
        inputs = {name: option(command, name) for name in ("runtime", "library", "checkpoint", "calibration")}
        selected = json.loads(selection.read_text())["candidates"]
        checkpoints = {row["label"]: row["checkpoint"] for row in selected}
        cases = case_spec(line, checkpoints)
        for checkpoint in {case["checkpoint"]["path"]: case["checkpoint"] for case in cases}.values():
            verify(checkpoint)
        teacher = identity(inputs["checkpoint"])
        history = {str((ROOT / row["path"]).resolve()): row for row in plan["source"]}
        library = json.loads(Path(inputs["library"]).read_text())
        rows = [dict(row, split=split) for split, values in library["splits"].items()
                for row in values if row["name"] in WALKS]
        if len(rows) != 2 or {row["name"] for row in rows} != set(WALKS):
            raise ValueError("Exactly the original two fixed walks are required")
        required = {original_plan, selection, *(Path(inputs[name]) for name in ("library", "checkpoint", "calibration"))}
        required.update(Path(case["checkpoint"]["path"]) for case in cases)
        for row in rows:
            path = Path(row["path"])
            required.update((path, path.with_suffix(".json")))
        runtime = Path(inputs["runtime"])
        for name, old in history.items():
            path = Path(name)
            if (path.is_relative_to(runtime) or path.is_relative_to(ROOT / "data/references/q1/model_20261023_P01")
                    or any(path.is_relative_to(ROOT / "script" / folder) for folder in SCRIPT_DIRS)
                    or path.name in ("diagnose_q1_upper_reference.py", "audit_q1_saved_sole_geometry.py",
                                     "guard_phuma_batch.py", "runtime_guard.py")):
                required.add(path)
        for path in sorted(required):
            actual = identity(path)
            old = history.get(str(path.resolve()))
            if old and any(actual[key] != old[key] for key in ("bytes", "sha256")):
                raise ValueError("Used original source differs from trial freeze: " + str(path))
            frozen[str(path)] = actual
        lines[line] = dict(original_plan=identity(original_plan), selection=identity(selection),
            inputs=inputs, teacher=teacher, rows=rows, cases=cases,
            execution_output=str(ROOT / "data/supermimic_for_q1" / f"Q1_GAIT_GOAL01_SUBSTEP_{line.upper()}_20261030"))
    current = [*HERE.glob("*.py"), *(SITE / name for name in BACKEND_FILES),
               HERE.parent / "supermimic_q1_gait_goal/supervise.py"]
    frozen.update({str(path): identity(path) for path in current})
    spec = dict(status="PREPARED_AWAITING_NEW_USER_DECISION", created_utc=stamp(),
        lines=lines, source=list(frozen.values()), backend_current_only=[identity(SITE / n) for n in BACKEND_FILES],
        source_base_commit=args.source_base_commit, optimizer_updates=0, policy_calls_executed=0,
        physical_substeps_executed=0, maximum_controls_per_line=3180,
        maximum_physical_substeps_per_line=12720, maximum_policy_batches_per_line=6360,
        maximum_worlds_at_once=1, seeds=list(SEEDS), control_dt_s=.02, physics_dt_s=.005,
        maximum_whole_wall_seconds_per_line=1800, grace_seconds=10, automatic_retry=False,
        automatic_training_or_adoption=False, user_accepted_checkpoint=None,
        expected_boot="4d859008-4c83-4a6d-814d-cbb9dfe885c9",
        expected_gpu="GPU-26028785-9fb9-45af-e614-4cc53af15d08", expected_oom_kill=0,
        limitations=["Current backend identities do not prove historical backend immutability",
            "Fresh one-world runs are not historical330-world replays",
            "Seed equality is not complete native solver-state equality",
            "Synthetic hook tests do not verify physical trajectory equivalence",
            "A failed native launch can leave an uncertain partial substep; no retry",
            "Budgets count rollout transitions; graph capture/reset forward are separately initialization"])
    if any(sum(case["maximum_controls"] for case in item["cases"]) != 3180 for item in lines.values()):
        raise AssertionError("Independent observation bound mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "observation_plan.json", spec)
    print(json.dumps(dict(status=spec["status"], plan=identity(args.output / "observation_plan.json"),
        frozen_inputs=len(frozen), physical_substeps_executed=0, optimizer_updates=0)), flush=True)


def array_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode("ascii"))
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def compare_initial(previous, current):
    import numpy as np
    if set(previous) != set(current):
        raise ValueError("Initial observation inventories differ")
    differences = {}
    for key in previous:
        left, right = previous[key], current[key]
        same_shape = left.shape == right.shape
        differences[key] = dict(shape_matches=same_shape,
            dtype_matches=left.dtype == right.dtype,
            exact=bool(same_shape and left.dtype == right.dtype and np.array_equal(left, right)),
            maximum_absolute_difference=float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if same_shape and left.size else None)
    return dict(all_recorded_inputs_exact=all(v["exact"] for v in differences.values()),
        fields=differences, complete_solver_state_equivalence_established=False,
        historical_rollout_replay_established=False)


def components(runtime):
    sys.path[:0] = [str(HERE.parent / folder) for folder in
        ("supermimic_q1_target_probe", "supermimic_q1_gait_goal", "supermimic_q1_contact_comp",
         "supermimic_q1_sole", "supermimic_q1_gait_comp", "supermimic_q1_hp", "project_tools")]
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    from probe import prepare_seeded
    from diagnose_q1_upper_reference import policy_digest
    from contact_reference import configure
    from physical import rollout
    from substep_observer import SubstepObserver, WarpSampler
    from worker import process_identity

    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES
    from gait_adapter import Head
    from reward_candidates import PhaseSoleReference
    from evaluation import clip_metrics
    from metrics import gait_clip
    from prepare_q1_gait_review import write_csv
    return SimpleNamespace(**{key: value for key, value in locals().items() if key != "runtime"})


def worker(args, plan):
    import random
    import numpy as np
    import torch

    item = plan["lines"][args.line]
    inputs = {key: Path(value) for key, value in item["inputs"].items()}
    loaded = components(inputs["runtime"])
    prepare_seeded, policy_digest, configure, rollout, SubstepObserver, WarpSampler = [
        getattr(loaded, name) for name in ("prepare_seeded", "policy_digest", "configure", "rollout", "SubstepObserver", "WarpSampler")]
    process_identity, load_policy, Q1_JOINT_NAMES, Head, PhaseSoleReference, clip_metrics, gait_clip, write_csv = [
        getattr(loaded, name) for name in ("process_identity", "load_policy", "Q1_JOINT_NAMES", "Head", "PhaseSoleReference", "clip_metrics", "gait_clip", "write_csv")]

    for source in plan["source"]:
        verify(source)
    run = args.output / "run"
    run.mkdir(exist_ok=False)
    status = dict(status="INITIALIZING", process=process_identity(), line=args.line,
        updated_utc=stamp(), optimizer_updates=0, controlled_transitions=0, completed_physical_substeps=0,
        attempted_native_substeps=0, returned_native_substeps=0)
    save(run / "status.json", status)
    summaries, actuator_rows, baselines = [], [], {}
    torch.set_num_threads(2)
    for index, case in enumerate(item["cases"]):
        prior_counts = {key: status[key] for key in ("controlled_transitions", "completed_physical_substeps",
            "attempted_native_substeps", "returned_native_substeps")}
        env = observer = sampler = hook = teacher = head = reference = trace = None
        initial, policy_calls = {}, dict(teacher=0, head=0)
        folder = run / f"case{index:02d}_{case['condition']}_repeat{case['repeat']}"
        folder.mkdir()
        row = next(row for row in item["rows"] if row["name"] == case["clip"])
        library = folder / "evaluation_library.json"
        save(library, dict(splits=dict(train=[row]), source=identity(inputs["library"])))
        status.update(case=index, condition=case["condition"], clip=case["clip"], frame=0, updated_utc=stamp())
        save(run / "status.json", status)
        try:
            _, env, _ = prepare_seeded(SimpleNamespace(runtime=inputs["runtime"], library=library, output=folder), 1, case["seed"])
            checkpoint = Path(case["checkpoint"]["path"]) if args.line == "base" else inputs["checkpoint"]
            teacher = load_policy(env, torch.load(checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
            def capture(module, values):
                policy_calls["teacher"] += 1
                if not initial:
                    initial.update(history=values[0][:1].detach().cpu().numpy().copy(),
                                   goal=values[1][:1].detach().cpu().numpy().copy())
                    python_rng, numpy_rng = random.getstate(), np.random.get_state()
                    initial.update(python_rng_words=np.array(python_rng[1], dtype=np.uint64),
                        python_rng_version=np.array(python_rng[0]),
                        python_rng_gaussian=np.array([python_rng[2] is not None, python_rng[2] or 0.]),
                        numpy_rng_algorithm=np.frombuffer(numpy_rng[0].encode("ascii"), dtype=np.uint8).copy(),
                        numpy_rng_words=numpy_rng[1].copy(), numpy_rng_tail=np.array(numpy_rng[2:]),
                        torch_rng_cpu=torch.get_rng_state().cpu().numpy().copy(),
                        torch_rng_cuda=torch.cuda.get_rng_state(env.device).cpu().numpy().copy())
            hook = teacher.register_forward_pre_hook(capture)
            teacher_digest = policy_digest(teacher)
            head = None
            head_digest = None
            if args.line == "ppo":
                saved = torch.load(case["checkpoint"]["path"], map_location=env.device, weights_only=True)
                if saved["source_base"] != item["teacher"] or saved["upper_reference_alpha"] != .5:
                    raise ValueError("PPO source teacher or upper-alpha contract changed")
                head = Head().to(env.device).requires_grad_(False).eval()
                head.load_state_dict({key[5:]: value for key, value in saved["actor"].items() if key.startswith("head.")})
                head_digest = array_digest({key: value.detach().cpu().numpy() for key, value in head.state_dict().items()})
                def count_head(module, values):
                    policy_calls["head"] += 1
                head.register_forward_pre_hook(count_head)
                del saved
            configure(library)
            reference = PhaseSoleReference(env.supermimic_mdp, inputs["runtime"] / "q1_assets/scene.xml")
            if int(env.supermimic_mdp.motion.clip_len[0]) - 1 != case["maximum_controls"]:
                raise ValueError("Fixed single-clip length differs from the prepared observation")
            for seed_fn in (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
                seed_fn(case["seed"])
            sampler = WarpSampler(env)
            observer = SubstepObserver(env.backend, sampler, case["maximum_controls"])
            def progress(frame):
                status.update(status="RUNNING", frame=frame, updated_utc=stamp(),
                    current_case_controls=observer.control_calls, current_case_substeps=observer.completed_substeps,
                    current_case_attempted_native_substeps=observer.attempted_native_substeps,
                    current_case_returned_native_substeps=observer.returned_native_substeps)
                save(run / "status.json", status)
            with torch.no_grad(), observer:
                trace = rollout(env, teacher, head, json.loads(inputs["calibration"].read_text())["thresholds"],
                    reference, mode="trained075" if case["condition"] == "trained600" else "static075",
                    upper_alpha=.5, progress=progress)
            archive = sampler.archive()
            if (len(archive["control_index"]) != 4 * observer.control_calls
                    or observer.completed_substeps != len(archive["control_index"])
                    or observer.attempted_native_substeps != observer.completed_substeps
                    or observer.returned_native_substeps != observer.completed_substeps
                    or observer.control_calls != int(trace["steps"][0])
                    or observer.faulted or policy_calls["teacher"] != observer.control_calls
                    or sum(policy_calls.values()) > 2 * observer.control_calls):
                raise AssertionError("Observed rollout/solve/policy counters disagree")
            first = sampler.records[0]
            for name in ("qpos", "qvel", "qacc_warmstart", "qacc", "time", "qfrc_applied", "xfrc_applied"):
                initial["native_" + name] = first["pre_" + name]
            initial["semantic_state"] = trace["initial"]
            for name, value in sampler.coefficients.items():
                initial["model_" + name] = value
            key = (case["clip"], case["repeat"])
            match = None
            if case["condition"] in ("initial", "zero_head"):
                baselines[key] = {name: value.copy() for name, value in initial.items()}
            else:
                match = compare_initial(baselines[key], initial)
            np.savez_compressed(folder / "initial_inputs.npz", **initial)
            np.savez_compressed(folder / "native_substeps.npz", **archive)
            np.savez_compressed(folder / "control_trajectory.npz", **trace)
            if policy_digest(teacher) != teacher_digest or (head is not None and
                    array_digest({key: value.detach().cpu().numpy() for key, value in head.state_dict().items()}) != head_digest):
                raise AssertionError("Observation changed frozen teacher/head/normalization")
            contract = sampler.contract()
            ids = sampler.actuation["actuator_ids"]
            force = archive["solve_actuator_force"][:, 0, ids]
            limits = sampler.actuation["force_range"]
            for joint, name in enumerate(Q1_JOINT_NAMES):
                near = np.isclose(force[:, joint], limits[joint, 0], rtol=0, atol=1e-4) | np.isclose(force[:, joint], limits[joint, 1], rtol=0, atol=1e-4)
                actuator_rows.append(dict(line=args.line, condition=case["condition"], clip=case["clip"], repeat=case["repeat"],
                    joint=name, native_actuator_id=int(ids[joint]), solved_substeps=len(force),
                    force_abs_p95_Nm=float(np.quantile(np.abs(force[:, joint]), .95)),
                    force_abs_max_Nm=float(np.abs(force[:, joint]).max()), limit_touch_substeps=int(near.sum()),
                    limit_touch_fraction=float(near.mean()), limit_touch_sample_duration_s=float(near.sum() * .005)))
            summary = dict(case=case, controlled_transitions=observer.control_calls, completed_physical_substeps=observer.completed_substeps,
                attempted_native_substeps=observer.attempted_native_substeps,
                returned_native_substeps=observer.returned_native_substeps,
                policy_batches=policy_calls, initial_inputs_sha256=array_digest(initial), initial_match=match,
                contract=contract, gait=gait_clip(trace, 0), tracking=clip_metrics(trace, 0), optimizer_updates=0,
                artifacts=[identity(folder / name) for name in ("initial_inputs.npz", "native_substeps.npz", "control_trajectory.npz")],
                storage="SERVER_LOCAL_ONLY", user_accepted_checkpoint=None)
            save(folder / "summary.json", summary)
            summaries.append(summary)
            status["controlled_transitions"] += observer.control_calls
            status["completed_physical_substeps"] += observer.completed_substeps
            status["attempted_native_substeps"] += observer.attempted_native_substeps
            status["returned_native_substeps"] += observer.returned_native_substeps
            if status["controlled_transitions"] > 3180 or status["completed_physical_substeps"] > 12720:
                raise AssertionError("Independent line exceeded its prepared observation bound")
            save(run / "status.json", status)
        except BaseException as error:
            if sampler is not None and sampler.records:
                np.savez_compressed(folder / "partial_native_substeps.npz", **sampler.archive())
            failure = dict(error=f"{type(error).__name__}: {error}",
                control_calls=observer.control_calls if observer else 0,
                completed_substeps_lower_bound=observer.completed_substeps if observer else 0,
                attempted_native_substeps=observer.attempted_native_substeps if observer else 0,
                returned_native_substeps=observer.returned_native_substeps if observer else 0,
                possible_partial_native_launch=bool(observer and
                    observer.attempted_native_substeps > observer.completed_substeps), automatic_retry=False)
            save(folder / "failure.json", failure)
            status.update(status="FAILED", updated_utc=stamp(), failed_case=failure,
                total_completed_substeps_lower_bound=prior_counts["completed_physical_substeps"] + failure["completed_substeps_lower_bound"],
                total_control_calls=prior_counts["controlled_transitions"] + failure["control_calls"],
                total_attempted_native_substeps=prior_counts["attempted_native_substeps"] + failure["attempted_native_substeps"],
                total_returned_native_substeps=prior_counts["returned_native_substeps"] + failure["returned_native_substeps"])
            save(run / "status.json", status)
            raise
        finally:
            if hook is not None:
                hook.remove()
            if env is not None:
                try:
                    env.close()
                finally:
                    env.backend.close()
            del env
            observer = sampler = teacher = head = reference = trace = None
            gc.collect()
            torch.cuda.empty_cache()
    for source in plan["source"]:
        verify(source)
    write_csv(args.output / "actuator_summary.csv", actuator_rows)
    save(args.output / "results.json", dict(status="OBSERVATION_COMPLETED", cases=summaries, optimizer_updates=0,
        controlled_transitions=status["controlled_transitions"], completed_physical_substeps=status["completed_physical_substeps"],
        attempted_native_substeps=status["attempted_native_substeps"], returned_native_substeps=status["returned_native_substeps"],
        physical_trajectory_equivalence_verified=False, historical_replay=False, user_accepted_checkpoint=None))
    status.update(status="COMPLETED", updated_utc=stamp())
    save(run / "status.json", status)


def supervise(args, plan):
    import fcntl
    import signal
    import subprocess
    import time
    sys.path[:0] = [str(HERE.parent / folder) for folder in ("supermimic_q1_hp", "supermimic_q1_gait_goal")]
    from run_trial import Q1Budget
    from supervise import memory_admission
    from runtime_guard import guarded_run

    class ObservationBudget(Q1Budget):
        def check(self):
            value = super().check()
            if (Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["expected_boot"]
                    or value["gpu"]["uuid"] != plan["expected_gpu"]):
                raise RuntimeError("Observation platform identity changed")
            if time.monotonic() - self.started >= plan["maximum_whole_wall_seconds_per_line"]:
                raise RuntimeError("Whole-wall observation limit reached; no retry")
            return value

    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status="INITIALIZING", line=args.line, started_utc=stamp(), plan=identity(args.plan),
        code_commit=args.code_commit, authorization=identity(args.authorization_record), automatic_retry=False)
    save(args.output / "status.json", status)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_args = SimpleNamespace(output=args.output, runtime=Path(plan["lines"][args.line]["inputs"]["runtime"]),
            mode="independent_observation", expected_oom_kill=plan["expected_oom_kill"])
        budget = ObservationBudget(budget_args)
        budget.check()
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU already in use; observation not started")
        admission = memory_admission()
        save(args.output / "memory_admission.json", admission)
        if not admission["admitted"]:
            raise RuntimeError("Initial memory reserve unavailable; no automatic retry")
        for source in plan["source"]:
            verify(source)
        command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u", str(Path(__file__).resolve()),
            "worker", "--plan", str(args.plan), "--line", args.line, "--output", str(args.output),
            "--authorization-record", str(args.authorization_record), "--code-commit", args.code_commit]
        status.update(status="RUNNING", command=command)
        save(args.output / "status.json", status)
        guarded = guarded_run(command, ROOT, budget, args.output / "observation_guard.json",
            ROOT / "logs" / (args.output.name + ".log"), check_interval_s=1, grace_s=10,
            stop_requested=lambda: stopped, progress_path=args.output / "run/status.json", maximum_idle_s=300)
        if guarded["status"] != "WORKER_COMPLETED" or guarded["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Observation failed; preserve partial evidence and do not retry")
        for source in plan["source"]:
            verify(source)
        status.update(status="COMPLETED", results=identity(args.output / "results.json"), user_accepted_checkpoint=None)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["ended_utc"] = stamp()
        save(args.output / "status.json", status)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("prepare", "check-imports", "supervise", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--ppo", type=Path, default=PPO)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--line", choices=("base", "ppo"))
    parser.add_argument("--authorization-record", type=Path)
    parser.add_argument("--source-base-commit")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use a new subtree of the unique server project root")
    if args.mode == "prepare":
        if (not args.source_base_commit or len(args.source_base_commit) != 40
                or any(value not in "0123456789abcdef" for value in args.source_base_commit)):
            parser.error("Record the current source base commit")
        prepare(args)
        return
    if args.mode == "check-imports":
        if args.plan is None:
            parser.error("Import check requires the prepared observation plan")
        plan = json.loads(args.plan.read_text())
        components(Path(plan["lines"]["base"]["inputs"]["runtime"]))
        print(json.dumps(dict(status="OBSERVATION_IMPORTS_PASSED", environments_constructed=0,
            policy_calls=0, physical_substeps=0, optimizer_updates=0)), flush=True)
        return
    if any(value is None for value in (args.plan, args.line, args.authorization_record, args.code_commit)):
        parser.error("Execution requires the frozen plan, independent line, new decision and published commit")
    if len(args.code_commit) != 40 or any(value not in "0123456789abcdef" for value in args.code_commit):
        parser.error("Expected an independently published40-digit source commit")
    plan = json.loads(args.plan.read_text())
    authorize(plan, args.plan, json.loads(args.authorization_record.read_text()), args.line)
    if args.output != Path(plan["lines"][args.line]["execution_output"]):
        parser.error("Execution must use the reviewed single-use output root")
    (worker if args.mode == "worker" else supervise)(args, plan)


if __name__ == "__main__":
    main()
