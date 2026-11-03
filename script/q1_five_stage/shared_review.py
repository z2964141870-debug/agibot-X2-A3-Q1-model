"""Serial native/common-plant evaluation after audited formal shared training."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys

from comparison import case_index
from contracts import REFERENCE_KEYS, valid_commit, validate_frozen
from controller import Runner, check_formal, final_gate, rooted
from delivery import read
from native_review import ready_training
from common import FIXED, ROOT, identity, now, require, save

HERE = Path(__file__).resolve().parent
MIB, GIB = 1024**2, 1024**3


def jobs():
    return [dict(name=f"{plant}/{robot}/{interface}/{stage}_{mode}", plant=plant, robot=robot,
                 interface=interface, stage=stage, mode=mode)
            for plant, robots in (("common", ("q1",)), ("native", ("g1", "q1")))
            for robot in robots for interface in ("motion", "human")
            for stage in ("initial", "final") for mode in ("fixed", "library")]


def phase_budget(job, rows):
    selected = rows if job["mode"] == "library" else [row for row in rows if Path(row["g1"]["path"]).stem in FIXED]
    frames = [row["frames"] for row in selected]
    require(frames and all(type(n) is int and n > 1 for n in frames), "Missing frozen frame counts")
    if job["mode"] == "fixed":
        require(len(frames) == 3, "Need all three fixed clips")
    transitions = sum(n - 1 for n in frames) * (3 if job["mode"] == "fixed" else 1)
    raw_video_bound = sum(frames) * 1280 * 392 * 3 if job["mode"] == "fixed" else 0
    return dict(seconds=21600 if job["mode"] == "library" else 7200,
                maximum_bytes=128 * MIB + transitions * 16384 + raw_video_bound,
                maximum_physical_transitions=transitions, videos=3 if job["mode"] == "fixed" else 0)


def phase_passed(value):
    cleanup = value.get("guard", {}).get("cleanup", {})
    require(value.get("status") == "COMPLETED" and value.get("inputs_unchanged") is True
            and value.get("guard", {}).get("status") == "WORKER_COMPLETED"
            and cleanup.get("worker_reaped") is True and cleanup.get("worker_return_code") == 0
            and value.get("cgroup_oom_kill_delta") == 0 and value.get("gpu_recovery_action") == "None",
            "Evaluation phase failed or did not cleanly return; no automatic retry or next job")


def preflight(config, pipeline_path):
    require(pipeline_path == Path(config["output"]) / "pipeline.json", "Training pipeline belongs to another experiment")
    frozen_path = Path(config["frozen_data"])
    require(identity(frozen_path)["sha256"] == config["frozen_data_sha256"], "Frozen data changed")
    frozen = read(frozen_path)
    names = validate_frozen(frozen)
    pipeline = read(pipeline_path)
    plan_path = Path(pipeline.get("formal_plan", {}).get("path", ""))
    require(plan_path.is_absolute() and plan_path.is_relative_to(ROOT / "data/supermimic_for_q1"),
            "Require a formal shared plan under the sole server root")
    plan = read(plan_path)
    require(Path(plan["output"]).is_relative_to(Path(config["output"]))
            and plan["human_root"] == config["human_root"], "Shared plan data or output root changed")
    plan_id = identity(plan_path)
    audits = {}
    inputs = [str(pipeline_path), str(plan_path), str(frozen_path)]
    for robot, subset in zip(("g1", "q1"), plan["subsets"]):
        expected = frozen[robot + "_subset"]
        path = rooted(expected["path"])
        require(Path(subset) == path and identity(path) == dict(expected, path=str(path)), "Frozen native subset changed")
        inputs.append(str(path))
    require(len(plan["subsets"]) == 2, "Need both frozen robot subsets")
    for stage, iteration in (("initial", 0), ("final", plan["cfg"]["iterations"])):
        path = pipeline_path.parent / f"formal_checkpoint_audit_{iteration}.json"
        audit = read(path)
        for robot in ("g1", "q1"):
            checkpoint = Path(plan["output"]) / "run" / robot / f"ckpt_{iteration}.pt"
            ready_training(plan, pipeline, plan_id, audit, robot, identity(checkpoint))
            inputs.append(str(checkpoint))
        audits[stage] = path
        inputs.append(str(path))
    workers = [Path(plan["output"]) / "workers" / f"{robot}.json" for robot in ("g1", "q1")]
    check_formal([read(path) for path in workers], plan, names)
    inputs.extend(map(str, workers))
    gate = final_gate(config, names)
    require(gate["passed"], "Original Primebot and Mac delivery gate no longer passes")
    for row in frozen["rows"]:
        for key in REFERENCE_KEYS:
            require(identity(row[key]["path"]) == row[key], "Paired evaluation reference changed")
            inputs.append(row[key]["path"])
    original = read(config["pinned_input_receipt"])["inputs"]
    source, asset_root = Path(plan["source"]), rooted(frozen["xml"]["path"]).parent.parent
    pinned = [row for row in original if Path(row["path"]).is_relative_to(source)
              or Path(row["path"]).is_relative_to(asset_root)]
    require(any(Path(row["path"]).suffix == ".xml" for row in pinned)
            and any(Path(row["path"]).is_relative_to(source) for row in pinned), "Missing fixed source/assets")
    for row in pinned:
        require(identity(row["path"]) == row, "Pinned native source/assets changed")
        inputs.append(row["path"])
    return plan_path, plan, frozen, names, audits, inputs


@contextmanager
def graphics(runtime):
    values = dict(LD_LIBRARY_PATH=str(Path(runtime["graphics_dir"]) / "lib")
                  + (os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT / "script/morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
        PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", Q1_REVIEW_FFMPEG=runtime["ffmpeg"])
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def replay_command(job, config, runtime, plan_path, plan, pipeline_path, audits, output, primebot_checkpoint, commit):
    iteration = 0 if job["stage"] == "initial" else plan["cfg"]["iterations"]
    checkpoint = Path(plan["output"]) / "run" / job["robot"] / f"ckpt_{iteration}.pt"
    python = config["morph_python"] if job["plant"] == "native" else runtime["primebot_python"]
    command = [python, "-u", str(HERE / ("native_review.py" if job["plant"] == "native" else "q1_native_review.py"))]
    for key, value in dict(plan=plan_path, pipeline=pipeline_path, audit=audits[job["stage"]], checkpoint=checkpoint,
                           frozen=config["frozen_data"], output=output, interface=job["interface"],
                           **{"evaluation-commit": commit}).items():
        command.extend(("--" + key, str(value)))
    if job["plant"] == "native":
        command.extend(("--robot", job["robot"]))
    else:
        command.extend(("--primebot-source", runtime["adapted_source"], "--primebot-checkpoint", str(primebot_checkpoint),
                        "--plant-python", runtime["primebot_python"], "--morph-python", config["morph_python"]))
    command.append("--library" if job["mode"] == "library" else "--render")
    return command


def comparisons():
    rows = []
    for plant, robots in (("common", ("q1",)), ("native", ("g1", "q1"))):
        for robot in robots:
            for interface in ("motion", "human"):
                for mode in ("fixed", "library"):
                    rows.append(dict(name=f"comparisons/{plant}_{robot}_{interface}_{mode}", plant=plant,
                                     robot=robot, interface=interface, mode=mode, direct=False))
    rows.extend(dict(name=f"comparisons/primebot_shared_motion_{mode}", plant="common", robot="q1",
                     interface="motion", mode=mode, direct=True) for mode in ("fixed", "library"))
    return rows


def comparison_command(job, config, output):
    target = output / job["name"] / "comparison.json"
    prefix = output / job["plant"] / job["robot"] / job["interface"]
    left = (Path(config["primebot_review"]) / f"final_{job['mode']}/result.json" if job["direct"]
            else prefix / f"initial_{job['mode']}/review/result.json")
    right = prefix / f"final_{job['mode']}/review/result.json"
    command = [config["morph_python"], "-u", str(HERE / ("native_compare.py" if job["plant"] == "native" else "comparison.py")),
               "--left", str(left), "--right", str(right), "--frozen", config["frozen_data"], "--output", str(target)]
    if job["plant"] == "native":
        command.extend(("--robot", job["robot"], "--interface", job["interface"]))
    else:
        command.extend(("--left-label", "Primebot final" if job["direct"] else "Shared initial", "--right-label", "Shared final",
                        "--right-interface", job["interface"]))
        if not job["direct"]:
            command.extend(("--left-interface", job["interface"]))
    if job["mode"] == "library":
        command.append("--library")
    return command, target, [left, right]


def execute(config_path, pipeline_path, runtime_path, output, commit):
    require(valid_commit(commit), "Publish and independently verify the complete evaluation controller first")
    require(output.is_absolute() and output.is_relative_to(ROOT / "data/supermimic_for_q1") and not output.exists(),
            "Use a fresh evaluation suite under the sole server root")
    config, runtime = read(config_path), read(runtime_path)
    plan_path, plan, frozen, names, audits, inputs = preflight(config, pipeline_path)
    require(runtime["morph_source"] == plan["source"] and runtime["morph_python"] == config["morph_python"]
            and all(runtime[key] == config[key] for key in ("expected_boot", "expected_gpu")), "Evaluation runtime differs")
    review = Path(config["primebot_review"])
    primebot_audit = read(review / "checkpoint_audit.json")
    primebot = primebot_audit["final"]["checkpoint"]
    require(identity(primebot["path"]) == primebot, "Primebot reference checkpoint changed")
    exposure = read(review / "exposure_comparison_checkpoint.json")
    require(exposure.get("status") == "UNAVAILABLE_EARLY_STOP_BEFORE_TARGET",
            "Exposure availability changed; audit and plan that comparison before replay")
    budgets = {job["name"]: phase_budget(job, frozen["rows"]) for job in jobs()}
    maximum = sum(row["maximum_bytes"] for row in budgets.values()) + 2 * GIB
    require(shutil.disk_usage(ROOT).free >= maximum + 100 * GIB, "Whole evaluation suite does not fit the disk reserve")
    inputs.extend([str(config_path), str(runtime_path), primebot["path"],
                   config["primebot_pipeline"], config["delivery_manifest"], config["mac_receipt"],
                   str(review / "checkpoint_audit.json"), str(review / "exposure_comparison_checkpoint.json"),
                   *map(str, HERE.glob("*.py")), *map(str, (HERE.parent / "morphmimic_q1").glob("*.py")),
                   *map(str, (HERE.parent / "primebot_q1_review").glob("*.py")),
                   *map(str, Path(runtime["adapted_source"]).rglob("*.py")),
                   str(HERE.parent / "morphmimic_repro/summarize_original.py")])
    output.mkdir(parents=True, exist_ok=False)
    pipeline_snapshot = output / "training_pipeline.json"
    for source, target in ((pipeline_path, pipeline_snapshot), (plan_path, output / "training_plan.json"),
                           (audits["initial"], output / "checkpoint_audit_initial.json"),
                           (audits["final"], output / "checkpoint_audit_final.json"),
                           (review / "checkpoint_audit.json", output / "primebot_checkpoint_audit.json"),
                           (Path(config["frozen_data"]), output / "frozen_data.json")):
        shutil.copyfile(source, target)
        require(identity(source)["sha256"] == identity(target)["sha256"], "Evaluation proof copy changed")
        inputs.append(str(target))
    state = dict(status="EVALUATING", started_utc=now(), evaluation_commit=commit, training_plan=identity(plan_path),
        training_pipeline=identity(pipeline_path), training_pipeline_snapshot=identity(pipeline_snapshot),
        frozen_data=identity(config["frozen_data"]), phases=[],
        maximum_suite_bytes=maximum, phase_budgets=budgets, automatic_retry=False, goal_completed=False,
        new_optimizer_updates=0, user_accepted_checkpoint=None)
    save(output / "guard_config.json", dict(adapted_source=plan["source"], expected_boot=config["expected_boot"],
        expected_gpu=config["expected_gpu"], maximum_source_bytes=config.get("maximum_source_bytes", 64 * MIB)))
    save(output / "evaluation.json", state)
    runner = Runner(config, output, inputs)
    try:
        with graphics(runtime):
            for job in jobs():
                path = output / job["name"] / "review"
                command = replay_command(job, config, runtime, plan_path, plan, pipeline_snapshot, audits, path, primebot["path"], commit)
                budget = budgets[job["name"]]
                phase, _ = runner.phase(job["name"], command, budget["seconds"], budget["maximum_bytes"])
                phase_passed(phase)
                result = read(path / "result.json")
                selected, seeds = (names, (0,)) if job["mode"] == "library" else (FIXED, (0, 1, 2))
                case_index(result, selected, seeds, job["interface"])
                state["phases"].append(dict(name=job["name"], result=identity(path / "result.json")))
                save(output / "evaluation.json", state)
            for job in comparisons():
                command, target, extra = comparison_command(job, config, output)
                phase, _ = runner.phase(job["name"], command, 1800, 64 * MIB, extra)
                phase_passed(phase)
                require(read(target).get("status") in ("COMPARABLE", "MISSING_COMPARABLE_PREFIX"), "Comparison did not finish")
                state["phases"].append(dict(name=job["name"], result=identity(target)))
                save(output / "evaluation.json", state)
            for robot in ("g1", "q1"):
                path = output / "curves" / robot / "plots"
                command = [config["morph_python"], "-u", str(HERE.parent / "morphmimic_repro/summarize_original.py"),
                    "--run", str(Path(plan["output"]) / "run" / robot), "--output", str(path),
                    "--label", f"Shared G1 + Q1 | {robot.upper()} | {plan['cfg']['num_envs']} envs | H32",
                    "--wm-only-iters", str(plan["cfg"]["wm_only_iters"])]
                events = sorted((Path(plan["output"]) / "run" / robot).glob("events.out.tfevents.*"))
                require(events, "Missing original shared TensorBoard events")
                phase, _ = runner.phase(f"curves/{robot}", command, 900, 256 * MIB, events)
                phase_passed(phase)
                require(read(path / "source_receipt.json")["log_rows"] > 0, "Missing training curves")
                state["phases"].append(dict(name=f"curves/{robot}", result=identity(path / "source_receipt.json")))
                save(output / "evaluation.json", state)
        from shared_delivery import prepare, write_conclusions
        write_conclusions(output, plan, config, exposure)
        state.update(status="SHARED_EVALUATION_COMPLETED_PENDING_MAC_DELIVERY", ended_utc=now(),
                     next_action="Copy only manifest members to Mac; complete SHA, full video decode and visual review")
        save(output / "evaluation.json", state)
        manifest = prepare(output, config, names, output / "delivery_manifest.json")
        return dict(status=state["status"], delivery_manifest=identity(output / "delivery_manifest.json"),
                    artifacts=len(manifest["artifacts"]), videos=len(manifest["media"]), goal_completed=False)
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        save(output / "evaluation.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "pipeline", "runtime-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    args = parser.parse_args()
    print(execute(args.config, args.pipeline, args.runtime_config, args.output, args.evaluation_commit))
