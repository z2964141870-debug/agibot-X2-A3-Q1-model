"""Continue only missing G1 review after the preserved static-terrain failure."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time

from review import (Budget, GIB, HERE, ITERATIONS, OLD_REVIEW, ROOT, TRAINING, allocated_tree,
    audit_record, checkpoint_audit, identity, now, observation, preflight, read, read_groups,
    require, save, guarded_run)


def source_ready(source):
    status = read(source / "status.json")
    require(status["status"] == "FAILED" and status["phase"] == "g1_16000_motion"
            and len(status["phases"]) == 10, "Require the preserved first G1 diagnostic failure")
    for index, phase in enumerate(status["phases"]):
        require(identity(phase["guard"]["path"]) == phase["guard"], "Old guard record changed")
        guard = read(phase["guard"]["path"])
        require(guard["cleanup"]["worker_reaped"] is True, "Old review child was not reaped")
        require(guard["status"] == ("WORKER_FAILED" if index == 9 else "WORKER_COMPLETED"),
                "Unexpected prior phase outcome")
    require(read(source / "source_integrity.json")["passed"] is True
            and (source / "training_after.json").is_file(), "Original cleanup/integrity not complete")
    error = (source / "g1_16000_motion.log").read_text()
    require("Native diagnostics require one actual static plane" in error, "Different G1 failure")
    groups = read_groups(source)
    require(all(len(groups["q1/" + face]) == 3 for face in ("motion", "human")), "Q1 cases are not complete")
    return groups


def execute(args):
    old_groups = source_ready(args.source_output)
    reused_g1 = 0
    if args.reuse_g1_output is not None:
        state = read(args.reuse_g1_output / "status.json")
        require(state["status"] == "FAILED" and state["phase"] == "g1_16000_motion"
                and len(state["phases"]) == 1, "Require the preserved first-case reset failure")
        guard = read(state["phases"][0]["guard"]["path"])
        require(guard["status"] == "WORKER_FAILED" and guard["cleanup"]["worker_reaped"] is True,
                "Prior G1 worker cleanup incomplete")
        require("Inplace update to inference tensor outside InferenceMode" in (
            args.reuse_g1_output / "g1_16000_motion.log").read_text(), "Different G1 continuation failure")
        require(read(args.reuse_g1_output / "source_integrity.json")["passed"] is True,
                "Prior G1 frozen-input integrity failed")
        saved = read(args.reuse_g1_output / "g1_native_cpu/16000/result.json")
        require(len(saved["cases"]) == 1 and saved["cases"][0]["interface"] == "motion", "Unexpected reused G1 case set")
        reused_g1 = len(saved["cases"])
    config, runtime, plan, frozen, sources, reused = preflight(args)
    audits = {iteration: read(checkpoint_audit(args.source_output, iteration)) for iteration in ITERATIONS}
    for iteration in ITERATIONS:
        for robot in ("g1", "q1"):
            audit_record(args.source_output, iteration, robot)
    old_auth = read(args.source_output / "authorization.json")
    started = datetime.fromisoformat(old_auth["started_utc"].replace("Z", "+00:00"))
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    require(0 <= elapsed < 7200, "Original whole-review wall budget exhausted")
    args.output.mkdir(exist_ok=False)
    save(args.output / "training_before.json", observation(config))
    for iteration, value in audits.items():
        target = checkpoint_audit(args.output, iteration)
        target.parent.mkdir(parents=True)
        save(target, value)
    save(args.output / "authorization.json", dict(started_utc=now(), evaluation_commit=args.evaluation_commit,
        original_started_utc=old_auth["started_utc"], source_status=identity(args.source_output / "status.json"),
        source_output=str(args.source_output), new_training=False, original_training_inputs_modified=False,
        correction="Static terrain may be in a body welded to world; require world-horizontal plane at z0",
        frozen_native_evaluator=identity(HERE.parent / "q1_five_stage/native_review.py"),
        reused_g1_cases=reused_g1, reused_g1_output=str(args.reuse_g1_output) if args.reuse_g1_output else None,
        reused_q1_cases=18, optimizer_updates=0, training_signals_sent=0,
        final_acceptance=False, user_accepted_checkpoint=None))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="1", MALLOC_TRIM_THRESHOLD_="131072", MALLOC_MMAP_THRESHOLD_="131072",
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", Q1_REVIEW_FFMPEG=runtime["ffmpeg"],
        __EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT / "script/morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
        LD_LIBRARY_PATH=str(Path(runtime["graphics_dir"]) / "lib") + (
            os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
        GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"))
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    budget, previous = Budget(config, args.output), dict(os.environ)
    budget.started -= elapsed
    original_check = budget.check
    def check():
        roots = [args.source_output, args.output] + ([args.reuse_g1_output] if args.reuse_g1_output else [])
        require(sum(allocated_tree(root)["unique_allocated_bytes"] for root in roots) < 3 * GIB,
                "Combined original and continuation output budget exceeded")
        return original_check()
    budget.check = check
    state = dict(status="RUNNING", started_utc=now(), source_output=str(args.source_output), phases=[],
        optimizer_updates=0, training_signals_sent=0, final_acceptance=False, user_accepted_checkpoint=None)
    try:
        os.environ.update(env)
        for job, iteration in [("g1", i) for i in ITERATIONS] + [("media_g1", 48000)]:
            label = f"{job}_{iteration}_motion"
            state["phase"] = label
            save(args.output / "status.json", state)
            command = [runtime["morph_python"], "-u", str(HERE / "review.py"), "--child", job,
                "--iteration", str(iteration), "--interface", "motion", "--config", str(args.config),
                "--runtime-config", str(args.runtime_config), "--output", str(args.output),
                "--evaluation-commit", args.evaluation_commit]
            if args.reuse_g1_output is not None:
                command.extend(("--reuse-g1-output", str(args.reuse_g1_output)))
            budget.phase_started = time.monotonic()
            result = guarded_run(command, ROOT, budget, args.output / (label + ".guard.json"),
                args.output / (label + ".log"), check_interval_s=.5, grace_s=5, stop_requested=lambda: requested)
            state["phases"].append(dict(name=label, guard=identity(args.output / (label + ".guard.json"))))
            require(result["status"] == "WORKER_COMPLETED" and result["cleanup"]["worker_reaped"] is True
                    and result["cleanup"]["worker_return_code"] == 0, "Continuation phase failed: " + label)
        require(all(identity(r["path"]) == r for r in sources), "Frozen original inputs changed")
        require(identity(OLD_REVIEW / "review/result.json") == old_auth["reused16k_motion"], "Reused Q1 motion changed")
        new_groups = read_groups(args.output)
        groups = {"q1/" + face: old_groups["q1/" + face] for face in ("motion", "human")}
        groups.update({"g1/" + face: new_groups["g1/" + face] for face in ("motion", "human")})
        require(all(len(records) == 3 for records in groups.values()), "Missing saved-checkpoint group")
        save(args.output / "comparisons.json", dict(groups=groups, final_acceptance=False,
            user_accepted_checkpoint=None, original_failure=identity(args.source_output / "status.json"),
            scope="Three deterministic training motions; frozen native physical scene; no final acceptance"))
        state.update(status="COMPLETED_PENDING_LOCAL_REVIEW", ended_utc=now())
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(args.output / "status.json", state)
        save(args.output / "source_integrity.json", dict(passed=all(identity(r["path"]) == r for r in sources),
            frozen_sources=len(sources), original_training_inputs_modified=False, training_signals_sent=0))
        save(args.output / "training_after.json", observation(config))
        os.environ.clear()
        os.environ.update(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "source-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--reuse-g1-output", type=Path)
    execute(parser.parse_args())
