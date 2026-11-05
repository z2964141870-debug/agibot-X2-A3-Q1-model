"""Independent guarded human H10/H5/H4/H2/H1 study during the original training.

Reuse the completed 16000 CPU save audit. Check pinned-runtime math and actual
H10 reference-state actions before any new physics. Each case releases both CPU
runtimes before its GPU EGL rendering phase. Failures are preserved, never retried.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from contracts import valid_commit
from delivery import read
from interim_review import (COMMIT, GIB, HANDLES, TRAINING, ReviewBudget, fixed_rows,
                            load_physical_robot, media, preflight, reusable_audit)
from runtime_guard import guarded_run

HORIZONS = (10, 5, 4, 2, 1)
CODE_ROOTS = ("script/supermimic_teleop/", "script/q1_five_stage/", "script/primebot_q1/",
              "script/primebot_q1_review/", "script/morphmimic_q1/", "script/supermimic_q1_sources/")
REQUIRED_CODE = ("script/supermimic_teleop/preview_review.py",
                 "script/supermimic_teleop/preview_worker.py",
                 "script/supermimic_teleop/preview_policy.py",
                 "script/supermimic_teleop/stream_contract.py",
                 "script/supermimic_teleop/test_preview_policy_runtime.py")


def validate_code_manifest(path, expected_sha256, commit, root=ROOT):
    require(valid_commit(commit) and isinstance(expected_sha256, str)
            and len(expected_sha256) == 64 and identity(path)["sha256"] == expected_sha256,
            "Evaluation code manifest SHA or published commit differs")
    value = read(path)
    rows = value.get("files", [])
    require(value.get("evaluation_commit") == commit and 1 <= len(rows) <= 256,
            "Evaluation code manifest has another commit or invalid file count")
    paths, actual = [], []
    for row in rows:
        relative = row.get("relative", "")
        item = Path(relative)
        require(isinstance(relative, str) and relative.startswith(CODE_ROOTS)
                and not item.is_absolute() and ".." not in item.parts and item.suffix == ".py"
                and set(row) == {"relative", "bytes", "sha256"}, "Invalid evaluation code manifest entry")
        path = root / item
        require(path.resolve() == path and not path.is_symlink(), "Evaluation code path must not traverse symlinks")
        found = identity(path)
        require((found["bytes"], found["sha256"]) == (row["bytes"], row["sha256"]),
                "Evaluation code differs from published manifest: " + relative)
        paths.append(relative)
        actual.append(found)
    require(len(set(paths)) == len(paths) and set(REQUIRED_CODE) <= set(paths),
            "Duplicate or missing required evaluation code")
    return actual


def runtime_paths(runtime):
    sys.path[:0] = [str(Path(runtime["adapted_source"]) / "src"),
                   str(HERE.parent / "primebot_q1_review")]


def require_plant(runtime):
    import torch
    torch.set_num_threads(1)
    require(Path(sys.executable).absolute() == Path(runtime["primebot_python"])
            and importlib.metadata.version("torch").split("+")[0] == "2.7.0"
            and importlib.metadata.version("mujoco") == "3.6.0"
            and not torch.cuda.is_initialized(), "Locked common CPU physics runtime differs")
    runtime_paths(runtime)
    return torch


def checkpoint_paths(config, audit_path):
    audited = read(audit_path)
    require(audited["status"] == "PASSED" and audited["iteration"] == 16000,
            "Require the reused trained16000 CPU save audit")
    checkpoint = next(row["checkpoint"] for row in audited["records"] if row["robot"] == "q1")
    baseline = read(Path(config["primebot_review"]) / "checkpoint_audit.json")["final"]["checkpoint"]
    for record in (checkpoint, baseline):
        require(identity(record["path"]) == record, "Audited policy or physical baseline checkpoint changed")
    return checkpoint, baseline


def policy_command(runtime, config, plan, checkpoint, baseline, horizon=None):
    worker = HERE.parent / "q1_five_stage/policy_process.py" if horizon is None else HERE / "preview_worker.py"
    command = [runtime["morph_python"], "-u", str(worker)]
    for key, value in {"plan": TRAINING / "e1024_train_plan.json", "checkpoint": checkpoint["path"],
            "frozen": config["frozen_data"], "primebot-checkpoint": baseline["path"],
            "morph-source": plan["source"], "primebot-source": runtime["adapted_source"]}.items():
        command.extend(("--" + key, str(value)))
    if horizon is not None:
        require(horizon in HORIZONS, "Unsupported preview study horizon")
        command.extend(("--preview-frames", str(horizon)))
    return command


def validate_stream(value, horizon, frame):
    from stream_contract import DT_NS
    require(isinstance(value, dict) and value.get("requested_preview_ns") == horizon*DT_NS
            and value.get("scheduled_origin_to_control_ns") == horizon*DT_NS
            and value.get("policy_prefix_frames") == horizon
            and value.get("source_preprocessing") == "offline_unverified"
            and value.get("measured_end_to_end_latency_ns") is None,
            "Worker changed the human preview timing or claimed live latency")
    provenance, sequences = value.get("provenance", []), value.get("used_sequences", [])
    require(len(provenance) == 10 and len(sequences) == horizon
            and all(type(index) is int and frame < index <= frame+horizon for index in sequences),
            "Worker used an out-of-budget or missing human source frame")
    terminal_sequence = None
    for offset, (kind, sequence) in enumerate(zip(provenance[:horizon], sequences), 1):
        require(kind in ("measured_exact", "declared_clip_terminal_hold"),
                "Active preview frame has invalid provenance")
        if kind == "measured_exact":
            require(terminal_sequence is None and sequence == frame+offset,
                    "Measured slot does not match its declared capture time")
        else:
            terminal_sequence = sequence if terminal_sequence is None else terminal_sequence
            require(sequence == terminal_sequence and offset > 1 and sequence == sequences[offset-2],
                    "Terminal hold must follow and repeat the final measured capture")
    require(provenance[horizon:] == ["masked_hold_placeholder"]*(10-horizon),
            "Hidden preview tail was enabled")
    measured = sum(kind == "measured_exact" for kind in provenance)
    require(value.get("measured_frames") == value.get("observed_prefix_frames") == measured
            and type(value.get("cpu_policy_wall_compute_ns")) is int
            and value["cpu_policy_wall_compute_ns"] >= 0, "Missing truthful stream timing")
    return value


def equivalence(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    from policy_process import PolicyProcess
    from supermimic.data import Q1Motions
    import mujoco
    from unittest.mock import patch

    checkpoint, baseline = checkpoint_paths(config, args.output / "checkpoint_audit.json")
    robot, _ = load_physical_robot(baseline["path"])
    records, original = [], {}
    for label, horizon in (("original_human_h10", None), ("stream_human_h10", 10)):
        actor = PolicyProcess(policy_command(runtime, config, plan, checkpoint, baseline, horizon),
                              args.output / (label + ".log"))
        try:
            require(actor.ready["checkpoint"] == checkpoint, "Equivalence worker loaded another policy")
            for item in fixed_rows(frozen):
                name = Path(item["q1_primebot"]["path"]).stem
                reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
                actor.start(item, 0, "human")
                actions = []
                # This checks inference along the reference; no new plant transitions.
                with patch.object(mujoco, "mj_step", side_effect=AssertionError("No equivalence physics")), \
                        patch.object(mujoco, "mj_step1", side_effect=AssertionError("No equivalence physics")), \
                        patch.object(mujoco, "mj_step2", side_effect=AssertionError("No equivalence physics")):
                    for frame in range(len(reference)-1):
                        reply = actor.action(frame, reference[frame])
                        action = reply["canonical_action"] + reply["native_action"]
                        require(len(action) == 51 and all(math.isfinite(x) for x in action),
                                "Invalid action in actual checkpoint equivalence")
                        if horizon is None:
                            actions.append(action)
                        else:
                            validate_stream(reply["stream"], 10, frame)
                            require(action == original[name][frame],
                                    f"Actual H10 preview action differs: {name} frame{frame}")
                if horizon is None:
                    original[name] = actions
                else:
                    records.append(dict(clip=name, frames=len(reference)-1, exact_equal=True,
                                        canonical_and_native_action_max_difference=0.))
            actor.close()
            require(actor.child.returncode == 0, "Equivalence policy worker did not exit normally")
        finally:
            if not actor.closed:
                actor.close(graceful=False)
    require(not torch.cuda.is_initialized(), "Equivalence initialized CUDA")
    save(args.output / "equivalence.json", dict(status="PASSED", cases=records,
        checkpoint=checkpoint, reference_state_action_comparisons=sum(row["frames"] for row in records),
        all_clamped_terminal_slots_covered=True, physical_transitions=0, optimizer_updates=0,
        cuda_initialized=False, both_workers_reaped=True, evaluation_commit=args.evaluation_commit))


def replay_case(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    from policy_process import PolicyProcess
    from q1_native_review import rollout

    checkpoint, baseline = checkpoint_paths(config, args.output / "checkpoint_audit.json")
    item = next(row for row in fixed_rows(frozen) if Path(row["q1_primebot"]["path"]).stem == args.clip)
    folder = args.output / f"h{args.horizon}" / args.clip
    require(not folder.exists(), "Case root exists; do not resume or repeat its physics")
    folder.mkdir(parents=True)
    result = dict(status="RUNNING", stage="LOW_LATENCY_HUMAN_PREVIEW_ONLY", interface="human",
        horizon_frames=args.horizon, iteration=16000, checkpoint=checkpoint, cases=[],
        evaluation_commit=args.evaluation_commit, final_acceptance=False, g1_quality_assessed=False,
        user_accepted_checkpoint=None, measured_end_to_end_latency_ns=None,
        plant="Locked Primebot CPU Torch2.7/MuJoCo3.6", preprocessing="offline_unverified")
    telemetry = (folder / "stream_timing.jsonl").open("x")

    class RecordingProcess(PolicyProcess):
        def action(self, frame, state):
            reply = super().action(frame, state)
            value = validate_stream(reply["stream"], args.horizon, frame)
            telemetry.write(json.dumps(dict(frame=frame, **value), allow_nan=False) + "\n")
            if frame % 50 == 0:
                telemetry.flush()
            return reply

    actor = None
    try:
        actor = RecordingProcess(policy_command(runtime, config, plan, checkpoint, baseline, args.horizon),
                                 folder / "policy_worker.log")
        require(actor.ready["checkpoint"] == checkpoint and actor.ready["metadata"]["iteration"] == 16000
                and actor.ready["preview"]["frames"] == args.horizon, "Wrong preview worker identity")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and Path(settings["robot_xml"]) == Path(runtime["xml"]),
                "Common physical action scale or XML changed")
        review = folder / "review"
        review.mkdir()
        row = rollout(robot, actor, settings, item, "human", 0, review / (args.clip + "_s0"))
        row.update(horizon_frames=args.horizon, preview_budget_ms=args.horizon*20,
                   measured_end_to_end_latency_ns=None)
        save(Path(row["trace"]).parent / "result.json", row)
        actor.close()
        telemetry.close()
        require(not torch.cuda.is_initialized() and actor.child.returncode == 0,
                "Common CPU replay initialized CUDA or policy failed to exit")
        result.update(status="PHYSICS_PASSED", cases=[row], render_xml=identity(settings["robot_xml"]),
            policy_worker_reaped=True, policy_worker_return_code=0, cuda_initialized=False,
            stream_telemetry=identity(folder / "stream_timing.jsonl"), asset_sha256=robot.asset_sha256,
            termination=robot.reward_identity(), ended_utc=now())
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        if actor is not None and not actor.closed:
            actor.close(graceful=False)
        telemetry.close()
        save(folder / "review/result.json", result)


def comparison(args, config, runtime, frozen):
    require_plant(runtime)
    from comparison import case_index, summary
    from metrics import compare

    values = {}
    for horizon in HORIZONS:
        cases = []
        for clip in FIXED:
            value = read(args.output / f"h{horizon}" / clip / "review/result.json")
            require(value["status"] == "PASSED" and value["horizon_frames"] == horizon
                    and value["checkpoint"] == read(args.output / "equivalence.json")["checkpoint"],
                    "Require the same audited policy and completed media for every horizon")
            cases.extend(value["cases"])
        values[horizon] = dict(status="PASSED", cases=cases)
    baseline = case_index(values[10], FIXED, (0,), "human")
    comparisons = []
    for horizon in HORIZONS[1:]:
        other = case_index(values[horizon], FIXED, (0,), "human")
        pairs = [compare(baseline[key], other[key]) for key in baseline]
        item = summary(baseline, other, pairs, FIXED)
        item.update(baseline_horizon=10, candidate_horizon=horizon,
                    baseline_preview_budget_ms=200, candidate_preview_budget_ms=horizon*20, pairs=pairs,
                    measured_end_to_end_latency_ms=None)
        comparisons.append(item)
    save(args.output / "comparison.json", dict(status="COMPLETED", interface="human", iteration=16000,
        horizons=HORIZONS, comparisons=comparisons, training_set_only=True, seed=0,
        repeated_seed_note="one nominal deterministic case per clip; no independent randomized trials",
        timing_note="H*20ms is the preview budget, not measured robot end-to-end latency",
        natural_gait_qualified=False, final_acceptance=False, user_accepted_checkpoint=None,
        optimizer_updates=0, new_physics_in_comparison=0))


def child(args):
    config, runtime = read(args.config), read(args.runtime_config)
    plan, frozen = read(TRAINING / "e1024_train_plan.json"), read(config["frozen_data"])
    if args.child == "equivalence":
        return equivalence(args, config, runtime, plan, frozen)
    if args.child == "replay":
        return replay_case(args, config, runtime, plan, frozen)
    if args.child == "comparison":
        return comparison(args, config, runtime, frozen)
    require(args.child == "media", "Unknown preview phase")
    runtime_paths(runtime)
    # The original GPU renderer reads trajectories only; physics/runtime exited.
    args.output = args.output / f"h{args.horizon}" / args.clip
    return media(args, config, runtime)


def environment(runtime):
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="1", MALLOC_TRIM_THRESHOLD_="131072",
        MALLOC_MMAP_THRESHOLD_="131072", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless",
        Q1_REVIEW_FFMPEG=runtime["ffmpeg"],
        __EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT / "script/morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
        LD_LIBRARY_PATH=str(Path(runtime["graphics_dir"]) / "lib")
            + (os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
        GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"))


def execute(args):
    if args.child:
        return child(args)
    args.iteration = 16000
    sources = validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
    config, runtime, plan, frozen, training_sources = preflight(args)
    audited = reusable_audit(args.reuse_audit, plan, 16000)
    control_inputs = [identity(path) for path in
                      (args.config, args.runtime_config, args.reuse_audit, args.code_manifest)]
    training_paths = {row["path"] for row in read(TRAINING / "e1024/train/status.json")["inputs"]}
    require(not {str(HERE / name) for name in ("preview_review.py", "preview_worker.py", "preview_policy.py",
                                              "stream_contract.py")} & training_paths,
            "New preview code must remain outside frozen training inputs")
    args.output.mkdir(exist_ok=False)
    result = dict(status="RUNNING", stage="LOW_LATENCY_PREVIEW_EXPLORATION", started_utc=now(),
        controller_pid=os.getpid(), evaluation_commit=args.evaluation_commit, phases=[],
        original_training_commit=COMMIT, training_inputs_modified=False,
        final_acceptance=False, g1_quality_assessed=False, user_accepted_checkpoint=None)
    save(args.output / "checkpoint_audit.json", audited)
    save(args.output / "authorization.json", dict(stage="LOW_LATENCY_PREVIEW_EXPLORATION",
        original_training_commit=COMMIT, evaluation_commit=args.evaluation_commit,
        plan=identity(TRAINING / "e1024_train_plan.json"), reused_audit=identity(args.reuse_audit),
        code_manifest=identity(args.code_manifest), evaluation_sources=sources,
        controller_inputs=control_inputs,
        training_source_inputs=training_sources, horizons=HORIZONS, clips=FIXED, interface="human", seed=0,
        maximum_wall_seconds=1800, maximum_output_bytes=3*GIB, memory_admission_bytes=22*GIB,
        memory_exit_bytes=24*GIB, gpu_exit_mib=18*1024, gpu_exit_temperature_c=80,
        training_gpu_lock_owner=55672, training_signals_sent=0, acquire_training_gpu_lock=False,
        final_acceptance=False, user_accepted_checkpoint=None))
    budget, requested = ReviewBudget(config, args.output), False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    previous_signals = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    previous = dict(os.environ)
    try:
        os.environ.update(environment(runtime))
        os.environ["Q1_NATIVE_TEST_SOURCE"] = plan["source"]
        runtime_command = [runtime["morph_python"], "-u", "-m", "unittest", "discover", "-s", str(HERE),
                           "-p", "test_preview_policy_runtime.py", "-v"]
        phases = [("runtime", runtime_command)]
        base = ["-u", str(Path(__file__).resolve()), "--config", str(args.config),
                "--runtime-config", str(args.runtime_config), "--output", str(args.output),
                "--evaluation-commit", args.evaluation_commit]
        phases.append(("equivalence", [runtime["primebot_python"], *base, "--child", "equivalence"]))
        for horizon in HORIZONS:
            for clip in FIXED:
                options = ["--horizon", str(horizon), "--clip", clip]
                for phase in ("replay", "media"):
                    phases.append((f"h{horizon}_{clip}_{phase}",
                        [runtime["primebot_python"], *base, "--child", phase, *options]))
        phases.append(("comparison", [runtime["primebot_python"], *base, "--child", "comparison"]))
        for label, command in phases:
            require(all(identity(row["path"]) == row for row in control_inputs),
                    "Preview controller configuration or reused audit changed")
            validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
            phase = guarded_run(command, ROOT, budget, args.output / (label + ".guard.json"),
                args.output / (label + ".log"), check_interval_s=.5, grace_s=5,
                stop_requested=lambda: requested)
            result["phases"].append(dict(label=label, guard=identity(args.output / (label + ".guard.json"))))
            require(phase["status"] == "WORKER_COMPLETED" and phase["cleanup"]["worker_reaped"] is True
                    and phase["cleanup"]["worker_return_code"] == 0, "Preview phase failed; no retry")
            if label == "runtime":
                text = (args.output / "runtime.log").read_text()
                require("Ran 1 test" in text and "skipped" not in text.lower() and "\nOK\n" in text,
                        "Native equivalence test was skipped or did not finish")
            if label == "equivalence":
                require(read(args.output / "equivalence.json")["status"] == "PASSED",
                        "Actual H10 equivalence did not pass before physics")
            save(args.output / "result.json", result)
        require(all(identity(row["path"]) == row for row in training_sources + sources + control_inputs),
                "Training or evaluation inputs changed during the preview study")
        result.update(status="COMPLETED", ended_utc=now(), comparison=identity(args.output / "comparison.json"),
            source_integrity="PASSED", training_signals_sent=0, gpu_lock_owner=55672,
            total_cases=15, media_videos=15, measured_end_to_end_latency_ms=None)
        return result
    except BaseException as error:
        result.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            save(args.output / "result.json", result)
        finally:
            os.environ.clear()
            os.environ.update(previous)
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("config", "runtime-config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--reuse-audit", type=Path)
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--child", choices=("equivalence", "replay", "media", "comparison"))
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--clip", choices=FIXED)
    options = parser.parse_args()
    if not options.child and (options.reuse_audit is None or options.code_manifest is None
                              or options.manifest_sha256 is None):
        parser.error("Controller requires the completed16000 audit and published code manifest/SHA")
    if options.child in ("replay", "media") and (options.horizon is None or options.clip is None):
        parser.error("Case phases require one horizon and one fixed clip")
    execute(options)
