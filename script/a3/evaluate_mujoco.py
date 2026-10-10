"""Bounded, thermally guarded paired MuJoCo evaluation of trusted A3 checkpoints."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from script.a3.autoresume import ROOT, VENDOR_COMMIT, monitor_child
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import command, sample
from script.a3.evaluation_common import checkpoint_metadata, preflight


def rollout_windows(summary, trace):
    ticks = np.asarray(trace["policy_tick"])
    state, reference = np.asarray(trace["q_state_29"]), np.asarray(trace["reference_q_29"])
    n = summary["num_policy_steps"]
    if state.shape != (n, 29) or reference.shape != state.shape or ticks.shape != (n,):
        raise ValueError("Incomplete rollout timeseries")
    if not np.array_equal(ticks, np.arange(n)) or not np.isfinite(state).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid rollout sequence or nonfinite joints")
    result = {}
    for label, mask in (("full_rollout", np.ones(n, dtype=bool)),
                        ("before_first_fall", ticks < summary["fall_tick"] if summary["fall"] else np.ones(n, dtype=bool))):
        count = int(mask.sum())
        errors = state[mask] - reference[mask]
        row = {"policy_steps": count,
               "joint_rmse_rad": float(np.sqrt(np.mean(errors ** 2))) if count else None}
        for key in ("root_pos_error_m", "root_quat_error_deg", "anchor_pos_error_m"):
            values = np.asarray(trace[key], dtype=float)
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError("Invalid pose error timeseries")
            row[key + "_mean"] = float(values[mask].mean()) if count else None
        if trace.get("sim_body_pos_w") is not None:
            names = trace["body_names"]
            groups = {"torso": ["torso_Link"],
                      "wrists": ["left_wrist_yaw_Link", "right_wrist_yaw_Link"],
                      "legs": [name for name in names if any(token in name for token in ("hip_", "knee_", "ankle_"))]}
            for frame, sim_key, ref_key in (("world", "sim_body_pos_w", "ref_body_pos_w"),
                                           ("anchor_local", "sim_body_pos_anchor", "ref_body_pos_anchor")):
                sim_pos, ref_pos = np.asarray(trace[sim_key]), np.asarray(trace[ref_key])
                if sim_pos.shape != (n, len(names), 3) or ref_pos.shape != sim_pos.shape:
                    raise ValueError("Body tracking timeseries shape mismatch")
                if not np.isfinite(sim_pos).all() or not np.isfinite(ref_pos).all():
                    raise ValueError("Nonfinite body tracking timeseries")
                errors_m = np.linalg.norm(sim_pos - ref_pos, axis=-1)
                for group, bodies in groups.items():
                    indices = [names.index(name) for name in bodies]
                    row[f"{group}_{frame}_mean_m"] = float(errors_m[mask][:, indices].mean()) if count else None
        result[label] = row
    if not math.isclose(result["full_rollout"]["joint_rmse_rad"], summary["tracking"]["all_29_rmse"], rel_tol=1e-6):
        raise ValueError("Timeseries disagrees with published tracking metrics")
    return result


def summarize_explicit(output):
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    manifest = json.loads((output / "explicit_manifest.json").read_text())
    protocol = manifest["protocol"]
    rows = []
    for motion in protocol["motions"]:
        name = Path(motion["path"]).stem
        run = manifest["runs"].get(name, {})
        metrics_path = output / f"{name}.metrics.json"
        if run.get("status") != "passed" or sha256(metrics_path) != run["metrics_sha256"]:
            raise ValueError("Cannot summarize incomplete or altered motion")
        timeseries_path = output / f"{name}.timeseries.json"
        trace = json.loads(timeseries_path.read_text())["motions"][0]
        summary = json.loads(metrics_path.read_text())["motions"][0]
        if summary["motion_name"] != Path(motion["path"]).name or trace["motion_name"] != summary["motion_name"]:
            raise ValueError("Motion identity mismatch")
        if sha256(Path(motion["path"])) != motion["sha256"]:
            raise ValueError("Reference changed since evaluation")
        root, rotation, joints, _ = sim.load_a3_flat_csv(Path(motion["path"]),
            source_fps=protocol["reference_fps_after_stride"], frame_stride=protocol["frame_stride"])
        expected = len(sim.resample_csv_motion(root, rotation, joints,
                       source_fps=protocol["reference_fps_after_stride"])[0])
        cap = protocol["max_policy_steps"]
        required = expected if cap is None else min(expected, cap)
        if summary["num_policy_steps"] != required:
            raise ValueError("Evaluation did not cover the requested policy steps")
        rows.append({"motion": summary["motion_name"], "fall": summary["fall"],
                     "first_fall_s": summary["fall_time_s"], "expected_full_policy_steps": expected,
                     "full_replay_completed": required == expected,
                     "motion_completed_without_fall": required == expected and not summary["fall"],
                     "windows": rollout_windows(summary, trace), "timeseries_sha256": sha256(timeseries_path)})
    report = {"protocol": protocol, "scope": "computational_completion_is_not_policy_acceptance", "motions": rows}
    atomic_json(output / "explicit_summary.json", report)
    return report


def evaluate_explicit(args):
    """Resume a motion-granular evaluation without changing legacy paired runs."""
    output, logs = args.output.resolve(), args.logs.resolve()
    if not output.is_relative_to(ROOT / "data") or not logs.is_relative_to(ROOT / "logs"):
        raise ValueError("Output and logs must remain inside project directories")
    checkpoint = args.checkpoint.resolve()
    metadata = checkpoint_metadata(checkpoint)
    motion_path = args.motion.resolve()
    motions = sorted(motion_path.glob("*.csv")) if motion_path.is_dir() else [motion_path]
    if not motions or any(not motion.is_file() or motion.suffix != ".csv" for motion in motions):
        raise ValueError("Expected one CSV or a directory of CSV motions")
    protocol = {"checkpoint": str(checkpoint), "checkpoint_sha256": metadata["sha256"],
                "motions": [{"path": str(m), "sha256": sha256(m)} for m in motions],
                "reference_fps_after_stride": args.reference_fps, "frame_stride": args.frame_stride,
                "max_policy_steps": args.max_policy_steps, "action_delay_ms": args.action_delay_ms,
                "encoder": "a3_fast", "vendor_commit": VENDOR_COMMIT}
    if args.reference_buffer_ms:
        protocol["reference_buffer_ms"] = args.reference_buffer_ms
        protocol["buffer_scope"] = "SIMULATED_ARRIVALS_FORWARD_VELOCITY_GUARDED_TARGET_METRICS"
    joint_mode = getattr(args, "joint_reference_mode", None)
    if joint_mode:
        if args.reference_buffer_ms:
            raise ValueError("Cannot combine joint ablation and buffer")
        predictor = getattr(args, "joint_predictor", None)
        protocol["joint_ablation"] = dict(mode=joint_mode, predictor_sha256=sha256(predictor) if predictor else None,
            noise_std_rad=args.joint_noise_std, noise_seed=args.joint_noise_seed,
            scope="JOINT_ONLY_FUTURE_ORIENTATION_ORACLE_NOT_FULL_CAUSAL_TELEOP")
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "explicit_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"protocol": protocol, "runs": {}}
    if manifest["protocol"] != protocol:
        raise ValueError("Evaluation protocol changed: create a new output directory")
    atomic_json(manifest_path, manifest)
    if args.summarize_only:
        print(json.dumps(summarize_explicit(output), indent=2))
        return 0
    vendor = ROOT / "script/vendor/sonic_for_a3"
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""))
    for motion in motions:
        name = motion.stem
        previous = manifest["runs"].get(name, {})
        metrics_path = output / f"{name}.metrics.json"
        if previous.get("status") == "passed" and metrics_path.exists():
            if sha256(metrics_path) != previous["metrics_sha256"]:
                raise ValueError("Completed metrics changed")
            continue
        health = preflight()
        ordinal = int(previous.get("attempt", 0)) + 1
        run_logs = logs / name / f"attempt_{ordinal:03d}"
        run_logs.mkdir(parents=True, exist_ok=False)
        argv = [str(ROOT / "data/environments/a3-sonic/bin/python"), "-m", "script.a3.sim_entry"]
        if args.reference_buffer_ms:
            argv += ["--reference-buffer-ms", str(args.reference_buffer_ms),
                     "--buffer-trace", str(output / f"{name}.buffer.npz")]
        if joint_mode:
            argv += ["--joint-reference-mode", joint_mode, "--joint-noise-std", str(args.joint_noise_std),
                     "--joint-noise-seed", str(args.joint_noise_seed),
                     "--joint-trace", str(output / f"{name}.joint.npz")]
            if args.joint_predictor:
                argv += ["--joint-predictor", str(args.joint_predictor)]
        if args.capture_inputs:
            argv += ["--capture", str(output / f"{name}.inputs.npz")]
        argv += ["--", "--checkpoint", str(checkpoint), "--motion", str(motion),
                 "--encoder-mode", "a3_fast", "--csv-source-fps", str(args.reference_fps),
                 "--csv-frame-stride", str(args.frame_stride), "--batch-once",
                 "--action-delay-ms", str(args.action_delay_ms),
                 "--mjcf", str(vendor / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"),
                 "--metrics-out", str(metrics_path), "--timeseries-out", str(output / f"{name}.timeseries.json")]
        if args.max_policy_steps is not None:
            argv += ["--max-policy-steps", str(args.max_policy_steps)]
        row = {"status": "running", "attempt": ordinal, "argv": argv, "started_at": time.time(),
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "initial_health": health}
        manifest["runs"][name] = row
        atomic_json(manifest_path, manifest)
        with (run_logs / "console.log").open("w") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, run_logs, {"max_cpu_c": 90, "max_gpu_c": 85,
                                                        "sample_seconds": 5, "expires_at": None})
        row.update(returncode=code, reason=reason, finished_at=time.time())
        if code != 0 or reason is not None:
            row["status"] = "failed"
            atomic_json(manifest_path, manifest)
            return 2
        payload = json.loads(metrics_path.read_text())
        if payload["fps"] != 50 or len(payload["motions"]) != 1 or payload["motions"][0]["num_policy_steps"] <= 0:
            raise ValueError("Incomplete evaluation output")
        summary = payload["motions"][0]
        if not math.isfinite(summary["tracking"]["all_29_rmse"]):
            raise ValueError("Nonfinite tracking metric")
        # Publish completion only after simulator outputs survive a host restart.
        for suffix in ("metrics.json", "timeseries.json", "inputs.npz", "buffer.npz", "joint.npz"):
            artifact = output / f"{name}.{suffix}"
            if artifact.exists():
                with artifact.open("rb") as stream:
                    os.fsync(stream.fileno())
        from script.a3.checkpoint_store import fsync_directory
        fsync_directory(output)
        row.update(status="passed", metrics_sha256=sha256(metrics_path), summary=summary)
        atomic_json(manifest_path, manifest)
        print(json.dumps({"motion": name, "status": "passed", "fall": summary["fall"]}), flush=True)
    summarize_explicit(output)
    return 0


def summarize(output, logs):
    manifest = json.loads((output / "manifest.json").read_text())
    expected = {item["name"] for item in manifest["motion_files"]}
    summary = {"scope": manifest["scope"], "aggregation": "errors include frames after first fall",
               "runs": [], "paired_motions": {}}
    for run in manifest["runs"]:
        step = run["step"]
        if run["status"] != "finished" or run["returncode"] != 0 or run["reason"] is not None:
            raise ValueError("Cannot summarize incomplete or interrupted evaluation")
        payload = json.loads((output / f"step{step}_metrics.json").read_text())
        motions = payload["motions"]
        if len(motions) != len(expected) or {m["motion_name"] for m in motions} != expected:
            raise ValueError("Missing or duplicated motion coverage")
        fall_times = [m["fall_time_s"] for m in motions if m["fall"]]
        total_steps = sum(m["num_policy_steps"] for m in motions)
        if payload["fps"] != 50 or any(m["num_policy_steps"] <= 0 for m in motions):
            raise ValueError("Unexpected FPS or empty rollout")
        for motion in motions:
            rmse = motion["tracking"]["all_29_rmse"]
            if not math.isfinite(rmse):
                raise ValueError("Nonfinite joint error")
            row = {"num_policy_steps": motion["num_policy_steps"], "duration_s": motion["num_policy_steps"] / 50,
                   "fall": motion["fall"], "fall_time_s": motion["fall_time_s"],
                   "joint_rmse_rad_full_rollout": rmse,
                   "root_position_mean_m_full_rollout": motion["root_pose"]["position_mean_m"]}
            summary["paired_motions"].setdefault(motion["motion_name"], {})[str(step)] = row
        health = [json.loads(line) for line in (logs / f"step{step}" / "health.jsonl").read_text().splitlines()]
        summary["runs"].append({
            "step": step, "checkpoint_sha256": run["checkpoint_sha256"],
            "motion_count": len(motions), "no_fall_count": len(motions) - len(fall_times),
            "fall_count": len(fall_times), "fall_time_median_s": statistics.median(fall_times) if fall_times else None,
            "fall_time_min_s": min(fall_times) if fall_times else None,
            "fall_time_max_s": max(fall_times) if fall_times else None,
            "total_policy_steps": total_steps,
            "joint_rmse_rad_full_rollout": math.sqrt(sum(
                m["num_policy_steps"] * m["tracking"]["all_29_rmse"] ** 2 for m in motions) / total_steps),
            "elapsed_wall_seconds": run["finished_at"] - run["started_at"],
            "sampled_cpu_max_c": max(h["temperatures_c"]["x86_pkg_temp"] for h in health),
            "sampled_gpu_max_c": max(g["temperature.gpu"] for h in health for g in h["gpus"]),
        })
    if len(summary["runs"]) != 2 or any(
            row["500"]["num_policy_steps"] != row["2000"]["num_policy_steps"]
            for row in summary["paired_motions"].values()):
        raise ValueError("Paired rollout horizons differ")
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint; omitting preserves legacy500/2000")
    parser.add_argument("--motion", type=Path, default=ROOT / "script/vendor/sonic_for_a3/a3_data/agibot_a3")
    parser.add_argument("--reference-fps", type=float, default=30, help="FPS after row stride selection")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-policy-steps", type=int)
    parser.add_argument("--action-delay-ms", type=float, default=0)
    parser.add_argument("--reference-buffer-ms", type=float, default=0)
    parser.add_argument("--capture-inputs", action="store_true")
    parser.add_argument("--joint-reference-mode", choices=("oracle_aligned", "hold", "cv", "smooth_cv", "E12", "E14"))
    parser.add_argument("--joint-predictor", type=Path)
    parser.add_argument("--joint-noise-std", type=float, default=0)
    parser.add_argument("--joint-noise-seed", type=int, default=0)
    args = parser.parse_args()
    if args.reference_buffer_ms and args.reference_buffer_ms < 180:
        parser.error("A3-fast causal replay requires at least180ms coverage")
    if args.reference_fps <= 0 or args.frame_stride < 1 or (args.max_policy_steps is not None and args.max_policy_steps < 1):
        parser.error("Invalid sampling or rollout length")
    if args.checkpoint is not None:
        return evaluate_explicit(args)
    output, logs = args.output.resolve(), args.logs.resolve()
    if not output.is_relative_to(ROOT / "data") or not logs.is_relative_to(ROOT / "logs"):
        parser.error("Evaluation output and logs must stay inside project data and logs")
    if args.summarize_only:
        summarize(output, logs)
        return 0
    if output.exists() or logs.exists():
        raise FileExistsError("Use a fresh evaluation directory")
    vendor = ROOT / "script/vendor/sonic_for_a3"
    if command(["git", "-C", str(vendor), "rev-parse", "HEAD"])["stdout"] != VENDOR_COMMIT:
        raise ValueError("Vendor version changed")
    occupied = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if occupied["returncode"] != 0 or occupied["stdout"]:
        raise RuntimeError("Cannot start evaluation: GPU unreadable or occupied")
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    motion_dir = vendor / "a3_data/agibot_a3"
    sources = [(500, "E005_resume_154to2000"), (2000, "E006_auto_03_s1800")]
    guard = {"max_cpu_c": 90, "max_gpu_c": 85, "sample_seconds": 5,
             "expires_at": time.time() + 900}
    manifest = {"vendor_commit": VENDOR_COMMIT, "guard": guard,
                "scope": "in-training-data paired MuJoCo sim2sim; not held-out or real-robot evaluation",
                "motion_files": [{"name": p.name, "sha256": sha256(p)}
                                 for p in sorted(motion_dir.glob("*.csv"))],
                "runs": []}
    if len(manifest["motion_files"]) != 20:
        raise ValueError("Expected pinned selected20 dataset")
    for step, run_id in sources:
        checkpoint = ROOT / "data/training/a3_20261009" / run_id / f"model_step_{step:06d}.pt"
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
        if sha256(checkpoint) != sidecar["sha256"] or checkpoint.stat().st_size != sidecar["bytes"]:
            raise ValueError("Checkpoint integrity mismatch")
        before = sample()
        if (before["temperatures_c"].get("x86_pkg_temp", 90) >= 90 or
                before["gpu_query"]["returncode"] != 0 or not before["gpus"] or
                any((gpu.get("temperature.gpu") or 85) >= 85 for gpu in before["gpus"])):
            raise RuntimeError("Unreadable or high temperature before evaluation")
        run_logs = logs / f"step{step}"
        run_logs.mkdir()
        argv = [str(ROOT / "data/environments/a3-sonic/bin/python"),
                str(vendor / "gear_sonic/scripts/sim2sim_a3_mujoco.py"),
                "--checkpoint", str(checkpoint), "--motion", str(motion_dir),
                "--encoder-mode", "a3_fast", "--csv-source-fps", "30", "--csv-frame-stride", "4",
                "--mjcf", str(vendor / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"),
                "--batch-once", "--action-delay-ms", "0", "--metrics-out", str(output / f"step{step}_metrics.json")]
        record = {"step": step, "checkpoint": str(checkpoint), "checkpoint_sha256": sidecar["sha256"],
                  "config_sha256": sha256(checkpoint.parent / "config.yaml"),
                  "argv": argv, "started_at": time.time(), "status": "running"}
        manifest["runs"].append(record)
        atomic_json(output / "manifest.json", manifest)
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (run_logs / "console.log").open("w") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, run_logs, guard)
        record.update(status="finished", returncode=code, reason=reason, finished_at=time.time())
        atomic_json(output / "manifest.json", manifest)
        print(json.dumps(record), flush=True)
        if code != 0 or reason is not None:
            return 2
    summarize(output, logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
