"""Bounded, thermally guarded paired MuJoCo evaluation of trusted A3 checkpoints."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from script.a3.autoresume import ROOT, VENDOR_COMMIT, monitor_child
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import command, sample


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
    args = parser.parse_args()
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
