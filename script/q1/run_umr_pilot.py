#!/usr/bin/env python3
"""Run bounded upstream UMR stages with an isolated dependency overlay and logs."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from dataset_input import identity


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=ROOT / "data/third_party/umr_q1_20261009")
    parser.add_argument("--overlay", type=Path, default=ROOT / "data/environments/umr_q1_overlay")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    args.vendor = args.vendor.resolve()
    args.overlay = args.overlay.resolve()
    args.config = args.config.resolve()
    args.log_dir = args.log_dir.resolve()
    config = json.loads(args.config.read_text())
    if config["correspondence"]["dataset"]["num_points"] > 512 or config["correspondence"]["train"]["epochs"] > 100:
        raise ValueError("This runner is bounded to a small correspondence pilot")
    if not 1 <= config["motion"].get("max_frames", 0) <= 64:
        raise ValueError("This runner requires 1-64 motion frames")
    if not (args.vendor / "LICENSE").is_file():
        parser.error("Missing upstream LICENSE; restore the checkout before running")
    status_path = args.config.parent / f"pilot_status_attempt{args.attempt}.json"
    if status_path.exists():
        parser.error("Preserve prior attempt; increment --attempt")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    environment["PYTHONPATH"] = str(args.overlay) + os.pathsep + environment.get("PYTHONPATH", "")
    report = {"usage": "TEST_ONLY", "training_allowed": False, "driver": identity(__file__),
              "config": identity(args.config), "upstream_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=args.vendor, text=True).strip(),
              "upstream_license": identity(args.vendor / "LICENSE"), "stages": [],
              "surface_correspondence_training_is_not_control_policy": True,
              "policy_trained": False, "hardware_control": False, "backup_status": "LOCAL_ONLY",
              "overlay": str(args.overlay), "python": sys.version}
    for stage, timeout in (("build", 180), ("train", 360), ("retarget", 240)):
        command = [sys.executable, str(args.vendor / "scripts/humanoid_retarget_pipeline.py"),
                   "--config", str(args.config), "--stage", stage, "--skip-view"]
        log = args.log_dir / f"{stage}_attempt{args.attempt}.log"
        row = {"stage": stage, "command": command, "log": str(log), "status": "RUNNING", "timeout_s": timeout}
        report["stages"].append(row)
        status_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Starting UMR {stage}, timeout {timeout}s, log {log}", flush=True)
        started = time.perf_counter()
        try:
            with log.open("x") as stream:
                process = subprocess.Popen(command, cwd=args.vendor, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                row["pid"] = process.pid
                status_path.write_text(json.dumps(report, indent=2) + "\n")
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            row.update(status="COMPLETE" if returncode == 0 else "FAILED", exit_code=returncode)
        except subprocess.TimeoutExpired:
            row.update(status="TIMEOUT", exit_code=None)
        row["wall_s"] = time.perf_counter() - started
        row["log_identity"] = identity(log)
        status_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"UMR {stage}: {row['status']} ({row['wall_s']:.1f}s)", flush=True)
        if row["status"] != "COMPLETE":
            return 1
    report["retarget_artifact"] = identity(Path(config["retarget"]["out"]))
    status_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
