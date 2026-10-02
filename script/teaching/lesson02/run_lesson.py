"""Run bounded training and an independent reload check in YUANQI directories."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment-id", required=True)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--timeout", type=int, default=600)
parser.add_argument("--python", default="/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python")
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_id):
    parser.error("experiment-id may contain letters, numbers, underscores and hyphens")
if min(args.num_envs, args.iterations, args.timeout) <= 0:
    parser.error("counts and timeout must be positive")
root = Path(__file__).resolve().parents[3]
run_dir = root / "data/training" / args.experiment_id
log_dir = root / "logs" / args.experiment_id
if run_dir.exists() or log_dir.exists():
    raise FileExistsError("Use a new experiment ID; previous results will be preserved.")
log_dir.mkdir(parents=True)
driver = Path(__file__).with_name("cartpole_ppo.py")
shutil.copyfile(driver, log_dir / "driver_source.py")
common = [args.python, "-u", str(driver), "--headless", "--device", "cuda:0",
          "--run-dir", str(run_dir), "--log-dir", str(log_dir / "tensorboard"),
          "--num-envs", str(args.num_envs), "--iterations", str(args.iterations),
          "--seed", "42", "--eval-seed", "10001"]


def execute(mode):
    command = common + ["--mode", mode, "--kit_args=--/log/file=" + str(log_dir / f"kit_{mode}.log")]
    if mode == "evaluate":
        command += ["--checkpoint", str(run_dir / "checkpoints/final.pt")]
    status_path = log_dir / f"job_{mode}.json"
    state = {"command": command, "cwd": str(root), "status": "RUNNING",
             "started_at": datetime.now(timezone.utc).isoformat()}
    with (log_dir / f"{mode}.log").open("w") as output:
        job = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        state["pid"] = job.pid
        status_path.write_text(json.dumps(state, indent=2))
        print(f"{mode}: PID {job.pid}; log {output.name}", flush=True)
        try:
            returncode = job.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            state["timed_out"] = True
            os.killpg(job.pid, signal.SIGTERM)
            try:
                returncode = job.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL)
                returncode = job.wait(timeout=10)
    marker = "YUANQI_TRAIN_COMPLETE" if mode == "train" else "YUANQI_RELOAD_VERIFIED"
    contents = (log_dir / f"{mode}.log").read_text(errors="replace")
    success = returncode == 0 and marker in contents and not state.get("timed_out")
    state.update(returncode=returncode, expected_marker_found=marker in contents,
                 status="PASSED" if success else "FAILED",
                 finished_at=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(state, indent=2))
    print("\n".join(contents.splitlines()[-18:]), flush=True)
    if not success:
        raise RuntimeError(f"{mode} did not finish verified work; inspect {log_dir}")


try:
    execute("train")
    execute("evaluate")
except Exception as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
print("LESSON_TRAIN_AND_RELOAD_PASSED", flush=True)
