"""Run an isolated, zero-update cross-robot command with the Q1 guards."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path("/opt/data/private/ZHY")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from run_trial import Q1Budget, process_identity
from runtime_guard import guarded_run, save


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


class Budget(Q1Budget):
    def check(self):
        result = super().check()
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.args.expected_boot:
            raise RuntimeError("Original boot identity changed")
        if result["gpu"]["uuid"] != self.args.expected_gpu:
            raise RuntimeError("Original GPU identity changed")
        if time.monotonic() - self.started > self.args.wall_seconds:
            raise RuntimeError("Command wall limit reached")
        return result


def run(args):
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status="INITIALIZING", process=process_identity(),
                  code_commit=args.code_commit, command=args.command,
                  cwd=str(args.cwd), started_utc=datetime.now(timezone.utc).isoformat(),
                  automatic_retry=False, optimizer_updates=0, goal_complete=False)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader"], text=True)
        if raw.strip():
            raise RuntimeError("GPU already has compute processes")
        cgroup = Path("/sys/fs/cgroup/memory")
        usage = int((cgroup / "memory.usage_in_bytes").read_text())
        limit = int((cgroup / "memory.limit_in_bytes").read_text())
        if usage > 24 * 1024**3 or limit <= 30 * 1024**3:
            raise RuntimeError("Original24GiB initial admission not satisfied")
        budget_args = SimpleNamespace(output=args.output, runtime=args.source,
            mode="independent", expected_oom_kill=0, expected_boot=args.expected_boot,
            expected_gpu=args.expected_gpu, wall_seconds=args.wall_seconds)
        budget = Budget(budget_args)
        record["initial_resources"] = budget.check()
        record["inputs"] = [identity(p) for p in args.input]
        record["status"] = "RUNNING"
        save(args.output / "status.json", record)
        result = guarded_run(args.command, args.cwd, budget, args.output / "guard.json",
            args.output / "stdout.log", check_interval_s=2, grace_s=10,
            stop_requested=lambda: requested)
        record["guard"] = result
        if result["status"] != "WORKER_COMPLETED" or result["cleanup"]["worker_return_code"]:
            raise RuntimeError("Command did not complete; retain original logs")
        if [identity(p) for p in args.input] != record["inputs"]:
            raise RuntimeError("Frozen command inputs changed")
        record["status"] = "COMPLETED"
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "status.json", record)
        lock.close()
    print(json.dumps(record, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("output", "source", "cwd"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--expected-boot", default="4d859008-4c83-4a6d-814d-cbb9dfe885c9")
    parser.add_argument("--expected-gpu", default="GPU-26028785-9fb9-45af-e614-4cc53af15d08")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.command = args.command[1:] if args.command[:1] == ["--"] else args.command
    args.output = args.output.resolve()
    args.source = args.source.resolve(strict=True)
    args.cwd = args.cwd.resolve(strict=True)
    args.input = [p.resolve(strict=True) for p in args.input]
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use a new output under the sole Q1 experiment root")
    if not all(p.is_relative_to(ROOT) for p in [args.source, args.cwd, *args.input]):
        parser.error("All command inputs must stay inside the sole project root")
    if (not args.command or not 0 < args.wall_seconds <= 7200
            or len(args.code_commit) != 40
            or any(c not in "0123456789abcdef" for c in args.code_commit)):
        parser.error("Require a command, bounded wall time and published full commit")
    run(args)


if __name__ == "__main__":
    main()
