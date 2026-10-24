#!/usr/bin/env python3
"""Run one owned worker with Q1 storage budgets and guaranteed child cleanup."""

import argparse
from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time


class StorageLimit(RuntimeError):
    pass


def allocated_tree(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Budget root must be a real directory: {root}")
    seen, allocated, logical, links = set(), 0, 0, 0
    for base, directories, files in os.walk(root, followlinks=False):
        for path in [Path(base), *(Path(base) / name for name in directories + files)]:
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            allocated += info.st_blocks * 512
            logical += info.st_size
            links += stat.S_ISLNK(info.st_mode)
    return {"root": str(root.resolve()), "unique_allocated_bytes": allocated,
            "unique_logical_bytes": logical, "unique_inodes": len(seen),
            "symlinks_not_followed": links}


@dataclass
class StorageBudget:
    experiment: Path
    source: Path
    minimum_free_bytes: int
    maximum_experiment_bytes: int
    maximum_source_bytes: int

    def check(self):
        experiment = allocated_tree(self.experiment)
        source = allocated_tree(self.source)
        free = {str(root.resolve()): shutil.disk_usage(root).free
                for root in (self.experiment, self.source)}
        evidence = {"experiment": experiment, "source": source, "free_bytes": free}
        if min(free.values()) < self.minimum_free_bytes:
            raise StorageLimit("Free space below the admitted reserve: " + json.dumps(evidence))
        if experiment["unique_allocated_bytes"] > self.maximum_experiment_bytes:
            raise StorageLimit("Experiment storage exceeds budget: " + json.dumps(evidence))
        if source["unique_allocated_bytes"] > self.maximum_source_bytes:
            raise StorageLimit("Source storage exceeds budget: " + json.dumps(evidence))
        return evidence


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def signal_group(child, signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def cleanup(child, grace_s):
    if child is None:
        return {"worker_started": False, "worker_reaped": True}
    signal_group(child, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while child.poll() is None and time.monotonic() < deadline:
        try:
            child.wait(timeout=min(1., max(.01, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    # The worker can exit before one of its subprocesses; close the entire owned
    # process group even after reaping the direct child.
    signal_group(child, signal.SIGKILL)
    child.wait()
    return {"worker_started": True, "worker_pid": child.pid,
            "worker_group": child.pid, "worker_reaped": True,
            "worker_return_code": child.returncode}


def guarded_run(command, workdir, budget, status_path, log_path,
                check_interval_s=5., grace_s=45., stop_requested=None,
                progress_path=None, maximum_idle_s=900.):
    if (not command or check_interval_s <= 0 or maximum_idle_s <= 0
            or not 0 <= grace_s <= 60):
        raise ValueError("Invalid guarded worker command or timing")
    status_path, log_path = Path(status_path), Path(log_path)
    if status_path.exists() or log_path.exists():
        raise FileExistsError("Guarded runs are single-use; existing status/log cannot be resumed")
    if status_path.resolve() == log_path.resolve():
        raise ValueError("Status and stdout log must be distinct")
    record = {"status": "PREFLIGHT", "command": command, "cwd": str(workdir),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "controller_pid": os.getpid(), "storage_status": "LOCAL_ONLY",
              "last_storage_check": None, "error": None}
    with status_path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    child = None
    try:
        record["last_storage_check"] = budget.check()
        if stop_requested and stop_requested():
            record["status"] = "STOPPED_SIGNAL"
            return record
        with log_path.open("x") as stream:
            child = subprocess.Popen(command, cwd=workdir, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status="RUNNING", worker_pid=child.pid, worker_group=child.pid)
            save(status_path, record)
            next_check = time.monotonic() + check_interval_s
            last_progress, marker = time.monotonic(), None
            while True:
                if stop_requested and stop_requested():
                    record["status"] = "STOPPED_SIGNAL"
                    break
                try:
                    code = child.wait(timeout=min(1., check_interval_s))
                    record["status"] = "WORKER_COMPLETED" if code == 0 else "WORKER_FAILED"
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= next_check:
                        record["last_storage_check"] = budget.check()
                        if progress_path is not None:
                            path = Path(progress_path)
                            changed = path.stat().st_mtime_ns if path.exists() else None
                            if changed != marker:
                                last_progress, marker = time.monotonic(), changed
                            if time.monotonic() - last_progress > maximum_idle_s:
                                raise RuntimeError("Worker made no progress within the admitted interval")
                        save(status_path, record)
                        next_check = time.monotonic() + check_interval_s
            if record["status"] == "WORKER_COMPLETED":
                record["last_storage_check"] = budget.check()
    except BaseException as exc:
        record.update(status="STOPPED_STORAGE" if isinstance(exc, StorageLimit) else "FAILED",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Cleanup precedes status I/O: even a full disk or broken status path
        # cannot leave our training worker running.
        record["cleanup"] = cleanup(child, grace_s)
        record["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(status_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--goal-config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    goal = json.loads(args.goal_config.read_text())
    storage = goal["storage"]
    budget = StorageBudget(args.experiment_root, args.source_root,
                           storage["minimum_free_bytes"],
                           storage["maximum_experiment_unique_allocated_bytes"],
                           storage["maximum_new_source_unique_allocated_bytes"])
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    result = guarded_run(command, args.workdir, budget, args.status, args.log,
                         stop_requested=lambda: requested, progress_path=args.progress_file)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
