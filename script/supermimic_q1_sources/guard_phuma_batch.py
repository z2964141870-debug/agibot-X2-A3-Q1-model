#!/usr/bin/env python3
"""Prepare one separate PHUMA batch with a shared-memory reserve for training."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys

from runtime_guard import StorageBudget, guarded_run


class MemoryLimit(RuntimeError):
    pass


class MemoryBoundBudget:
    def __init__(self, storage, maximum_bytes, expected_oom, cgroup=Path("/sys/fs/cgroup/memory")):
        self.storage = storage
        self.maximum_bytes = maximum_bytes
        self.expected_oom = expected_oom
        self.cgroup = cgroup

    def memory(self):
        current = int((self.cgroup / "memory.usage_in_bytes").read_text())
        counters = dict(line.split() for line in (self.cgroup / "memory.oom_control").read_text().splitlines())
        oom = int(counters["oom_kill"])
        if current >= self.maximum_bytes or oom != self.expected_oom:
            raise MemoryLimit(f"Data worker reserve reached: shared_bytes={current}, oom_kill={oom}")
        return {"shared_bytes": current, "oom_kill": oom, "maximum_bytes": self.maximum_bytes}

    def check(self):
        self.memory()
        evidence = self.storage.check()
        evidence["shared_memory"] = self.memory()
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--maximum-memory-gib", type=float, default=28.75)
    args = parser.parse_args()
    if not 0 < args.maximum_memory_gib <= 29:
        parser.error("The data worker must leave at least 1 GiB below the preserved trainer's 30 GiB limit")
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("A new independent batch under the Q1 data directory is required")
    selection = json.loads(args.selection.read_text())
    if any(row["split"] != "train" for row in selection["selected"]):
        raise ValueError("This concurrent preparation accepts additional training candidates only")
    if args.count < 1 or args.offset < 0 or args.offset + args.count > len(selection["selected"]):
        raise ValueError("Invalid fixed candidate slice")
    maximum_bytes = int(args.maximum_memory_gib * 1024 ** 3)
    source = Path(selection["archive"]["path"]).resolve(strict=True).parent
    storage = StorageBudget(output, source, 100 * 1024 ** 3, 12 * 1024 ** 3, 250 * 1024 ** 3)
    budget = MemoryBoundBudget(storage, maximum_bytes, args.expected_oom_kill)
    budget.memory()
    output.mkdir()
    os.nice(10)
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(root / "script/supermimic_q1_phuma/prepare_expansion_batch.py"),
               "--root", str(root), "--selection", str(args.selection.resolve()),
               "--vendor", str(args.vendor.resolve()), "--output", str(output / "batch"),
               "--offset", str(args.offset), "--count", str(args.count),
               "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json",
                         root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "batch/references", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
