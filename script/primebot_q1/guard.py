"""One-use Q1 phase with the existing cleanup, memory and storage protections."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import HERE, ROOT, identity, now, require, save

sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from runtime_guard import StorageBudget, guarded_run
from guard_phuma_batch import MemoryBoundBudget


def check_initial_cache(cache_bytes, maximum_bytes):
    require(type(maximum_bytes) is int and 0 < maximum_bytes <= 4 * 1024**3,
            "Profile cache limit must be a positive integer at most4GiB")
    require(type(cache_bytes) is int and cache_bytes >= 0, "Profile cache measurement is invalid")
    require(cache_bytes <= maximum_bytes, "Initial cache exceeds the approved profile admission")
    return dict(cache_bytes=cache_bytes, maximum_cache_bytes=maximum_bytes)


def profile_cache_admission(maximum_bytes):
    fields = dict(line.split() for line in Path("/sys/fs/cgroup/memory/memory.stat").read_text().splitlines())
    return check_initial_cache(int(fields["total_cache"]), maximum_bytes)


class Budget:
    def __init__(self, config, output, seconds, maximum_bytes):
        self.config, self.started, self.seconds = config, time.monotonic(), seconds
        source_bytes = config.get("maximum_source_bytes", 64 * 1024**2)
        require(type(source_bytes) is int and 0 < source_bytes <= 256 * 1024**2,
                "Source storage budget must be a positive integer at most256MiB")
        self.inner = MemoryBoundBudget(StorageBudget(output, Path(config["adapted_source"]),
            100 * 1024**3, maximum_bytes, source_bytes), 30 * 1024**3, 0)
        require(type(config.get("record_memory_breakdown", False)) is bool, "Invalid memory recorder setting")
        self.recorder = None
        if config.get("record_memory_breakdown", False):
            sys.path.insert(0, str(HERE.parent / "q1_five_stage"))
            from memory_watch import GuardMemoryRecorder
            self.recorder = GuardMemoryRecorder(Path("/proc"), Path("/sys/fs/cgroup/memory"),
                                               os.getpid(), output / "memory.jsonl")

    def check(self):
        if self.recorder is not None:
            self.recorder.record()
        evidence = self.inner.check()
        require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == self.config["expected_boot"],
                "Frozen server boot changed")
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,temperature.gpu,gpu_recovery_action,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip().split(",")
        gpu, temperature, recovery, used = [v.strip() for v in raw]
        require(gpu == self.config["expected_gpu"] and recovery == "None", "GPU identity or health changed")
        require(int(temperature) < 86, "GPU temperature reserve reached")
        require(time.monotonic() - self.started < self.seconds, "Phase wall-time guard reached")
        evidence["gpu"] = dict(uuid=gpu, temperature_c=int(temperature), recovery=recovery, used_mib=int(used))
        return evidence


def run(config, output, command, seconds, maximum_bytes, inputs=(), maximum_initial_cache_bytes=None):
    require(output.resolve().is_relative_to(ROOT / "data/supermimic_for_q1"), "Invalid phase output root")
    output.mkdir(parents=True, exist_ok=False)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    record = dict(status="INITIALIZING", started_utc=now(), command=command,
                  automatic_retry=False, user_accepted_checkpoint=None)
    budget = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                  "--format=csv,noheader"], text=True).strip(), "GPU is occupied")
        memory = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
        require(memory <= 24 * 1024**3, "Initial memory exceeds the original24GiB admission")
        source = Path(config["adapted_source"])
        paths = sorted(set([*HERE.glob("*.py"), *source.rglob("*.py"), *map(Path, inputs)]))
        record["inputs"] = [identity(p) for p in paths if "__pycache__" not in p.parts]
        os.environ.update(PYTHONPATH=os.pathsep.join((str(HERE), str(source / "src"))),
            PYTHONUNBUFFERED="1", MUJOCO_GL="egl", OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="1")
        if maximum_initial_cache_bytes is not None:
            record["cache_admission"] = profile_cache_admission(maximum_initial_cache_bytes)
        record["status"] = "RUNNING"
        save(output / "status.json", record)
        budget = Budget(config, output, seconds, maximum_bytes)
        result = guarded_run(command, ROOT, budget,
            output / "guard.json", output / "stdout.log", check_interval_s=2, grace_s=10,
            stop_requested=lambda: stopped)
        record["guard"] = result
        require(result["status"] == "WORKER_COMPLETED" and result["cleanup"]["worker_return_code"] == 0,
                "Guarded worker failed; preserve the original phase")
        require([identity(p["path"]) for p in record["inputs"]] == record["inputs"], "Frozen phase input changed")
        record["status"] = "COMPLETED"
        return record
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if budget is not None and budget.recorder is not None:
                budget.recorder.close()
            memory_summary = output / "memory.summary.json"
            if memory_summary.exists():
                record["memory_observation"] = dict(summary=identity(memory_summary),
                                                    samples=identity(output / "memory.jsonl"))
            record["ended_utc"] = now()
            save(output / "status.json", record)
        finally:
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--maximum-bytes", type=int, required=True)
    parser.add_argument("--maximum-initial-cache-bytes", type=int)
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(command and 0 < args.seconds <= 30 * 86400, "Invalid bounded phase command")
    run(json.loads(args.config.read_text()), args.output, command, args.seconds, args.maximum_bytes,
        [args.config, *args.input], args.maximum_initial_cache_bytes)
