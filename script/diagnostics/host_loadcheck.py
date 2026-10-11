"""Bounded, non-resuming CPU/memory diagnosis with durable host evidence."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from script.a3.host_health import append_record, command, sample
from script.a3.idle_reboot_watch import save_json, utc


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/experiments/host_loadcheck_20261011_E01"
LOGS = ROOT / "logs/host_loadcheck_20261011_E01"
TOOLROOT = ROOT / "data/tools/host_loadcheck_20261011/root"
STRESS = TOOLROOT / "usr/bin/stress-ng"
FATAL = re.compile(r"Hardware Error|Machine check events logged|Kernel panic|"
                   r"NVRM: Xid|GPU has fallen off|Out of memory: Killed process|"
                   r"BUG:|watchdog:.*lockup", re.I)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(TOOLROOT / "usr/lib/x86_64-linux-gnu")
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def protection(health, high_count):
    cpu = health.get("temperatures_c", {}).get("x86_pkg_temp")
    gpus = health.get("gpus", [])
    if cpu is None or not gpus or any(gpu.get("temperature.gpu") is None for gpu in gpus):
        return "TELEMETRY_UNREADABLE", high_count
    hot = cpu >= 90 or any(gpu["temperature.gpu"] >= 85 for gpu in gpus)
    high_count = high_count + 1 if hot else 0
    if high_count >= 2:
        return "TEMPERATURE_PROTECTION", high_count
    if health["memory_bytes"].get("MemAvailable", 0) < 4 * 1024**3:
        return "MEMORY_RESERVE_PROTECTION", high_count
    return None, high_count


def cpu_times():
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if re.match(r"cpu(?:\d+)? ", line):
            fields = line.split()
            values = list(map(int, fields[1:9]))
            result[fields[0]] = (sum(values), values[3] + values[4])
    return result


def utilization(previous, current):
    result = {}
    for name, (total, idle) in current.items():
        if name in previous:
            delta = total - previous[name][0]
            result[name] = 100 * (delta - idle + previous[name][1]) / delta if delta else None
    return result


def initialize():
    if (OUTPUT / "job.json").exists():
        raise FileExistsError("Existing experiment must not be reset")
    if not STRESS.is_file():
        raise FileNotFoundError(STRESS)
    version = subprocess.run([str(STRESS), "--version"], env=environment(),
                             capture_output=True, text=True, check=True).stdout.strip()
    health = sample()
    reason, count = protection(health, 0)
    if reason or count:
        raise RuntimeError("Initial health is unsafe: " + str(reason or "HIGH_TEMPERATURE"))
    stages = []
    for workers, load in [(1, 25), (1, 50), (1, 100), (2, 100), (4, 100)]:
        stages.append(dict(id=f"cpu_w{workers}_load{load}", kind="cpu", workers=workers,
                           load_percent_per_worker=load, durations_seconds=[30, 120, 600]))
    for mib in (256, 1024):
        stages.append(dict(id=f"memory_{mib}MiB", kind="memory", mib=mib,
                           durations_seconds=[30, 120, 600]))
    job = dict(experiment="HOST_LOADCHECK_20261011_E01", initialized_at_utc=utc(time.time()),
               initial_boot=health["boot_id"], tool=str(STRESS), version=version,
               tool_sha256=sha(STRESS), code_sha256=sha(Path(__file__)), stages=stages,
               sample_seconds=5, cooldown_seconds=300, max_cpu_c=90, max_gpu_c=85,
               consecutive_hot_samples=2, memory_reserve_gib=4, auto_resume=False,
               a3_training_allowed=False, reinforcement_learning_allowed=False,
               stop_on_first_failure=True, backup_status="LOCAL_ONLY")
    save_json(OUTPUT / "job.json", job)
    save_json(OUTPUT / "state.json", dict(status="READY", stages={}, boot_id=health["boot_id"]))
    append_record(LOGS / "health.jsonl", dict(stage="registration", health=health))
    print(json.dumps(job, indent=2), flush=True)


def stop(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def arguments(stage, seconds):
    args = [str(STRESS), "--timeout", f"{seconds}s", "--verify", "--metrics-brief", "--abort"]
    if stage["kind"] == "cpu":
        args += ["--cpu", str(stage["workers"]), "--cpu-load", str(stage["load_percent_per_worker"]),
                 "--cpu-method", "int64"]
    else:
        args += ["--vm", "1", "--vm-bytes", f"{stage['mib']}M", "--vm-keep", "--vm-method", "write64"]
    return args


def execute(job, state, stage, seconds):
    name = f"{stage['id']}_{seconds}s"
    path = LOGS / f"{name}.log"
    before = sample()
    reason, high = protection(before, 0)
    if reason or high:
        raise RuntimeError(reason or "HIGH_INITIAL_TEMPERATURE")
    if before["boot_id"] != job["initial_boot"]:
        raise RuntimeError("BOOT_CHANGED_BEFORE_STAGE")
    start = time.monotonic()
    started_epoch = time.time()
    started = utc(started_epoch)
    record = dict(status="RUNNING", started_at_utc=started, boot_id=before["boot_id"],
                  command=arguments(stage, seconds), log=str(path.relative_to(ROOT)))
    state.update(status="RUNNING", current_stage=name)
    state["stages"][name] = record
    save_json(OUTPUT / "state.json", state)
    append_record(LOGS / "health.jsonl", dict(stage=name, phase="before_launch", health=before))
    previous = cpu_times()
    child = None
    reason = None
    try:
        with path.open("w") as stream:
            child = subprocess.Popen(record["command"], env=environment(), stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state["pid"] = child.pid
            save_json(OUTPUT / "state.json", state)
            while True:
                health = sample()
                current = cpu_times()
                kernel = command(["journalctl", "-k", "-b", "--since", f"@{int(started_epoch)}", "--no-pager", "-o", "cat"])
                critical = [line for line in kernel.get("stdout", "").splitlines() if FATAL.search(line)]
                append_record(LOGS / "health.jsonl", dict(stage=name, phase="running", health=health,
                              cpu_busy_percent=utilization(previous, current),
                              kernel_readable=kernel["returncode"] == 0, critical_kernel_lines=critical))
                previous = current
                os.fsync(stream.fileno())
                reason, high = protection(health, high)
                if health["boot_id"] != job["initial_boot"]:
                    reason = "BOOT_CHANGED"
                if critical:
                    reason = "CRITICAL_KERNEL_EVENT"
                if kernel["returncode"] != 0:
                    reason = "KERNEL_TELEMETRY_UNREADABLE"
                if reason:
                    break
                if child.poll() is not None:
                    break
                if time.monotonic() - start > seconds + 30:
                    reason = "WORKLOAD_TIMEOUT"
                    break
                time.sleep(5)
            stop(child)
        text = path.read_text()
        success = child.returncode == 0 and "successful run completed" in text.lower()
        if not reason and not success:
            reason = "WORKLOAD_FAILED_OR_UNVERIFIED"
        record.update(status="FAILED" if reason else "PASSED", ended_at_utc=utc(time.time()),
                      elapsed_seconds=time.monotonic()-start, returncode=child.returncode,
                      stop_reason=reason, log_sha256=sha(path), log_bytes=path.stat().st_size)
        state.pop("pid", None)
        save_json(OUTPUT / "state.json", state)
        print(json.dumps(dict(stage=name, result=record)), flush=True)
        if reason:
            raise RuntimeError(reason)
    finally:
        if child is not None:
            stop(child)


def cooldown(job, state):
    state.update(status="COOLDOWN", cooldown_started_at_utc=utc(time.time()))
    save_json(OUTPUT / "state.json", state)
    end = time.monotonic() + job["cooldown_seconds"]
    high = 0
    while time.monotonic() < end:
        health = sample()
        append_record(LOGS / "health.jsonl", dict(stage="cooldown", health=health))
        reason, high = protection(health, high)
        if reason or health["boot_id"] != job["initial_boot"]:
            raise RuntimeError(reason or "BOOT_CHANGED")
        time.sleep(min(5, max(0, end-time.monotonic())))


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = json.loads((OUTPUT / "job.json").read_text())
        state = json.loads((OUTPUT / "state.json").read_text())
        if state["status"] != "READY":
            raise RuntimeError("No automatic resumption or repeat execution")
        if sha(Path(__file__)) != job["code_sha256"] or sha(STRESS) != job["tool_sha256"]:
            raise RuntimeError("Registered code or binary changed")
        try:
            for index, stage in enumerate(job["stages"]):
                for seconds in stage["durations_seconds"]:
                    execute(job, state, stage, seconds)
                if index != len(job["stages"])-1:
                    cooldown(job, state)
            state.update(status="COMPLETED", ended_at_utc=utc(time.time()))
        except BaseException as error:
            state.update(status="STOPPED", stop_reason=str(error), ended_at_utc=utc(time.time()))
            raise
        finally:
            save_json(OUTPUT / "state.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    initialize() if args.initialize else run()
