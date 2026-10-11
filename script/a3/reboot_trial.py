"""A single authorized A3 smoke run with no recovery after interruption."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from script.a3.host_health import append_record, command, sample
from script.a3.idle_reboot_watch import save_json, utc
from script.diagnostics.host_loadcheck import FATAL, protection


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/experiments/a3_reboot_trial_20261011_E22"
LOGS = ROOT / "logs/a3_reboot_trial_20261011_E22"
UNIT = "yuanqi-a3-reboot-trial-E22.service"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def launch_allowed(state, job, boot):
    return state.get("status") == "READY" and job["initial_boot"] == boot


def register():
    if (OUTPUT / "trial_job.json").exists() or (OUTPUT / "launch_state.json").exists():
        raise FileExistsError("Existing launch registration cannot be reset")
    if list((ROOT / "data/training/a3_finetune_20261010").glob("R10_*")):
        raise FileExistsError("R10 is already allocated")
    health = sample()
    reason, hot = protection(health, 0)
    if reason or hot:
        raise RuntimeError(reason or "HIGH_INITIAL_TEMPERATURE")
    gpu = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    if gpu["returncode"] != 0 or gpu["stdout"]:
        raise RuntimeError("GPU occupied or unreadable")
    source = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
    job = dict(experiment="A3_REBOOT_TRIAL_20261011_E22", trial="R10",
               registered_at_utc=utc(time.time()), initial_boot=health["boot_id"],
               source=str(source), source_sha256=digest(source), source_bytes=source.stat().st_size,
               target_step=2, num_envs=16, num_mini_batches=4, num_learning_epochs=1,
               lr_scale=0.1, seed=0, save_frequency=1, cpu_quota_percent=100,
               max_cpu_c=90, max_gpu_c=85, sample_seconds=5,
               consecutive_hot_samples=2, memory_reserve_gib=4,
               runtime_limit_seconds=900, auto_resume=False,
               repeat_launch_allowed=False, stop_on_first_fault=True,
               a3_training_authorized=True, evaluation_authorized=False,
               runner_sha256=digest(Path(__file__)),
               reused_code_sha256={p:digest(ROOT / p) for p in (
                   "script/a3/corrected_trial.py", "script/a3/verified_finetune.py",
                   "script/a3/resumable_sonic.py", "script/diagnostics/host_loadcheck.py")},
               backup_status="LOCAL_ONLY")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT / "trial_job.json", job)
    save_json(OUTPUT / "launch_state.json", dict(status="READY", boot_id=health["boot_id"]))
    append_record(LOGS / "preflight.jsonl", health)
    print(json.dumps(job, indent=2))


def guarded_monitor(child, log_dir, config):
    stopped = False
    hot = 0
    started = time.time()
    boot = read(OUTPUT / "trial_job.json")["initial_boot"]

    def handle(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {s:signal.signal(s, handle) for s in (signal.SIGTERM, signal.SIGINT)}
    reason = None
    try:
        while child.poll() is None:
            health = sample()
            kernel = command(["journalctl", "-k", "-b", "--since", f"@{int(started)}",
                              "--no-pager", "-o", "cat"])
            critical = [line for line in kernel.get("stdout", "").splitlines() if FATAL.search(line)]
            append_record(log_dir / "health.jsonl", health)
            append_record(LOGS / "guard.jsonl", dict(health=health, critical_kernel_lines=critical,
                          kernel_readable=kernel["returncode"] == 0))
            reason, hot = protection(health, hot)
            if health["boot_id"] != boot:
                reason = "CONFIRMED_NEW_BOOT"
            elif stopped:
                reason = "OPERATOR_STOP"
            elif kernel["returncode"] != 0:
                reason = "KERNEL_TELEMETRY_UNREADABLE"
            elif critical:
                reason = "CRITICAL_KERNEL_EVENT"
            elif time.time() - started >= 900:
                reason = "RUNTIME_LIMIT"
            if reason:
                break
            time.sleep(config["sample_seconds"])
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    return child.wait(), reason


def configure():
    from script.a3 import corrected_trial as trial
    trial.configure_trial("R09")
    trial.TRIAL = "R10"
    trial.OUTPUT = OUTPUT
    trial.LOGS = LOGS
    trial.monitor_child = guarded_monitor
    return trial


def run():
    with (OUTPUT / "launch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = read(OUTPUT / "trial_job.json")
        state = read(OUTPUT / "launch_state.json")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not launch_allowed(state, job, boot):
            raise RuntimeError("No repeat launch or automatic reboot recovery")
        if digest(Path(__file__)) != job["runner_sha256"]:
            raise RuntimeError("Registered runner changed")
        for path, sha in job["reused_code_sha256"].items():
            if digest(ROOT / path) != sha:
                raise RuntimeError("Registered dependency changed: " + path)
        state.update(status="RUNNING", started_at_utc=utc(time.time()), phase="IMPORT_AND_PREFLIGHT")
        save_json(OUTPUT / "launch_state.json", state)
        try:
            trial = configure()
            code = trial.train()
            actual = read(OUTPUT / "state.json") if (OUTPUT / "state.json").exists() else {}
            state.update(status="TRAINING_COMPLETE" if code == 0 and actual.get("status") == "complete"
                         else "STOPPED", returncode=code, ended_at_utc=utc(time.time()),
                         verified_step=actual.get("verified_step", 0), reason=actual.get("reason"),
                         phase="TRAINING_EXIT")
            return code
        except BaseException as error:
            state.update(status="STOPPED", reason=repr(error), ended_at_utc=utc(time.time()))
            raise
        finally:
            save_json(OUTPUT / "launch_state.json", state)


def verify():
    state = read(OUTPUT / "launch_state.json")
    job = read(OUTPUT / "trial_job.json")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != job["initial_boot"]:
        raise RuntimeError("New boot observed; no further compute allowed")
    if state["status"] != "TRAINING_COMPLETE":
        raise RuntimeError("Training has not completed")
    return configure().verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "verify"))
    action = parser.parse_args().action
    result = dict(register=register, run=run, verify=verify)[action]()
    raise SystemExit(result if isinstance(result, int) else 0)
