"""Bounded A3 recovery across host reboots; no reboot, driver, or power changes."""

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

from script.a3.checkpoint_store import atomic_json, select_checkpoint
from script.a3.host_health import append_record, collect_snapshot, command, sample


ROOT = Path(__file__).resolve().parents[2]
VENDOR_COMMIT = "fe6868ba37034f89b912f0fb851bce19f120266d"


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def decision(config, state, step, now):
    if step >= config["target_step"]:
        return "complete"
    if now >= config["expires_at"]:
        return "expired"
    attempts = state.get("attempts", [])
    if len(attempts) >= config["max_attempts"]:
        return "attempt_limit"
    no_progress = 0
    for attempt in reversed(attempts):
        if step > attempt["start_step"]:
            break
        no_progress += 1
    if no_progress >= config["max_no_progress"]:
        return "no_progress"
    return "run"


def best_checkpoint(config, state):
    candidates = []
    directories = list(config["source_dirs"]) + [item["run_dir"] for item in state.get("attempts", [])]
    for directory in dict.fromkeys(directories):
        try:
            path, payload, rejected = select_checkpoint(ROOT / directory)
            if payload["state"].cur_episode_length.numel() != config["num_envs"]:
                raise ValueError("Checkpoint environment count differs")
            candidates.append((int(payload["state"].global_step), str(path), rejected))
            del payload
        except RuntimeError:
            continue
    if not candidates:
        raise RuntimeError("No verified checkpoint in this registered experiment lineage")
    return max(candidates, key=lambda item: item[0])


def read_state(path, config):
    if not path.exists():
        raise FileNotFoundError("Missing recovery ledger: refusing to reset the attempt budget")
    state = json.loads(path.read_text())
    if state["config_sha256"] != fingerprint(config):
        raise ValueError("Job changed: create a new job directory instead of resetting its budget")
    return state


def finish_attempt(config, state, attempt, step, returncode, reason):
    attempt.update(status="finished", returncode=returncode, reason=reason, verified_step=step,
                   finished_at=time.time())
    if step >= config["target_step"] and returncode == 0 and reason is None:
        state.update(status="complete", verified_step=step)
        return 0
    if reason is None and returncode in {-9, 137, 75}:
        state.update(status="interrupted", verified_step=step)
        return 75
    state.update(status="blocked", reason=reason or f"training_exit_{returncode}", verified_step=step)
    return 2


def rearm_temperature(config, state, authorization, now):
    if not authorization or not authorization.strip():
        raise ValueError("Explicit authorization record is required")
    if state.get("status") != "blocked" or state.get("reason") != "temperature_stop":
        raise ValueError("Only a temperature protection stop can be rearmed here")
    action = decision(config, state, state["verified_step"], now)
    if action != "run":
        raise ValueError(f"Cannot rearm: {action}")
    state.setdefault("operator_authorizations", []).append({
        "at": now, "authorization": authorization, "previous_reason": state["reason"],
        "verified_step": state["verified_step"], "attempts_used": len(state["attempts"]),
    })
    state.update(status="armed")
    state.pop("reason")
    return state


def monitor_child(child, log_dir, config):
    stopped = False
    high_temperature = 0

    def stop_handler(signum, frame):
        nonlocal stopped
        stopped = True

    original = {name: signal.signal(name, stop_handler) for name in (signal.SIGTERM, signal.SIGINT)}
    reason = None
    try:
        while child.poll() is None:
            health = sample()
            append_record(log_dir / "health.jsonl", health)
            gpu_hot = any((gpu.get("temperature.gpu") or 0) >= config["max_gpu_c"]
                          for gpu in health["gpus"])
            cpu_hot = health["temperatures_c"].get("x86_pkg_temp", 0) >= config["max_cpu_c"]
            high_temperature = high_temperature + 1 if gpu_hot or cpu_hot else 0
            expired = time.time() >= config["expires_at"]
            if stopped or expired or high_temperature >= 2 or health["gpu_query"]["returncode"] != 0:
                reason = ("operator_stop" if stopped else "expired" if expired else
                          "temperature_stop" if high_temperature >= 2 else "gpu_query_failed")
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(config["sample_seconds"])
        return child.wait(), reason
    finally:
        for name, handler in original.items():
            signal.signal(name, handler)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def run_job(job_dir):
    config = json.loads((job_dir / "job.json").read_text())
    state_path = job_dir / "state.json"
    state = read_state(state_path, config)
    if state.get("status") in {"blocked", "complete"}:
        print("Job is already " + state["status"], flush=True)
        return 0 if state["status"] == "complete" else 2
    step, checkpoint, rejected = best_checkpoint(config, state)
    action = decision(config, state, step, time.time())
    if action != "run":
        state.update(status="complete" if action == "complete" else "blocked", reason=action, verified_step=step)
        atomic_json(state_path, state)
        print(json.dumps({"action": action, "verified_step": step}), flush=True)
        return 0 if action == "complete" else 2
    if command(["git", "-C", str(ROOT / "script/vendor/sonic_for_a3"), "rev-parse", "HEAD"])["stdout"] != VENDOR_COMMIT:
        raise ValueError("Vendor commit changed")
    disk = os.statvfs(ROOT)
    if disk.f_bavail * disk.f_frsize < 20 * 1024 ** 3:
        raise RuntimeError("Less than 20 GiB free: do not start training")
    gpu_processes = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if gpu_processes["returncode"] != 0:
        raise RuntimeError("GPU process query failed")
    if gpu_processes["stdout"]:
        print("GPU is occupied; waiting without spending an attempt", flush=True)
        return 75
    health = sample()
    if health["gpu_query"]["returncode"] != 0 or not health["gpus"]:
        raise RuntimeError("No readable GPU before launch")
    if any((gpu.get("temperature.gpu") or 0) >= config["max_gpu_c"] for gpu in health["gpus"]):
        raise RuntimeError("GPU temperature too high before launch")
    if health["temperatures_c"].get("x86_pkg_temp", 0) >= config["max_cpu_c"]:
        raise RuntimeError("CPU temperature too high before launch")
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    delay = max(0, config["cooldown_seconds"] - uptime)
    if state["attempts"]:
        delay = max(delay, state["attempts"][-1]["started_at"] + config["cooldown_seconds"] - time.time())
    if delay > 0:
        print(f"Cooling/waiting {delay:.0f}s after startup/previous attempt", flush=True)
        time.sleep(delay)
        return 75
    for previous in state["attempts"]:
        if previous.get("status") == "running":
            previous.update(status="interrupted", reason="supervisor_or_host_interrupted")
    ordinal = len(state["attempts"]) + 1
    run_id = f"{config['run_prefix']}_{ordinal:02d}_s{step}"
    run_dir = f"data/training/a3_20261009/{run_id}"
    log_dir = ROOT / "logs/a3_training_20261009" / run_id
    if (ROOT / run_dir).exists() or log_dir.exists():
        raise FileExistsError("Run ID already exists; do not reuse or overwrite it")
    log_dir.mkdir(parents=True)
    collect_snapshot(log_dir / "diagnostics")
    attempt = {"run_id": run_id, "run_dir": run_dir, "start_step": step,
               "checkpoint": checkpoint, "rejected": rejected, "started_at": time.time(),
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "status": "running"}
    state["attempts"].append(attempt)
    state["status"] = "running"
    atomic_json(state_path, state)
    environment = os.environ.copy()
    environment.update(RUN_ID=run_id, RESUME_MODE="true", CHECKPOINT=checkpoint,
                       NUM_ENVS=str(config["num_envs"]), NUM_MINI_BATCHES="4",
                       NUM_LEARNING_ITERATIONS=str(config["target_step"]),
                       SAVE_FREQUENCY="25", SAVE_LAST_FREQUENCY="25",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    print(json.dumps(attempt), flush=True)
    with (log_dir / "supervisor.log").open("a") as output:
        child = subprocess.Popen(["bash", str(ROOT / "script/a3/run_official_training.sh")],
                                 cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        returncode, reason = monitor_child(child, log_dir, config)
    new_step, _, _ = best_checkpoint(config, state)
    result = finish_attempt(config, state, attempt, new_step, returncode, reason)
    atomic_json(state_path, state)
    print(json.dumps(state), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--source-dir", type=Path, help="New job's trusted source checkpoint directory")
    parser.add_argument("--target-step", type=int, help="New job's cumulative update target")
    parser.add_argument("--run-prefix", help="New job's distinct run prefix")
    parser.add_argument("--rearm-temperature", metavar="AUTHORIZATION",
                        help="Record explicit authorization without resetting the budget or launching")
    options = parser.parse_args()
    if options.init and options.rearm_temperature is not None:
        parser.error("Init and rearm are mutually exclusive")
    if not options.init and any(value is not None for value in (
            options.source_dir, options.target_step, options.run_prefix)):
        parser.error("Source, target and prefix can only be set when initializing a new job")
    job_dir = options.job_dir.resolve()
    if not job_dir.is_relative_to(ROOT / "data/training"):
        parser.error("Job directory must be inside this project's data/training")
    if options.init:
        job_dir.mkdir(parents=True, exist_ok=True)
        if (job_dir / "job.json").exists() or (job_dir / "state.json").exists():
            raise FileExistsError("Existing job budget cannot be reset by init")
        source = (ROOT / (options.source_dir or Path("data/training/a3_20261009/E005_resume_154to2000"))).resolve()
        target = options.target_step if options.target_step is not None else 2000
        prefix = options.run_prefix if options.run_prefix is not None else "E006_auto"
        if not source.is_relative_to(ROOT / "data/training"):
            parser.error("Source must be inside this project's data/training")
        if target <= 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            parser.error("Target must be positive and prefix must contain only letters, numbers, underscore or hyphen")
        config = {"source_dirs": [str(source.relative_to(ROOT))],
                  "run_prefix": prefix, "num_envs": 64, "target_step": target,
                  "max_attempts": 3, "max_no_progress": 2, "expires_at": time.time() + 86400,
                  "cooldown_seconds": 300, "sample_seconds": 5, "max_gpu_c": 85, "max_cpu_c": 90}
        atomic_json(job_dir / "job.json", config)
        atomic_json(job_dir / "state.json", {"config_sha256": fingerprint(config), "attempts": [], "status": "armed"})
        print(json.dumps(config, indent=2))
        return 0
    with (job_dir / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another supervisor owns this job", flush=True)
            return 2
        try:
            if options.rearm_temperature is not None:
                try:
                    config = json.loads((job_dir / "job.json").read_text())
                    state = read_state(job_dir / "state.json", config)
                    rearm_temperature(config, state, options.rearm_temperature, time.time())
                except (ValueError, KeyError, OSError) as error:
                    print(f"Rearm refused: {error}", flush=True)
                    return 2
                atomic_json(job_dir / "state.json", state)
                print(json.dumps(state), flush=True)
                return 0
            return run_job(job_dir)
        except Exception as error:
            # Preserve a valid ledger's budget and stop state, even after controller errors.
            try:
                config = json.loads((job_dir / "job.json").read_text())
                state = read_state(job_dir / "state.json", config)
                state.update(status="blocked", reason=f"controller_{type(error).__name__}",
                             error=str(error))
                atomic_json(job_dir / "state.json", state)
            except Exception:
                pass
            try:
                append_record(ROOT / "logs/a3_autoresume_20261009/errors.jsonl",
                              {"at": time.time(), "type": type(error).__name__, "error": str(error)})
            except OSError:
                print("Could not persist error log; recovery still stops", flush=True)
            print(f"Blocked: {type(error).__name__}: {error}", flush=True)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
