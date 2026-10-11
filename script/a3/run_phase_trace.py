"""Single E25 traced smoke, gated by completion of E24, no reboot recovery."""

import argparse
import ast
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
from script.a3.phase_trace import summarize_files
from script.diagnostics.host_loadcheck import FATAL, protection, stop

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "script/vendor/sonic_for_a3"
OUT = ROOT / "data/experiments/a3_phase_trace_20261011_E25"
LOG = ROOT / "logs/a3_phase_trace_20261011_E25"
TRAINING = ROOT / "data/training/a3_phase_trace_20261011_E25"
CONTROL = ROOT / "data/experiments/a3_isolation_20261011_E24"
PYTHON = ROOT / "data/environments/a3-sonic/bin/python"
OFFICIAL = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
SOURCES = ("script/a3/phase_trace.py", "script/a3/traced_entry.py", "script/a3/traced_training.py",
           "script/a3/run_phase_trace.py", "script/a3/verified_finetune.py", "script/a3/finetune_trainer.py",
           "script/a3/resumable_sonic.py", "script/a3/checkpoint_store.py",
           "script/a3/host_health.py", "script/diagnostics/host_loadcheck.py",
           "script/a3/idle_reboot_watch.py",
           "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh",
           "script/vendor/sonic_for_a3/gear_sonic/train_agent_trl.py",
           "script/vendor/sonic_for_a3/gear_sonic/trl/trainer/ppo_trainer.py",
           "script/vendor/sonic_for_a3/gear_sonic/trl/trainer/ppo_trainer_aux_loss.py")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def control_ready(job, state):
    return (job["boot_id"] == read(CONTROL / "job.json")["boot_id"]
            and state.get("status") == "COMPLETED" and len(state.get("results", [])) == 15
            and all(x["status"] == "PASSED" for x in state["results"]))


def validate_source_methods():
    # Read source only; importing Isaac here could launch a context before the guard.
    from script.a3.traced_entry import TARGETS
    methods = {}
    isaac = Path("/home/yu/IsaacLab/source/isaaclab")
    for module, targets in TARGETS.items():
        path = (isaac if module.startswith("isaaclab.") else VENDOR).joinpath(*module.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text())
        classes = {node.name:node for node in tree.body if isinstance(node, ast.ClassDef)}
        for clsname, wanted in targets.items():
            present = {node.name for node in classes[clsname].body if isinstance(node, ast.FunctionDef)}
            if not set(wanted) <= present:
                raise ValueError("Missing trace target: " + module + ":" + clsname + ":" + str(set(wanted)-present))
        methods[str(path)] = sha(path)
    trainer = ast.parse((VENDOR / "gear_sonic/trl/trainer/ppo_trainer.py").read_text())
    present = {node.name for node in ast.walk(trainer) if isinstance(node, ast.FunctionDef)}
    wanted = {"_setup_storage", "policy_step", "_rollout_step", "_get_rollout_data", "_get_mb_rollout_data",
              "_forward_model", "_compute_loss", "_gradient_clipping", "_compute_returns",
              "sync_running_mean_std", "sync_adaptive_sampling"}
    if not wanted <= present:
        raise ValueError("Trainer target contract changed")
    return methods


def register():
    if OUT.exists() or TRAINING.exists():
        raise FileExistsError("Existing trace trial cannot be reset")
    health = sample()
    reason, hot = protection(health, 0)
    if reason or hot:
        raise RuntimeError(reason or "HOT_PREFLIGHT")
    control = read(CONTROL / "state.json")
    if control["status"] not in ("RUNNING", "COOLDOWN", "COMPLETED"):
        raise RuntimeError("E24 has not succeeded or is no longer running")
    if read(CONTROL / "job.json")["boot_id"] != health["boot_id"]:
        raise RuntimeError("Control boot differs")
    external = validate_source_methods()
    source = (VENDOR / "train_a3_035_fromscratch.sh").read_text()
    original = '${REPO_DIR}/gear_sonic/train_agent_trl.py'
    if source.count(original) != 1:
        raise ValueError("Pinned launcher entry contract changed")
    generated = source.replace(original, str(ROOT / "script/a3/traced_entry.py"))
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    launcher = OUT / "traced_launcher.sh"
    launcher.write_text(generated)
    job = dict(experiment="A3_PHASE_TRACE_20261011_E25", registered_at_utc=utc(time.time()),
               boot_id=health["boot_id"], control_job_sha256=sha(CONTROL / "job.json"),
               source_sha256=sha(OFFICIAL), source_bytes=OFFICIAL.stat().st_size,
               target_updates=2, num_envs=16, minibatches=4, epochs=1, lr_scale=0.1, seed=0,
               runtime_limit_seconds=900, control_wait_limit_seconds=7200,
               max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
               memory_reserve_gib=4, auto_resume=False, repeat_allowed=False,
               stop_on_first_fault=True, synchronized_diagnostic=True,
               sources_sha256={p:sha(ROOT / p) for p in SOURCES},
               external_sources_sha256=external, launcher_sha256=sha(launcher),
               backup_status="LOCAL_ONLY")
    save_json(OUT / "job.json", job)
    save_json(OUT / "launch_state.json", dict(status="READY", boot_id=health["boot_id"]))
    save_json(ROOT / "data/manifests/a3_phase_trace_registration_20261011_E25.json", job)
    print(json.dumps(job), flush=True)


def health_check(job, state, started, high):
    health = sample()
    kernel = command(["journalctl", "-k", "-b", "--since", f"@{int(started)}", "--no-pager", "-o", "cat"])
    critical = [x for x in kernel.get("stdout", "").splitlines() if FATAL.search(x)]
    append_record(LOG / "health.jsonl", dict(phase=state["status"], health=health,
                  critical_kernel_lines=critical, kernel_readable=kernel["returncode"] == 0))
    reason, high = protection(health, high)
    if health["boot_id"] != job["boot_id"]:
        reason = "CONFIRMED_NEW_BOOT"
    elif kernel["returncode"] != 0:
        reason = "KERNEL_TELEMETRY_UNREADABLE"
    elif critical:
        reason = "CRITICAL_KERNEL_EVENT"
    if reason:
        raise RuntimeError(reason)
    return high


def validate_hashes(job):
    for path, expected in job["sources_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Registered source changed: " + path)
    for path, expected in job["external_sources_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Registered Isaac/motion source changed: " + path)
    if sha(OUT / "traced_launcher.sh") != job["launcher_sha256"] or sha(CONTROL / "job.json") != job["control_job_sha256"]:
        raise ValueError("Launcher/control registration changed")
    if sha(OFFICIAL) != job["source_sha256"]:
        raise ValueError("Official checkpoint changed")


def run():
    with (OUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read(OUT / "job.json"), read(OUT / "launch_state.json")
        if state["status"] != "READY" or sample()["boot_id"] != job["boot_id"]:
            raise RuntimeError("No repeat or reboot recovery")
        state.update(status="WAITING_GPU_CONTROL", queued_at_utc=utc(time.time()))
        save_json(OUT / "launch_state.json", state)
        def interrupted(signum, frame):
            raise RuntimeError("OPERATOR_OR_SERVICE_STOP")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        started, high, child = time.time(), 0, None
        try:
            validate_hashes(job)
            while True:
                high = health_check(job, state, started, high)
                control = read(CONTROL / "state.json")
                if control_ready(job, control):
                    break
                if control["status"] not in ("RUNNING", "COOLDOWN"):
                    raise RuntimeError("CONTROL_FAILED_OR_INCOMPLETE_" + control["status"])
                if time.time() - started > job["control_wait_limit_seconds"]:
                    raise RuntimeError("CONTROL_WAIT_TIMEOUT")
                time.sleep(5)
            for row in control["results"]:
                if sha(CONTROL / (row["stage"] + ".result.json")) != row["result_sha256"]:
                    raise ValueError("Control result SHA changed")
            # Let any terminating E24 CUDA process leave; no overlap.
            deadline = time.monotonic() + 60
            while True:
                gpu = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
                if gpu["returncode"] != 0:
                    raise RuntimeError("GPU_TELEMETRY_UNREADABLE")
                if not gpu.get("stdout"):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("GPU_OCCUPIED")
                time.sleep(5)
            validate_hashes(job)
            reason, high = protection(sample(), 0)
            if reason or high:
                raise RuntimeError(reason or "HOT_PREFLIGHT")
            state.update(status="TRAINING", started_at_utc=utc(time.time()))
            save_json(OUT / "launch_state.json", state)
            TRAINING.mkdir(parents=True, exist_ok=False)
            env = os.environ.copy()
            env.update(ISAAC_PYTHON=str(PYTHON), REPO_DIR=str(VENDOR.resolve()),
                       MOTION_FILE=str(ROOT / "data/experiments/a3_fullchain_20261010/motionlib/train"),
                       EXPERIMENT_DIR=str(TRAINING), CHECKPOINT=str(OFFICIAL), NUM_ENVS="16",
                       NUM_MINI_BATCHES="4", NUM_LEARNING_ITERATIONS="200", NUM_PROCESSES="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       WANDB_MODE="disabled", PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
                       YUANQI_RESUME_LOG_DIR=str(LOG), YUANQI_PHASE_TRACE_DIR=str(LOG / "phases"))
            argv = ["bash", str(OUT / "traced_launcher.sh"), "seed=0",
                    "trainer._target_=script.a3.traced_training.TracedFineTuneTrainer",
                    "callbacks.model_save._target_=script.a3.traced_training.TracedSaveCallback",
                    "callbacks.model_save.save_frequency=1", "+callbacks.model_save.target_global_step=2",
                    "algo.config.num_learning_epochs=1", "algo.config.actor_learning_rate=2e-6",
                    "algo.config.adaptive_lr_min=1e-6", "algo.config.adaptive_lr_max=2e-5"]
            state["argv"] = argv
            save_json(OUT / "launch_state.json", state)
            with (LOG / "console.log").open("w") as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                started, high = time.time(), 0
                while True:
                    high = health_check(job, state, started, high)
                    if child.poll() is not None:
                        break
                    if time.time() - started > job["runtime_limit_seconds"]:
                        raise RuntimeError("TRAINING_TIMEOUT")
                    time.sleep(5)
                state["returncode"] = child.returncode
                if child.returncode != 0:
                    raise RuntimeError("TRAINING_FAILED_" + str(child.returncode))
            sidecar = TRAINING / "model_step_000002.json"
            if not sidecar.exists():
                raise RuntimeError("TARGET_CHECKPOINT_MISSING")
            record = read(sidecar)
            checkpoint = TRAINING / "model_step_000002.pt"
            if checkpoint.stat().st_size != record["bytes"] or sha(checkpoint) != record["sha256"]:
                raise ValueError("TARGET_CHECKPOINT_HASH_MISMATCH")
            state.update(status="COMPLETED", verified_target_sha256=record["sha256"])
        except BaseException as error:
            state.update(status="STOPPED", reason=repr(error))
            raise
        finally:
            if child is not None:
                stop(child)
            state.update(ended_at_utc=utc(time.time()))
            save_json(OUT / "launch_state.json", state)
            summarize()


def summarize():
    job, state = read(OUT / "job.json"), read(OUT / "launch_state.json")
    health = sample()
    if health["boot_id"] != job["boot_id"]:
        state.update(status="STOPPED_REBOOT", reason="CONFIRMED_NEW_BOOT", observed_boot=health["boot_id"])
        save_json(OUT / "launch_state.json", state)
    result = dict(experiment=job["experiment"], observed_at_utc=utc(time.time()),
                  initial_boot=job["boot_id"], observed_boot=health["boot_id"], state=state,
                  trace=summarize_files(LOG / "phases"), backup_status="LOCAL_ONLY",
                  scope="SYNCHRONIZED_SHORT_DIAGNOSTIC_NOT_POLICY_EFFECT_OR_ROOT_CAUSE_PROOF")
    save_json(ROOT / "data/manifests/a3_phase_trace_result_20261011_E25.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "summarize"))
    dict(register=register, run=run, summarize=summarize)[parser.parse_args().action]()
