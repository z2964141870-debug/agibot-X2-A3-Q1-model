"""Independent official-PT 2-to-200 update diagnostic job; no legacy resumes."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.autoresume import monitor_child
from script.a3.checkpoint_store import select_checkpoint
from script.a3.evaluation_common import checkpoint_metadata, preflight
from script.a3.fullchain_support import DATA, LOGS, ROOT, VENDOR, digest, mark, write_json


def select_trial_checkpoint(attempts):
    valid = []
    for attempt in attempts:
        directory = Path(attempt["run_dir"]).resolve()
        if directory.parent != ROOT / "data/training/a3_finetune_20261010" or not directory.name.startswith("R04_"):
            raise ValueError("Checkpoint directory is outside this trial lineage")
        try:
            path, payload, _ = select_checkpoint(directory)
        except RuntimeError:
            continue
        step = int(payload["state"].global_step)
        if payload["state"].cur_episode_length.numel() != 16 or not 0 <= step <= 200:
            raise ValueError("Trial checkpoint escaped its 16-env, 0-to-200 lineage")
        valid.append((step, path))
        del payload
    return max(valid, key=lambda row: row[0]) if valid else (0, None)


def run():
    job_dir = DATA / "finetune_R04"
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervisor owns this trial")
        return run_locked(job_dir)


def recovery_remaining(state, config, boot, uptime, now):
    if not state["attempts"]:
        return max(0, config["cooldown_seconds"] - uptime)
    last = state["attempts"][-1]
    if last["boot_id"] != boot:
        return max(0, config["cooldown_seconds"] - uptime)
    if last["status"] in ("running", "interrupted"):
        detected = last.setdefault("interruption_detected_at", now)
        return max(0, config["cooldown_seconds"] - (now - detected))
    return 0


def run_locked(job_dir):
    config = {"source": str(ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"),
              "target_step": 200, "smoke_target": 2, "num_envs": 16, "num_mini_batches": 4,
              "save_frequency": 10, "max_attempts": None, "expires_at": None,
              "max_no_progress": 2, "max_cpu_c": 90, "max_gpu_c": 85, "sample_seconds": 5,
              "cooldown_seconds": 300, "dataset": str(DATA / "motionlib/train"),
              "split_sha256": digest(DATA / "split.json"), "old_lineages_allowed": False}
    config_path = job_dir / "job.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Job changed; do not rewrite its ledger")
    write_json(config_path, config)
    state_path = job_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "pending", "attempts": []}
    if state["status"] in ("blocked", "complete"):
        return 0 if state["status"] == "complete" else 2
    official = Path(config["source"])
    if not official.exists():
        state.update(status="deferred", reason="official_pt_unavailable")
        write_json(state_path, state)
        mark("finetune", "deferred", reason="official_pt_unavailable", job=str(config_path),
             user_action="Provide SHA-verified official PT/config; then rerun finetune_job")
        return 2
    if checkpoint_metadata(official).get("source") != "official_035":
        raise ValueError("This trial requires the released official PT")
    mark("finetune", "running", job=str(config_path))
    while True:
        step, checkpoint = select_trial_checkpoint(state["attempts"])
        if step >= 200:
            state.update(status="complete", verified_step=step, checkpoint=str(checkpoint))
            write_json(state_path, state)
            mark("finetune", "passed", verified_step=step, checkpoint=str(checkpoint),
                 scope="short_official_sample_trial_not_effect_acceptance")
            return 0
        count = 0
        for previous in reversed(state["attempts"]):
            if step > previous["start_step"]:
                break
            count += 1
        if count >= 2:
            state.update(status="blocked", reason="two_attempts_without_saved_progress")
            write_json(state_path, state)
            mark("finetune", "deferred", reason=state["reason"])
            return 2
        preflight()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        remaining = recovery_remaining(state, config, boot, uptime, time.time())
        if remaining > 0:
            state.update(status="deferred", reason="recovery_cooldown", boot_id=boot,
                         cooldown_remaining_seconds=remaining)
            write_json(state_path, state)
            mark("finetune", "deferred", reason="recovery_cooldown", remaining_seconds=remaining)
            return 75
        target = 2 if step < 2 else 200
        ordinal = len(state["attempts"]) + 1
        run_dir = ROOT / "data/training/a3_finetune_20261010" / f"R04_{ordinal:03d}_s{step}"
        logs = LOGS / "finetune" / run_dir.name
        if run_dir.exists() or logs.exists():
            raise FileExistsError("Do not overwrite a previous training attempt")
        run_dir.mkdir(parents=True)
        logs.mkdir(parents=True)
        for previous in state["attempts"]:
            if previous["status"] == "running":
                previous.update(status="interrupted", reason="supervisor_or_host_interrupted")
        row = {"start_step": step, "target": target, "run_dir": str(run_dir), "logs": str(logs),
               "boot_id": boot, "started_at": time.time(), "status": "running",
               "checkpoint": str(checkpoint or official), "full_state_resume": checkpoint is not None}
        state["attempts"].append(row)
        state["status"] = "running"
        write_json(state_path, state)
        environment = os.environ.copy()
        environment.update(ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
                           MOTION_FILE=config["dataset"], EXPERIMENT_DIR=str(run_dir),
                           CHECKPOINT=row["checkpoint"], NUM_ENVS="16", NUM_MINI_BATCHES="4",
                           NUM_LEARNING_ITERATIONS="200", NUM_PROCESSES="1",
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           WANDB_MODE="disabled", PYTHONPATH=str(ROOT),
                           YUANQI_RESUME_LOG_DIR=str(logs))
        argv = ["bash", str(VENDOR / "train_a3_035_fromscratch.sh"), "seed=0",
                "trainer._target_=script.a3.finetune_trainer.FineTuneTrainer",
                "callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback",
                "callbacks.model_save.save_frequency=10", "+callbacks.model_save.save_last_frequency=10",
                f"+callbacks.model_save.target_global_step={target}"]
        if checkpoint:
            argv += ["++resume=true", "++resume_in_place=false"]
        with (logs / "console.log").open("w") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, logs, config)
        new_step, _ = select_trial_checkpoint(state["attempts"])
        row.update(status="finished", returncode=code, reason=reason, verified_step=new_step, finished_at=time.time())
        write_json(state_path, state)
        if code != 0 or reason is not None:
            state.update(status="blocked", reason=reason or f"training_exit_{code}")
            write_json(state_path, state)
            mark("finetune", "deferred", reason=state["reason"])
            return 2


if __name__ == "__main__":
    raise SystemExit(run())
