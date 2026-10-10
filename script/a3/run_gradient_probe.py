"""Guarded independent E07 execution, with existing thermal monitoring."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.autoresume import monitor_child
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import OFFICIAL
from script.a3.evaluation_common import checkpoint_metadata, preflight
from script.a3.fullchain_support import DATA, ROOT, VENDOR


def main(experiment="E07"):
    output = ROOT / "data/experiments" / f"a3_gradient_20261010_{experiment}"
    logs = ROOT / "logs" / f"a3_gradient_20261010_{experiment}"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_name = "likelihood_diagnostic.json" if experiment == "E10" else "gradient_diagnostic.json"
        if (output / result_name).exists():
            return 0
        state_path = output / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="pending", attempts=[])
        if state.get("status") == "blocked":
            return 2
        if experiment == "E10" and (output / "initial_rollout").exists() and not state.get("attempts"):
            initial = json.loads((output / "job.json").read_text())
            state["attempts"] = [dict(run=str(output / "initial_rollout"), boot_id=initial["boot_id"],
                                      status="interrupted", reason="boot_changed_before_result")]
        attempts = state.get("attempts", [])
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for previous in attempts:
            if previous.get("status") == "running" and previous["boot_id"] != boot:
                previous.update(status="interrupted", reason="boot_changed_before_result")
        if experiment == "E10" and len(attempts) >= 2:
            state.update(status="blocked", reason="two_attempts_without_saved_diagnostic")
            atomic_json(state_path, state)
            return 2
        if attempts and attempts[-1]["boot_id"] != boot:
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            if uptime < 300:
                state.update(status="cooldown", cooldown_remaining_seconds=300-uptime)
                atomic_json(state_path, state)
                return 75
        preflight()
        ordinal = len(attempts) + 1
        run = output / ("initial_rollout" if ordinal == 1 else f"initial_rollout_{ordinal:03d}")
        if run.exists():
            raise ValueError("Existing unfinished attempt: inspect rather than overwrite")
        run.mkdir()
        config = dict(source=str(OFFICIAL), source_sha256=checkpoint_metadata(OFFICIAL)["sha256"],
                      num_envs=16, max_cpu_c=90, max_gpu_c=85, sample_seconds=5,
                      probe_sha256=sha256(ROOT / "script/a3/gradient_probe.py"),
                      runtime_probe_sha256=sha256(ROOT / "script/a3/runtime_likelihood_probe.py") if experiment == "E10" else None,
                      split_sha256=sha256(DATA / "split.json"), seed=0, optimizer_steps_allowed=0,
                      max_attempts=None, expires_at=None, backup_status="LOCAL_ONLY",
                      boot_id=boot, max_no_progress=2, cooldown_seconds=300)
        atomic_json(output / ("job.json" if ordinal == 1 else f"attempt_{ordinal:03d}_job.json"), config)
        if ordinal > 1:
            logs = logs / f"attempt_{ordinal:03d}"
            logs.mkdir(parents=True, exist_ok=False)
        attempt = dict(run=str(run), boot_id=boot, status="running", logs=str(logs), started_at=time.time())
        attempts.append(attempt)
        state.update(status="running", attempts=attempts)
        atomic_json(state_path, state)
        environment = os.environ.copy()
        environment.update(ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
            REPO_DIR=str(VENDOR.resolve()), MOTION_FILE=str(DATA / "motionlib/train"),
            EXPERIMENT_DIR=str(run), CHECKPOINT=str(OFFICIAL), NUM_ENVS="16",
            NUM_MINI_BATCHES="4", NUM_LEARNING_ITERATIONS="200", NUM_PROCESSES="1",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            WANDB_MODE="disabled", PYTHONPATH=str(ROOT), YUANQI_RESUME_LOG_DIR=str(logs),
            YUANQI_PROBE_OUTPUT=str(output))
        argv = ["bash", str(VENDOR / "train_a3_035_fromscratch.sh"), "seed=0",
                "trainer._target_=" + ("script.a3.runtime_likelihood_probe.RuntimeLikelihoodTrainer"
                                       if experiment == "E10" else "script.a3.gradient_probe.GradientProbeTrainer"),
                "algo.config.num_learning_epochs=1", "algo.config.actor_learning_rate=2e-6",
                "algo.config.adaptive_lr_min=1e-6", "algo.config.adaptive_lr_max=2e-5",
                "callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback",
                "callbacks.model_save.save_frequency=1", "+callbacks.model_save.target_global_step=2"]
        with (logs / "console.log").open("w") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, logs, config)
        completed = code == 0 and reason is None and (output / result_name).exists()
        attempt.update(status="complete" if completed else "failed", returncode=code, reason=reason)
        state.update(status="complete" if completed else "failed", returncode=code,
                     reason=reason, optimizer_steps_allowed=0)
        atomic_json(state_path, state)
        print(json.dumps(dict(completed=completed, returncode=code, reason=reason)), flush=True)
        return 0 if completed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E07", "E08", "E10"), default="E07")
    raise SystemExit(main(parser.parse_args().experiment))
