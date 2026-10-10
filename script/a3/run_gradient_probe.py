"""Guarded independent E07 execution, with existing thermal monitoring."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

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
        if (output / "gradient_diagnostic.json").exists():
            return 0
        preflight()
        run = output / "initial_rollout"
        if run.exists():
            raise ValueError("Existing unfinished attempt: inspect rather than overwrite")
        run.mkdir()
        config = dict(source=str(OFFICIAL), source_sha256=checkpoint_metadata(OFFICIAL)["sha256"],
                      num_envs=16, max_cpu_c=90, max_gpu_c=85, sample_seconds=5,
                      probe_sha256=sha256(ROOT / "script/a3/gradient_probe.py"),
                      split_sha256=sha256(DATA / "split.json"), seed=0, optimizer_steps_allowed=0,
                      max_attempts=None, expires_at=None, backup_status="LOCAL_ONLY",
                      boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        atomic_json(output / "job.json", config)
        environment = os.environ.copy()
        environment.update(ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
            REPO_DIR=str(VENDOR.resolve()), MOTION_FILE=str(DATA / "motionlib/train"),
            EXPERIMENT_DIR=str(run), CHECKPOINT=str(OFFICIAL), NUM_ENVS="16",
            NUM_MINI_BATCHES="4", NUM_LEARNING_ITERATIONS="200", NUM_PROCESSES="1",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            WANDB_MODE="disabled", PYTHONPATH=str(ROOT), YUANQI_RESUME_LOG_DIR=str(logs),
            YUANQI_PROBE_OUTPUT=str(output))
        argv = ["bash", str(VENDOR / "train_a3_035_fromscratch.sh"), "seed=0",
                "trainer._target_=script.a3.gradient_probe.GradientProbeTrainer",
                "algo.config.num_learning_epochs=1", "algo.config.actor_learning_rate=2e-6",
                "algo.config.adaptive_lr_min=1e-6", "algo.config.adaptive_lr_max=2e-5",
                "callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback",
                "callbacks.model_save.save_frequency=1", "+callbacks.model_save.target_global_step=2"]
        with (logs / "console.log").open("w") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, logs, config)
        completed = code == 0 and reason is None and (output / "gradient_diagnostic.json").exists()
        atomic_json(output / "state.json", dict(status="complete" if completed else "failed",
                    returncode=code, reason=reason, optimizer_steps_allowed=0))
        print(json.dumps(dict(completed=completed, returncode=code, reason=reason)), flush=True)
        return 0 if completed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E07", "E08"), default="E07")
    raise SystemExit(main(parser.parse_args().experiment))
