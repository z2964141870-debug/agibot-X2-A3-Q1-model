"""Guarded E19 saved-state policy sensitivity diagnostic; zero updates."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.autoresume import monitor_child
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.controller_reference_probe import OUTPUT, SOURCE
from script.a3.evaluation_common import preflight
from script.a3.fullchain_support import DATA, ROOT


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs/a3_controller_reference_20261010_E19"
    logs.mkdir(parents=True, exist_ok=True)
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = OUTPUT / "state.json"
        if (OUTPUT / "result.json").exists():
            return 0
        if state_path.exists():
            raise RuntimeError("Existing E19 attempt must be inspected; no automatic restart")
        preflight()
        guard = dict(max_cpu_c=90, max_gpu_c=85, sample_seconds=5, expires_at=None)
        job = dict(experiment="E19", frozen_policy=True, optimizer_updates=0, states_per_motion=100,
            validation_only=True, candidate_modes=["hold", "cv", "smooth_cv", "E12", "E14"], noise_std_rad=[0., .01],
            source_e18_sha256=sha256(SOURCE / "result.json"),
            source_inputs_verified_sha256=sha256(SOURCE / "independent_inputs.json"),
            code_sha256=sha256(ROOT / "script/a3/controller_reference_probe.py"), guard=guard,
            max_attempts=None, max_no_progress=2, cooldown_seconds=300, backup_status="LOCAL_ONLY")
        atomic_json(OUTPUT / "job.json", job)
        state = dict(status="running", started_at=time.time(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        atomic_json(state_path, state)
        with (logs / "console.log").open("w") as stream:
            child = subprocess.Popen([str(ROOT / "data/environments/a3-sonic/bin/python"), "-m",
                "script.a3.controller_reference_probe"], cwd=ROOT, env=os.environ.copy(), stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, logs, guard)
        complete = code == 0 and reason is None and (OUTPUT / "result.json").exists()
        state.update(status="complete" if complete else "blocked", returncode=code, reason=reason,
                     finished_at=time.time())
        atomic_json(state_path, state)
        return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
