"""Reuse the frozen joint coordinator with a narrowly scoped owned worker selector."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CONFIGS = {
    "control": dict(variant="control", threads=4, spinning=True, batched_fk=False),
    "sleep1": dict(variant="sleep1", threads=1, spinning=False, batched_fk=False),
    "sleep1_fk": dict(variant="sleep1_fk", threads=1, spinning=False, batched_fk=True),
}


def select_worker(command):
    command = list(command)
    if "script.x2_joint_latency.worker" in command:
        i = command.index("script.x2_joint_latency.worker")
        if i == 0 or command[i-1] != "-m" or command[i+1] not in ("fgp", "policy"):
            raise ValueError("Unexpected owned worker command")
        command[i] = "script.x2_joint_perf.worker"
    return command


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--variant", choices=CONFIGS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve existing experiment")
    original = subprocess.Popen

    def spawn(command, *args, **kwargs):
        changed = select_worker(command)
        if changed != list(command):
            env = dict(kwargs.get("env") or os.environ)
            env["X2_JOINT_PERF_CONFIG"] = json.dumps(CONFIGS[a.variant])
            kwargs["env"] = env
        return original(changed, *args, **kwargs)

    # Only this coordinator's two explicit Python worker commands are redirected.
    # The existing native observe command, guard and lifetime remain unchanged.
    subprocess.Popen = spawn
    from script.x2_joint_latency.run import main as joint_main
    sys.argv = [sys.argv[0], "--output", str(a.output), "--fixture", str(a.fixture),
                "--scheduling", "latest"]
    try:
        joint_main()
    finally:
        subprocess.Popen = original
        if a.output.is_dir():
            (a.output/"performance_config.json").write_text(json.dumps(CONFIGS[a.variant], indent=2)+"\n")


if __name__ == "__main__":
    main()
