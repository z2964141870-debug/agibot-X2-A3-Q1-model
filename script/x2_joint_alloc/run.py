"""New bounded prefix/joint roots using R1's guards and an owned affinity wrapper."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .allocation import PLAN, sources, validate_available, validate_prefix


def select(command, taskset):
    command = list(command)
    for module, mode in (("script.x2_joint_pipeline.worker", "fgp"),
                         ("script.x2_joint_perf.worker", "policy")):
        if module in command:
            i = command.index(module)
            if i == 0 or command[i-1] != "-m" or command[i+1] != mode:
                raise ValueError("Unexpected owned Python command")
            command[i] = "script.x2_joint_alloc.worker"
            return command
    if Path(command[0]).name == "x2_morph_native":
        if command[command.index("--mode")+1] != "observe":
            raise ValueError("Only native observe may enter this experiment")
        return [taskset, "--cpu-list", ",".join(map(str, PLAN["native"])), *command]
    return command


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("prefix", "joint"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--parity", type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve prior experiment")
    if a.mode == "prefix" and (a.reference is None or a.parity is not None):
        p.error("Prefix requires reference only")
    if a.mode == "joint" and (a.parity is None or a.reference is not None):
        p.error("Joint requires parity only")
    validate_available(os.sched_getaffinity(0))
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset unavailable")
    if a.mode == "joint":
        validate_prefix(a.parity)
    original = subprocess.Popen
    def spawn(command, *args, **kwargs):
        return original(select(command, taskset), *args, **kwargs)
    subprocess.Popen = spawn
    sys.argv = [sys.argv[0], "--output", str(a.output), "--fixture", str(a.fixture)]
    try:
        if a.mode == "prefix":
            from script.x2_joint_pipeline.probe import main as probe
            sys.argv += ["--reference", str(a.reference)]
            probe()
        else:
            from script.x2_joint_pipeline.run import main as joint
            sys.argv += ["--parity", str(a.parity)]
            joint()
    finally:
        subprocess.Popen = original
        if a.output.is_dir():
            (a.output/"allocation.json").write_text(json.dumps(dict(variant="bounded_affinity_r2",
                mode=a.mode, plan=PLAN, source_sha256=sources(), hardware_commands=0,
                request_rate_hz=50, original_native_deadline_ms=20,
                additional_analysis_budgets_ms=[25,30,1000/30]), indent=2)+"\n")


if __name__ == "__main__":
    main()
