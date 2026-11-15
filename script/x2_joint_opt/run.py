"""Fresh prefix then bounded joint comparison, preserving all original guards."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import PLANS, settings, sources, validate_prefix


def select(command, output, variant):
    command = list(command)
    for module, mode in (("script.x2_joint_pipeline.worker", "fgp"),
                         ("script.x2_joint_perf.worker", "policy")):
        if module in command:
            i = command.index(module)
            if i < 1 or command[i-1] != "-m" or command[i+1] != mode:
                raise ValueError("Unexpected owned Python command")
            command[i] = "script.x2_joint_opt.worker"
            return command
    if Path(command[0]).name == "x2_morph_native":
        if command[command.index("--mode")+1] != "observe":
            raise ValueError("Only observe")
        return [shutil.which("taskset"), "--cpu-list", ",".join(map(str, PLANS[variant]["native"])),
                sys.executable, "-B", "-m", "script.x2_joint_opt.worker", "native",
                "--output", str(output), "--", *command]
    return command


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("prefix", "joint"))
    p.add_argument("--variant", choices=tuple(PLANS), required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--parity", type=Path)
    a = p.parse_args()
    config = settings(dict(variant=a.variant, release=a.release))
    if a.output.exists() or not set(range(8)).issubset(os.sched_getaffinity(0)) or not shutil.which("taskset"):
        p.error("Require fresh output, CPUs 0..7 and taskset")
    if a.mode == "prefix" and (a.reference is None or a.parity is not None):
        p.error("Prefix requires reference only")
    if a.mode == "joint":
        if a.parity is None or a.reference is not None:
            p.error("Joint requires matching new prefix only")
        validate_prefix(a.parity, config)
    original = subprocess.Popen
    from script.x2_joint_latency import run as coordinator
    from script.x2_joint_startup.run import read_ready
    old_ready = coordinator.read_ready
    before = sources()

    def spawn(command, *args, **kwargs):
        changed = select(command, a.output, a.variant)
        if changed != list(command):
            env = dict(kwargs.get("env") or os.environ)
            env["X2_JOINT_OPT_CONFIG"] = json.dumps(config)
            kwargs["env"] = env
        return original(changed, *args, **kwargs)
    subprocess.Popen = spawn
    sys.argv = [sys.argv[0], "--output", str(a.output), "--fixture", str(a.fixture)]
    try:
        if a.mode == "prefix":
            from script.x2_joint_pipeline.probe import main as prefix
            sys.argv += ["--reference", str(a.reference)]
            prefix()
        else:
            from script.x2_joint_pipeline.run import main as joint
            sys.argv += ["--parity", str(a.parity)]
            with (a.output.parent/(a.output.name+"_protocol.jsonl")).open("x") as tape:
                coordinator.read_ready = lambda child,timeout=40: read_ready(child,tape,timeout)
                joint()
    finally:
        subprocess.Popen, coordinator.read_ready = original, old_ready
        if a.output.is_dir():
            after = sources()
            (a.output/"opt_config.json").write_text(json.dumps(dict(mode=a.mode, config=config,
                plan=PLANS[a.variant], source_sha256=before, source_unchanged=before==after,
                profiled=True, gpu_frame_bound=64 if a.mode=="prefix" else 900,
                graph_initialization_enqueues_bound=2, preview_span_ms=180,
                hardware_commands=0, training_updates=0), indent=2)+"\n")
            if after != before:
                raise RuntimeError("Frozen experimental source changed")


if __name__ == "__main__":
    main()
