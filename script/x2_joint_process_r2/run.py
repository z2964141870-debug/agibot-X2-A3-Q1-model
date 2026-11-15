"""New candidate gates around the frozen resource-guarded observer coordinator."""
import argparse
import json
from pathlib import Path
import sys
import time

from .config import DEADLINE_UTC, sources


def validate(directory, release):
    record = json.loads((directory/"process_config.json").read_text())
    reap = json.loads((directory/"cpu_reap.json").read_text())
    if (record["mode"] != "prefix" or not record["completed"] or record["release"] != release
            or record["sources"] != sources() or not record["source_unchanged"]
            or reap["alive"] or reap["exitcode"] != 0 or reap["completed"] != 64):
        raise ValueError("Require this candidate's fresh prefix and clean CPU child")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("prefix", "joint"))
    p.add_argument("--release", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--parity", type=Path)
    a = p.parse_args()
    if time.time()+120 >= DEADLINE_UTC:
        p.error("Not enough time before the fixed three-hour deadline")
    if a.mode == "joint":
        if a.parity is None:
            p.error("Require new matching CPU-process prefix")
        validate(a.parity, a.release)
    from script.x2_joint_opt import run as coordinator
    original = coordinator.select
    before = sources()
    complete = False
    def select(command, output, variant):
        result = original(command, output, variant)
        module = "script.x2_joint_opt.worker"
        if module in result and result[result.index(module)+1] == "fgp":
            result[result.index(module)] = "script.x2_joint_process_r2.worker"
        return result
    coordinator.select = select
    sys.argv = [sys.argv[0], a.mode, "--variant", "dedicated_hosts", "--release", a.release,
                "--output", str(a.output), "--fixture", str(a.fixture)]
    if a.reference:
        sys.argv += ["--reference", str(a.reference)]
    if a.parity:
        sys.argv += ["--parity", str(a.parity)]
    try:
        coordinator.main()
        complete = True
    finally:
        coordinator.select = original
        if a.output.is_dir():
            after = sources()
            (a.output/"process_config.json").write_text(json.dumps(dict(mode=a.mode,
                release=a.release, sources=before, source_unchanged=before==after,
                completed=complete, deadline_utc=DEADLINE_UTC, two_frame_credit=True,
                gpu_frame_bound=64 if a.mode=="prefix" else 900, hardware_commands=0), indent=2)+"\n")
            if before != after:
                raise RuntimeError("Own frozen candidate changed")


if __name__ == "__main__":
    main()
