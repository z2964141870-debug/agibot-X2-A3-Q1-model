#!/usr/bin/env python3
"""Launch one explicitly configured training process with recoverable local records."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("training_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command_args = args.training_args
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    if not command_args:
        raise ValueError("Explicit training arguments are required")
    command = [str(args.python.resolve()), "-u", str(Path(__file__).with_name("train.py")), *command_args]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    if args.record.exists():
        raise FileExistsError(args.record)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
               TORCHINDUCTOR_COMPILE_THREADS="1")
    with args.log.open("x") as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    result = {"pid": proc.pid, "pgid": proc.pid, "command": command, "log": str(args.log.resolve()),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "automatic_restart": False, "status": "LAUNCHED_NOT_COMPLETED"}
    with args.record.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
