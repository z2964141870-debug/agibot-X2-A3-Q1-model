"""Record the taskset-applied mask, then exec the unchanged observe binary."""
import argparse
import json
import os
from pathlib import Path
import threading
import time


def validate(command, cpus):
    if (not command or Path(command[0]).name != "x2_morph_native"
            or "--mode" not in command or command[command.index("--mode")+1:] == []
            or command[command.index("--mode")+1] != "observe" or set(cpus) != {0,1}):
        raise ValueError("Require the original observe binary on CPUs 0,1")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    cpus = sorted(os.sched_getaffinity(0))
    validate(command, cpus)
    with args.receipt.open("x") as stream:
        json.dump(dict(role="native",pid=os.getpid(),tid=threading.get_native_id(),cpus=cpus,
            command=command,monotonic_ns=time.monotonic_ns(),hardware_commands=0),stream,indent=2)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
