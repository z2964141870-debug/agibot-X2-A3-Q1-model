"""Apply only owned role affinity and diagnostics around the existing pipeline."""
import json
import os
from pathlib import Path
import sys
import threading
import time

from .config import PLANS, settings


def native(command, output, config):
    if (not command or Path(command[0]).name != "x2_morph_native" or "--mode" not in command
            or command[command.index("--mode")+1] != "observe"):
        raise ValueError("Only the frozen observe command may exec")
    cpus = sorted(os.sched_getaffinity(0))
    if cpus != PLANS[config["variant"]]["native"]:
        raise ValueError("Native affinity differs")
    with (output/"native_kernel_binding.json").open("x") as f:
        json.dump(dict(role="native",pid=os.getpid(),tid=threading.get_native_id(),cpus=cpus,
                       command=command,monotonic_ns=time.monotonic_ns(),hardware_commands=0),f)
    os.execv(command[0], command)


def main():
    config = settings(json.loads(os.environ["X2_JOINT_OPT_CONFIG"]))
    mode = sys.argv[1]
    output = Path(sys.argv[sys.argv.index("--output")+1])
    if mode == "native":
        return native(sys.argv[sys.argv.index("--")+1:], output, config)
    if mode not in ("fgp", "policy"):
        raise ValueError("Unknown owned worker")
    from script.x2_joint_alloc import allocation, worker
    from script.x2_joint_startup.worker import diagnostic_channel
    allocation.PLAN.clear()
    allocation.PLAN.update(PLANS[config["variant"]])
    original = worker.bind
    with (output/("kernel_binding_"+mode+".jsonl")).open("x") as tape:
        def bind(role):
            cpus = original(role)
            tape.write(json.dumps(dict(role=role,pid=os.getpid(),tid=threading.get_native_id(),
                cpus=cpus,monotonic_ns=time.monotonic_ns(),hardware_commands=0))+"\n")
            tape.flush()
            return cpus
        worker.bind = bind
        try:
            with diagnostic_channel(python_protocol=(mode == "policy")):
                if mode == "fgp":
                    from .profile import instrument
                    source = Path(sys.argv[sys.argv.index("--source-root")+1])
                    with instrument(source, output):
                        worker.main()
                else:
                    worker.main()
        finally:
            worker.bind = original


if __name__ == "__main__":
    main()
