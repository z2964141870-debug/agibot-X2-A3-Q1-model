"""Apply CPU masks around the unchanged ordered R1 pipeline and policy worker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from .allocation import PLAN, bind, sources, validate_available


def physics_command(command):
    command = list(command)
    if "stage_tran_gp.worker" not in command:
        return command
    i = command.index("stage_tran_gp.worker")
    if i == 0 or command[i-1] != "-m":
        raise ValueError("Unexpected author physics worker command")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset unavailable")
    return [taskset, "--cpu-list", ",".join(map(str, PLAN["cpu"])), *command]


def main():
    validate_available(os.sched_getaffinity(0))
    if sys.argv[1] == "policy":
        bind("policy")
        from script.x2_joint_perf.worker import main as original
        original()
        return
    if sys.argv[1] != "fgp":
        raise ValueError("Only registered owned workers may be allocated")
    output = Path(sys.argv[sys.argv.index("--output")+1])
    if (output/"affinity.json").exists():
        raise ValueError("Preserve prior affinity evidence")
    roles, mutex = {}, threading.Lock()
    def enter(role):
        tid = threading.get_native_id()
        with mutex:
            existing = roles.get(role)
            if existing is not None:
                if existing["tid"] != tid:
                    raise RuntimeError("Execution role changed owner thread")
                return
            roles[role] = dict(tid=tid, cpus=bind(role))
    enter("gpu")
    from script.x2_joint_pipeline import worker
    original_scheduler, original_spawn = worker.BoundedPipeline, subprocess.Popen
    class AllocatedPipeline(original_scheduler):
        def __init__(self, process, send, capacity=2, close_sender=None):
            def cpu(item):
                enter("cpu")
                return process(item)
            def sender(item):
                enter("sender")
                return send(item)
            super().__init__(cpu, sender, capacity, close_sender)
    def spawn(command, *args, **kwargs):
        return original_spawn(physics_command(command), *args, **kwargs)
    status = "WORKER_FAILED"
    worker.BoundedPipeline, subprocess.Popen = AllocatedPipeline, spawn
    try:
        worker.main()
        status = "WORKER_CLEAN_EXIT"
    finally:
        worker.BoundedPipeline, subprocess.Popen = original_scheduler, original_spawn
        (output/"affinity.json").write_text(json.dumps(dict(status=status, pid=os.getpid(),
            plan=PLAN, roles=roles, source_sha256=sources(), hardware_commands=0,
            scope="Owned tasks only; no exclusive cpuset or system service modification"), indent=2)+"\n")


if __name__ == "__main__":
    main()
