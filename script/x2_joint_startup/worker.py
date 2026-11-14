"""Route file-descriptor diagnostics while preserving explicit readiness JSON."""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import sys
import threading
import time


@contextmanager
def diagnostic_channel(python_protocol=False):
    saved_fd = os.dup(1)
    protocol = os.fdopen(os.dup(saved_fd), "w", buffering=1)
    old_stdout, old_original = sys.stdout, sys.__stdout__
    libc = ctypes.CDLL(None)
    libc.fflush.argtypes, libc.fflush.restype = [ctypes.c_void_p], ctypes.c_int
    try:
        old_stdout.flush()
        libc.fflush(None)
        os.dup2(2, 1)
        sys.stdout = protocol if python_protocol else sys.stderr
        sys.__stdout__ = protocol
        yield
    finally:
        try:
            sys.stdout.flush()
            libc.fflush(None)
            protocol.flush()
        finally:
            os.dup2(saved_fd, 1)
            sys.stdout, sys.__stdout__ = old_stdout, old_original
            protocol.close()
            os.close(saved_fd)


def main():
    mode = sys.argv[1]
    if mode not in ("fgp", "policy"):
        raise ValueError("Only owned FGP and advisory policy workers")
    output = Path(sys.argv[sys.argv.index("--output")+1])
    from script.x2_joint_alloc import worker
    original_bind = worker.bind
    with (output/("kernel_binding_"+mode+".jsonl")).open("x") as tape:
        def bind(role):
            cpus = original_bind(role)
            tape.write(json.dumps(dict(role=role,pid=os.getpid(),tid=threading.get_native_id(),
                cpus=cpus,monotonic_ns=time.monotonic_ns(),hardware_commands=0))+"\n")
            tape.flush()
            return cpus
        worker.bind = bind
        try:
            with diagnostic_channel(python_protocol=(mode == "policy")):
                worker.main()
        finally:
            worker.bind = original_bind


if __name__ == "__main__":
    main()
