"""Independent admission and GPU ownership after the explicit V04 user pause."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess

from preview_review import COMMIT, GIB, HANDLES, ROOT, TRAINING
from common import require
from delivery import read
from memory_watch import process


LOCK = ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock"


def paused_state(config, receipt, *, proc=Path("/proc"), training=TRAINING, root=ROOT):
    require(Path.cwd().resolve() == root and Path(config["output"]) == training,
            "Require the original sole server root and shared experiment")
    require((proc / "sys/kernel/random/boot_id").read_text().strip() == config["expected_boot"],
            "Server boot changed after the user pause")
    require(receipt.get("pause_verified") is True and receipt.get("training_commit") == COMMIT
            and receipt.get("boot") == config["expected_boot"]
            and receipt.get("gpu") == config["expected_gpu"]
            and Path(receipt.get("training_root", "")) == training,
            "Missing matching verified user-pause receipt")
    require(all(process(proc, pid) is None for pid in HANDLES), "An original training PID is present")
    pipeline, guard = read(training / "pipeline.json"), read(training / "e1024/train/guard.json")
    require(pipeline.get("status") == "FAILED"
            and "Controller stopped after a phase" in pipeline.get("error", "")
            and guard.get("status") == "STOPPED_SIGNAL", "Original training is not in its explicit stopped state")


def assert_owner(pid, ticks, *, lock=LOCK, proc=Path("/proc")):
    require(type(pid) is int and pid > 1 and isinstance(ticks, str) and ticks.isdigit(),
            "Missing controller GPU-owner identity")
    owner = process(proc, pid)
    require(owner is not None and owner["start_ticks"] == ticks, "GPU owner exited or PID was reused")
    stat = lock.stat()
    expected = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    for line in (proc / "locks").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "FLOCK" and fields[3] == "WRITE" and fields[4] == str(pid):
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == expected:
                return owner
    raise RuntimeError("Controller does not own the original exclusive GPU lock")


@contextmanager
def own_gpu(lock=LOCK):
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another job owns the original GPU lock; do not contend or restart") from error
        try:
            require(not compute_pids(), "GPU has an existing compute process")
            yield process(Path("/proc"), os.getpid())
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def compute_pids():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    return {int(line.strip()) for line in output.splitlines() if line.strip()}
