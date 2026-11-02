"""Guard one native Q1 integration phase without imposing old short training gates."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path("/opt/data/private/ZHY")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from run_trial import Q1Budget, process_identity
from runtime_guard import guarded_run, save


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


class Budget(Q1Budget):
    def check(self):
        result = super().check()
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.args.expected_boot:
            raise RuntimeError("Original boot changed")
        if result["gpu"]["uuid"] != self.args.expected_gpu:
            raise RuntimeError("Original GPU changed")
        if time.monotonic() - self.started > self.args.wall_seconds:
            raise RuntimeError("Frozen phase wall limit reached")
        return result


def run(args):
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record = dict(status="INITIALIZING", process=process_identity(),
        code_commit=args.code_commit, purpose=args.purpose, command=args.command,
        started_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False,
        optimizer_updates="reported_by_worker" if args.purpose in ("profile", "train") else 0,
        user_accepted_checkpoint=None)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True)
        if running.strip():
            raise RuntimeError("GPU already has compute processes")
        cgroup = Path("/sys/fs/cgroup/memory")
        usage = int((cgroup / "memory.usage_in_bytes").read_text())
        limit = int((cgroup / "memory.limit_in_bytes").read_text())
        if usage > 24 * 1024**3 or limit <= 30 * 1024**3:
            raise RuntimeError("Original24GiB initial admission not satisfied")
        sources = sorted(set([*args.input, *HERE.glob("*.py"),
            HERE.parent / "morphmimic_cross_robot/collective_transport.py",
            *args.source.rglob("*.py")]))
        record["inputs"] = [identity(path) for path in sources]
        budget = Budget(SimpleNamespace(output=args.output, runtime=args.source,
            mode="independent", expected_oom_kill=0, expected_boot=args.expected_boot,
            expected_gpu=args.expected_gpu, wall_seconds=args.wall_seconds))
        record["initial_resources"] = budget.check()
        record["status"] = "RUNNING"
        save(args.output / "status.json", record)
        env = os.environ.copy()
        env.update(GIT_PYTHON_GIT_EXECUTABLE=str(
            ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            TORCHINDUCTOR_COMPILE_THREADS="1", MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
        os.environ.update(env)
        result = guarded_run(args.command, args.cwd, budget, args.output / "guard.json",
            args.output / "stdout.log", check_interval_s=2, grace_s=10,
            stop_requested=lambda: requested)
        record["guard"] = result
        if result["status"] != "WORKER_COMPLETED" or result["cleanup"]["worker_return_code"]:
            raise RuntimeError("Phase did not complete; preserve original logs")
        if [identity(path) for path in sources] != record["inputs"]:
            raise RuntimeError("Frozen phase sources changed")
        record["status"] = "COMPLETED"
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "status.json", record)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("output", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--purpose", choices=("retarget", "preflight", "profile", "train"), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--expected-boot", default="4d859008-4c83-4a6d-814d-cbb9dfe885c9")
    parser.add_argument("--expected-gpu", default="GPU-26028785-9fb9-45af-e614-4cc53af15d08")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.command = args.command[1:] if args.command[:1] == ["--"] else args.command
    args.output = args.output.resolve()
    args.source, args.cwd = args.source.resolve(strict=True), args.cwd.resolve(strict=True)
    args.input = [p.resolve(strict=True) for p in args.input]
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use a new phase inside the sole Q1 data root")
    if not all(p.is_relative_to(ROOT) for p in (args.source, args.cwd, *args.input)):
        parser.error("All inputs must stay inside the sole project root")
    if (len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit)
            or not args.command or not 0 < args.wall_seconds <= 7 * 86400):
        parser.error("Require a full published commit, command and bounded phase wall time")
    if not args.controller:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        forwarded = [a for a in sys.argv[1:] if a != "--detach"]
        with (args.output / "controller.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--controller", *forwarded], stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", dict(controller_pid=child.pid, code_commit=args.code_commit))
        print(json.dumps(dict(controller_pid=child.pid, output=str(args.output))), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
