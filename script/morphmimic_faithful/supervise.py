"""Run one MorphMimic phase under the established project resource guards."""

import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE.parent / "supermimic_q1_hp"),
                str(HERE.parent / "supermimic_q1_ppo")]
from run_trial import Q1Budget, process_identity
from runtime_guard import guarded_run, save


def verify_upstream(source, tree):
    pinned = json.loads(tree.read_text())
    if pinned["sha"] != "aa40fe46a7fa995ba7a7354ad065a2c8ad64a302":
        raise ValueError("Different upstream version")
    count = 0
    for row in pinned["tree"]:
        if row["type"] != "blob":
            continue
        path = source / row["path"]
        payload = path.read_bytes()
        header = f"blob {len(payload)}\0".encode()
        if len(payload) != row["size"] or hashlib.sha1(header + payload).hexdigest() != row["sha"]:
            raise ValueError("Upstream source identity differs: " + row["path"])
        count += 1
    return count


def own_sources(config, tree):
    paths = [*HERE.glob("*.py"), config, tree,
             HERE.parent / "supermimic_q1_hp/run_trial.py",
             HERE.parent / "supermimic_q1_hp/worker.py",
             HERE.parent / "supermimic_q1_sources/runtime_guard.py",
             HERE.parent / "supermimic_q1_sources/guard_phuma_batch.py"]
    return [dict(path=str(p), bytes=p.stat().st_size,
                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


class Budget(Q1Budget):
    def check(self):
        record = super().check()
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.args.expected_boot:
            raise RuntimeError("Server boot identity changed")
        if record["gpu"]["uuid"] != self.args.expected_gpu:
            raise RuntimeError("GPU identity changed")
        if self.args.phase in ("data", "profile") and time.monotonic() - self.started > 1800:
            raise RuntimeError("Preparation/profile whole-wall limit reached")
        return record


def execute(args):
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record = dict(status="INITIALIZING", mode=args.mode, process=process_identity(),
                  code_commit=args.code_commit, automatic_retry=False,
                  goal="MORPHMIMIC_FAITHFUL01", goal_complete=False)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    save(args.output / "status.json", record)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads(args.config.read_text())
        source = Path(config["source"])
        if source.resolve() != ROOT / "script/third_party/morphmimic_aa40fe46":
            raise ValueError("Unexpected source root")
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if raw.strip():
            raise RuntimeError("GPU already has compute processes")
        cgroup = Path("/sys/fs/cgroup/memory")
        usage = int((cgroup / "memory.usage_in_bytes").read_text())
        limit = int((cgroup / "memory.limit_in_bytes").read_text())
        if usage > 24 * 1024**3 or limit <= 30 * 1024**3:
            raise RuntimeError("Original24GiB initial admission not satisfied")
        record.update(source_blobs_verified=verify_upstream(source, args.upstream_tree),
                      configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      own_sources=own_sources(args.config, args.upstream_tree))
        os.environ.update(PYTHONPATH=str(source / "src") + os.pathsep + str(source),
            GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="1",
            MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
        python = ROOT / "data/environments/morphmimic_original_20261029/bin/python"
        command = [str(python), "-u", str(HERE / "worker.py"), "--mode", args.mode,
                   "--config", str(args.config), "--output", str(args.output / "run")]
        budget_args = SimpleNamespace(output=args.output, runtime=source,
            mode="independent", phase=args.mode, expected_oom_kill=0,
            expected_boot=args.expected_boot, expected_gpu=args.expected_gpu)
        record.update(status="RUNNING", command=command)
        save(args.output / "status.json", record)
        guarded = guarded_run(command, ROOT, Budget(budget_args), args.output / "guard.json",
            ROOT / "logs" / (args.output.name + ".log"), grace_s=10,
            check_interval_s=2, stop_requested=lambda: requested,
            progress_path=args.output / "run/progress.json", maximum_idle_s=1800)
        if guarded["status"] != "WORKER_COMPLETED" or guarded["cleanup"]["worker_return_code"]:
            raise RuntimeError("Worker did not complete; retain original failure")
        record.update(status="COMPLETED", source_blobs_unchanged=verify_upstream(source, args.upstream_tree))
        if hashlib.sha256(args.config.read_bytes()).hexdigest() != record["configuration_sha256"]:
            raise RuntimeError("Configuration changed during execution")
        if own_sources(args.config, args.upstream_tree) != record["own_sources"]:
            raise RuntimeError("Execution sources changed during phase")
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_unix"] = time.time()
        save(args.output / "status.json", record)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("data", "profile", "train"), required=True)
    for key in ("config", "output", "upstream-tree"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-boot", default="4d859008-4c83-4a6d-814d-cbb9dfe885c9")
    parser.add_argument("--expected-gpu", default="GPU-26028785-9fb9-45af-e614-4cc53af15d08")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.config = args.config.resolve(strict=True)
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("New phase must be inside the sole Q1 data root")
    if len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit):
        parser.error("An independently published full40-character commit is required")
    if not args.controller:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        forwarded = [a for a in sys.argv[1:] if a != "--detach"]
        with (ROOT / "logs" / (args.output.name + "_controller.log")).open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(HERE / "supervise.py"),
                *forwarded, "--controller"], stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", dict(controller_pid=child.pid, code_commit=args.code_commit))
        print(json.dumps(dict(controller_pid=child.pid, output=str(args.output))))
    else:
        execute(args)


if __name__ == "__main__":
    main()
