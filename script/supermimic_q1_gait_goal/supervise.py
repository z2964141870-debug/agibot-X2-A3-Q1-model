"""Single-use GAIT_GOAL01 worker with independent resource and active-time caps."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "supermimic_q1_ppo")]
from run_trial import Q1Budget
from runtime_guard import guarded_run
from worker import process_identity
from resources import identity, save, stamp


def active_training_seconds(clock, now):
    if clock["absolute_maximum_seconds"] != 1200 or clock["current_cap_seconds"] != 1200:
        raise ValueError("The independent per-line cap must remain1200s")
    if clock["inherited_training_seconds"] != 0:
        raise ValueError("No old training budget may be inherited")
    total = float(clock["accumulated_training_seconds"])
    start = clock["active_started_monotonic"]
    if start is not None:
        start = float(start)
        if not math.isfinite(start) or start < 0 or start > now:
            raise ValueError("Invalid active clock origin")
        total += now - start
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid active training seconds")
    return total


class Budget(Q1Budget):
    def __init__(self, args):
        super().__init__(SimpleNamespace(output=args.output, runtime=args.runtime, mode="independent",
                                        expected_oom_kill=args.expected_oom_kill))
        self.original = args

    def check(self):
        evidence = super().check()
        args = self.original
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != args.expected_boot:
            raise RuntimeError("Boot identity changed")
        if evidence["gpu"]["uuid"] != args.expected_gpu:
            raise RuntimeError("GPU identity changed")
        cap = 7200 if args.mode == "train" else (600 if args.mode == "preflight16" else 900)
        if time.monotonic() - self.started >= cap:
            raise RuntimeError("Owned worker whole-wall cap reached")
        path = args.output / "run/training_clock.json"
        if args.mode == "train" and path.exists():
            elapsed = active_training_seconds(json.loads(path.read_text()), time.monotonic())
            evidence["actual_training_seconds"] = elapsed
            if elapsed >= 1150:
                raise RuntimeError("External1150s stop;10s cleanup inside1200s; no extension")
        return evidence


def unchanged(items):
    for item in items:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen source changed: " + item["path"])


def execute(args):
    status = dict(status="INITIALIZING", mode=args.mode, line=args.line, process=process_identity(),
                  started_utc=stamp(), code_commit=args.code_commit, automatic_retry=False)
    save(args.output / "status.json", status)
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("Existing GPU computation; worker not started")
        previous = json.loads(args.frozen_plan.read_text())["source"]
        unchanged(previous)
        own = [*HERE.glob("*.py"), *(HERE.parent / "supermimic_q1_sole").glob("*.py"),
               args.library, args.checkpoint, args.calibration]
        if args.mode == "train":
            for count, path in ((16, args.preflight16), (4096, args.preflight4096)):
                parent = json.loads(path.read_text())
                if parent["status"] != "COMPLETED" or parent["line"] != args.line:
                    raise ValueError("Matching completed preflight required")
                plan_path = path.parent / "plan.json"
                plan = json.loads(plan_path.read_text())
                if plan["mode"] != f"preflight{count}" or plan["line"] != args.line:
                    raise ValueError("Wrong preflight environment count")
                unchanged(plan["source"])
                own.extend((path, plan_path, path.parent / "run/receipt.json"))
        sources = {item["path"]: item for item in previous}
        sources.update({str(path): identity(path) for path in own})
        command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
                   str(HERE / ("train.py" if args.mode == "train" else "preflight.py")), "--line", args.line]
        for name in ("runtime", "library", "checkpoint", "calibration"):
            command.extend(("--" + name, str(getattr(args, name))))
        command.extend(("--output", str(args.output / "run")))
        command.extend(("--code-commit", args.code_commit) if args.mode == "train" else
                       ("--envs", "16" if args.mode == "preflight16" else "4096"))
        plan = dict(source=list(sources.values()), command=command, mode=args.mode, line=args.line,
                    code_commit=args.code_commit, process=process_identity(), started_utc=stamp(),
                    hard_training_seconds=1200, soft_training_seconds=1100, physical_gates_seconds=[300, 600, 1100],
                    external_stop_seconds=1150, grace_seconds=10, automatic_extension=False, automatic_retry=False,
                    cross_environment_h32_direction_established=False)
        save(args.output / "plan.json", plan)
        status.update(status="RUNNING")
        save(args.output / "status.json", status)
        guard = guarded_run(command, ROOT, Budget(args), args.output / "train_guard.json",
            ROOT / "logs" / (args.output.name + ".log"), check_interval_s=1, grace_s=10,
            stop_requested=lambda: requested, progress_path=args.output / "run/status.json", maximum_idle_s=600)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Owned worker did not complete; no automatic retry")
        receipt = json.loads((args.output / "run/receipt.json").read_text())
        expected = ("TRAINING_COMPLETED" if args.mode == "train" else
                    "GOAL_BASE_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASSED" if args.line == "base" else
                    "ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED")
        if receipt["status"] != expected:
            raise ValueError("Worker receipt disagrees with requested phase")
        unchanged(list(sources.values()))
        status.update(status="COMPLETED", completed_utc=stamp(), frozen_sources_unchanged=len(sources),
                      receipt=identity(args.output / "run/receipt.json"),
                      training_seconds=receipt.get("training_seconds", 0), user_accepted_checkpoint=None)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["ended_utc"] = stamp()
        save(args.output / "status.json", status)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "frozen-plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("preflight16", "preflight4096"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--mode", choices=("preflight16", "preflight4096", "train"), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-boot", required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    if len(args.code_commit) != 40 or any(char not in "0123456789abcdef" for char in args.code_commit):
        parser.error("Independently published full40-digit commit required")
    if args.mode == "train" and (args.preflight16 is None or args.preflight4096 is None):
        parser.error("Both completed line-specific preflights required")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use the unique server project root")
    if args.controller:
        execute(args)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[value for value in sys.argv[1:] if value != "--detach"], "--controller"]
        with (ROOT / "logs" / (args.output.name + "_controller.log")).open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", dict(pid=child.pid, command=command, started_utc=stamp(), automatic_retry=False))
        if not args.detach:
            return child.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
