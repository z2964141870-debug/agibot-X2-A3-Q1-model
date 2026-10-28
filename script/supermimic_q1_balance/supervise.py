"""Own one independent residual experiment; absolute training cap 3600 seconds."""

from __future__ import annotations

import argparse
import json
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
from resources import save, stamp, identity


class Budget(Q1Budget):
    def __init__(self, args):
        super().__init__(SimpleNamespace(output=args.output, runtime=args.runtime, mode="independent",
                                         expected_oom_kill=args.expected_oom_kill))
        self.original = args

    def check(self):
        result = super().check()
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.original.expected_boot:
            raise RuntimeError("Container boot identity changed")
        if result["gpu"]["uuid"] != self.original.expected_gpu:
            raise RuntimeError("GPU identity changed")
        if time.monotonic() - self.started > (10800 if self.original.mode == "train" else 1800):
            raise RuntimeError("Independent pipeline wall cap reached")
        path = self.original.output / "run/training_clock.json"
        if path.exists():
            clock = json.loads(path.read_text())
            if clock["absolute_maximum_seconds"] != 3600 or clock["current_cap_seconds"] != 3600:
                raise RuntimeError("Invalid independent one-hour training clock")
            elapsed = clock["accumulated_training_seconds"]
            if clock["active_started_monotonic"] is not None:
                elapsed += time.monotonic() - clock["active_started_monotonic"]
            result["actual_training_seconds"] = elapsed
            if elapsed >= 3540:
                raise RuntimeError("External cumulative3540s stop; 45s grace, no retry")
        return result


def check_sources(items):
    for item in items:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen source changed: " + item["path"])


def execute(a):
    status = {"status": "INITIALIZING", "started_utc": stamp(), "process": process_identity(),
              "mode": a.mode, "code_commit": a.code_commit, "no_automatic_retry": True}
    save(a.output / "status.json", status)
    previous = json.loads(a.frozen_plan.read_text())["source"]
    check_sources(previous)
    own_paths = list(HERE.glob("*.py")) + [a.calibration, a.library, a.checkpoint]
    own_paths += [HERE.parent / "project_tools" / n for n in
                  ("diagnose_q1_upper_reference.py", "q1_upper_reference.py")]
    if a.video_suite is not None:
        own_paths.append(a.video_suite)
    merged = {item["path"]: item for item in previous}
    merged.update({str(p): identity(p) for p in own_paths})
    sources = list(merged.values())
    entry = "train.py" if a.mode == "train" else "diagnose.py" if a.mode == "diagnose" else "preflight.py"
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u", str(HERE / entry)]
    for name in ("runtime", "library", "checkpoint", "calibration"):
        command += ["--" + name, str(getattr(a, name))]
    command += ["--output", str(a.output / "run")]
    if a.mode.startswith("preflight"):
        command += ["--envs", "16" if a.mode == "preflight16" else "4096"]
    if a.mode == "train":
        command += ["--code-commit", a.code_commit]
    save(a.output / "plan.json", {"source": sources, "command": command, "code_commit": a.code_commit,
        "started_utc": stamp(), "process": process_identity(), "old_frozen_sources_checked": len(previous),
        "boot": a.expected_boot, "gpu": a.expected_gpu, "mode": a.mode,
        "environment_count": 16 if a.mode == "preflight16" else 33 if a.mode == "diagnose" else 4096,
        "maximum_training_seconds": 3600, "soft_training_seconds": 3500,
        "external_training_stop_seconds": 3540, "grace_seconds": 45, "no_automatic_retry": True,
        "scope": "Independent RF13 only; no RF12 continuation, expansion or five-hour extension."})
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    budget = Budget(a)
    try:
        budget.check()
        status.update(status="RUNNING")
        save(a.output / "status.json", status)
        guard = guarded_run(command, ROOT, budget, a.output / "train_guard.json",
            ROOT / "logs" / (a.output.name + ".log"), check_interval_s=2, grace_s=45,
            stop_requested=lambda: requested, progress_path=a.output / "run/status.json", maximum_idle_s=900)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0 or not guard["cleanup"]["worker_reaped"]:
            raise RuntimeError("Owned worker failed/stopped; preserve its terminal outcome")
        receipt = json.loads((a.output / "run/receipt.json").read_text())
        expected = "TRAINING_COMPLETED" if a.mode == "train" else "DIAGNOSTIC_COMPLETED" if a.mode == "diagnose" else "ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED"
        if receipt["status"] != expected:
            raise RuntimeError("Worker receipt does not match actual completion")
        check_sources(sources)
        if a.mode == "train":
            status.update(status="TRAINING_COMPLETED_PENDING_DELIVERY", phase="delivery")
            save(a.output / "status.json", status)
            post = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
                    str(HERE / "deliver.py"), "--runtime", str(a.runtime), "--trial", str(a.output),
                    "--video-suite", str(a.video_suite)]
            delivered = guarded_run(post, ROOT, budget, a.output / "delivery_guard.json",
                ROOT / "logs" / (a.output.name + "_delivery.log"), grace_s=45,
                progress_path=a.output / "delivery_status.json", maximum_idle_s=900,
                stop_requested=lambda: requested)
            if delivered["status"] != "WORKER_COMPLETED" or delivered["cleanup"]["worker_return_code"] != 0:
                raise RuntimeError("Training saved; nontraining delivery failed")
            check_sources(sources)
        status.update(status="COMPLETED", completed_utc=stamp(), receipt=identity(a.output / "run/receipt.json"),
                      frozen_sources_unchanged=len(sources))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(a.output / "status.json", status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight16", "preflight4096", "diagnose", "train"), required=True)
    for name in ("runtime", "library", "checkpoint", "calibration", "frozen-plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--video-suite", type=Path)
    p.add_argument("--code-commit")
    p.add_argument("--expected-oom-kill", type=int, default=0)
    p.add_argument("--expected-boot", required=True)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--controller", action="store_true")
    a = p.parse_args()
    if a.mode == "train" and (a.video_suite is None or a.code_commit is None or len(a.code_commit) != 40):
        p.error("Training needs fixed video suite and published40-digit source commit")
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        p.error("Use sole ZHY data tree")
    if not a.controller:
        a.output.mkdir(parents=True, exist_ok=False)
    if a.detach:
        forwarded = [v for v in sys.argv[1:] if v != "--detach"]
        log = ROOT / "logs" / (a.output.name + "_controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(HERE / "supervise.py"), *forwarded, "--controller"],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(a.output / "launch.json", {"controller_pid": child.pid, "started_utc": stamp(), "log": str(log)})
        print(json.dumps({"controller_pid": child.pid, "output": str(a.output)}))
    else:
        try:
            execute(a)
        except BaseException as error:
            path = a.output / "status.json"
            value = json.loads(path.read_text()) if path.exists() else {"process": process_identity()}
            value.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
            save(path, value)
            raise


if __name__ == "__main__":
    main()
