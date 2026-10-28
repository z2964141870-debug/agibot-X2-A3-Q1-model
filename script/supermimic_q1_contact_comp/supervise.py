"""Own RF16 only, with an independent physical-gated four-hour budget."""

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
from contact_limits import OPTIMIZATION_DEADLINE, DELIVERY_DEADLINE, CHECKS


def unchanged(items):
    for item in items:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen source changed: "+item["path"])


class Budget(Q1Budget):
    def __init__(self, a):
        super().__init__(SimpleNamespace(output=a.output, runtime=a.runtime, mode="independent",
                                         expected_oom_kill=a.expected_oom_kill))
        self.original = a

    def check(self):
        evidence = super().check()
        a = self.original
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != a.expected_boot or evidence["gpu"]["uuid"] != a.expected_gpu:
            raise RuntimeError("Boot/GPU identity changed")
        if time.time() >= DELIVERY_DEADLINE:
            raise RuntimeError("Absolute09:00 Beijing overnight deadline reached")
        if time.monotonic()-self.started > (5400 if a.mode == "train" else 2400):
            raise RuntimeError("Whole pipeline wall cap reached")
        path = a.output / "run/training_clock.json"
        if path.exists():
            clock = json.loads(path.read_text())
            cap = clock["current_cap_seconds"]
            if clock["absolute_maximum_seconds"] != 1300 or cap !=1300:
                raise RuntimeError("Invalid explicitly authorized overnight RF16 budget")
            if clock["optimization_wall_deadline_unix"] != OPTIMIZATION_DEADLINE:
                raise RuntimeError("Optimization deadline changed")
            actual = clock["accumulated_training_seconds"]
            if clock["active_started_monotonic"] is not None:
                actual += time.monotonic()-clock["active_started_monotonic"]
            evidence.update(actual_training_seconds=actual, current_training_cap_seconds=cap)
            if time.time() >= OPTIMIZATION_DEADLINE-55 and a.mode == "train":
                raise RuntimeError("External07:00 optimization stop with45s grace")
            if actual >= cap-60:
                raise RuntimeError("External conditional training stop with45s grace")
        return evidence


def execute(a):
    status = {"status": "INITIALIZING", "process": process_identity(), "started_utc": stamp(),
              "mode": a.mode, "code_commit": a.code_commit, "no_automatic_retry": True}
    save(a.output / "status.json", status)
    previous = json.loads(a.frozen_plan.read_text())["source"]
    unchanged(previous)
    own = list(HERE.glob("*.py")) + [a.checkpoint, a.library, a.calibration, a.video_suite]
    library_rows = [r for values in json.loads(a.library.read_text())["splits"].values() for r in values]
    own += [Path(r["path"]).with_suffix(".json") for r in library_rows]
    sources = {item["path"]: item for item in previous}
    sources.update({str(p):identity(p) for p in own})
    sources = list(sources.values())
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
               str(HERE / ("train.py" if a.mode == "train" else "preflight.py"))]
    for name in ("runtime", "library", "checkpoint", "calibration"):
        command += ["--"+name, str(getattr(a,name))]
    command += ["--output", str(a.output / "run"), "--upper-alpha", str(a.upper_alpha)]
    if a.mode == "train":
        command += ["--code-commit", a.code_commit]
    else:
        command += ["--envs", "16" if a.mode == "preflight16" else "4096"]
    save(a.output / "plan.json", {"source": sources, "command": command, "code_commit": a.code_commit,
         "process": process_identity(), "started_utc": stamp(), "boot": a.expected_boot, "gpu": a.expected_gpu,
         "maximum_training_seconds": 1300, "upper_reference_alpha": a.upper_alpha,
         "check_at_seconds": list(CHECKS), "external_reserve_seconds": 60, "grace_seconds": 45,
         "optimization_wall_deadline_unix": OPTIMIZATION_DEADLINE,
         "delivery_wall_deadline_unix": DELIVERY_DEADLINE,
         "authorization": "User autonomous exploration2026-10-29 until09:00 Beijing",
         "extension_allowed": False,
         "user_accepted_checkpoint": None, "no_automatic_retry": True})
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    budget = Budget(a)
    try:
        status.update(status="RUNNING")
        save(a.output / "status.json", status)
        guard = guarded_run(command, ROOT, budget, a.output / "train_guard.json",
            ROOT / "logs" / (a.output.name+".log"), check_interval_s=2, grace_s=45,
            stop_requested=lambda:requested, progress_path=a.output / "run/status.json", maximum_idle_s=900)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("RF16 worker failed; no automatic retry")
        result = json.loads((a.output / "run/receipt.json").read_text())
        expected = "TRAINING_COMPLETED" if a.mode == "train" else "ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED"
        if result["status"] != expected:
            raise RuntimeError("RF16 completion receipt mismatch")
        unchanged(sources)
        if a.mode == "train":
            status.update(status="TRAINING_COMPLETED_PENDING_DELIVERY", phase="delivery")
            save(a.output / "status.json", status)
            command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u", str(HERE / "deliver.py"),
                       "--runtime", str(a.runtime), "--trial", str(a.output), "--video-suite", str(a.video_suite)]
            delivered = guarded_run(command, ROOT, budget, a.output / "delivery_guard.json",
                ROOT / "logs" / (a.output.name+"_delivery.log"), grace_s=45,
                progress_path=a.output / "delivery_status.json", maximum_idle_s=900, stop_requested=lambda:requested)
            if delivered["status"] != "WORKER_COMPLETED":
                raise RuntimeError("Training saved; independent nontraining delivery failed")
            unchanged(sources)
        status.update(status="COMPLETED", completed_utc=stamp(), frozen_sources_unchanged=len(sources),
                      receipt=identity(a.output / "run/receipt.json"))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(a.output / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight16", "preflight4096", "train"), required=True)
    for name in ("runtime", "library", "checkpoint", "calibration", "video-suite", "frozen-plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--code-commit")
    p.add_argument("--upper-alpha", type=float, default=.5)
    p.add_argument("--expected-oom-kill", type=int, default=0)
    p.add_argument("--expected-boot", required=True)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--controller", action="store_true")
    a = p.parse_args()
    if a.mode == "train" and (a.code_commit is None or len(a.code_commit) != 40):
        p.error("Formal training requires published40-digit source commit")
    if not 0 <= a.upper_alpha <= 1:
        p.error("Upper blend must be in [0,1]")
    if time.time() >= OPTIMIZATION_DEADLINE-300:
        p.error("Overnight launch window has closed")
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        p.error("Use sole ZHY data root")
    if not a.controller:
        a.output.mkdir(parents=True, exist_ok=False)
    if a.detach:
        command = [sys.executable, "-u", str(HERE / "supervise.py"),
                   *[v for v in sys.argv[1:] if v != "--detach"], "--controller"]
        log = ROOT / "logs" / (a.output.name+"_controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        save(a.output / "launch.json", {"controller_pid":child.pid, "started_utc":stamp(), "log":str(log)})
        print(json.dumps({"controller_pid":child.pid, "output":str(a.output)}))
    else:
        execute(a)

