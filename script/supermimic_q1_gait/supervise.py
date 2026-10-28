"""Single-use RF14 controller with an independent cumulative training guard."""

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
from resources import identity, save, stamp


class Budget(Q1Budget):
    def __init__(self, a):
        super().__init__(SimpleNamespace(output=a.output, runtime=a.runtime, mode="independent",
                                        expected_oom_kill=a.expected_oom_kill))
        self.original = a

    def check(self):
        result = super().check()
        a = self.original
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != a.expected_boot:
            raise RuntimeError("Boot identity changed")
        if result["gpu"]["uuid"] != a.expected_gpu:
            raise RuntimeError("GPU identity changed")
        if time.monotonic() - self.started >= (21600 if a.mode == "base" else 1800):
            raise RuntimeError("Pipeline wall-time cap reached")
        path = a.output / "run/training_clock.json"
        if path.exists():
            clock = json.loads(path.read_text())
            if clock["absolute_maximum_seconds"] != 14400:
                raise RuntimeError("Invalid cumulative four-hour clock")
            elapsed = clock["accumulated_training_seconds"]
            if clock["active_started_monotonic"] is not None:
                elapsed += time.monotonic() - clock["active_started_monotonic"]
            result["cumulative_training_seconds"] = elapsed
            if elapsed >= 14340:
                raise RuntimeError("External cumulative14340s stop; 45s grace within14400s")
        return result


def unchanged(items):
    for item in items:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen source changed: " + item["path"])


def execute(a):
    status = {"status": "INITIALIZING", "started_utc": stamp(), "process": process_identity(),
              "mode": a.mode, "code_commit": a.code_commit, "no_automatic_retry": True}
    save(a.output / "status.json", status)
    previous = json.loads(a.frozen_plan.read_text())["source"]
    unchanged(previous)
    paths = list(HERE.glob("*.py")) + [a.library, a.checkpoint, a.video_suite]
    merged = {v["path"]: v for v in previous}
    merged.update({str(p): identity(p) for p in paths})
    sources = list(merged.values())
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
               str(HERE / ("base.py" if a.mode == "base" else "preflight.py"))]
    for name in ("runtime", "library", "checkpoint"):
        command += ["--" + name, str(getattr(a, name))]
    command += ["--output", str(a.output / "run")]
    if a.mode == "base":
        command += ["--code-commit", a.code_commit]
    else:
        command += ["--envs", "16" if a.mode == "preflight16" else "4096"]
    save(a.output / "plan.json", {"source": sources, "command": command, "code_commit": a.code_commit,
         "process": process_identity(), "started_utc": stamp(), "maximum_training_seconds": 14400,
         "inherited_seconds": 3500.2659895760007, "soft_training_seconds": 14300,
         "external_training_stop_seconds": 14340, "grace_seconds": 45,
         "no_automatic_retry": True, "old_frozen_sources_checked": len(previous),
         "boot": a.expected_boot, "gpu": a.expected_gpu})
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    budget = Budget(a)
    try:
        status.update(status="RUNNING")
        save(a.output / "status.json", status)
        guard = guarded_run(command, ROOT, budget, a.output / "train_guard.json",
            ROOT / "logs" / (a.output.name + ".log"), check_interval_s=2, grace_s=45,
            stop_requested=lambda: requested, progress_path=a.output / "run/status.json", maximum_idle_s=900)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Owned worker failed; no automatic restart")
        receipt = json.loads((a.output / "run/receipt.json").read_text())
        expected = "TRAINING_COMPLETED" if a.mode == "base" else "ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED"
        if receipt["status"] != expected:
            raise RuntimeError("Actual worker completion receipt mismatch")
        unchanged(sources)
        if a.mode == "base":
            status.update(status="TRAINING_COMPLETED_PENDING_DELIVERY", phase="delivery")
            save(a.output / "status.json", status)
            post = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
                    str(HERE / "deliver.py"), "--runtime", str(a.runtime), "--trial", str(a.output),
                    "--video-suite", str(a.video_suite)]
            delivered = guarded_run(post, ROOT, budget, a.output / "delivery_guard.json",
                ROOT / "logs" / (a.output.name + "_delivery.log"), grace_s=45,
                progress_path=a.output / "delivery_status.json", maximum_idle_s=900,
                stop_requested=lambda: requested)
            if delivered["status"] != "WORKER_COMPLETED":
                raise RuntimeError("Saved training complete; nontraining delivery failed")
            unchanged(sources)
        status.update(status="COMPLETED", completed_utc=stamp(), receipt=identity(a.output / "run/receipt.json"),
                      frozen_sources_unchanged=len(sources))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(a.output / "status.json", status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight16", "preflight4096", "base"), required=True)
    for name in ("runtime", "library", "checkpoint", "video-suite", "frozen-plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--code-commit")
    p.add_argument("--expected-oom-kill", type=int, default=0)
    p.add_argument("--expected-boot", required=True)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--controller", action="store_true")
    a = p.parse_args()
    if a.mode == "base" and (a.code_commit is None or len(a.code_commit) != 40):
        p.error("Formal training needs published40-digit code commit")
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        p.error("Use sole ZHY data tree")
    if not a.controller:
        a.output.mkdir(parents=True, exist_ok=False)
    if a.detach:
        command = [sys.executable, "-u", str(HERE / "supervise.py"),
                   *[v for v in sys.argv[1:] if v != "--detach"], "--controller"]
        log = ROOT / "logs" / (a.output.name + "_controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        save(a.output / "launch.json", {"controller_pid": child.pid, "started_utc": stamp(), "log": str(log)})
        print(json.dumps({"controller_pid": child.pid, "output": str(a.output)}))
    else:
        execute(a)


if __name__ == "__main__":
    main()
