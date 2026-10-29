"""Single-use frozen training, saved-only delivery and audit under one GPU lock."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_gait_goal"))
from supervise import Budget, ROOT, guarded_run, identity, memory_admission, process_identity, save, stamp, unchanged


class PhaseBudget(Budget):
    def __init__(self, args, maximum_seconds):
        super().__init__(args)
        self.maximum_seconds = maximum_seconds

    def check(self):
        evidence = super().check()
        if time.monotonic() - self.started >= self.maximum_seconds:
            raise RuntimeError("Owned phase wall limit reached; no automatic retry")
        return evidence


def preflight_sources(args):
    sources = {}
    for count, path in ((16, args.preflight16), (4096, args.preflight4096)):
        status = json.loads(path.read_text())
        plan_path, receipt_path = path.parent / "plan.json", path.parent / "run/receipt.json"
        plan, receipt = json.loads(plan_path.read_text()), json.loads(receipt_path.read_text())
        expected = "GOAL_BASE_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASSED" if args.line == "base" else "ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED"
        if (status["status"] != "COMPLETED" or status["line"] != args.line
                or plan["line"] != args.line or plan["mode"] != f"preflight{count}"
                or receipt["status"] != expected or receipt["optimizer_steps"] != 0):
            raise ValueError("Matching completed zero-update preflights required")
        unchanged(plan["source"])
        source_map = {row["path"]: row for row in plan["source"]}
        for value in (args.library, args.checkpoint, args.calibration):
            if source_map.get(str(value)) != identity(value):
                raise ValueError("Preflight used a different pinned dependency")
        sources.update(source_map)
        sources.update({str(value): identity(value) for value in (path, plan_path, receipt_path)})
    return sources


def execute(args):
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    status = dict(status="INITIALIZING", line=args.line, process=process_identity(),
                  started_utc=stamp(), code_commit=args.code_commit, automatic_retry=False)
    save(args.output / "status.json", status)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU computation already active")
        budget = PhaseBudget(args, 7200)
        budget.check()
        admission = memory_admission()
        save(args.output / "memory_admission.json", admission)
        if not admission["admitted"]:
            raise RuntimeError("Original memory admission not met; no worker started")
        sources = preflight_sources(args)
        own = [*HERE.glob("*.py"), args.video_suite,
               HERE.parent / "project_tools/plot_training_dashboard.py",
               HERE.parent / "project_tools/audit_q1_rf13_media.py",
               HERE.parent / "project_tools/audit_q1_hp_artifacts.py"]
        sources.update({str(path): identity(path) for path in own})
        def command(entry, flags):
            return [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
                    "-u", str(HERE / entry), *flags]
        train_flags = ["--line", args.line, "--code-commit", args.code_commit]
        for name in ("runtime", "library", "checkpoint", "calibration"):
            train_flags.extend(("--" + name, str(getattr(args, name))))
        train_flags.extend(("--output", str(args.output / "run")))
        stages = [
            ("imports", "train.py", [*train_flags, "--check-imports"], 120, "import_guard.json", None),
            ("training", "train.py", train_flags, 7200, "train_guard.json", args.output / "run/status.json"),
            ("delivery", "deliver.py", ["--line", args.line, "--runtime", str(args.runtime),
                "--trial", str(args.output), "--video-suite", str(args.video_suite)],
                1800, "delivery_guard.json", args.output / "delivery_status.json"),
            ("saved_audit", "audit_saved.py", ["--trial", str(args.output), "--line", args.line],
                300, "audit_guard.json", None)]
        plan = dict(source=list(sources.values()), code_commit=args.code_commit, line=args.line,
            hard_training_seconds=1200, soft_training_seconds=1100, external_stop_seconds=1150,
            grace_seconds=10, physical_gates_seconds=[300, 600, 1100], automatic_retry=False,
            automatic_extension=False, original_preflight_sources_unchanged=True,
            training_only_reporting_adapter=True, preflight_repeated=False,
            phases=[dict(name=name, command=command(entry, flags), maximum_wall_seconds=cap)
                    for name, entry, flags, cap, _, _ in stages])
        save(args.output / "plan.json", plan)
        for name, entry, flags, cap, guard_name, progress_path in stages:
            status.update(status="RUNNING", phase=name, updated_utc=stamp())
            save(args.output / "status.json", status)
            guard = guarded_run(command(entry, flags), ROOT, budget if name == "training" else PhaseBudget(args, cap),
                args.output / guard_name, ROOT / f"logs/{args.output.name}_{name}.log", check_interval_s=1,
                grace_s=10, stop_requested=lambda: requested, progress_path=progress_path,
                maximum_idle_s=600)
            if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
                raise RuntimeError(name + " did not complete; no automatic retry")
            if name == "training":
                receipt = json.loads((args.output / "run/receipt.json").read_text())
                if receipt["status"] != "TRAINING_COMPLETED":
                    raise ValueError("Missing normal training terminal receipt")
                status["training_seconds"] = receipt["training_seconds"]
            unchanged(list(sources.values()))
        if json.loads((args.output / "saved_audit.json").read_text())["status"] != "SAVED_EVIDENCE_AUDIT_PASSED":
            raise ValueError("Saved delivery audit not passed")
        status.update(status="COMPLETED", phase="pipeline_completed", user_accepted_checkpoint=None,
            goal_complete=False, visual_review_completed=False, frozen_sources_unchanged=len(sources))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["ended_utc"] = stamp()
        save(args.output / "status.json", status)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "video-suite", "output", "preflight16", "preflight4096"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-boot", required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    args.mode = "train"
    for name in ("runtime", "library", "checkpoint", "calibration", "video_suite", "preflight16", "preflight4096"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if len(args.code_commit) != 40 or any(value not in "0123456789abcdef" for value in args.code_commit):
        parser.error("Full40-digit independently published commit required")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use the unique server project root")
    if args.controller:
        execute(args)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               *[value for value in sys.argv[1:] if value != "--detach"], "--controller"]
        with (ROOT / f"logs/{args.output.name}_controller.log").open("x") as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", dict(pid=child.pid, command=cmd, started_utc=stamp(), automatic_retry=False))
        if not args.detach:
            raise SystemExit(child.wait())
