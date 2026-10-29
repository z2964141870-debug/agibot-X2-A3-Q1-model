"""Single-use native trial and saved rendering, awaiting explicit Mac media proof."""

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / "supermimic_q1_gait_goal_review"
spec = importlib.util.spec_from_file_location("goal_review_pipeline", REVIEW / "pipeline.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
from supervise import ROOT, guarded_run, identity, memory_admission, process_identity, save, stamp, unchanged


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
        budget = review.PhaseBudget(args, 7200)
        budget.check()
        admission = memory_admission()
        save(args.output / "memory_admission.json", admission)
        if not admission["admitted"]:
            raise RuntimeError("Original memory admission not met; no worker started")
        sources = review.preflight_sources(args)
        own = [*HERE.glob("*.py"), *REVIEW.glob("*.py"), args.video_suite,
               HERE.parent / "project_tools/plot_training_dashboard.py",
               HERE.parent / "project_tools/audit_q1_rf13_media.py",
               HERE.parent / "project_tools/audit_q1_hp_artifacts.py"]
        sources.update({str(path): identity(path) for path in own})

        def command(entry, flags):
            return [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
                    "-u", str(entry), *flags]

        flags = ["--line", args.line, "--code-commit", args.code_commit]
        for name in ("runtime", "library", "checkpoint", "calibration"):
            flags.extend(("--" + name, str(getattr(args, name))))
        flags.extend(("--output", str(args.output / "run")))
        stages = [
            ("imports", REVIEW / "train.py", [*flags, "--check-imports"], 120, "import_guard.json", None),
            ("training", REVIEW / "train.py", flags, 7200, "train_guard.json", args.output / "run/status.json"),
            ("delivery", HERE / "render_saved.py", ["--line", args.line, "--runtime", str(args.runtime),
                "--trial", str(args.output), "--video-suite", str(args.video_suite)],
                1800, "delivery_guard.json", args.output / "delivery_status.json")]
        plan = dict(source=list(sources.values()), code_commit=args.code_commit, line=args.line,
            hard_training_seconds=1200, soft_training_seconds=1100, external_stop_seconds=1150,
            grace_seconds=10, physical_gates_seconds=[300, 600, 1100], automatic_retry=False,
            automatic_extension=False, original_preflight_sources_unchanged=True,
            training_only_reporting_adapter=True, preflight_repeated=False,
            external_mac_media_audit_required=True,
            phases=[dict(name=name, command=command(entry, values), maximum_wall_seconds=cap)
                    for name, entry, values, cap, _, _ in stages])
        save(args.output / "plan.json", plan)
        for name, entry, values, cap, guard_name, progress_path in stages:
            status.update(status="RUNNING", phase=name, updated_utc=stamp())
            save(args.output / "status.json", status)
            guard = guarded_run(command(entry, values), ROOT,
                budget if name == "training" else review.PhaseBudget(args, cap),
                args.output / guard_name, ROOT / f"logs/{args.output.name}_{name}.log",
                check_interval_s=1, grace_s=10, stop_requested=lambda: requested,
                progress_path=progress_path, maximum_idle_s=600)
            if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
                raise RuntimeError(name + " did not complete; no automatic retry")
            if name == "training":
                receipt = json.loads((args.output / "run/receipt.json").read_text())
                if receipt["status"] != "TRAINING_COMPLETED":
                    raise ValueError("Normal training terminal required")
                status["training_seconds"] = receipt["training_seconds"]
            unchanged(list(sources.values()))
        status.update(status="AWAITING_MAC_MEDIA_AUDIT", phase="server_artifacts_ready",
            frozen_sources_unchanged=len(sources), goal_complete=False, user_accepted_checkpoint=None)
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
    if len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit):
        parser.error("Full independently published40-digit commit required")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use the unique server project root")
    if args.controller:
        execute(args)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[value for value in sys.argv[1:] if value != "--detach"], "--controller"]
        with (ROOT / f"logs/{args.output.name}_controller.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", dict(pid=child.pid, command=command, started_utc=stamp(), automatic_retry=False))
        if not args.detach:
            raise SystemExit(child.wait())
