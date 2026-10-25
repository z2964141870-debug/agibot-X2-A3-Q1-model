#!/usr/bin/env python3
"""Evaluate stopped R02D and run P15 using the explicit M01 relocation view."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from guard_phuma_batch import MemoryBoundBudget
from relocated_io import identity, migration_view, JsonView, TorchView
from runtime_guard import StorageBudget, guarded_run, save


HERE = Path(__file__).resolve().parent
R02D = "PHUMA_R02D_bounded_partial5000_extend7000_20261025"


def budget(args):
    return MemoryBoundBudget(StorageBudget(args.output,
        args.root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)


def evaluate_worker(args, view):
    import torch

    sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))
    import run_contact_corrected as driver
    import evaluate

    driver.json, evaluate.json = JsonView(view), JsonView(view)
    driver.torch, evaluate.torch = TorchView(torch, view), TorchView(torch, view)
    sys.argv = [driver.__file__, "evaluate", "--root", str(args.root),
        "--motions", str(args.motions), "--checkpoint", str(args.checkpoint),
        "--output", str(args.destination), "--clips", *args.clips]
    driver.main()


def development(args, view):
    from evaluate_phuma_R02_checkpoint import metrics

    base = args.root / "data/supermimic_for_q1"
    experiment = base / R02D
    spec = view.load(experiment / "spec.json")
    if len(spec["split_clips"]["development"]) != 6:
        raise ValueError("Expected the original six development clips")
    names = spec["split_clips"]["development"]
    if set(names) & set(spec["split_clips"]["train"] + spec["split_clips"]["test"]):
        raise ValueError("Development clip membership overlaps the original other splits")
    checkpoints = {"parent5000": experiment / "run/step_000000.pt",
                   "bounded10282": experiment / "run/final.pt"}
    audit_path = args.root / "data/manifests/supermimic_q1_R02D_exit_save_readonly_audit_20261025.json"
    audit = json.loads(audit_path.read_text())
    if (audit["status"] != "STOPPED_R02D_EXIT_SAVE_READ_ONLY_AUDIT_PASSED"
            or not all(audit["gates"].values()) or audit["final_budget_completed"]):
        raise ValueError("Require the original successful exit-save audit, not completed training")
    view.verify_records(audit["inputs"])
    source = {row["clip"]: row for row in view.load(Path(spec["motions"]) / "manifest.json")["motions"]}
    if any(source[name]["split"] != "development" for name in names):
        raise ValueError("Development source membership changed")
    originals = [identity(p) for p in checkpoints.values()]
    results, evidence = {}, []
    save(args.output / "plan.json", {"status": "REGISTERED_RELOCATED_DEVELOPMENT_COMPARISON",
        "checkpoints": originals, "clips": names, "test_evaluated": False,
        "training_complete": False, "optimizer_updates": 0,
        "tracking_gates": {"complete_without_early_failure": True, "joint_rmse_rad": .15,
                           "root_position_rmse_m": .10}, "relocation": view.evidence()})
    core = args.root / "script/supermimic_for_q1"
    for label, checkpoint in checkpoints.items():
        destination = args.output / ("evaluation_" + label)
        audit_file = args.output / ("evaluation_audit_" + label + ".json")
        commands = [
            ("evaluate_" + label, [str(Path(__file__).resolve()), "evaluate-worker", "--root", str(args.root),
                "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill),
                "--checkpoint", str(checkpoint), "--motions", spec["motions"],
                "--destination", str(destination), "--clips", *names]),
            ("audit_" + label, [str(core / "audit_evaluation.py"), "--root", str(args.root),
                "--evaluation", str(destination), "--output", str(audit_file)])]
        for phase, command in commands:
            budget(args).check()
            log = args.root / "logs" / (args.output.name + "_" + phase + ".log")
            with log.open("x") as stream:
                completed = subprocess.run([sys.executable, "-u", *command], cwd=args.root,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
            if completed.returncode:
                raise RuntimeError(phase + " exited " + str(completed.returncode))
            print(json.dumps({"completed_phase": phase}), flush=True)
        receipt_file = destination / "receipt.json"
        receipt, trace_audit = json.loads(receipt_file.read_text()), json.loads(audit_file.read_text())
        if (trace_audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
                or trace_audit["receipt"] != identity(receipt_file)
                or receipt["checkpoint"] != identity(checkpoint)
                or sorted(row["clip"] for row in receipt["rows"]) != sorted(names)):
            raise ValueError("Independent development trace verification failed")
        view.verify_records(receipt["rows"])
        results[label] = {row["clip"]: metrics(row) for row in receipt["rows"]}
        evidence.extend([identity(receipt_file), identity(audit_file)])
    if originals != [identity(p) for p in checkpoints.values()]:
        raise ValueError("A checkpoint changed during the relocated evaluation")
    view.verify_records(audit["inputs"])
    view.verify_records(spec["frozen_files"])
    pairs = [{"clip": name, "split": "development", "bucket": source[name]["bucket"],
              **{label: result[name] for label, result in results.items()}} for name in names]
    report = {"status": "RELOCATED_R02D_DEVELOPMENT_MATCHED_PHYSICS_COMPLETED",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "pairs": pairs, "evidence": evidence,
        "test_evaluated": False, "training_complete": False, "optimizer_updates": 0,
        "health_after": budget(args).check(), "relocation": view.evidence(), "storage_status": "LOCAL_ONLY",
        "limits": "Six development clips only; source-person isolation is not certified. No hardware claim."}
    save(args.output / "comparison.json", report)
    print(json.dumps({"status": report["status"], "pairs": pairs}), flush=True)


def worker(args):
    view = migration_view(args.root)
    if args.phase == "evaluate-worker":
        evaluate_worker(args, view)
        return
    save(args.output / "relocation_admission.json", view.evidence())
    if args.phase == "development":
        development(args, view)
    else:
        import probe_walking_floor_recovery as adapter

        # Only JSON reads get an explicit migrated view; original bytes and checks stay intact.
        adapter.engine.json = JsonView(view)
        adapter.worker(args)
        save(args.output / "relocation_after.json", view.evidence())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("development", "p15", "evaluate-worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--clips", nargs="+")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use the unified root for all new artifacts")
    if args.worker or args.phase == "evaluate-worker":
        worker(args)
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    budget(args).memory()
    migration_view(args.root)
    if os.statvfs(args.root).f_bavail * os.statvfs(args.root).f_frsize < 100 * 1024 ** 3:
        raise RuntimeError("The original 100GiB free-space reserve is required")
    args.output.mkdir()
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), args.phase, "--worker",
        "--root", str(args.root), "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill)]
    result = guarded_run(command, args.root, budget(args), args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=1., grace_s=5.,
        stop_requested=lambda: requested, progress_path=args.output, maximum_idle_s=900.)
    if args.phase == "p15" and result["status"] == "WORKER_COMPLETED":
        command = [sys.executable, str(HERE / "audit_walking_root_offset.py"), "--root", str(args.root),
            "--experiment", str(args.output), "--output", str(args.output / "independent_audit.json"),
            "--expected-oom-kill", str(args.expected_oom_kill)]
        return subprocess.run(command, cwd=args.root, timeout=600, check=False).returncode
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
