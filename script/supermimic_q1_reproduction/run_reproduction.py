#!/usr/bin/env python3
"""Prepare BONES references and run the paper-first Q1 training entry."""

import argparse
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parent / "supermimic_q1_sources"
sys.path.insert(0, str(SOURCES))
from bounded_archive_io import bounded_identity
from guard_phuma_batch import MemoryBoundBudget
from relocated_io import JsonView, migration_view
from runtime_guard import StorageBudget, guarded_run, save
from reference_admission import POLICY, assemble


def worker(args):
    if args.phase == "prepare":
        import bones_retarget as converter
        import mink
        import relocated_io

        relocated_io.identity = bounded_identity
        view = migration_view(args.root)
        converter.json = JsonView(view)
        original_solver = converter.grounded_solver

        def native_speed_solver(assets):
            solver, floor = original_solver(assets)
            solver.sim.lower = solver.sim.model.jnt_range[solver.sim.joint_ids, 0].copy()
            solver.sim.upper = solver.sim.model.jnt_range[solver.sim.joint_ids, 1].copy()
            solver.sim.config["reference_velocity_limit_rad_s"] = float("inf")
            solver.limits = [mink.ConfigurationLimit(solver.sim.model, min_distance_from_limits=0.)]
            return solver, floor

        converter.grounded_solver = native_speed_solver
        converter.PROTOCOL = "BONES_G1_120HZ_Q1_50HZ_NO_FIXED_SPEED_CAP_V2"
        motions = args.output / "converted"
        converter.convert(SimpleNamespace(root=args.root, intake=args.intake,
            model_comparison=args.root / "data/supermimic_for_q1/BONES_P02_model_20261024/official_comparison.json",
            output=motions, offset=args.offset, count=args.count, expected_oom_kill=args.expected_oom_kill))
        manifest_path = motions / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["retarget_runtime_overrides"] = {"entry": bounded_identity(__file__),
            "fixed_joint_speed_cap_rad_s": None, "joint_limits": "ACTUAL_Q1_XML_RANGES",
            "local_geometry_gates": "DIAGNOSTICS_ONLY_NOT_TRAINING_ADMISSION",
            "native_time_scale": 1., "admission_policy": POLICY}
        save(manifest_path, manifest)
        admission = assemble(motions, args.output / "library")
        save(args.output / "summary.json", {"status": "BONES_REPRODUCTION_REFERENCE_PREPARATION_COMPLETED",
            "admission": admission, "conversion": bounded_identity(manifest_path),
            "entry": bounded_identity(__file__), "relocation": view.evidence(),
            "requested_clips": args.count, "storage_status": "LOCAL_ONLY"})
        print(json.dumps({"status": "BONES_REPRODUCTION_REFERENCE_PREPARATION_COMPLETED",
            "admitted_clips": len(admission["admitted_clips"]), "duration_s": admission["admitted_duration_s"]}), flush=True)
        return
    import train_reproduction
    sys.argv = [str(HERE / "train_reproduction.py"), "--root", str(args.root),
        "--motions", str(args.library / "references"), "--reference-receipt", str(args.library / "admission.json"),
        "--output", str(args.output / "run"), "--backend-receipt", str(args.backend_receipt),
        "--steps", str(args.steps), "--num-envs", "64", "--threads", "4", "--batch", "128",
        "--wm-batch", "64", "--wm-batches", "10", "--checkpoint-every", "1000",
        "--seed", "2026102513", "--code-commit", args.code_commit]
    train_reproduction.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare", "train"))
    for name in ("root", "output", "intake", "library", "backend-receipt"):
        parser.add_argument("--" + name, type=Path, required=name in ("root", "output"))
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--code-commit")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if not args.output.is_relative_to(args.root / "data/supermimic_for_q1") or args.output == args.root / "data/supermimic_for_q1":
        raise ValueError("Keep the independent output in the unified Q1 project")
    if args.phase == "prepare" and (args.intake is None or not 1 <= args.count <= 128):
        parser.error("prepare requires a real intake and a finite frozen candidate slice")
    if args.phase == "train" and any(x is None for x in (args.library, args.backend_receipt, args.code_commit)):
        parser.error("train requires a prepared library, existing backend receipt and published commit")
    if args.worker:
        worker(args)
        return 0
    maximum_memory = int((30. if args.phase == "train" else 28.75) * 1024 ** 3)
    budget = MemoryBoundBudget(StorageBudget(args.output, args.root / "data/references/bones_seed/archives",
        100 * 1024 ** 3, 12 * 1024 ** 3, 250 * 1024 ** 3), maximum_memory, args.expected_oom_kill)
    budget.memory()
    args.output.mkdir()
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), args.phase, "--worker",
        "--root", str(args.root), "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill),
        "--offset", str(args.offset), "--count", str(args.count), "--steps", str(args.steps)]
    for name in ("intake", "library", "backend_receipt", "code_commit"):
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    progress = args.output / ("converted/progress.json" if args.phase == "prepare" else "run/status.json")
    result = guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), grace_s=10., stop_requested=lambda: requested,
        progress_path=progress, maximum_idle_s=900.)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
