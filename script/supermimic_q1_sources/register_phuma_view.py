#!/usr/bin/env python3
"""Admit a documented source-exclusion view to the preserved expanded Q1 pipeline."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from phuma_source_view import identity, validated_view


HERE = Path(__file__).resolve().parent
EXPANDED = HERE.parent / "supermimic_q1_expanded"


def preflight(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget

    budget = MemoryBoundBudget(StorageBudget(args.library, args.library, 100 * 1024 ** 3,
                                            512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    before = budget.check()
    evidence = validated_view(args.root, args.library, args.source_aliases)
    evidence.update(captured_utc=datetime.now(timezone.utc).isoformat(),
                    source=identity(__file__), health_before=before, health_after=budget.check(),
                    new_physics_steps=0, new_optimizer_updates=0, storage_status="LOCAL_ONLY")
    return evidence


def register(args):
    root, output = args.root.resolve(strict=True), args.experiment.resolve()
    if (output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1")
            or args.steps < 300 or len(args.code_commit) != 40):
        raise ValueError("Use a new independent Q1 experiment, finite budget and published commit")
    evidence = preflight(args)
    backend = json.loads(args.backend_receipt.read_text())
    if (backend["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
            or backend["model_slots"] != args.model_slots
            or backend["resource_probe64"]["environments"] != 64):
        raise ValueError("Preserve the verified 64-environment model-slot backend")
    sys.path.insert(0, str(EXPANDED))
    import expanded_pipeline as pipeline

    args.motions = args.library / "references"
    args.reference_receipt, args.scan_check = (args.library / name for name in
                                              ("reference_check.json", "scan_check.json"))
    pipeline.register_legacy_template(args)
    spec_path = args.experiment / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec.update(num_envs=64, threads=4, model_slots=args.model_slots,
                library_completion=evidence["library_completion"], source_view=evidence,
                registration_entry=identity(__file__), training_entry=identity(EXPANDED / "train_expanded.py"),
                training_method="Learned differentiable world model, imagined actor BPTT, critic; H32 unchanged.",
                runtime_backend="VERIFIED_MUJOCO_MODEL_SLOTS_NO_DISCRETIZED_DR")
    files = {row["path"]: row for row in spec["frozen_files"] + evidence["evidence_files"]}
    for path in [Path(__file__), HERE / "guard_phuma_batch.py", HERE / "runtime_guard.py",
                 *EXPANDED.glob("*.py")]:
        record = identity(path)
        files[record["path"]] = record
    spec["frozen_files"] = [files[key] for key in sorted(files)]
    pipeline.verify_bindings(spec)
    pipeline.save(spec_path, spec)
    print(json.dumps({"status": "SOURCE_VIEW_RUN_REGISTERED_NOT_STARTED", "spec": identity(spec_path),
                      "num_envs": 64, "model_slots": args.model_slots, "steps": args.steps,
                      "training_transitions": args.steps * 64,
                      "split_counts": {key: len(value) for key, value in spec["split_clips"].items()},
                      "source_isolation_certified": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("check", "register", "run"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--source-aliases", type=Path)
    parser.add_argument("--expected-oom-kill", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--backend-receipt", type=Path)
    parser.add_argument("--model-slots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2026102509)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.phase in ("check", "register"):
        if any(getattr(args, key) is None for key in ("root", "library", "source_aliases", "expected_oom_kill")):
            parser.error("check/register need root, library, source aliases and expected OOM count")
    if args.phase == "check":
        if args.output is None or args.output.exists():
            parser.error("check needs a new output receipt")
        if not args.output.resolve().is_relative_to(args.root.resolve() / "data/supermimic_for_q1"):
            parser.error("check output must stay under Q1 experiment data")
        evidence = preflight(args)
        with args.output.open("x") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": evidence["status"], "receipt": identity(args.output),
                          "split_counts": {key: len(value) for key, value in evidence["split_clips"].items()},
                          "new_optimizer_updates": 0}), flush=True)
    elif args.phase == "register":
        if any(getattr(args, key) is None for key in ("experiment", "backend_receipt", "code_commit")):
            parser.error("register needs experiment, backend receipt and published code commit")
        register(args)
    else:
        if args.experiment is None:
            parser.error("run needs the previously registered experiment")
        experiment = args.experiment.resolve(strict=True)
        spec = json.loads((experiment / "spec.json").read_text())
        if (spec["registration_entry"] != identity(__file__)
                or spec["source_view"]["source_isolation_certified"] is not False):
            raise ValueError("Require this entry's independently registered source-exclusion contract")
        sys.path.insert(0, str(EXPANDED))
        import expanded_pipeline as pipeline

        pipeline.run(experiment)


if __name__ == "__main__":
    main()
