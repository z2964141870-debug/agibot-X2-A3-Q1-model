#!/usr/bin/env python3
"""Prepare a separately reserved PHUMA walking test without changing source splits."""

import argparse
import collections
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget, guarded_run


def worker(args):
    import audit_scan
    import retarget
    import verify_reference
    from prepare_expansion_batch import uncached_identity

    selected = json.loads(args.selection.read_text())
    contract = selected["walking_holdout"]
    if (selected["status"] != "PHUMA_P10_SUPPLEMENTARY_WALKING_HOLDOUT_SELECTED"
            or contract["role"] != "SUPPLEMENTARY_TEST_ONLY_NEVER_TRAIN"
            or not contract["source_split_preserved"]):
        raise ValueError("Require a separately reserved walking test selection")
    for record in contract["inputs"] + contract["sources"]:
        if identity(record["path"]) != record:
            raise ValueError("Walking selection input or source changed")
    archive = Path(selected["archive"]["path"]).resolve()
    old_identity = retarget.identity

    def checked_identity(path):
        return uncached_identity(path) if Path(path).resolve() == archive else old_identity(path)

    retarget.identity = checked_identity
    audit_scan.file_id = checked_identity
    output = args.output / "batch"
    output.mkdir()
    audit_scan.audit(args.selection, output / "scan_check.json")
    vendor = args.root / "data/references/phuma_source_2faf15b5"
    retarget.convert(args.root, args.selection, vendor, output / "references", False)
    verify_reference.verify(args.root, output / "references", output / "reference_check.json")
    check = json.loads((output / "reference_check.json").read_text())
    admitted = [row for row in check["clips"] if row["qualified_for_training_pilot"]]
    receipt = {"status": "SUPPLEMENTARY_WALKING_GEOMETRY_PREPARATION_COMPLETED",
               "captured_utc": datetime.now(timezone.utc).isoformat(),
               "selection": identity(args.selection), "scan_check": identity(output / "scan_check.json"),
               "manifest": identity(output / "references/manifest.json"),
               "reference_check": identity(output / "reference_check.json"),
               "candidate_count": len(selected["selected"]), "qualified_count": len(admitted),
               "qualified_clips": [row["clip"] for row in admitted],
               "qualified_duration_s": sum(row["duration_s"] for row in admitted),
               "qualified_roles": {row["clip"]: "SUPPLEMENTARY_TEST_ONLY_NEVER_TRAIN" for row in admitted},
               "original_source_split_counts": dict(collections.Counter(row["split"] for row in admitted)),
               "rejection_gate_counts": dict(collections.Counter(key for row in check["clips"]
                    for key, passed in row.get("gates", {}).items() if not passed)),
               "conversion_error_count": sum("conversion_error" in row for row in check["clips"]),
               "source_isolation_certified": False, "new_physics_steps": 0, "new_optimizer_updates": 0,
               "new_training_references": 0, "storage_status": "LOCAL_ONLY",
               "sources": [identity(__file__), identity(HERE.parent / "supermimic_q1_phuma/audit_scan.py"),
                           identity(HERE.parent / "supermimic_q1_phuma/retarget.py"),
                           identity(HERE.parent / "supermimic_q1_phuma/verify_reference.py")],
               "limits": "All original nineteen geometric checks retained; their training-pilot field means geometric admission only. Every new reference is reserved exclusively for supplementary evaluation. Name semantics, full cross-source identity and actual walking tracking are not certified."}
    with (output / "completion.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


def main(args):
    args.root = args.root.resolve(strict=True)
    args.selection = args.selection.resolve(strict=True)
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new bounded walking-preparation directory")
    selected = json.loads(args.selection.read_text())
    source = Path(selected["archive"]["path"]).resolve(strict=True).parent
    budget = MemoryBoundBudget(StorageBudget(args.output, source,
        100 * 1024 ** 3, 256 * 1024 ** 2, 250 * 1024 ** 3), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    args.output.mkdir()
    os.nice(10)
    stop = False

    def requested(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
               "--selection", str(args.selection), "--output", str(args.output),
               "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    result = guarded_run(command, args.root, budget, args.output / "guard.json",
                         args.root / "logs" / (args.output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: stop,
                         progress_path=args.output / "batch/references", maximum_idle_s=900.)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
