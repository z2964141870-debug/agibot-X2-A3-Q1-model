#!/usr/bin/env python3
"""Run the existing inverse/contact diagnostic on fixed admitted PHUMA train clips."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys

HERE = Path(__file__).resolve().parent
SELECTION_SEED = 2026102508


def select_rows(manifest, checked, splits, count=8):
    names = [name for group in splits.values() for name in group]
    if (set(splits) != {"train", "development", "test"}
            or len(names) != len(set(names)) or not 1 <= count <= 16):
        raise ValueError("Require three disjoint registered splits and a bounded count")
    records = {row["clip"]: row for row in manifest["motions"]}
    checks = {row["clip"]: row for row in checked["clips"]}
    if len(records) != len(manifest["motions"]) or len(checks) != len(checked["clips"]):
        raise ValueError("Repeated conversion or check identity")
    admitted = {name for name, row in checks.items() if row["qualified_for_training_pilot"]}
    if admitted != set(names) or admitted != set(checked["qualified_clips"]):
        raise ValueError("Independent admission differs from the registered split coverage")
    rows = []
    for name in sorted(admitted):
        row, check = records[name], checks[name]
        expected = next(split for split, members in splits.items() if name in members)
        if (row["split"] != expected or check["split"] != expected
                or row["phuma_source"]["source_group"] != check["source_group"]
                or row["bucket"] != check["bucket"]
                or not row["qualified_for_training_pilot"] or row["output"] != check["reference"]
                or len(check["gates"]) != 19 or not all(check["gates"].values())):
            raise ValueError("Reference origin, split, identity or all 19 gates must agree")
        if expected == "train" and row["bucket"] == "locomotion":
            rows.append(row)
    if len(rows) < count:
        raise ValueError("Not enough independently admitted train displacement candidates")
    rows.sort(key=lambda row: hashlib.sha256(
        (str(SELECTION_SEED) + ":" + row["clip"]).encode()).digest())
    return rows[:count]


def worker(args):
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_analysis"))
    import mujoco as mj
    import numpy as np
    from inspect_reference_dynamics import (
        LIMITS, checked, configured_model, container_guard, identity, inspect_clip)
    from runtime_guard import save

    root, experiment = args.root.resolve(strict=True), args.experiment.resolve(strict=True)
    spec_path, admission_path = experiment / "spec.json", experiment / "run/admission.json"
    spec, admission = (json.loads(path.read_text()) for path in (spec_path, admission_path))
    manifest_path = Path(spec["motions"]) / "manifest.json"
    check_path, completion_record = Path(spec["reference_receipt"]), spec["library_completion"]
    completion = json.loads(checked(root, completion_record).read_text())
    if (completion["manifest"] != identity(manifest_path)
            or completion["reference_check"] != identity(check_path)
            or admission["motion_manifest"] != completion["manifest"]):
        raise ValueError("Use the unchanged registered library and independent geometry receipt")
    manifest, reference_check = (json.loads(path.read_text()) for path in (manifest_path, check_path))
    if (reference_check["input_manifest"] != completion["manifest"]
            or manifest["q1_assets"] != admission["assets"]):
        raise ValueError("Library geometry or Q1 control assets differ from the trainer")
    rows = select_rows(manifest, reference_check, spec["split_clips"])
    helper = HERE.parent / "supermimic_q1_analysis/inspect_reference_dynamics.py"
    inputs = [identity(path) for path in (Path(__file__), helper, spec_path, admission_path)]
    inputs += [completion_record, completion["manifest"], completion["reference_check"],
               admission["assets"]["mjcf"], *(row["output"] for row in rows)]
    for record in inputs:
        checked(root, record)
    baseline = container_guard(args.expected_oom_kill, reserve=512 * 1024 ** 2)
    model, feet, floor, caps, meshes = configured_model(root, admission, mj)
    inputs += meshes
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in rows:
        result = inspect_clip(model, feet, floor, caps, row, args.output, mj, args.expected_oom_kill)
        result.update(split="train", source_group=row["phuma_source"]["source_group"],
                      bucket="source_displacement_candidate_not_certified_walking")
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    for record in inputs:
        checked(root, record)
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected Torch import during reference-only diagnostics")
    report = {"status": "REFERENCE_INVERSE_CONTACT_DIAGNOSTIC_COMPLETED",
              "captured_utc": datetime.now(timezone.utc).isoformat(),
              "stage": "PHUMA_D04_TRAIN_ONLY", "source": identity(__file__),
              "admission": identity(admission_path), "spec": identity(spec_path),
              "selection_seed": SELECTION_SEED, "selection_rule": "First 8 SHA-ordered admitted train displacement candidates; no policy results or held-out clips used.",
              "inputs": inputs, "inputs_verified_before_after": len(inputs), "mesh_inputs": meshes,
              "versions": {"mujoco": mj.__version__, "numpy": np.__version__},
              "active_joint_names": admission["assets"]["active_joint_names"],
              "effective_effort_nm": caps.tolist(), "friction": 1.,
              "contact_protocol": "Q1_FOOT_PRIORITY_V2", "clips": results,
              "acceleration_method": "central3/central5 derivatives of saved interval qvel; two boundary frames excluded per end",
              "health_before": baseline, "health_after": container_guard(args.expected_oom_kill),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "torch_imported": False, "references_modified": False,
              "physics_tracking_validated": False, "storage_status": "LOCAL_ONLY", "limits": LIMITS}
    save(args.output / "receipt.json", report)


def guarded(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, guarded_run

    root, output = args.root.resolve(strict=True), args.output.resolve()
    experiment = args.experiment.resolve(strict=True)
    if (output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1")
            or not experiment.is_relative_to(root / "data/supermimic_for_q1")):
        raise ValueError("Require a new diagnostic output and an existing Q1 experiment")
    spec = json.loads((experiment / "spec.json").read_text())
    budget = MemoryBoundBudget(StorageBudget(output, Path(spec["motions"]).parent,
                                            100 * 1024 ** 3, 512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    output.mkdir()
    os.nice(10)
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--root", str(root),
               "--experiment", str(experiment), "--output", str(output / "diagnosis"),
               "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json", root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "diagnosis", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(guarded(args))
