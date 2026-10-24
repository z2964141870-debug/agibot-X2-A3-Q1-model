#!/usr/bin/env python3
"""Compare one body-local target mapping on the same four unassigned AMASS clips."""

import argparse
import collections
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "q1", "supermimic_q1_phuma"):
    sys.path.insert(0, str(HERE.parent / directory))

from prepare_amass_body import identity


def worker(args):
    import numpy as np
    from amass_local_offsets import fit_and_solve, independent_local_offsets, local_offset_targets
    from contact_protocol import ContactCorrectedAssets
    from prepare_motion import export_q1
    from q1_retarget import BodyReference
    from retarget import grounded_solver
    from retarget_amass_body import body_rows
    from runtime_guard import save
    from validate_egolocate import leg_scale
    from verify_amass_reference import check

    previous_path = args.baseline / "manifest.json"
    previous = json.loads(previous_path.read_text())
    body_manifest = json.loads(Path(previous["body_manifest"]["path"]).read_text())
    rows = body_rows(body_manifest)
    if len(rows) != 4 or previous["qualified_for_training"] or previous["converted_count"] != 4:
        raise ValueError("Use the completed four-clip unassigned P04 baseline")
    before = json.loads((args.baseline_checks / "reference_check.json").read_text())
    if before["retarget_manifest"] != identity(previous_path):
        raise ValueError("The baseline geometry receipt belongs to another conversion")
    frozen = [identity(previous_path), previous["body_manifest"], identity(args.baseline_checks / "reference_check.json")]
    frozen += previous["sources"] + body_manifest["sources"]
    frozen += [row["body_reference"] for row in rows] + [row["source"]["source"] for row in rows]
    sources = [identity(path) for path in (Path(__file__), HERE / "amass_local_offsets.py",
               HERE / "verify_amass_reference.py", HERE / "retarget_amass_body.py")]
    if any(identity(row["path"]) != row for row in frozen):
        raise ValueError("A bound baseline, source or body preparation file changed")
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != previous["q1_assets"]:
        raise ValueError("Q1 model or control contract changed")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnostics").mkdir()
    candidates, checked = [], []
    old = {row["clip"]: row for row in before["clips"]}
    for row in rows:
        solver, floor = grounded_solver(assets)
        body = BodyReference.load(row["body_reference"]["path"], solver.names)
        scale, scaling = leg_scale(body, solver)
        targets, rotations, roots = local_offset_targets(body, solver.robot_positions, solver.robot_rotations, scale)
        qposes, errors, initial_iterations = fit_and_solve(solver, body, targets, rotations, roots)
        trace_path = args.output / "diagnostics" / (row["clip"] + ".npz")
        np.savez_compressed(trace_path, time_s=body.times, qpos_before_ground_origin=qposes,
                            target_position_m_world=targets, target_rotation_matrix=rotations,
                            position_error_m=errors, source_position_m_world=body.positions,
                            source_orientation_wxyz_world=body.orientations, source_body_names=np.asarray(body.names))
        candidate = {"clip": row["clip"], "source": row["source"], "body_reference": row["body_reference"],
                     "split": "UNASSIGNED_DIAGNOSTIC_ONLY", "qualified_for_training_pilot": False,
                     "source_hz": 120., "target_hz": 50., "time_scale": 1., "scale": scale, "scaling": scaling,
                     "target_calibration_floor_m": floor, "initial_pose_fit_iterations": initial_iterations,
                     "kinematic_trace": identity(trace_path)}
        candidate.update(export_q1(assets, qposes, errors, args.output / (row["clip"] + ".npz")))
        candidate["converter_kinematic_candidate"] = candidate.pop("qualified_for_training_pilot")
        candidate["qualified_for_training_pilot"] = False
        result = check(assets, candidate, BodyReference.load(row["body_reference"]["path"], solver.names),
                       target_builder=independent_local_offsets)
        candidates.append(candidate)
        checked.append(result)
        print(json.dumps({"clip": row["clip"], "baseline_residual_p95_m": old[row["clip"]]["IK_position_p95_m"],
                          "local_offset_residual_p95_m": result["IK_position_p95_m"],
                          "baseline_support_fraction": old[row["clip"]]["support_within_15mm_fraction"],
                          "local_offset_support_fraction": result["support_within_15mm_fraction"],
                          "geometry_qualified": result["geometry_qualified"],
                          "failed_gates": [key for key, value in result["gates"].items() if not value]}), flush=True)
    if any(identity(row["path"]) != row for row in frozen + sources):
        raise ValueError("Baseline, body or implementation identity changed during comparison")
    report = {"status": "AMASS_FOUR_CLIP_LOCAL_OFFSET_COMPARISON_COMPLETED", "captured_utc": datetime.now(timezone.utc).isoformat(),
              "reference_protocol": "AMASS_SMPLX16_NATIVE120_Q1_50HZ_BODY_LOCAL_OFFSETS_DIAGNOSTIC_V1",
              "baseline_manifest": frozen[0], "baseline_check": frozen[2], "q1_assets": assets.manifest(),
              "motions": candidates, "clips": checked, "baseline_clips": before["clips"],
              "geometry_qualified_count": sum(row["geometry_qualified"] for row in checked),
              "rejection_gate_counts": dict(collections.Counter(key for row in checked for key, value in row["gates"].items() if not value)),
              "qualified_for_training": False, "qualified_training_duration_s": 0., "new_physics_steps": 0,
              "new_optimizer_updates": 0, "sources": sources, "storage_status": "LOCAL_ONLY",
              "limits": "Same four purposive diagnostic clips and all 22 prior geometry/export thresholds. Only the calibration offset frame changes, with first-pose fitting matched to those targets. Not a full GMR run, dynamic tracking validation, split assignment or independent test."}
    save(args.output / "receipt.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("motions", "clips", "baseline_clips", "q1_assets")}), flush=True)


def guarded(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, guarded_run

    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent comparison directory")
    budget = MemoryBoundBudget(StorageBudget(output, args.baseline.resolve(strict=True),
                                            100 * 1024 ** 3, 512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    output.mkdir()
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    os.nice(10)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--root", str(root),
               "--baseline", str(args.baseline.resolve()), "--baseline-checks", str(args.baseline_checks.resolve()),
               "--output", str(output / "comparison"), "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json", root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "comparison", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(guarded(args))
