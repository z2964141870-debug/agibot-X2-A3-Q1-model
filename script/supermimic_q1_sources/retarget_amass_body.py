#!/usr/bin/env python3
"""Retarget prepared AMASS bodies to diagnostic Q1 candidates under a memory guard."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "q1", "supermimic_q1_phuma"):
    sys.path.insert(0, str(HERE.parent / directory))

from prepare_amass_body import identity


PROTOCOL = "AMASS_SMPLX16_NATIVE120_TO_Q1_50HZ_UNASSIGNED_V1"


def body_rows(manifest):
    if (manifest["status"] != "AMASS_BODY_REFERENCES_PREPARED_Q1_PENDING"
            or manifest["qualified_for_q1_training"] or manifest["new_q1_references"] != 0
            or not 1 <= len(manifest["motions"]) <= 16):
        raise ValueError("Expected bounded AMASS body references without training admission")
    rows = manifest["motions"]
    if (len({row["clip"] for row in rows}) != len(rows)
            or len({row["source"]["source"]["sha256"] for row in rows}) != len(rows)):
        raise ValueError("Repeated source or body clip")
    for row in rows:
        source = row["source"]
        if (row["split"] != "UNASSIGNED_DIAGNOSTIC_ONLY" or row["qualified_for_q1_training"]
                or source["split"] != row["split"] or source["qualified_for_q1_training"]
                or source["source_hz"] != 120. or source["num_betas"] != 16
                or source["source_start_frame"] != 0 or source["source_stop_frame_exclusive"] != 481
                or row["frames"] != 201 or row["duration_s"] != 4.):
            raise ValueError("AMASS native-time, shape or unassigned-split contract changed")
    return rows


def worker(args):
    import mujoco
    import numpy as np
    from contact_protocol import ContactCorrectedAssets
    from prepare_motion import export_q1
    from q1_retarget import BodyReference
    from retarget import grounded_solver, initialize_first_pose
    from validate_egolocate import leg_scale
    from runtime_guard import save

    manifest = json.loads(args.body_manifest.read_text())
    rows = body_rows(manifest)
    frozen = [identity(args.body_manifest), manifest["selection"], manifest["model_intake"]]
    frozen += manifest["sources"]
    frozen += [row["body_reference"] for row in rows]
    frozen += [row["source"]["source"] for row in rows]
    sources = [identity(path) for path in (
        Path(__file__), HERE / "prepare_amass_body.py", HERE.parent / "q1/validate_egolocate.py",
        HERE.parent / "q1/q1_retarget.py", HERE.parent / "q1/q1_sim.py",
        HERE.parent / "supermimic_q1_phuma/retarget.py",
        HERE.parent / "supermimic_q1_phuma/retarget_control.json",
        HERE.parent / "supermimic_q1_phuma/retarget_mapping.json",
        HERE.parent / "supermimic_for_q1/prepare_motion.py")]
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("AMASS body preparation inputs or code changed")
    assets = ContactCorrectedAssets(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnostics").mkdir()
    motions, started = [], time.monotonic()
    for row in rows:
        item = {"clip": row["clip"], "body_reference": row["body_reference"],
                "source": row["source"], "split": "UNASSIGNED_DIAGNOSTIC_ONLY",
                "bucket": "locomotion_candidate", "qualified_for_training_pilot": False}
        try:
            solver, floor = grounded_solver(assets)
            body = BodyReference.load(row["body_reference"]["path"], solver.names)
            if (len(body.times) != 201
                    or not np.allclose(body.times, np.arange(201) / 50., rtol=0, atol=1e-12)):
                raise ValueError("Body reference no longer has the verified four-second native clock")
            scale, scaling = leg_scale(body, solver)
            item.update(scale=scale, scaling=scaling, target_calibration_floor_m=floor,
                        first_pose_initialization=initialize_first_pose(solver, body, scale))
            result = solver.retarget(body, scale)
            trace = args.output / "diagnostics" / (row["clip"] + ".npz")
            np.savez_compressed(trace, time_s=body.times, qpos_before_ground_origin=result[1],
                                target_position_m_world=result[2], target_rotation_matrix=result[3],
                                position_error_m=result[4], source_position_m_world=body.positions,
                                source_orientation_wxyz_world=body.orientations,
                                source_body_names=np.asarray(body.names))
            item.update(export_q1(assets, result[1], result[4], args.output / (row["clip"] + ".npz")),
                        kinematic_trace=identity(trace), source_hz=120., target_hz=50., time_scale=1.)
            item["converter_kinematic_candidate"] = item.pop("qualified_for_training_pilot")
            item["qualified_for_training_pilot"] = False
        except Exception as exc:
            item.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        motions.append(item)
        save(args.output / "progress.json", {"completed": len(motions), "requested": len(rows),
                                            "elapsed_s": time.monotonic() - started})
        print(json.dumps({key: item[key] for key in ("clip", "scale", "diagnostics", "error")
                          if key in item}, allow_nan=False), flush=True)
    if any(identity(record["path"]) != record for record in frozen + sources):
        raise ValueError("AMASS input or Q1 converter identity changed during retargeting")
    report = {"status": "AMASS_Q1_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING",
              "reference_protocol": PROTOCOL, "captured_utc": datetime.now(timezone.utc).isoformat(),
              "body_manifest": frozen[0], "q1_assets": assets.manifest(), "motions": motions,
              "sources": sources, "converted_count": sum("output" in row for row in motions),
              "conversion_errors": sum("error" in row for row in motions),
              "qualified_for_training": False, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "elapsed_s": time.monotonic() - started, "storage_status": "LOCAL_ONLY",
              "limits": "Diagnostic Q1 candidates only. Independent native-time, mapping and geometry checks, source alias reconciliation and split assignment remain required."}
    save(args.output / "manifest.json", report)
    summary = {key: value for key, value in report.items() if key != "motions"}
    summary["manifest"] = identity(args.output / "manifest.json")
    save(args.output / "completion.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def guarded(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, guarded_run

    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent Q1 candidate directory")
    body_rows(json.loads(args.body_manifest.read_text()))
    budget = MemoryBoundBudget(StorageBudget(output, args.body_manifest.resolve().parent,
                                            100 * 1024 ** 3, 512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    output.mkdir()
    os.nice(10)
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--root", str(root),
               "--body-manifest", str(args.body_manifest.resolve()), "--output", str(output / "references"),
               "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json", root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "references/progress.json", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--body-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(guarded(args))
