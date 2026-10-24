#!/usr/bin/env python3
"""Prepare native-speed AMASS body references before loading a Q1 robot model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q1"))


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def selected_rows(selection, offset, count):
    if (selection["status"] != "AMASS_BOUNDED_RAW_MOVING_CANDIDATES_SELECTED"
            or selection["qualified_for_q1_training"] or offset < 0 or not 1 <= count <= 16):
        raise ValueError("Expected a bounded, unadmitted AMASS candidate selection")
    rows = selection["selected"][offset:offset + count]
    if len(rows) != count or any(row["split"] != "UNASSIGNED_DIAGNOSTIC_ONLY"
                                 or row["qualified_for_q1_training"] or not row["raw_window_candidate"]
                                 or row["source_start_frame"] != 0 or row["source_stop_frame_exclusive"] != 481
                                 or row["source_hz"] != 120. or row["num_betas"] != 16 for row in rows):
        raise ValueError("Window, shape or unassigned-split contract changed")
    return rows


def resampled_reference(times, positions, rotations, names):
    from egolocate_input import resample
    from q1_retarget import BodyReference

    target, points, quats = resample(times, positions, rotations, hz=50.)
    return BodyReference(target, names, points, quats, points[0], quats[0], names,
                         "AMASS_SMPLX16_NATIVE120_TO50_DIAGNOSTIC_ONLY_UNASSIGNED")


def worker(args):
    import gc
    import numpy as np
    import torch
    from dataset_input import paired_smplx_standing, smplx_stageii

    selection = json.loads(args.selection.read_text())
    rows = selected_rows(selection, args.offset, args.count)
    intake = json.loads(args.model_intake.read_text())
    if intake["status"] != "SMPLX_PACKAGE_AND_16_BETA_AMASS_FK_PASSED":
        raise ValueError("The previously verified sixteen-beta body models are required")
    model_folder = args.model_intake.resolve().parent
    models = {row["gender"]: row for row in intake["models"]}
    for gender in {row["gender"] for row in rows}:
        record = {key: models[gender][key] for key in ("path", "bytes", "sha256")}
        if (Path(record["path"]).resolve() != model_folder / "smplx" / f"SMPLX_{gender.upper()}.npz"
                or identity(record["path"]) != record):
            raise ValueError("Verified body-model identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    names = list(json.loads((HERE.parent / "supermimic_q1_phuma/retarget_mapping.json").read_text())["body_map"])
    frozen = [identity(args.selection), identity(args.model_intake)]
    source_paths = [Path(__file__), HERE / "prepare_amass_body.py", HERE.parent / "q1/dataset_input.py",
                    HERE.parent / "q1/egolocate_input.py", HERE.parent / "q1/q1_retarget.py",
                    HERE.parent / "q1/inspect_egolocate.py", HERE.parent / "supermimic_q1_phuma/retarget_mapping.json"]
    sources = [identity(path) for path in dict.fromkeys(source_paths)]
    motions, started = [], time.monotonic()
    for row in rows:
        if identity(row["source"]["path"]) != row["source"]:
            raise ValueError("Frozen AMASS source bytes changed")
        times, positions, rotations, conversion = smplx_stageii(
            Path(row["source"]["path"]), model_folder, 4., names)
        if (len(times) != 481 or times[-1] != 4. or conversion["source_fps"] != 120.
                or conversion["num_betas"] != 16 or conversion["gender"] != row["gender"]):
            raise ValueError("FK changed the frozen native-time or shape contract")
        body = resampled_reference(times, positions, rotations, names)
        body, calibration = paired_smplx_standing(body, conversion)
        clip = "amass_" + hashlib.sha256(row["relative_path"].encode()).hexdigest()[:16]
        destination = args.output / (clip + ".npz")
        body.save(destination)
        item = {"clip": clip, "source": row, "body_reference": identity(destination),
                "conversion": conversion, "calibration": calibration,
                "frames": len(body.times), "duration_s": float(body.times[-1]),
                "horizontal_displacement_m": float(np.linalg.norm(
                    body.positions[:, names.index("pelvis"), :2]
                    - body.positions[:1, names.index("pelvis"), :2], axis=-1).max()),
                "split": "UNASSIGNED_DIAGNOSTIC_ONLY", "qualified_for_q1_training": False}
        motions.append(item)
        print(json.dumps({key: value for key, value in item.items()
                          if key not in ("source", "conversion", "calibration")}), flush=True)
        del times, positions, rotations, body
        gc.collect()
    if any(identity(row["path"]) != row for row in frozen + sources):
        raise ValueError("Input receipt or body-preparation source changed during FK")
    if torch.cuda.is_initialized():
        raise ValueError("AMASS body preparation must not create a CUDA context")
    manifest = {"status": "AMASS_BODY_REFERENCES_PREPARED_Q1_PENDING", "captured_utc": datetime.now(timezone.utc).isoformat(),
                "selection": frozen[0], "model_intake": frozen[1], "offset": args.offset, "count": args.count,
                "motions": motions, "sources": sources, "new_q1_references": 0, "new_physics_steps": 0,
                "new_optimizer_updates": 0, "qualified_for_q1_training": False, "storage_status": "LOCAL_ONLY"}
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {key: value for key, value in manifest.items() if key != "motions"}
    summary.update({"manifest": identity(args.output / "manifest.json"), "body_reference_count": len(motions),
                    "body_frames": sum(row["frames"] for row in motions),
                    "source_window_duration_s": sum(row["duration_s"] for row in motions),
                    "fk_vs_full_library_max_abs_error_m": max(row["conversion"]["fk_vs_full_library_max_abs_error_m"] for row in motions),
                    "calibration_fk_max_abs_error_m": max(row["calibration"]["fk_vs_official_forward_max_abs_error_m"] for row in motions),
                    "body_references": [{key: row[key] for key in ("clip", "body_reference", "frames", "duration_s", "split", "qualified_for_q1_training")} for row in motions],
                    "no_cuda_context": True, "elapsed_s": time.monotonic() - started,
                    "limits": "Human body references only; prior SMPL-X full-forward checks at three source frames and synthetic standing calibration retained. No Q1 model, retargeting, geometry admission, split assignment or policy success. Shape parameters remain only in the ignored full manifest."})
    with (args.output / "completion.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


def guarded(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, guarded_run

    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent AMASS body-preparation directory")
    selected_rows(json.loads(args.selection.read_text()), args.offset, args.count)
    budget = MemoryBoundBudget(StorageBudget(output, args.model_intake.resolve().parent,
                                            100 * 1024 ** 3, 512 * 1024 ** 2, 2 * 1024 ** 3),
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
               "--selection", str(args.selection.resolve()), "--model-intake", str(args.model_intake.resolve()),
               "--output", str(output / "body"), "--offset", str(args.offset), "--count", str(args.count),
               "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json", root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "body", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model-intake", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(guarded(args))
