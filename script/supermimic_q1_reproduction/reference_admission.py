"""Admit readable native-time Q1 references without local geometry thresholds."""

import json
from pathlib import Path
import sys

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "supermimic_for_q1"
sys.path.insert(0, str(CORE))
from supermimic.q1_assets import TRACKED, identity


POLICY = "USER_20261025_PAPER_FIRST_NO_LOCAL_GEOMETRY_GATES"


def validate_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        time = data["time_s"]
        frames = len(time)
        shapes = {"qpos": (frames, 29), "qvel": (frames, 28),
                  "joint_pos": (frames, 22), "joint_vel": (frames, 22),
                  "body_pos_w": (frames, len(TRACKED), 3),
                  "body_quat_w": (frames, len(TRACKED), 4),
                  "body_lin_vel_w": (frames, len(TRACKED), 3),
                  "body_ang_vel_w": (frames, len(TRACKED), 3)}
        if (time.ndim != 1 or frames < 2 or not np.isfinite(time).all() or time[0] != 0.
                or not np.allclose(np.diff(time), .02, atol=1e-8, rtol=0)
                or float(data["fps"]) != 50. or data["body_names"].tolist() != list(TRACKED)):
            raise ValueError("Reference time axis or Q1 body order cannot be loaded")
        for key, shape in shapes.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError("Reference array is malformed or nonfinite: " + key)
        return {"frames": frames, "duration_s": float(time[-1]), "io_valid": True}


def assemble(motions, output):
    manifest_path = motions / "manifest.json"
    original = json.loads(manifest_path.read_text())
    if original["status"] != "BONES_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING":
        raise ValueError("Complete the actual BONES conversion first")
    output.mkdir()
    references = output / "references"
    references.mkdir()
    rows, checks = [], []
    for source in original["motions"]:
        row = dict(source)
        if "output" not in row:
            checks.append({"clip": row["clip"], "io_valid": False, "error": row.get("error")})
            continue
        if identity(row["output"]["path"]) != row["output"]:
            raise ValueError("Converted reference bytes changed")
        try:
            checked = validate_arrays(row["output"]["path"])
        except (ValueError, KeyError) as exc:
            checks.append({"clip": row["clip"], "io_valid": False, "error": str(exc)})
            continue
        (references / (row["clip"] + ".npz")).symlink_to(row["output"]["path"])
        # The converter's original quality flags remain untouched and are diagnostic only.
        row.update(admitted_for_training=True, admission_policy=POLICY)
        rows.append(row)
        checks.append({"clip": row["clip"], "split": row["split"], **checked})
    if not rows:
        raise ValueError("No readable Q1 reference arrays were produced")
    manifest = {"status": "Q1_REPRODUCTION_LIBRARY_ASSEMBLED", "admission_policy": POLICY,
        "original_conversion": identity(manifest_path), "motions": rows,
        "reference_protocol": original["reference_protocol"],
        "q1_assets": original["q1_assets"], "local_geometry_gates_used_for_admission": False,
        "legacy_diagnostics_preserved": True, "storage_status": "LOCAL_ONLY"}
    target = references / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    report = {"status": "Q1_REFERENCE_IO_ADMISSION_COMPLETED", "admission_policy": POLICY,
        "input_manifest": identity(target), "clips": checks,
        "admitted_clips": [row["clip"] for row in rows],
        "admitted_duration_s": sum(row["duration_s"] for row in checks if row["io_valid"]),
        "local_geometry_gates_used_for_admission": False,
        "validation_scope": "Readable finite arrays, Q1 dimensions, body order and native 50Hz time only. No support, floor gap, self-collision, IK residual or 2rad/s admission threshold.",
        "source": identity(__file__), "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    (output / "admission.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
