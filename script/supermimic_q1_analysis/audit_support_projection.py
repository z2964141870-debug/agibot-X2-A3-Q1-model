#!/usr/bin/env python3
"""Independently check that a support candidate changes only vertical root motion."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_id(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def verify(root, item):
    path = Path(item["path"]).resolve()
    if not path.is_relative_to(root) or file_id(path) != item:
        raise ValueError("Projection audit input identity/scope mismatch")
    return path


def run(root, receipt_path, output):
    root, output = root.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(root / "data"):
        raise ValueError("Fresh project audit output required")
    receipt_id = file_id(receipt_path)
    report = json.loads(verify(root, receipt_id).read_text())
    if report["status"] != "SUPPORT_HEIGHT_CANDIDATE_BUILT_NOT_ADMITTED":
        raise ValueError("Unexpected candidate receipt")
    inputs = [receipt_id, *report["inputs"], report["candidate_dynamics"]]
    for item in inputs:
        verify(root, item)
    dynamics = json.loads(Path(report["candidate_dynamics"]["path"]).read_text())
    if [row["clip"] for row in dynamics["clips"]] != [row["clip"] for row in report["clips"]]:
        raise ValueError("Candidate diagnostic coverage mismatch")
    checks, max_difference, results = 0, 0., []
    def equal(actual, expected, tolerance=0):
        nonlocal checks, max_difference
        checks += 1
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise ValueError("Projection audit shape mismatch")
        if actual.dtype.kind in "USb" or expected.dtype.kind in "USb":
            if not np.array_equal(actual, expected):
                raise ValueError("Projection audit exact field mismatch")
            return
        difference = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
        max_difference = max(max_difference, difference)
        if not np.isfinite(difference) or difference > tolerance:
            raise ValueError(f"Projection invariant mismatch: {difference} > {tolerance}")
    for row, diagnostic in zip(report["clips"], dynamics["clips"]):
        inputs += [row["parent"], row["candidate"], row["projection_arrays"], diagnostic["raw_arrays"]]
        for item in inputs[-4:]:
            verify(root, item)
        with np.load(row["parent"]["path"], allow_pickle=False) as saved:
            parent = {key: saved[key].copy() for key in saved.files}
        with np.load(row["candidate"]["path"], allow_pickle=False) as saved:
            candidate = {key: saved[key].copy() for key in saved.files}
        with np.load(row["projection_arrays"]["path"], allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        if parent.keys() != candidate.keys() or diagnostic["reference"] != row["candidate"]:
            raise ValueError("Projection schema or diagnostic reference mismatch")
        shift = candidate["qpos"][:, 2] - parent["qpos"][:, 2].astype(float)
        equal(shift, arrays["root_height_shift_m"], 1e-15)
        equal(candidate["qpos"][:, np.arange(29) != 2], parent["qpos"][:, np.arange(29) != 2])
        equal(candidate["qvel"][:, np.arange(28) != 2], parent["qvel"][:, np.arange(28) != 2])
        for key in set(parent) - {"qpos", "qvel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}:
            equal(candidate[key], parent[key])
        dt = np.diff(parent["time_s"])
        equal(dt, np.full(len(dt), .02), 1e-10)
        derivative = np.diff(candidate["qpos"][:, 2]) / dt
        equal(candidate["qvel"][:-1, 2], derivative, 1e-11)
        equal(candidate["qvel"][-1, 2], derivative[-1], 1e-11)
        delta_v = candidate["qvel"][:, 2] - parent["qvel"][:, 2]
        translation = np.zeros((len(shift), 1, 3)); translation[:, 0, 2] = shift
        delta_velocity = np.zeros_like(translation); delta_velocity[:, 0, 2] = delta_v
        equal(candidate["body_pos_w"], parent["body_pos_w"].astype(float) + translation, 2e-6)
        equal(candidate["body_quat_w"], parent["body_quat_w"], 2e-6)
        equal(candidate["body_lin_vel_w"], parent["body_lin_vel_w"].astype(float) + delta_velocity, 2e-6)
        equal(candidate["body_ang_vel_w"], parent["body_ang_vel_w"], 2e-6)
        equal(arrays["candidate_foot_gap_m"], arrays["parent_foot_gap_m"] + shift[:, None], 1e-12)
        equal(arrays["candidate_self_contact_count"], arrays["parent_self_contact_count"])
        lower = arrays["candidate_foot_gap_m"].min(axis=1)
        if (lower.min() < report["settings"]["minimum_lower_foot_z_m"] - 1e-10
                or lower.max() > report["settings"]["maximum_lower_foot_z_m"] + 1e-10
                or np.max(np.abs(shift)) > report["settings"]["maximum_root_correction_m"]
                or row["qualified_for_training_pilot"] is not False):
            raise ValueError("Support band, correction bound or candidate-only status mismatch")
        with np.load(diagnostic["raw_arrays"]["path"], allow_pickle=False) as saved:
            equal(saved["foot_gap_m"], arrays["candidate_foot_gap_m"][saved["frame_index"]], 1e-12)
        results.append({"clip": row["clip"], "frames": len(shift), "only_vertical_root_motion_changed": True,
                        "maximum_absolute_root_correction_m": float(np.max(np.abs(shift))),
                        "candidate": row["candidate"], "self_collision_counts_unchanged": True})
    for item in inputs:
        verify(root, item)
    result = {"status": "SUPPORT_PROJECTION_EXPORT_AUDIT_PASSED", "source": file_id(__file__),
              "candidate_receipt": receipt_id, "checks": checks, "max_absolute_difference": max_difference,
              "clips": results, "inputs_verified_before_after": len(inputs), "new_physics_steps": 0,
              "new_inverse_calls": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY", "physics_tracking_validated": False,
              "limits": "Checks preserved fields, translation/velocity export identities and the measured support band. No integration, force feasibility or policy improvement is established."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "max_absolute_difference": max_difference, "output": file_id(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.receipt, args.output)


if __name__ == "__main__":
    main()
