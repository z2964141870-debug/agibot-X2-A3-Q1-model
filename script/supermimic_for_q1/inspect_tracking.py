#!/usr/bin/env python3
"""Describe root displacement and per-joint tracking on preserved evaluation traces."""

import argparse
import json
from pathlib import Path

import numpy as np

from supermimic.q1_assets import Q1Assets, identity


def describe(row, names):
    for key in ("reference", "trace"):
        if identity(row[key]["path"])["sha256"] != row[key]["sha256"]:
            raise ValueError("Preserved reference or evaluation trace changed")
    with np.load(row["reference"]["path"], allow_pickle=False) as data:
        reference = data["qpos"].astype(np.float64)
    with np.load(row["trace"]["path"], allow_pickle=False) as data:
        state = data["state"].astype(np.float64)
        action = data["actions"].astype(np.float64)
    count = len(state)
    if count != row["executed_steps"] or len(reference) != row["expected_steps"]:
        raise ValueError("Saved trace and full-clip denominators differ")
    target = reference[:count]
    root_error = np.linalg.norm(state[:, :3] - target[:, :3], axis=-1)
    actual_quat, target_quat = state[:, 3:7], target[:, 3:7]
    cosine = np.abs(np.sum(actual_quat * target_quat, axis=-1))
    cosine /= np.linalg.norm(actual_quat, axis=-1) * np.linalg.norm(target_quat, axis=-1)
    orientation = 2 * np.arccos(np.clip(cosine, 0., 1.))
    error = state[:, 7:29] - target[:, 7:29]
    joint_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    groups = {"legs": list(range(12)), "waist": [12, 13], "arms": list(range(14, 22))}
    result = {"clip": row["clip"], "split": row["split"], "executed_steps": count,
              "expected_steps": len(reference), "failed": row["failed"],
              "root_position_rmse_norm_m": float(np.sqrt(np.mean(root_error ** 2))),
              "root_position_p95_norm_m": float(np.percentile(root_error, 95)),
              "root_position_last_error_m": float(root_error[-1]),
              "root_orientation_rmse_rad": float(np.sqrt(np.mean(orientation ** 2))),
              "reference_full_horizontal_displacement_m": float(np.linalg.norm(reference[-1, :2] - reference[0, :2])),
              "reference_executed_horizontal_displacement_m": float(np.linalg.norm(target[-1, :2] - reference[0, :2])),
              "actual_horizontal_displacement_m": float(np.linalg.norm(state[-1, :2] - reference[0, :2])),
              "group_joint_rmse_rad": {group: float(np.sqrt(np.mean(error[:, ids] ** 2))) for group, ids in groups.items()},
              "action_abs_max": float(np.abs(action).max()),
              "action_fraction_outside_registered_clip": float(np.mean(np.abs(action) > 20.)),
              "joints": [{"name": name, "rmse_rad": float(joint_rmse[index]),
                          "reference_executed_range_rad": float(np.ptp(target[:, 7 + index])),
                          "actual_range_rad": float(np.ptp(state[:, 7 + index]))} for index, name in enumerate(names)]}
    if abs(float(np.sqrt(np.mean(error ** 2))) - row["executed_joint_rmse_rad"]) > 2e-5:
        raise ValueError("Independent joint-error arithmetic differs from the evaluation receipt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evaluation = json.loads(args.evaluation.read_text())
    assets = Q1Assets(args.root)
    expected_names = [f"{side}_{joint}_joint" for side in ("left", "right")
                      for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    expected_names += ["waist_roll_joint", "waist_yaw_joint"]
    expected_names += [f"{side}_{joint}_joint" for side in ("left", "right")
                       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    if assets.names != expected_names:
        raise ValueError("Registered Q1 limb group ordering changed")
    rows = [describe(row, assets.names) for row in evaluation["rows"]]
    report = {"status": "SAVED_TRACKING_DIAGNOSTIC_COMPLETED", "rows": rows,
              "evaluation": identity(args.evaluation), "source": identity(__file__),
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY",
              "limits": "Post-hoc descriptive diagnostic; original usability gates unchanged. Errors use executed steps only, with full denominators and failure flags retained. Post-control state is compared with that control step's reference."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != "joints"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
