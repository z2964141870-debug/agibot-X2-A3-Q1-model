#!/usr/bin/env python3
"""Compare saved TEST_ONLY predictions with labels; never feed labels to inference."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from infer_clotho_fgp import identity, read_sequence


JOINT_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
               "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
               "neck", "left_collar", "right_collar", "head", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hand", "right_hand")


def diagnose(directory, archive, body):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["usage"] != "TEST_ONLY" or summary["training_allowed"]:
        raise ValueError("Only TEST_ONLY predictions can be diagnosed here")
    bundle_path = directory / "body_only.npz"
    if identity(bundle_path)["sha256"] != summary["bundle"]["sha256"]:
        raise ValueError("Bundle no longer matches its inference summary")
    with np.load(bundle_path, allow_pickle=False) as source:
        indices = source["frame_index"].copy()
        predicted_rotation = source["pose_matrix"].reshape(-1, 24, 3, 3).copy()
        predicted_root = source["trans"].copy()
    _, _, labels, _, _ = read_sequence(archive, summary["source_sequence"])
    if (len(indices) != len(predicted_rotation) or np.any(np.diff(indices) <= 0)
            or indices[0] < 0 or indices[-1] >= len(labels["pose.pt"])):
        raise ValueError("Prediction and label frame indices do not align")
    label_pose = labels["pose.pt"].numpy()[indices]
    label_rotation = Rotation.from_rotvec(label_pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    label_joint = labels["joint.pt"].numpy()[indices]
    label_relative = label_joint - label_joint[:, 0:1]
    with torch.inference_mode():
        _, predicted_fk = body.forward_kinematics(torch.from_numpy(predicted_rotation).float(), calc_joint=True)
        _, label_fk = body.forward_kinematics(torch.from_numpy(label_rotation).float(), calc_joint=True)
    predicted_fk, label_fk = predicted_fk.numpy(), label_fk.numpy()
    # Remove each pose's own root orientation, without fitting to label joints.
    predicted_local = np.einsum("tji,tik->tjk", predicted_fk, predicted_rotation[:, 0])
    label_local = np.einsum("tji,tik->tjk", label_fk, label_rotation[:, 0])
    rotation_errors = (Rotation.from_matrix(predicted_rotation.reshape(-1, 3, 3))
                       * Rotation.from_matrix(label_rotation.reshape(-1, 3, 3)).inv()).magnitude().reshape(-1, 24)
    position_errors = np.linalg.norm(predicted_fk - label_relative, axis=-1)
    root_error = ((predicted_root - predicted_root[0])
                  - (label_joint[:, 0] - label_joint[0, 0]))
    arrays = (rotation_errors, position_errors, root_error, predicted_local, label_local)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite diagnostic values")
    return {
        "experiment": directory.name, "sequence": summary["source_sequence"],
        "frames": len(indices), "frame_index_first_last": [int(indices[0]), int(indices[-1])],
        "same_template_prediction_vs_label_pose_fk_position_p95_m": float(np.quantile(np.linalg.norm(predicted_fk - label_fk, axis=-1), .95)),
        "label_pose_fk_vs_supplied_label_joint_position_p95_m": float(np.quantile(np.linalg.norm(label_fk - label_relative, axis=-1), .95)),
        "root_orientation_removed_articulation_position_p95_m": float(np.quantile(np.linalg.norm(predicted_local - label_local, axis=-1), .95)),
        "pelvis_rotation_error_p95_rad": float(np.quantile(rotation_errors[:, 0], .95)),
        "relative_root_trajectory_rmse_m": float(np.sqrt(np.mean(np.sum(root_error ** 2, axis=-1)))),
        "relative_root_trajectory_axis_rmse_m_smpl_xyz": np.sqrt(np.mean(root_error ** 2, axis=0)).tolist(),
        "joint_errors": [{"name": name, "position_p95_m": float(np.quantile(position_errors[:, i], .95)),
                          "local_rotation_p95_rad": float(np.quantile(rotation_errors[:, i], .95))}
                         for i, name in enumerate(JOINT_NAMES)],
        "input_summary": identity(directory / "summary.json"), "input_bundle": identity(bundle_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fgp-root", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    sys.path.insert(0, str(args.fgp_root.resolve()))
    from Aplus.tools.smpl_light import SMPLight
    torch.set_num_threads(2)
    body = SMPLight()
    results = [diagnose(path, args.archive, body) for path in args.experiments]
    report = {"usage": "TEST_ONLY", "training_allowed": False,
              "labels_used_for_inference": False, "independent_benchmark": False,
              "existing_training_overlap_possible": True,
              "template_test": "LABEL_POSE_FK_VS_LABEL_JOINT_ROOT_RELATIVE",
              "root_orientation_test": "REMOVE_EACH_POSES_OWN_ROOT_ROTATION_NO_ALIGNMENT_FIT",
              "source_files": [identity(Path(__file__).resolve()),
                               identity(Path(__file__).with_name("infer_clotho_fgp.py")),
                               identity(args.fgp_root / "Aplus/tools/smpl_light.py")],
              "archive": identity(args.archive), "results": results,
              "training_started": False, "hardware_control": False, "baidu_backup_status": "LOCAL_ONLY"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
