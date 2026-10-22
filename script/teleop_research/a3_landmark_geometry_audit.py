#!/usr/bin/env python3
"""Audit the saved T10 target geometry and MuJoCo FK, without re-solving IK."""

import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation


PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
IDS = [0, 9, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
BASIS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source_from_paired(artifact, data_zip, names, pose_name, start=30, stop=60):
    from q1_retarget import BodyReference

    with np.load(artifact, allow_pickle=False) as saved:
        pose = saved[pose_name][start:stop].copy()
        body = saved["label_rootrelative_joints" if pose_name == "label_pose" else
                     "predicted_rootrelative_joints"][start:stop].copy()
        label_body = saved["label_rootrelative_joints"][start:stop].copy()
        rest = saved["rest_bone_vectors"].copy()
    with zipfile.ZipFile(data_zip) as package:
        payload = package.read("train_released/G_0113_zj_train/183926_zj_walk/joint.pt")
        joints = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True).numpy()[start:stop]
    root = joints[:, :1]
    neutral = np.zeros((24, 3))
    for joint in range(1, 24):
        neutral[joint] = neutral[PARENTS[joint]] + rest[joint - 1]
    calibration = (neutral[IDS] + root[0, 0]) @ BASIS.T
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    world = np.empty_like(local)
    for joint, parent in enumerate(PARENTS):
        world[:, joint] = local[:, joint] if parent < 0 else world[:, parent] @ local[:, joint]
    positions = (body + root)[:, IDS] @ BASIS.T
    orientations = BASIS @ world[:, IDS] @ BASIS.T
    quaternions = Rotation.from_matrix(orientations.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 14, 4)
    source = BodyReference(np.arange(len(pose)) / 30, list(names), positions, quaternions,
                           calibration, np.tile([1., 0, 0, 0], (14, 1)), list(names),
                           "OFFLINE_LABEL_ROOT_CONTACT_ASSISTED_GEOMETRY_DIAGNOSIS")
    return source, label_body + root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--t10", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "data/references/a3_smpl_q1_helpers_20261012"))
    import mujoco
    from script.a3.smpl_hull_floor import HullA3Retargeter
    from script.a3.mocap_adapter import BODY_MAP, MJCF

    previous_path = Path(args.t10) / "receipt.json"
    previous = json.loads(previous_path.read_text())
    for path, sha in previous["asset_sha256"].items():
        if identity(path)["sha256"] != sha:
            raise ValueError("T10 robot asset changed")
    paired = previous["paired_artifact"]
    if identity(paired["path"])["sha256"] != paired["sha256"]:
        raise ValueError("T09 paired source changed")
    retargeter = HullA3Retargeter()
    model = retargeter.model
    names = retargeter.names
    body_ids = np.array(retargeter.bodies)
    data = mujoco.MjData(model)
    id_for = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    source_files = {Path(inspect.getfile(cls)).resolve() for cls in type(retargeter).__mro__ if cls is not object}
    rows = []
    for arm, pose_key in (("label_pose", "label_pose"), ("onnx_pose", "pred_pose")):
        trace_path = Path(args.t10) / arm / "kinematic_trace.npz"
        with np.load(trace_path, allow_pickle=False) as saved:
            qpos = saved["qpos"].copy()
            targets = saved["target_position_m"].copy()
            saved_actual = saved["actual_position_m"].copy()
        source, contacts = source_from_paired(paired["path"], args.data, names, pose_key)
        retargeter.set_human_source(contacts)
        recomputed, rotations = retargeter.targets(source, previous["scale"])
        reconstruction_error = float(np.max(np.abs(recomputed - targets)))
        if reconstruction_error > 1e-8:
            raise ValueError("Reconstructed T10 targets differ")
        actual, actual_rotation, all_position = [], [], []
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            actual.append(data.xpos[body_ids].copy())
            actual_rotation.append(data.xmat[body_ids].reshape(-1, 3, 3).copy())
            all_position.append(data.xpos.copy())
        actual, actual_rotation, all_position = map(np.asarray, (actual, actual_rotation, all_position))
        fk_error = float(np.max(np.abs(actual - saved_actual)))
        if fk_error > 1e-8:
            raise ValueError("Saved IK position disagrees with independent MuJoCo FK")
        error = np.linalg.norm(actual - targets, axis=-1)
        per_body = {name: dict(mean_m=float(error[:, i].mean()), p95_m=float(np.quantile(error[:, i], .95)),
                               max_m=float(error[:, i].max())) for i, name in enumerate(names)}
        shoulders = {}
        for side in ("left", "right"):
            torso = names.index("spine3")
            shoulder = names.index(side + "_shoulder")
            elbow = names.index(side + "_elbow")
            roll = id_for(side + "_shoulder_roll_Link")
            yaw = id_for(side + "_shoulder_yaw_Link")
            elbow_id = id_for(side + "_elbow_Link")
            torso_id = body_ids[torso]
            retargeter.configuration.update(retargeter.data.qpos.copy())
            rest_data = retargeter.configuration.data
            roll_local = rest_data.xmat[torso_id].reshape(3, 3).T @ (rest_data.xpos[roll] - rest_data.xpos[torso_id])
            pivot = targets[:, torso] + np.einsum("tij,j->ti", rotations[:, torso], roll_local)
            actual_pivot = actual[:, torso] + np.einsum("tij,j->ti", actual_rotation[:, torso], roll_local)
            fixed_pivot_error = float(np.max(np.linalg.norm(actual_pivot - all_position[:, roll], axis=-1)))
            if fixed_pivot_error > 1e-8:
                raise ValueError("Shoulder roll pivot is not rigid relative to torso")
            b, c = model.body_pos[yaw], model.body_pos[elbow_id]
            if (not np.allclose(model.body_quat[yaw], [1, 0, 0, 0]) or
                    not np.allclose(model.body_quat[elbow_id], [1, 0, 0, 0])):
                raise ValueError("Shoulder geometric formula requires the pinned body orientations")
            radius = np.linalg.norm(b[:2])
            axis = rotations[:, shoulder, :, 2]
            delta = targets[:, shoulder] - pivot
            parallel = np.einsum("ti,ti->t", delta, axis)
            perpendicular = np.linalg.norm(delta - parallel[:, None] * axis, axis=-1)
            position_at_exact_orientation = np.hypot(parallel - b[2], perpendicular - radius)
            # Rotation about the yaw joint changes only the transverse dot product.
            minimum_reach = np.sqrt((b[2] + c[2])**2 + (np.linalg.norm(b[:2]) - np.linalg.norm(c[:2]))**2)
            maximum_reach = np.sqrt((b[2] + c[2])**2 + (np.linalg.norm(b[:2]) + np.linalg.norm(c[:2]))**2)
            target_reach = np.linalg.norm(targets[:, elbow] - pivot, axis=-1)
            elbow_reach_mismatch = np.maximum(minimum_reach - target_reach, target_reach - maximum_reach).clip(0)
            actual_axis = actual_rotation[:, shoulder, :, 2]
            actual_delta = actual[:, shoulder] - all_position[:, roll]
            actual_parallel = np.einsum("ti,ti->t", actual_delta, actual_axis)
            actual_perp = np.linalg.norm(actual_delta - actual_parallel[:, None] * actual_axis, axis=-1)
            real_circle_error = float(np.max(np.hypot(actual_parallel - b[2], actual_perp - radius)))
            if real_circle_error > 1e-8:
                raise ValueError("Analytic shoulder geometry disagrees with actual FK")
            shoulders[side] = dict(mapped_human_landmark=side + "_shoulder", robot_body=BODY_MAP[side + "_shoulder"],
                robot_yaw_offset_from_roll_m=b.tolist(), rigid_roll_pivot_fk_error_m=fixed_pivot_error,
                exact_geometry_fk_error_m=real_circle_error,
                target_position_at_exact_target_orientation_min_error_mean_m=float(position_at_exact_orientation.mean()),
                target_position_at_exact_target_orientation_min_error_p95_m=float(np.quantile(position_at_exact_orientation, .95)),
                roll_pivot_to_elbow_robot_reach_interval_m=[float(minimum_reach), float(maximum_reach)],
                target_roll_pivot_to_elbow_range_m=[float(target_reach.min()), float(target_reach.max())],
                elbow_target_reach_mismatch_p95_m=float(np.quantile(elbow_reach_mismatch, .95)))
        rows.append(dict(arm=arm, saved_trace=identity(trace_path), target_reconstruction_maxabs_m=reconstruction_error,
            independent_fk_maxabs_m=fk_error, ik_position_p95_m=float(np.quantile(error, .95)),
            per_body=per_body, shoulders=shoulders))
    report = dict(experiment="T12_GEOMETRY_AUDIT", source_frames=[30, 60], prior_receipt=identity(previous_path),
        driver=identity(__file__), source_files=[identity(p) for p in sorted(source_files)], mjcf=identity(MJCF), rows=rows,
        finding="Human shoulder is anchored to robot shoulder_yaw origin 147.5mm downstream of shoulder roll pivot",
        formula_scope="Necessary geometry at prescribed torso frame; exact-orientation bound is conditional, not position-only global infeasibility",
        policy_loaded=False, dynamics_step_executed=False, ik_resolved=False, training_started=False,
        independent_test=False, development_only=True)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
