#!/usr/bin/env python3
"""Convert existing SMPL-X or robot motion into the TEST_ONLY Q1 body contract."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egolocate_input import SMPL_BODY_IDS, SMPL_TO_ROBOT_BASIS, resample, save_test_only
from q1_retarget import BodyReference, heading_rotation


HERE = Path(__file__).resolve().parent


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def knee_geometry(positions, names):
    result = {}
    for side in ("left", "right"):
        hip, knee, ankle = [names.index(f"{side}_{part}") for part in ("hip", "knee", "ankle")]
        upper = positions[:, hip] - positions[:, knee]
        lower = positions[:, ankle] - positions[:, knee]
        length = np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1)
        if np.any(length < 1e-8):
            raise ValueError("Degenerate source leg chain")
        flexion = np.pi - np.arccos(np.clip(np.sum(upper * lower, axis=-1) / length, -1, 1))
        result[side] = {"knee_flexion_min_deg": float(np.rad2deg(flexion.min())),
                        "knee_flexion_max_deg": float(np.rad2deg(flexion.max())),
                        "knee_flexion_range_deg": float(np.rad2deg(np.ptp(flexion))),
                        "first_frame_knee_flexion_deg": float(np.rad2deg(flexion[0]))}
    return result


def paired_smplx_standing(source, conversion):
    """Pair a synthetic arms-down SMPL-X stance with the Q1 neutral stance."""
    import smplx
    import torch
    from smplx.lbs import batch_rigid_transform

    if conversion.get("format") != "SMPLX_STAGEII":
        raise ValueError("Synthetic standing calibration requires original SMPL-X and shape parameters")
    model_path = Path(conversion["model"]["path"])
    if identity(model_path)["sha256"] != conversion["model"]["sha256"]:
        raise ValueError("Calibration body model identity changed")
    betas = np.asarray(conversion["betas"], dtype=float)
    model = smplx.SMPLX(str(model_path), num_betas=len(betas), use_pca=False,
                       flat_hand_mean=True, batch_size=1)
    pose = np.zeros((55, 3))
    pose[16, 2], pose[17, 2] = -np.pi / 2, np.pi / 2
    pelvis = source.names.index("pelvis")
    source_heading = heading_rotation(Rotation.from_quat(source.orientations[0, pelvis], scalar_first=True).as_matrix())
    pose[0] = Rotation.from_matrix(source_heading @ SMPL_TO_ROBOT_BASIS).as_rotvec()
    with torch.no_grad():
        beta_tensor = torch.as_tensor(betas, dtype=model.v_template.dtype)
        shaped = model.v_template + torch.einsum("l,vcl->vc", beta_tensor, model.shapedirs)
        rest = torch.einsum("jv,vc->jc", model.J_regressor, shaped)
        rotations = torch.as_tensor(Rotation.from_rotvec(pose).as_matrix()[None], dtype=rest.dtype)
        joints, transforms = batch_rigid_transform(rotations, rest[None], model.parents)
        expected = model(betas=beta_tensor[None], global_orient=torch.as_tensor(pose[None, 0], dtype=rest.dtype),
                         body_pose=torch.as_tensor(pose[None, 1:22].reshape(1, 63), dtype=rest.dtype)).joints[0, :22].numpy()
        fk_error = float(np.max(np.abs(joints[0, :22].numpy() - expected)))
    if fk_error > 1e-5:
        raise ValueError("Synthetic calibration FK disagrees with official SMPL-X forward")
    ids = [SMPL_BODY_IDS[n] for n in source.names]
    positions = joints[0, ids].numpy()
    positions += source.positions[0, pelvis] - positions[pelvis]
    body_rotations = transforms[0, ids, :3, :3].numpy() @ SMPL_TO_ROBOT_BASIS.T
    quats = Rotation.from_matrix(body_rotations).as_quat(scalar_first=True)
    calibrated = BodyReference(source.times, source.names, source.positions, source.orientations,
                               positions, quats, source.names, source.source_kind + "_SYNTHETIC_STANDING_PAIR")
    recipe = {"method": "MODEL_SYNTHESIZED_ARMS_DOWN_STANDING_PAIR_NOT_HARDWARE_CALIBRATION",
              "root_heading": "SOURCE_FIRST_FRAME_WORLD_Z_HEADING_ONLY",
              "translation_origin": "SOURCE_FIRST_FRAME_PELVIS",
              "body_pose": "ZERO_EXCEPT_LEFT_SHOULDER_Z_MINUS_90_RIGHT_SHOULDER_Z_PLUS_90_DEG",
              "fk_vs_official_forward_max_abs_error_m": fk_error,
              "same_shape_as_input": True, "hardware_paired_pose_verified": False}
    return calibrated, recipe


def smplx_stageii(path, model_folder, seconds, names):
    import smplx
    import torch
    from smplx.lbs import batch_rigid_transform

    torch.set_num_threads(2)
    with np.load(path, allow_pickle=False) as z:
        if z["surface_model_type"].item() != "smplx":
            raise ValueError("This adapter requires explicit SMPL-X stageii metadata")
        fps = float(z["mocap_frame_rate"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source FPS")
        count = min(len(z["poses"]), int(np.floor(seconds * fps)) + 1)
        pose = z["poses"][:count].copy()
        trans = z["trans"][:count].copy()
        betas = z["betas"].copy()
        gender = z["gender"].item()
        fields = [("root_orient", 0, 3), ("pose_body", 3, 66), ("pose_jaw", 66, 69),
                  ("pose_eye", 69, 75), ("pose_hand", 75, 165)]
        for key, a, b in fields:
            if not np.allclose(z[key][:count], pose[:, a:b], atol=1e-8, rtol=0):
                raise ValueError(f"Full poses disagree with stageii field {key}")
    if (pose.shape != (count, 165) or trans.shape != (count, 3) or betas.ndim != 1
            or not all(np.isfinite(a).all() for a in (pose, trans, betas))):
        raise ValueError("Invalid SMPL-X numeric arrays")
    model_file = model_folder / "smplx" / f"SMPLX_{gender.upper()}.npz"
    model = smplx.SMPLX(str(model_file), num_betas=len(betas), use_pca=False,
                       flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        beta_tensor = torch.as_tensor(betas, dtype=model.v_template.dtype)
        shaped = model.v_template + torch.einsum("l,vcl->vc", beta_tensor, model.shapedirs)
        rest = torch.einsum("jv,vc->jc", model.J_regressor, shaped)
        local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(count, 55, 3, 3)
        rotations = torch.as_tensor(local, dtype=rest.dtype)
        joints, transforms = batch_rigid_transform(rotations, rest.expand(count, -1, -1), model.parents)
        positions = joints.numpy() + trans[:, None]
        world_rotations = transforms[:, :, :3, :3].numpy()
        # Check the efficient joint path against the library's complete mesh forward pass.
        picks = np.unique(np.linspace(0, count - 1, 3, dtype=int))
        poses_t = torch.as_tensor(pose[picks], dtype=rest.dtype)
        expected = model(betas=beta_tensor.expand(len(picks), -1),
                         expression=model.expression.expand(len(picks), -1), global_orient=poses_t[:, :3],
                         body_pose=poses_t[:, 3:66], jaw_pose=poses_t[:, 66:69],
                         leye_pose=poses_t[:, 69:72], reye_pose=poses_t[:, 72:75],
                         left_hand_pose=poses_t[:, 75:120], right_hand_pose=poses_t[:, 120:165],
                         transl=torch.as_tensor(trans[picks], dtype=rest.dtype)).joints[:, :22].numpy()
        fk_error = float(np.max(np.abs(positions[picks, :22] - expected)))
    if fk_error > 1e-5:
        raise ValueError(f"SMPL-X joint FK differs from complete library forward: {fk_error}")
    ids = [SMPL_BODY_IDS[n] for n in names]
    # AMASS positions are already Z-up. Change body-local semantic axes only.
    orientations = world_rotations[:, ids] @ SMPL_TO_ROBOT_BASIS.T
    selected_positions = positions[:, ids]
    pelvis = names.index("pelvis")
    ankles = [names.index(f"{s}_ankle") for s in ("left", "right")]
    gap = float(np.median(selected_positions[:, pelvis, 2] - selected_positions[:, ankles, 2].mean(axis=1)))
    if gap < 0.2:
        raise ValueError("AMASS Z-up standing/walking convention failed the leg-height check")
    return (np.arange(count) / fps, selected_positions, orientations,
            {"format": "SMPLX_STAGEII", "source_fps": fps, "gender": gender,
             "num_betas": len(betas), "betas": betas.tolist(), "model": identity(model_file),
             "smplx_version": importlib.metadata.version("smplx"),
             "fk_vs_full_library_max_abs_error_m": fk_error,
             "median_pelvis_above_ankles_z_m": gap,
             "axis_conversion": "WORLD_Z_UP_UNCHANGED_BODY_NATIVE_TO_X_FORWARD_Y_LEFT_Z_UP",
             "body_native_to_canonical_basis": SMPL_TO_ROBOT_BASIS.tolist()})


def robot_motion(path, model_path, kind, seconds, names):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError("Expected robot floating root at qpos[:7]")
    joint_ids = np.arange(1, model.njnt)
    if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Expected scalar named robot joints")
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in joint_ids]
    if kind == "g1-csv":
        qpos = np.loadtxt(path / "qpos_29dof.csv", delimiter=",")
        times = np.atleast_1d(np.loadtxt(path / "time.csv", delimiter=","))
        prefix = "robot/"
        if model.nq != 36 or qpos.shape != (len(times), 36):
            raise ValueError("BONES-SEED G1 CSV must match the pinned 29 DOF MuJoCo qpos contract")
        sources = [identity(path / "qpos_29dof.csv"), identity(path / "time.csv")]
        order_evidence = "CSV_MUJOCO_QPOS_ORDER_AS_DOCUMENTED_BY_SOURCE_REPLAY_PY"
    else:
        import joblib
        library = joblib.load(path)
        if not isinstance(library, dict) or len(library) != 1:
            raise ValueError("Expected an existing single-clip PHUMA-derived X2 library file")
        clip = next(iter(library.values()))
        source_names = clip["joint_names_mujoco"]
        if len(set(source_names)) != len(source_names) or set(source_names) != set(model_names):
            raise ValueError("PHUMA-derived X2 joint metadata differs from the selected model")
        fps = float(clip["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid robot library FPS")
        dof = np.asarray(clip["dof"], dtype=float)
        qpos = np.repeat(model.qpos0[None], len(dof), axis=0)
        qpos[:, :3] = clip["root_trans_offset"]
        root_quat = np.asarray(clip["root_rot"], dtype=float)
        if not np.allclose(np.linalg.norm(root_quat, axis=-1), 1, atol=1e-4, rtol=0):
            raise ValueError("Expected normalized PHUMA-derived X2 xyzw root quaternions")
        qpos[:, 3:7] = Rotation.from_quat(root_quat).as_quat(scalar_first=True)
        qpos[:, model.jnt_qposadr[joint_ids]] = dof[:, [source_names.index(n) for n in model_names]]
        times = np.arange(len(dof)) / fps
        prefix, sources = "", [identity(path)]
        order_evidence = "EXPLICIT_JOINT_NAMES_MUJOCO_AND_XYZW_ROOT_FROM_EXISTING_CONVERTER"
    if (len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(qpos).all() or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-4, rtol=0)):
        raise ValueError("Invalid robot motion time/state arrays")
    times = times - times[0]
    keep = times <= seconds + 1e-9
    times, qpos = times[keep], qpos[keep]
    body_map = json.loads((HERE / "retarget_config.json").read_text())["body_map"]
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + body_map[n]) for n in names]
    if any(i < 0 for i in bodies):
        raise ValueError("Mapped source robot body is missing")
    positions, orientations = [], []
    data = mujoco.MjData(model)
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[bodies].copy())
        orientations.append(data.xmat[bodies].reshape(-1, 3, 3).copy())
    limits = model.jnt_range[joint_ids]
    joints = qpos[:, model.jnt_qposadr[joint_ids]]
    return times, np.asarray(positions), np.asarray(orientations), {
        "format": "G1_MUJOCO_QPOS_CSV" if kind == "g1-csv" else "PHUMA_DERIVED_X2_ROBOT_LIBRARY",
        "sources": sources, "model": identity(model_path), "source_joint_names": model_names,
        "joint_order_evidence": order_evidence,
        "source_joint_limit_violations": int(np.count_nonzero((joints < limits[:, 0]) | (joints > limits[:, 1]))),
        "source_max_joint_speed_rad_s": float(np.max(np.abs(np.diff(joints, axis=0) / np.diff(times)[:, None]))),
        "axis_conversion": "ROBOT_WORLD_AND_BODY_X_FORWARD_Y_LEFT_Z_UP_UNCHANGED",
        "raw_human_pose_available": False,
        "prior_retargeting_distortion_included": True}


def load_dataset_reference(path, names):
    details = json.loads(path.with_suffix(".json").read_text())
    if (details["usage"] != "TEST_ONLY" or details["training_allowed"] is not False
            or identity(path)["sha256"] != details["body_reference"]["sha256"]):
        raise ValueError("Dataset reference identity or usage mismatch")
    with np.load(path, allow_pickle=False) as z:
        if z["usage"].item() != "TEST_ONLY" or bool(z["training_allowed"].item()):
            raise ValueError("Dataset body reference must be TEST_ONLY")
    source = BodyReference.load(path, names)
    if len(source.times) != details["resampled_frames"]:
        raise ValueError("Dataset reference frame count mismatch")
    return source, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("smplx-stageii", "g1-csv", "phuma-x2"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Preserve existing outputs")
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    names = list(json.loads((HERE / "retarget_config.json").read_text())["body_map"])
    if args.kind == "smplx-stageii":
        times, positions, rotations, details = smplx_stageii(args.input, args.model, args.seconds, names)
        details["sources"] = [identity(args.input)]
    else:
        times, positions, rotations, details = robot_motion(args.input, args.model, args.kind, args.seconds, names)
    times_out, positions_out, quats = resample(times, positions, rotations, hz=50)
    source = BodyReference(times_out, names, positions_out, quats, positions_out[0], quats[0], names,
                           f"TEST_ONLY_{args.kind.upper()}_FIRST_FRAME_RELATIVE_DIAGNOSTIC")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_test_only(source, args.output)
    root_positions = positions_out[:, names.index("pelvis")]
    details.update({"usage": "TEST_ONLY", "training_allowed": False,
                    "raw_frames_used": len(times), "source_duration_used_s": float(times[-1] - times[0]),
                    "resampled_frames": len(times_out), "reference_hz": 50,
                    "resampled_duration_s": float(times_out[-1]),
                    "calibration": "FIRST_FRAME_RELATIVE_DIAGNOSTIC_NOT_SAME_POSE_VALIDATION",
                    "root_translation_range_world_m": np.ptp(root_positions, axis=0).tolist(),
                    "root_path_length_m": float(np.linalg.norm(np.diff(root_positions, axis=0), axis=-1).sum()),
                    "source_leg_geometry": knee_geometry(positions_out, names),
                    "source_kind": source.source_kind, "converter": identity(__file__),
                    "body_reference": identity(args.output)})
    args.output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2), flush=True)


if __name__ == "__main__":
    main()
