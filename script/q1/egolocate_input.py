"""TEST_ONLY recorded SMPL body conversion; no model download or robot transport."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from inspect_egolocate import SMPL_PARENTS, global_rotations
from q1_retarget import BodyReference


SMPL_BODY_IDS = {"pelvis": 0, "spine3": 9, "left_hip": 1, "right_hip": 2,
                 "left_knee": 4, "right_knee": 5, "left_ankle": 7, "right_ankle": 8,
                 "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
                 "right_elbow": 19, "left_wrist": 20, "right_wrist": 21}
SMPL_TO_ROBOT_BASIS = np.asarray([[0., 0, 1], [1, 0, 0], [0, 1, 0]])


def repair_local_rotations(matrices, max_error=0.01):
    raw = np.asarray(matrices, dtype=float)
    if raw.ndim != 4 or raw.shape[1:] != (24, 3, 3) or not np.isfinite(raw).all():
        raise ValueError("Expected finite [frames,24,3,3] local SMPL rotations")
    error = np.linalg.norm(raw.transpose(0, 1, 3, 2) @ raw - np.eye(3), axis=(-2, -1))
    determinant = np.linalg.det(raw)
    if np.any(error > max_error) or np.any(determinant <= 0) or np.any(np.abs(determinant - 1) > max_error):
        raise ValueError("Source rotations exceed the explicit SO(3) repair budget")
    repaired = Rotation.from_matrix(raw.reshape(-1, 3, 3)).as_matrix().reshape(raw.shape)
    return repaired, {"repair_budget_orthogonality_frobenius": max_error,
                      "source_max_orthogonality_error": float(error.max()),
                      "source_max_determinant_error": float(np.abs(determinant - 1).max()),
                      "max_matrix_repair_frobenius_change": float(np.linalg.norm(repaired - raw, axis=(-2, -1)).max())}


def resample(times, positions, rotations, hz=50, max_gap=0.2):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Source times must be finite and strictly increasing")
    if not np.isfinite(hz) or hz <= 0 or np.diff(times).max() > max_gap:
        raise ValueError("Invalid output rate or source gap exceeds interpolation budget")
    relative = times - times[0]
    target_times = np.arange(int(np.floor(relative[-1] * hz)) + 1, dtype=float) / hz
    if len(target_times) < 2:
        raise ValueError("Recording too short for the output rate")
    positions = np.asarray(positions, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    if (positions.ndim != 3 or positions.shape[0] != len(times) or positions.shape[-1] != 3
            or rotations.shape != positions.shape[:2] + (3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
        raise ValueError("Inconsistent source position/rotation arrays")
    result_positions = np.empty((len(target_times), positions.shape[1], 3))
    quaternions = np.empty((len(target_times), positions.shape[1], 4))
    for body in range(positions.shape[1]):
        for axis in range(3):
            result_positions[:, body, axis] = np.interp(target_times, relative, positions[:, body, axis])
        quaternions[:, body] = Slerp(relative, Rotation.from_matrix(rotations[:, body]))(target_times).as_quat(scalar_first=True)
    return target_times, result_positions, quaternions


def convert_bundle(path, names, hz=50):
    path = Path(path)
    inspection = json.loads(path.with_name("inspection.json").read_text())
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != inspection["bundle"]["sha256"]:
        raise ValueError("Body bundle SHA-256 differs from extraction manifest")
    with np.load(path, allow_pickle=False) as data:
        if data["usage"].item() != "TEST_ONLY" or data["training_allowed"].item() is not False:
            raise ValueError("This test adapter requires explicit TEST_ONLY data")
        if not np.array_equal(data["parents"], SMPL_PARENTS):
            raise ValueError("Unexpected parent tree")
        times, joints = data["source_time_s"].copy(), data["joints"].astype(float)
        raw = data["pose_matrix"].astype(float).reshape(-1, 24, 3, 3)
        axis_angle = data["axis_angle"].astype(float).reshape(-1, 3)
        translation = data["trans"].astype(float)
        source_sha = data["source_sha256"].item()
    local, repair = repair_local_rotations(raw)
    from_axis_angle = Rotation.from_rotvec(axis_angle)
    angle_error = (Rotation.from_matrix(local.reshape(-1, 3, 3)) * from_axis_angle.inv()).magnitude()
    if angle_error.max() > 0.01:
        raise ValueError("Recorded axis-angle disagrees with local matrices beyond 0.01 rad")
    world = global_rotations(local)
    basis = SMPL_TO_ROBOT_BASIS
    positions = joints @ basis.T
    orientations = basis @ world @ basis.T
    ids = [SMPL_BODY_IDS[name] for name in names]
    target_times, target_positions, target_quaternions = resample(times, positions[:, ids], orientations[:, ids], hz)
    source = BodyReference(target_times, names, target_positions, target_quaternions,
                           target_positions[0], target_quaternions[0], names,
                           "TEST_ONLY_REAL_CLOTH_FGP_SMPL_FIRST_FRAME_RELATIVE_CALIBRATION")
    details = {
        "usage": "TEST_ONLY", "training_allowed": False, "source_raw_sha256": source_sha,
        "source_bundle_sha256": actual_sha, "raw_frames": len(times),
        "source_duration_s": float(times[-1] - times[0]),
        "source_average_hz": float((len(times) - 1) / (times[-1] - times[0])),
        "source_max_gap_s": float(np.diff(times).max()),
        "resampled_frames": len(target_times), "reference_hz": hz,
        "resampled_duration_s": float(target_times[-1]),
        "interpolation": "POSITION_LINEAR_ROTATION_SLERP_NO_EXTRAPOLATION",
        "axis_basis": "SMPL_X_LEFT_Y_UP_Z_FORWARD_TO_ROBOT_X_FORWARD_Y_LEFT_Z_UP",
        "basis_matrix": basis.tolist(), "axes_verified_on_hardware": False,
        "calibration": "FIRST_FRAME_RELATIVE_DIAGNOSTIC_NOT_PAIRED_HUMAN_ROBOT_CALIBRATION",
        "root_translation_range_m": np.ptp(translation, axis=0).tolist(),
        "root_translation_all_zero": bool(np.all(translation == 0)),
        "axis_angle_vs_projected_matrix_max_error_rad": float(angle_error.max()),
        "rotation_repair": repair,
    }
    return source, details


def save_test_only(reference, path):
    reference.save(path)
    with np.load(path, allow_pickle=False) as saved:
        fields = {name: saved[name].copy() for name in saved.files}
    np.savez_compressed(path, **fields, usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
