"""Decode documented BONES G1 degree/cm CSVs without a frame-rate default."""

import csv
import hashlib
import io
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROTOCOL = "BONES_G1_NAMED_CSV_DEG_CM_EXPLICIT_NATIVE_HZ_V2"
ROOT_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ",
                "root_rotateX", "root_rotateY", "root_rotateZ")
JOINT_NAMES_29 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint",
    "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def positive_rate(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"An explicit numeric {name} is required")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {name}")
    return float(value)


def decode_bones_g1(payload, *, source_hz, expected_frames=None, joint_names=JOINT_NAMES_29,
                    maximum_bytes=64 * 1024 ** 2, maximum_frames=100000):
    source_hz = positive_rate(source_hz, "source_hz")
    joint_names = tuple(joint_names)
    if (len(joint_names) != 29 or len(set(joint_names)) != 29
            or any(not name.endswith("_joint") for name in joint_names)):
        raise ValueError("Expected 29 unique named G1 joints")
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum_bytes:
        raise ValueError("CSV is empty, not bytes, or exceeds the admitted bound")
    if expected_frames is not None and (isinstance(expected_frames, bool)
                                       or not isinstance(expected_frames, int) or expected_frames < 2):
        raise ValueError("Invalid expected frame count")
    text = payload.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    fields = next(reader, None)
    if fields is None:
        raise ValueError("CSV header is missing")
    header = [item.strip() for item in fields]
    joint_columns = tuple(name + "_dof" for name in joint_names)
    expected = ("Frame", *ROOT_COLUMNS, *joint_columns)
    if len(header) != len(expected) or len(set(header)) != len(header) or set(header) != set(expected):
        raise ValueError("CSV columns differ from the named BONES G1 schema")
    order = [header.index(name) for name in expected]
    rows = []
    for line, values in enumerate(reader, 2):
        if len(values) != len(header):
            raise ValueError(f"CSV field count differs at line {line}")
        if len(rows) >= maximum_frames:
            raise ValueError("CSV frame count exceeds the admitted bound")
        try:
            row = [float(values[index]) for index in order]
        except ValueError as exc:
            raise ValueError(f"Nonnumeric CSV value at line {line}") from exc
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"Nonfinite CSV value at line {line}")
        rows.append(row)
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < 2 or (expected_frames is not None and len(values) != expected_frames):
        raise ValueError("CSV and expected metadata frame counts differ, or fewer than two frames")
    frame_ids = values[:, 0]
    if (np.any(frame_ids != np.floor(frame_ids)) or np.any(frame_ids < 0)
            or np.any(np.diff(frame_ids) != 1)):
        raise ValueError("Frame IDs must be nonnegative consecutive integers; missing frames are not retimed")
    time_s = (frame_ids - frame_ids[0]) / source_hz
    qpos = np.empty((len(values), 36), dtype=np.float64)
    qpos[:, :3] = values[:, 1:4] * .01
    # Official viewer uses intrinsic ZYX. For stored X/Y/Z angle columns this
    # equals scipy's extrinsic xyz, Rz(z) @ Ry(y) @ Rx(x).
    qpos[:, 3:7] = Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat()[:, [3, 0, 1, 2]]
    qpos[:, 7:] = np.deg2rad(values[:, 7:])
    return {"protocol": PROTOCOL, "time_s": time_s, "qpos_wxyz": qpos,
            "source_hz": source_hz, "source_frames": len(values),
            "source_frame_ids": frame_ids.astype(np.int64), "joint_names": joint_names,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "source_bytes": len(payload), "source_root_origin_m": qpos[0, :3].tolist(),
            "convention": "x_forward_y_left_z_up_m_rad_wxyz",
            "qualified_for_q1_training": False}


def resample_native_motion(motion, *, target_hz, maximum_frames=1000000):
    target_hz = positive_rate(target_hz, "target_hz")
    if motion["protocol"] != PROTOCOL or motion["convention"] != "x_forward_y_left_z_up_m_rad_wxyz":
        raise ValueError("Unexpected decoded motion contract")
    times = np.asarray(motion["time_s"], dtype=np.float64)
    qpos = np.asarray(motion["qpos_wxyz"], dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or np.any(np.diff(times) <= 0) or not np.isfinite(times).all()
            or qpos.shape != (len(times), 36) or not np.isfinite(qpos).all()
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError("Invalid native-time motion arrays")
    output_count = float(times[-1]) * target_hz
    if not math.isfinite(output_count) or output_count + 1 > maximum_frames:
        raise ValueError("Resampling exceeds admitted output frame count")
    output_time = np.arange(int(math.floor(output_count + 1e-9)) + 1) / target_hz
    if len(output_time) < 2:
        raise ValueError("Source duration is shorter than one target control period")
    # Bound tiny floating-point overshoots without adding or stretching an endpoint.
    output_time = output_time[output_time <= times[-1] + 1e-12]
    query = np.minimum(output_time, times[-1])
    result = np.empty((len(output_time), 36), dtype=np.float64)
    for column in (*range(3), *range(7, 36)):
        result[:, column] = np.interp(query, times, qpos[:, column])
    result[:, 3:7] = Slerp(times, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(query).as_quat()[:, [3, 0, 1, 2]]
    return {"time_s": output_time, "qpos_wxyz": result, "joint_names": motion["joint_names"],
            "source_hz": motion["source_hz"], "target_hz": target_hz,
            "source_duration_s": float(times[-1]), "target_duration_s": float(output_time[-1]),
            "endpoint_truncation_s": float(times[-1] - output_time[-1]),
            "time_scale": 1., "root_origin_preserved": True,
            "qualified_for_q1_training": False}
