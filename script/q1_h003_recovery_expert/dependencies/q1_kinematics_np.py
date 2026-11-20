"""Pure NumPy SO(3) geometry and Q1 22-DOF forward kinematics (ZERO PyTorch / SciPy / MuJoCo)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

Q1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

SOLE_LOCAL_OFFSET = np.array([0.020, 0.0, -0.022], dtype=np.float32)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) (..., 4) in [w, x, y, z] order to rotation matrix (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    )
    return out.reshape(q.shape[:-1] + (3, 3)).astype(np.float32)


def quat_xyzw_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert ROS2 sensor_msgs/Imu quaternion [x, y, z, w] to rotation matrix (..., 3, 3)."""
    q_xyzw = np.asarray(q_xyzw, dtype=np.float32)
    q_wxyz = np.stack((q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]), axis=-1)
    return quat_wxyz_to_mat(q_wxyz)


def mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (..., 3, 3) to canonical quaternion (..., 4) [w, x, y, z] with w >= 0."""
    m = np.asarray(m, dtype=np.float32)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = np.sqrt(np.clip(1.0 + m00 + m11 + m22, 1e-10, None)) * 0.5
    qx = np.sqrt(np.clip(1.0 + m00 - m11 - m22, 1e-10, None)) * 0.5
    qy = np.sqrt(np.clip(1.0 - m00 + m11 - m22, 1e-10, None)) * 0.5
    qz = np.sqrt(np.clip(1.0 - m00 - m11 + m22, 1e-10, None)) * 0.5
    cand = np.stack(
        (
            np.stack((qw, (m21 - m12) / (4.0 * qw), (m02 - m20) / (4.0 * qw), (m10 - m01) / (4.0 * qw)), axis=-1),
            np.stack(((m21 - m12) / (4.0 * qx), qx, (m01 + m10) / (4.0 * qx), (m02 + m20) / (4.0 * qx)), axis=-1),
            np.stack(((m02 - m20) / (4.0 * qy), (m01 + m10) / (4.0 * qy), qy, (m12 + m21) / (4.0 * qy)), axis=-1),
            np.stack(((m10 - m01) / (4.0 * qz), (m02 + m20) / (4.0 * qz), (m12 + m21) / (4.0 * qz), qz), axis=-1),
        ),
        axis=-2,
    )
    idx = np.argmax(np.stack((qw, qx, qy, qz), axis=-1), axis=-1)
    q = np.take_along_axis(cand, idx[..., None, None], axis=-2).squeeze(-2)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    sign = np.where(q[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return (q * sign).astype(np.float32)


def so3_log_np(rotation: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) -> R^3 matching unisupermimic.rl.algos.supermimic.geometry.so3_log."""
    q = mat_to_quat_wxyz(rotation)
    vec = q[..., 1:]
    sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)
    half = np.arctan2(sin_half, q[..., :1])
    scale = np.where(
        sin_half < 1e-6,
        2.0 / np.clip(q[..., :1], 1e-6, None),
        2.0 * half / (sin_half + 1e-12),
    )
    return (vec * scale).astype(np.float32)


def axis_angle_to_mat_np(axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Convert unit axis (..., 3) and angle (...) to rotation matrix (..., 3, 3)."""
    axis = np.asarray(axis, dtype=np.float32)
    ang = np.asarray(angle, dtype=np.float32)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(ang), np.sin(ang)
    comp = 1.0 - c
    out = np.stack(
        (
            c + x * x * comp,
            x * y * comp - z * s,
            x * z * comp + y * s,
            y * x * comp + z * s,
            c + y * y * comp,
            y * z * comp - x * s,
            z * x * comp - y * s,
            z * y * comp + x * s,
            c + z * z * comp,
        ),
        axis=-1,
    )
    return out.reshape(ang.shape + (3, 3)).astype(np.float32)


def yaw_mat_np(rotation: np.ndarray) -> np.ndarray:
    """Extract pure Z-yaw rotation matrix matching unisupermimic geometry.yaw_mat."""
    r = np.asarray(rotation, dtype=np.float32)
    fx, fy = r[..., 0, 0], r[..., 1, 0]
    norm = np.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / norm, fy / norm
    z = np.zeros_like(c)
    o = np.ones_like(c)
    return np.stack((c, -s, z, s, c, z, z, z, o), axis=-1).reshape(r.shape[:-2] + (3, 3)).astype(np.float32)


def yaw_angle_from_mat(rotation: np.ndarray) -> float:
    """Extract scalar yaw angle (rad) from a single (3, 3) rotation matrix."""
    return float(np.arctan2(float(rotation[1, 0]), float(rotation[0, 0])))


def rotz_np(yaw_rad: float) -> np.ndarray:
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def mat_to_6d_np(rotation: np.ndarray) -> np.ndarray:
    """Row-major first two columns [M00, M01, M10, M11, M20, M21] matching geometry.mat_to_6d!"""
    r = np.asarray(rotation, dtype=np.float32)
    return r[..., :, :2].reshape(r.shape[:-2] + (6,)).astype(np.float32)


def parse_mjcf_kinematics(mjcf_path: str | Path) -> dict[str, Any]:
    """Parse supermimic.xml directly with xml.etree (zero MuJoCo dependency!) into exact FK tables."""
    tree = ET.parse(Path(mjcf_path))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {mjcf_path}")

    jname_to_idx = {n: i for i, n in enumerate(Q1_JOINT_NAMES)}
    bodies: list[str] = []
    parent: list[int] = []
    joint_idx: list[int] = []
    axes: list[list[float]] = []
    offsets_p: list[list[float]] = []
    offsets_q: list[list[float]] = []
    q_lo = [0.0] * len(Q1_JOINT_NAMES)
    q_hi = [0.0] * len(Q1_JOINT_NAMES)

    def _visit_body(elem: ET.Element, parent_local_idx: int) -> None:
        bname = elem.attrib["name"]
        my_idx = len(bodies)
        bodies.append(bname)

        pos = [float(v) for v in elem.attrib.get("pos", "0 0 0").split()]
        quat = [float(v) for v in elem.attrib.get("quat", "1 0 0 0").split()]
        # Normalize body_quat the same way MuJoCo compiler does on load
        q_arr = np.asarray(quat, dtype=np.float64)
        q_arr = q_arr / np.linalg.norm(q_arr)

        free_j = elem.find("freejoint")
        j_elems = elem.findall("joint")
        if free_j is not None:
            parent.append(-1)
            joint_idx.append(-1)
            axes.append([0.0, 0.0, 0.0])
        elif j_elems:
            j_el = j_elems[0]
            jname = j_el.attrib["name"]
            j_i = jname_to_idx[jname]
            parent.append(parent_local_idx)
            joint_idx.append(j_i)
            ax = [float(v) for v in j_el.attrib.get("axis", "0 0 1").split()]
            ax_arr = np.asarray(ax, dtype=np.float64)
            ax_arr = ax_arr / np.linalg.norm(ax_arr)
            axes.append(ax_arr.tolist())
            if "range" in j_el.attrib:
                lo_s, hi_s = j_el.attrib["range"].split()
                q_lo[j_i] = float(lo_s)
                q_hi[j_i] = float(hi_s)
        else:
            parent.append(parent_local_idx)
            joint_idx.append(-1)
            axes.append([0.0, 0.0, 0.0])

        offsets_p.append(pos)
        offsets_q.append(q_arr.tolist())

        for child in elem.findall("body"):
            _visit_body(child, my_idx)

    for top_body in worldbody.findall("body"):
        _visit_body(top_body, -1)

    nom_h = 0.42132428111574716
    custom = root.find("custom")
    if custom is not None:
        for num in custom.findall("numeric"):
            if num.attrib.get("name") == "nominal_root_height":
                nom_h = float(num.attrib["data"].split()[0])

    return {
        "body_names": bodies,
        "parent": parent,
        "joint_idx": joint_idx,
        "axes": axes,
        "off_p": offsets_p,
        "off_quat_wxyz": offsets_q,
        "tracked_bodies": list(TRACKED_BODIES),
        "tracked_idx": [bodies.index(n) for n in TRACKED_BODIES],
        "q_lo": q_lo,
        "q_hi": q_hi,
        "nominal_root_height": nom_h,
    }


class Q1KinematicsNP:
    """Fast single-robot & batched pure-NumPy forward kinematics matching Torch Q1Kinematics."""

    def __init__(self, spec_or_mjcf: str | Path | dict[str, Any]):
        if isinstance(spec_or_mjcf, dict):
            spec = spec_or_mjcf
        else:
            p = Path(spec_or_mjcf)
            if p.suffix.lower() == ".json":
                spec = json.loads(p.read_text(encoding="utf-8"))
            else:
                spec = parse_mjcf_kinematics(p)

        self.body_names: list[str] = list(spec["body_names"])
        self.parent: list[int] = [int(x) for x in spec["parent"]]
        self.joint_idx: list[int] = [int(x) for x in spec["joint_idx"]]
        self.axes = np.array(spec["axes"], dtype=np.float32)
        self.off_p = np.array(spec["off_p"], dtype=np.float32)
        self.off_R = quat_wxyz_to_mat(np.array(spec["off_quat_wxyz"], dtype=np.float32))
        self.tracked_idx: list[int] = [int(x) for x in spec["tracked_idx"]]
        self.num_bodies = len(self.body_names)
        self.q_lo = np.array(spec["q_lo"], dtype=np.float32)
        self.q_hi = np.array(spec["q_hi"], dtype=np.float32)
        self.nominal_root_height = float(spec["nominal_root_height"])

        self.left_ankle_tracked_idx = TRACKED_BODIES.index("left_ankle_roll_link")
        self.right_ankle_tracked_idx = TRACKED_BODIES.index("right_ankle_roll_link")
        self.torso_body_idx = self.body_names.index("torso_link")
        self.waist_roll_body_idx = self.body_names.index("waist_roll_link")

    def forward_single(
        self,
        p_root: np.ndarray,
        R_root: np.ndarray,
        v_root: np.ndarray,
        w_root_world: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        *,
        all_bodies: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Single-step unbatched FK (shapes: (3,), (3,3), (3,), (3,), (22,), (22,))."""
        p_list = [np.asarray(p_root, dtype=np.float32)]
        R_list = [np.asarray(R_root, dtype=np.float32)]
        v_list = [np.asarray(v_root, dtype=np.float32)]
        w_list = [np.asarray(w_root_world, dtype=np.float32)]
        q_f = np.asarray(q, dtype=np.float32)
        qd_f = np.asarray(qd, dtype=np.float32)

        for idx in range(1, self.num_bodies):
            p_idx = self.parent[idx]
            j_idx = self.joint_idx[idx]
            R_par = R_list[p_idx]
            dp = R_par @ self.off_p[idx]
            pos = p_list[p_idx] + dp
            off_rot = R_par @ self.off_R[idx]
            if j_idx < 0:
                rot = off_rot
                ang_vel = w_list[p_idx]
            else:
                rot = off_rot @ axis_angle_to_mat_np(self.axes[idx], q_f[j_idx])
                axis_world = off_rot @ self.axes[idx]
                ang_vel = w_list[p_idx] + axis_world * qd_f[j_idx]
            lin_vel = v_list[p_idx] + np.cross(w_list[p_idx], dp)
            p_list.append(pos.astype(np.float32))
            R_list.append(rot.astype(np.float32))
            v_list.append(lin_vel.astype(np.float32))
            w_list.append(ang_vel.astype(np.float32))

        sel = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            np.stack([p_list[i] for i in sel], axis=0),
            np.stack([R_list[i] for i in sel], axis=0),
            np.stack([v_list[i] for i in sel], axis=0),
            np.stack([w_list[i] for i in sel], axis=0),
        )

    def torso_imu_to_pelvis(
        self,
        R_torso_world: np.ndarray,
        w_torso_body: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Torso-mounted IMU (R_torso_world, w_torso_body) to Pelvis (R_pelvis_world, w_pelvis_body).

        Kinematic chain: pelvis -> waist_roll_link (q[12]) -> torso_link (q[13]).
        R_torso = R_pelvis @ R_pelvis_to_torso  =>  R_pelvis = R_torso @ R_pelvis_to_torso^T
        w_torso_world = w_pelvis_world + R_pelvis @ w_rel_pelvis.
        """
        wr_b = self.waist_roll_body_idx
        to_b = self.torso_body_idx
        q_wr = float(q[12])
        q_wy = float(q[13])
        qd_wr = float(qd[12])
        qd_wy = float(qd[13])

        R_0_wr_off = self.off_R[wr_b]
        ax_wr_pelvis = R_0_wr_off @ self.axes[wr_b]
        R_0_wr = R_0_wr_off @ axis_angle_to_mat_np(self.axes[wr_b], q_wr)

        R_wr_to_off = R_0_wr @ self.off_R[to_b]
        ax_wy_pelvis = R_wr_to_off @ self.axes[to_b]
        R_pelvis_to_torso = R_wr_to_off @ axis_angle_to_mat_np(self.axes[to_b], q_wy)

        R_pelvis_world = (R_torso_world @ R_pelvis_to_torso.T).astype(np.float32)
        w_rel_pelvis = ax_wr_pelvis * qd_wr + ax_wy_pelvis * qd_wy
        w_pelvis_body = (R_pelvis_to_torso @ np.asarray(w_torso_body, dtype=np.float32) - w_rel_pelvis).astype(np.float32)
        return R_pelvis_world, w_pelvis_body
