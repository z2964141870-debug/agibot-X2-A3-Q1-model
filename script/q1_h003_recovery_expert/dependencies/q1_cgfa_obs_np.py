"""Pure NumPy CGFA (`cgfa_clipped_norm`), Yaw Aligner, `obs_80` Builder, and 4-Modality Streamers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from q1_kinematics_np import (
    SOLE_LOCAL_OFFSET,
    TRACKED_BODIES,
    Q1KinematicsNP,
    mat_to_6d_np,
    rotz_np,
    so3_log_np,
    yaw_angle_from_mat,
    yaw_mat_np,
)

Q1_NOMINAL_PELVIS_Z = 0.4213
Q1_NOMINAL_TORSO_Z = 0.5513
Q1_NOMINAL_ARM_LEN = 0.275

IDX = {name: i for i, name in enumerate(TRACKED_BODIES)}

BONE_PAIRS: tuple[tuple[int, int], ...] = (
    (IDX["pelvis"], IDX["torso_link"]),
    (IDX["left_hip_roll_link"], IDX["left_knee_link"]),
    (IDX["left_knee_link"], IDX["left_ankle_roll_link"]),
    (IDX["right_hip_roll_link"], IDX["right_knee_link"]),
    (IDX["right_knee_link"], IDX["right_ankle_roll_link"]),
    (IDX["left_shoulder_roll_link"], IDX["left_elbow_link"]),
    (IDX["left_elbow_link"], IDX["left_wrist_yaw_link"]),
    (IDX["right_shoulder_roll_link"], IDX["right_elbow_link"]),
    (IDX["right_elbow_link"], IDX["right_wrist_yaw_link"]),
)


class YawAlignerNP:
    """Align real-robot IMU world yaw to reference trajectory initial yaw on reset() (`Card 01 §3.2`)."""

    def __init__(self) -> None:
        self.R_align = np.eye(3, dtype=np.float32)
        self.initialized = False

    def reset(self, R_imu_raw: np.ndarray, R_ref_0: np.ndarray | None = None) -> None:
        yaw_imu0 = yaw_angle_from_mat(R_imu_raw)
        yaw_ref0 = 0.0 if R_ref_0 is None else yaw_angle_from_mat(R_ref_0)
        self.R_align = rotz_np(yaw_ref0 - yaw_imu0)
        self.initialized = True

    def align(self, R_imu_raw: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.reset(R_imu_raw)
        return (self.R_align @ np.asarray(R_imu_raw, dtype=np.float32)).astype(np.float32)


class ContactGatedFootAnchorNP:
    """Bit-exact pure-NumPy CGFA estimator defaulting to non-contracting `cgfa_clipped_norm` (`g=0.96, R=0.25m`)."""

    def __init__(
        self,
        kin: Q1KinematicsNP,
        dt: float = 0.02,
        *,
        height_temp: float = 50.0,
        angvel_weight: float = 2.5,
        flat_weight: float = 6.0,
        loco_speed_thresh: float = 0.15,
        loco_speed_temp: float = 0.04,
        yaw_rate_equiv_m: float = 0.20,
        gamma_stat: float = 0.998,
        gamma_loco: float = 0.96,
        bound_m: float = 0.25,
    ):
        self.kin = kin
        self.dt = float(dt)
        self.height_temp = float(height_temp)
        self.angvel_weight = float(angvel_weight)
        self.flat_weight = float(flat_weight)
        self.loco_speed_thresh = float(loco_speed_thresh)
        self.loco_speed_temp = float(loco_speed_temp)
        self.yaw_rate_equiv_m = float(yaw_rate_equiv_m)
        self.gamma_stat = float(gamma_stat)
        self.gamma_loco = float(gamma_loco)
        self.bound_m = float(bound_m)

        self.e_world_xy = np.zeros(2, dtype=np.float32)
        self.prev_r_sole = np.zeros((2, 3), dtype=np.float32)
        self.prev_p_ref_xy = np.zeros(2, dtype=np.float32)
        self.prev_R_ref: np.ndarray | None = None
        self.v_ref_ema = 0.0
        self.initialized = False

    def compute_sole_fk(
        self,
        R_imu: np.ndarray,
        w_body: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute left/right sole positions (2,3), rotations (2,3,3), world angvels (2,3), and lateral foot width (cm).

        IMPORTANT: `w_body` is IMU/pelvis local gyroscope angular velocity (as reported by ROS2 `sensor_msgs/Imu`).
        FK requires world-frame root angular velocity `w_world = R_imu @ w_body`.
        """
        zeros3 = np.zeros(3, dtype=np.float32)
        w_world = (R_imu @ np.asarray(w_body, dtype=np.float32)).astype(np.float32)
        body_p, body_R, _, body_w = self.kin.forward_single(zeros3, R_imu, zeros3, w_world, q, qd)
        la, ra = self.kin.left_ankle_tracked_idx, self.kin.right_ankle_tracked_idx
        r_ankles = body_p[(la, ra), :]
        R_feet = body_R[(la, ra), :, :]
        w_feet = body_w[(la, ra), :]
        r_sole = r_ankles + (R_feet @ SOLE_LOCAL_OFFSET[:, None]).squeeze(-1)

        # Heading-frame lateral foot separation (cm) for real-time narrow-stance monitoring
        heading_t = yaw_mat_np(R_imu).T
        sole_h = (heading_t @ r_sole.T).T
        foot_width_cm = float(abs(float(sole_h[0, 1] - sole_h[1, 1])) * 100.0)
        return r_sole.astype(np.float32), R_feet.astype(np.float32), w_feet.astype(np.float32), foot_width_cm

    def reset(
        self,
        R_imu: np.ndarray,
        w_body: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        p_ref_xy: np.ndarray,
        *,
        init_error_xy: np.ndarray | None = None,
        R_ref: np.ndarray | None = None,
    ) -> None:
        r_sole, _, _, _ = self.compute_sole_fk(R_imu, w_body, q, qd)
        self.prev_r_sole = r_sole.copy()
        self.prev_p_ref_xy = np.asarray(p_ref_xy, dtype=np.float32).copy()
        self.prev_R_ref = None if R_ref is None else np.asarray(R_ref, dtype=np.float32).copy()
        self.v_ref_ema = 0.0
        self.e_world_xy = (
            np.zeros(2, dtype=np.float32)
            if init_error_xy is None
            else np.asarray(init_error_xy, dtype=np.float32).copy()
        )
        self.initialized = True

    def step(
        self,
        R_imu: np.ndarray,
        w_body: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        p_ref_xy: np.ndarray,
        *,
        R_ref: np.ndarray | None = None,
        mode: str = "cgfa_clipped_norm",
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Update CGFA state and return heading-frame `obs_root_xy` (2,) + diagnostic dict."""
        p_ref_xy_f = np.asarray(p_ref_xy, dtype=np.float32)
        if mode == "zeroed":
            _, _, _, fw_cm = self.compute_sole_fk(R_imu, w_body, q, qd)
            return np.zeros(2, dtype=np.float32), {
                "gamma": 0.0,
                "w_left": 0.5,
                "e_world_norm": 0.0,
                "foot_width_cm": fw_cm,
            }

        if not self.initialized:
            self.reset(R_imu, w_body, q, qd, p_ref_xy_f, R_ref=R_ref)

        r_sole, R_feet, w_feet, fw_cm = self.compute_sole_fk(R_imu, w_body, q, qd)
        rz = r_sole[:, 2]
        dz = np.maximum(rz - float(np.min(rz)), 0.0)
        foot_w_mag = np.linalg.norm(w_feet, axis=-1)
        foot_up_z = np.minimum(R_feet[:, 2, 2], 1.0)
        tilt_pen = np.maximum(1.0 - foot_up_z, 0.0)
        logits = -self.height_temp * dz - self.angvel_weight * foot_w_mag - self.flat_weight * tilt_pen
        logits_shift = logits - float(np.max(logits))
        exp_l = np.exp(logits_shift)
        weights = (exp_l / float(np.sum(exp_l))).astype(np.float32)

        dr_xy = r_sole[:, :2] - self.prev_r_sole[:, :2]
        dp_fk_xy = -np.sum(weights[:, None] * dr_xy, axis=0)

        dp_ref_xy = p_ref_xy_f - self.prev_p_ref_xy
        v_inst = float(np.linalg.norm(dp_ref_xy)) / self.dt
        if R_ref is not None:
            R_ref_f = np.asarray(R_ref, dtype=np.float32)
            if self.prev_R_ref is None:
                self.prev_R_ref = R_ref_f.copy()
            d_R = R_ref_f @ self.prev_R_ref.T
            yaw_rate = abs(float(np.arctan2(float(d_R[1, 0]), float(d_R[0, 0])))) / self.dt
            v_inst += self.yaw_rate_equiv_m * yaw_rate
            self.prev_R_ref = R_ref_f.copy()

        self.v_ref_ema = 0.7 * self.v_ref_ema + 0.3 * v_inst
        alpha_loco = float(1.0 / (1.0 + np.exp(-(self.v_ref_ema - self.loco_speed_thresh) / self.loco_speed_temp)))
        gamma = (1.0 - alpha_loco) * self.gamma_stat + alpha_loco * self.gamma_loco

        if mode == "cgfa_clipped_norm":
            raw_e = gamma * self.e_world_xy + dp_ref_xy - dp_fk_xy
            nrm = max(float(np.linalg.norm(raw_e)), 1e-8)
            e_next = raw_e * (self.bound_m / nrm) if nrm > self.bound_m else raw_e
        elif mode == "cgfa_unbounded":
            e_next = gamma * self.e_world_xy + dp_ref_xy - dp_fk_xy
        elif mode == "unfiltered_odom":
            gamma = 1.0
            e_next = self.e_world_xy + dp_ref_xy - dp_fk_xy
        else:
            raise ValueError(f"Unsupported deploy CGFA mode {mode!r}; use 'cgfa_clipped_norm'")

        self.prev_r_sole = r_sole.copy()
        self.prev_p_ref_xy = p_ref_xy_f.copy()
        self.e_world_xy = e_next.astype(np.float32)

        heading_t = yaw_mat_np(R_imu).T
        e3 = np.array([self.e_world_xy[0], self.e_world_xy[1], 0.0], dtype=np.float32)
        obs_root_xy = (heading_t @ e3)[:2].astype(np.float32)
        return obs_root_xy, {
            "gamma": float(gamma),
            "w_left": float(weights[0]),
            "e_world_norm": float(np.linalg.norm(self.e_world_xy)),
            "foot_width_cm": fw_cm,
        }


def build_obs_80_np(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    default_q: np.ndarray,
    R_imu: np.ndarray,
    w_body: np.ndarray,
    a_prev: np.ndarray,
    R_ref: np.ndarray,
    cgfa_exy: np.ndarray,
    xi_joint_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Construct exact 80-D policy observation matching `unisupermimic` `MDP.obs(..., observation_mode='root_xy')`.

    Layout:
      [0:22]  joint_pos - default_q - default_offset
      [22:44] joint_vel (rad/s)
      [44:47] root_angular_velocity_body (w_body in pelvis IMU frame, rad/s)
      [47:50] projected_gravity_body (-R_imu[2, :], upright = [0, 0, -1])
      [50:72] previous_action ((q_tgt_safe - default_q) / action_scale)
      [72:78] mat_to_6d(R_imu^T @ R_ref) (row-major first 2 columns)
      [78:80] cgfa_exy (heading-frame root XY anchor error, m)
    """
    offset = np.zeros(22, dtype=np.float32) if xi_joint_offset is None else np.asarray(xi_joint_offset, dtype=np.float32)
    q_rel = np.asarray(q, dtype=np.float32) - np.asarray(default_q, dtype=np.float32) - offset
    qd_f = np.asarray(qd, dtype=np.float32)
    w_b = np.asarray(w_body, dtype=np.float32)
    grav = -np.asarray(R_imu, dtype=np.float32)[2, :]
    a_p = np.asarray(a_prev, dtype=np.float32)
    rel_rot_6d = mat_to_6d_np(np.asarray(R_imu, dtype=np.float32).T @ np.asarray(R_ref, dtype=np.float32))
    exy = np.asarray(cgfa_exy, dtype=np.float32)
    obs = np.concatenate((q_rel, qd_f, w_b, grav, a_p, rel_rot_6d, exy), axis=0).astype(np.float32)
    if obs.shape != (80,):
        raise ValueError(f"Expected obs_80 shape (80,), got {obs.shape}")
    return obs


# ---------------------------------------------------------------------------
# Pure-NumPy 0ms Causal 4-Modality Feature Builders (bit-exact with PyTorch)
# ---------------------------------------------------------------------------


def _window_indices(num_total: int, f_idx: int, num_frames: int = 10) -> tuple[int, np.ndarray, np.ndarray]:
    f_now = int(np.clip(f_idx, 0, num_total - 1))
    offsets = np.arange(-(num_frames - 1), 1, dtype=np.int64)
    hist_f = np.clip(f_now + offsets, 0, num_total - 1)
    prev_f = np.clip(hist_f - 1, 0, num_total - 1)
    return f_now, hist_f, prev_f


def build_robot_pose_590_np(
    root_p: np.ndarray,
    root_R: np.ndarray,
    q: np.ndarray,
    f_idx: int,
    *,
    dt: float = 0.02,
    num_frames: int = 10,
) -> np.ndarray:
    """Build 590D (10 x 59) causal `robot_pose` input matching `build_causal_history_goal(..., latest_offset=0)`."""
    f_now, hist_f, prev_f = _window_indices(len(root_p), f_idx, num_frames)
    heading_t = yaw_mat_np(root_R[f_now]).T  # (3, 3)

    p_hist, p_prev = root_p[hist_f], root_p[prev_f]
    R_hist, R_prev = root_R[hist_f], root_R[prev_f]
    q_hist, q_prev = q[hist_f], q[prev_f]

    v_world = (p_hist - p_prev) / dt
    w_world = so3_log_np(R_hist @ np.swapaxes(R_prev, -1, -2)) / dt
    qd_causal = (q_hist - q_prev) / dt

    R_hist_t = np.swapaxes(R_hist, -1, -2)
    v_local = (R_hist_t @ v_world[..., None]).squeeze(-1)
    w_local = (R_hist_t @ w_world[..., None]).squeeze(-1)

    pos_delta = (heading_t[None, :, :] @ (p_hist - root_p[f_now][None, :])[..., None]).squeeze(-1)
    rot_6d = mat_to_6d_np(heading_t[None, :, :] @ R_hist)

    feat = np.concatenate((pos_delta, rot_6d, v_local, w_local, q_hist, qd_causal * 0.1), axis=-1)
    return feat.reshape(-1).astype(np.float32)


def build_human_smpl_430_np(
    root_p: np.ndarray,
    root_R: np.ndarray,
    body_p: np.ndarray,
    f_idx: int,
    *,
    dt: float = 0.02,
    num_frames: int = 10,
    h_nom: float = Q1_NOMINAL_PELVIS_Z,
) -> np.ndarray:
    """Build 430D (10 x 43) causal `human_smpl` feature matching `build_human_smpl_goal`."""
    f_now, hist_f, prev_f = _window_indices(len(root_p), f_idx, num_frames)
    heading_t = yaw_mat_np(root_R[f_now]).T

    p_hist, p_prev = root_p[hist_f], root_p[prev_f]
    R_hist, R_prev = root_R[hist_f], root_R[prev_f]
    bp_hist = body_p[hist_f]

    v_world = (p_hist - p_prev) / dt
    w_world = so3_log_np(R_hist @ np.swapaxes(R_prev, -1, -2)) / dt
    R_hist_t = np.swapaxes(R_hist, -1, -2)

    p_delta = (heading_t[None, :, :] @ (p_hist - root_p[f_now][None, :])[..., None]).squeeze(-1) / h_nom
    z_norm = p_hist[..., 2:3] / h_nom
    rot_6d = mat_to_6d_np(heading_t[None, :, :] @ R_hist)
    v_local = (R_hist_t @ v_world[..., None]).squeeze(-1) / h_nom
    w_local = (R_hist_t @ w_world[..., None]).squeeze(-1) * 0.2

    bones = []
    for a, b in BONE_PAIRS:
        vec = bp_hist[:, b, :] - bp_hist[:, a, :]
        vec_h = (heading_t[None, :, :] @ vec[..., None]).squeeze(-1)
        unit_v = vec_h / np.clip(np.linalg.norm(vec_h, axis=-1, keepdims=True), 1e-4, None)
        bones.append(unit_v)
    bone_feat = np.concatenate(bones, axis=-1)
    feat = np.concatenate((p_delta, z_norm, rot_6d, v_local, w_local, bone_feat), axis=-1)
    return feat.reshape(-1).astype(np.float32)


def build_vr_3pt_360_np(
    body_p: np.ndarray,
    body_R: np.ndarray,
    f_idx: int,
    *,
    dt: float = 0.02,
    num_frames: int = 10,
    h_nom: float = Q1_NOMINAL_PELVIS_Z,
    h_torso_nom: float = Q1_NOMINAL_TORSO_Z,
    arm_len: float = Q1_NOMINAL_ARM_LEN,
) -> np.ndarray:
    """Build 360D (10 x 36) causal `vr_3pt` feature from Torso + Left/Right Wrist 6DoF poses."""
    f_now, hist_f, prev_f = _window_indices(len(body_p), f_idx, num_frames)
    t_idx = IDX["torso_link"]
    lw_idx = IDX["left_wrist_yaw_link"]
    rw_idx = IDX["right_wrist_yaw_link"]

    R_t_now = body_R[f_now, t_idx]
    heading_t = yaw_mat_np(R_t_now).T

    pt_hist = body_p[hist_f, t_idx]
    pt_prev = body_p[prev_f, t_idx]
    Rt_hist = body_R[hist_f, t_idx]
    pt_now = body_p[f_now, t_idx]

    torso_dxy = (heading_t[None, :, :] @ (pt_hist - pt_now[None, :])[..., None]).squeeze(-1)[..., :2] / h_nom
    torso_z = pt_hist[..., 2:3] / h_torso_nom
    torso_rot6d = mat_to_6d_np(heading_t[None, :, :] @ Rt_hist)
    vt_local = (np.swapaxes(Rt_hist, -1, -2) @ (((pt_hist - pt_prev) / dt)[..., None])).squeeze(-1) / h_nom

    wrist_feats = []
    for w_idx in (lw_idx, rw_idx):
        pw_hist = body_p[hist_f, w_idx]
        pw_prev = body_p[prev_f, w_idx]
        Rw_hist = body_R[hist_f, w_idx]
        rel_p = (heading_t[None, :, :] @ (pw_hist - pt_hist)[..., None]).squeeze(-1) / arm_len
        w_rot6d = mat_to_6d_np(heading_t[None, :, :] @ Rw_hist)
        vw_rel = (pw_hist - pt_hist) - (pw_prev - pt_prev)
        vw_h = (heading_t[None, :, :] @ ((vw_rel / dt)[..., None])).squeeze(-1) / arm_len * 0.2
        wrist_feats.extend((rel_p, w_rot6d, vw_h))

    feat = np.concatenate((torso_dxy, torso_z, torso_rot6d, vt_local, *wrist_feats), axis=-1)
    return feat.reshape(-1).astype(np.float32)


def build_planner_cmd_single_10d_np(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    z_ratio: float = 1.0,
    reach_x: float = 0.04,
) -> np.ndarray:
    """Single-step 10D PlannerCmd feature (`Card 00 Pitfall 7`: static stand uses neutral arm `reach_x=0.04`)."""
    sp = float(np.sqrt(vx * vx + vy * vy + (0.25 * wz) ** 2))
    stand_f = float(1.0 / (1.0 + np.exp(-(0.12 - sp) / 0.03)))
    crouch_f = float(1.0 / (1.0 + np.exp(-(0.88 - z_ratio) / 0.04)))
    return np.array(
        [vx, vy, wz, z_ratio, 0.10 * crouch_f, 0.0, stand_f, reach_x, reach_x, crouch_f],
        dtype=np.float32,
    )


PLANNER_PROFILES: tuple[str, ...] = (
    "stand_only",
    "gantry_step_forward",
    "gantry_turn_in_place",
    "gantry_squat",
    "card04_basic_18s",
)


def get_planner_profile_cmd(profile: str, t_s: float) -> tuple[float, float, float, float, float]:
    """Deterministic safe gantry PlannerCmd schedules `(vx, vy, wz, z_ratio, reach_x)` as a function of `t_s`."""
    s = max(0.0, float(t_s))
    if profile == "stand_only":
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if profile == "gantry_step_forward":
        # 0-3s stand -> 3-7s slow 0.25 m/s forward (~1.0m within gantry) -> 7s+ smooth stop & stand
        if s < 3.0:
            return (0.0, 0.0, 0.0, 1.0, 0.04)
        if s < 7.0:
            return (0.25, 0.0, 0.0, 1.0, 0.04)
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if profile == "gantry_turn_in_place":
        # 0-3s stand -> 3-5s left +0.45 rad/s -> 5-6s pause -> 6-8s right -0.45 rad/s -> 8s+ stand
        if s < 3.0:
            return (0.0, 0.0, 0.0, 1.0, 0.04)
        if s < 5.0:
            return (0.0, 0.0, 0.45, 1.0, 0.04)
        if s < 6.0:
            return (0.0, 0.0, 0.0, 1.0, 0.04)
        if s < 8.0:
            return (0.0, 0.0, -0.45, 1.0, 0.04)
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if profile == "gantry_squat":
        # 0-3s stand -> 3-5s squat down to 0.82 -> 5-6s hold -> 6-8s rise to 1.0 -> 8s+ stand
        if s < 3.0:
            return (0.0, 0.0, 0.0, 1.0, 0.04)
        if s < 5.0:
            alpha = (s - 3.0) / 2.0
            return (0.0, 0.0, 0.0, 1.0 - 0.18 * alpha, 0.04)
        if s < 6.0:
            return (0.0, 0.0, 0.0, 0.82, 0.04)
        if s < 8.0:
            alpha = (s - 6.0) / 2.0
            return (0.0, 0.0, 0.0, 0.82 + 0.18 * alpha, 0.04)
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if profile == "card04_basic_18s":
        # Card 04 §3 full 18.0s sequence: stand -> 0.35 m/s walk -> +-0.50 rad/s turn -> 0.80 squat -> stop
        if s < 3.0:
            return (0.0, 0.0, 0.0, 1.0, 0.04)
        if s < 7.0:
            return (0.35, 0.0, 0.0, 1.0, 0.04)
        if s < 9.0:
            return (0.0, 0.0, 0.50, 1.0, 0.04)
        if s < 11.0:
            return (0.0, 0.0, -0.50, 1.0, 0.04)
        if s < 13.0:
            alpha = (s - 11.0) / 2.0
            return (0.0, 0.0, 0.0, 1.0 - 0.20 * alpha, 0.04)
        if s < 15.0:
            alpha = (s - 13.0) / 2.0
            return (0.0, 0.0, 0.0, 0.80 + 0.20 * alpha, 0.04)
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    raise ValueError(f"Unknown planner profile {profile!r}; choose from {PLANNER_PROFILES}")


class LivePlannerControllerNP:
    """50Hz causal `PlannerCmd` (`100D`) streamer + world reference integrator + CGFA stop re-anchoring (`Card 04`)."""

    def __init__(
        self,
        *,
        dt: float = 0.02,
        num_frames: int = 10,
        vx_bounds: tuple[float, float] = (-0.25, 0.45),
        vy_bounds: tuple[float, float] = (-0.25, 0.25),
        wz_bounds: tuple[float, float] = (-0.65, 0.65),
        z_ratio_bounds: tuple[float, float] = (0.78, 1.02),
        smooth_slew: bool = True,
        max_dv_s: float = 0.90,
        max_dw_s: float = 1.80,
        max_dz_s: float = 0.40,
        reanchor_yaw_on_stop: bool = False,
    ):
        self.dt = float(dt)
        self.num_frames = int(num_frames)
        self.vx_bounds = vx_bounds
        self.vy_bounds = vy_bounds
        self.wz_bounds = wz_bounds
        self.z_ratio_bounds = z_ratio_bounds
        self.smooth_slew = bool(smooth_slew)
        self.max_dv_s = float(max_dv_s)
        self.max_dw_s = float(max_dw_s)
        self.max_dz_s = float(max_dz_s)
        self.reanchor_yaw_on_stop = bool(reanchor_yaw_on_stop)

        self.buf10: np.ndarray | None = None
        self.p_plan_xy = np.zeros(2, dtype=np.float32)
        self.yaw_plan = 0.0
        self.cur_cmd = np.array([0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float32)
        self.was_moving = False

    def reset(self, p0_xy: np.ndarray | None = None, yaw0_rad: float = 0.0) -> None:
        self.buf10 = None
        self.p_plan_xy = np.zeros(2, dtype=np.float32) if p0_xy is None else np.asarray(p0_xy, dtype=np.float32).copy()
        self.yaw_plan = float(yaw0_rad)
        self.cur_cmd = np.array([0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float32)
        self.was_moving = False

    def push_and_build(
        self,
        vx: float,
        vy: float,
        wz: float,
        z_ratio: float = 1.0,
        reach_x: float = 0.04,
        *,
        cgfa: ContactGatedFootAnchorNP | None = None,
        R_imu_aligned: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance 1 step (dt=0.02s) and return `(planner_100d, p_ref_xy, R_ref)`."""
        vx_t = float(np.clip(vx, *self.vx_bounds))
        vy_t = float(np.clip(vy, *self.vy_bounds))
        wz_t = float(np.clip(wz, *self.wz_bounds))
        zr_t = float(np.clip(z_ratio, *self.z_ratio_bounds))
        rx_t = float(np.clip(reach_x, -0.10, 0.50))

        if self.smooth_slew:
            dv = self.max_dv_s * self.dt
            dw = self.max_dw_s * self.dt
            dz = self.max_dz_s * self.dt
            self.cur_cmd[0] += float(np.clip(vx_t - float(self.cur_cmd[0]), -dv, dv))
            self.cur_cmd[1] += float(np.clip(vy_t - float(self.cur_cmd[1]), -dv, dv))
            self.cur_cmd[2] += float(np.clip(wz_t - float(self.cur_cmd[2]), -dw, dw))
            self.cur_cmd[3] += float(np.clip(zr_t - float(self.cur_cmd[3]), -dz, dz))
            self.cur_cmd[4] = rx_t
        else:
            self.cur_cmd[:] = (vx_t, vy_t, wz_t, zr_t, rx_t)

        vx_c, vy_c, wz_c, zr_c, rx_c = (float(x) for x in self.cur_cmd)
        moving = (abs(vx_c) + abs(vy_c) + 0.25 * abs(wz_c)) > 0.015

        # Card 04 §2.2: Zero-GT smooth stop re-anchoring prevents pullback overshoot on stop
        if self.was_moving and not moving and cgfa is not None and cgfa.initialized:
            self.p_plan_xy = (self.p_plan_xy - cgfa.e_world_xy).astype(np.float32)
            cgfa.prev_p_ref_xy = self.p_plan_xy.copy()
            cgfa.e_world_xy[:] = 0.0
            if self.reanchor_yaw_on_stop and R_imu_aligned is not None:
                self.yaw_plan = yaw_angle_from_mat(R_imu_aligned)
                cgfa.prev_R_ref = rotz_np(self.yaw_plan)
        self.was_moving = moving

        cy, sy = float(np.cos(self.yaw_plan)), float(np.sin(self.yaw_plan))
        self.p_plan_xy[0] += (cy * vx_c - sy * vy_c) * self.dt
        self.p_plan_xy[1] += (sy * vx_c + cy * vy_c) * self.dt
        self.yaw_plan += wz_c * self.dt
        R_ref = rotz_np(self.yaw_plan)

        row10 = build_planner_cmd_single_10d_np(vx=vx_c, vy=vy_c, wz=wz_c, z_ratio=zr_c, reach_x=rx_c)
        if self.buf10 is None:
            self.buf10 = np.repeat(row10[None, :], self.num_frames, axis=0)
        else:
            self.buf10 = np.concatenate((self.buf10[1:], row10[None, :]), axis=0)

        return self.buf10.reshape(-1).astype(np.float32), self.p_plan_xy.copy(), R_ref


class LiveVR3PtBufferNP:
    """Real-time 50Hz ring buffer for live PICO / VR 3-point (Torso/Head + L/R Wrist 6DoF) streaming (`Card 01 §4.4`)."""

    def __init__(
        self,
        *,
        dt: float = 0.02,
        num_frames: int = 10,
        h_nom: float = Q1_NOMINAL_PELVIS_Z,
        h_torso_nom: float = Q1_NOMINAL_TORSO_Z,
        arm_len: float = Q1_NOMINAL_ARM_LEN,
    ):
        self.dt = float(dt)
        self.num_frames = int(num_frames)
        self.h_nom = float(h_nom)
        self.h_torso_nom = float(h_torso_nom)
        self.arm_len = float(arm_len)
        self.body_p_buf: np.ndarray | None = None
        self.body_R_buf: np.ndarray | None = None

    def reset(self) -> None:
        self.body_p_buf = None
        self.body_R_buf = None

    def push_and_build(
        self,
        p_torso: np.ndarray,
        R_torso: np.ndarray,
        p_lwrist: np.ndarray,
        R_lwrist: np.ndarray,
        p_rwrist: np.ndarray,
        R_rwrist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Push newest 3-point pose, return `(vr_3pt_360d, p_ref_xy, R_ref)` for CGFA & `obs_80`."""
        bp = np.zeros((len(TRACKED_BODIES), 3), dtype=np.float32)
        bR = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(TRACKED_BODIES), axis=0)
        t_idx, lw_idx, rw_idx = IDX["torso_link"], IDX["left_wrist_yaw_link"], IDX["right_wrist_yaw_link"]
        bp[t_idx], bR[t_idx] = p_torso, R_torso
        bp[lw_idx], bR[lw_idx] = p_lwrist, R_lwrist
        bp[rw_idx], bR[rw_idx] = p_rwrist, R_rwrist

        K = self.num_frames + 1  # Need 1 extra prior frame for backward velocity at hist[0]
        if self.body_p_buf is None:
            self.body_p_buf = np.repeat(bp[None, :, :], K, axis=0)
            self.body_R_buf = np.repeat(bR[None, :, :], K, axis=0)
        else:
            self.body_p_buf = np.concatenate((self.body_p_buf[1:], bp[None, :, :]), axis=0)
            self.body_R_buf = np.concatenate((self.body_R_buf[1:], bR[None, :, :]), axis=0)

        feat_360 = build_vr_3pt_360_np(
            self.body_p_buf,
            self.body_R_buf,
            K - 1,
            dt=self.dt,
            num_frames=self.num_frames,
            h_nom=self.h_nom,
            h_torso_nom=self.h_torso_nom,
            arm_len=self.arm_len,
        )
        p_ref_xy = np.asarray(p_torso[:2], dtype=np.float32) * (Q1_NOMINAL_PELVIS_Z / max(self.h_nom, 1e-4))
        R_ref = yaw_mat_np(np.asarray(R_torso, dtype=np.float32))
        return feat_360, p_ref_xy, R_ref


class PrebakedClipLibraryNP:
    """Loads pre-baked 50Hz `.npz` reference clips (`root_p, root_R, q, body_p, body_R`) with zero MotionLib/Torch."""

    def __init__(self, npz_path: str | Path):
        data = np.load(Path(npz_path))
        self.root_p = np.asarray(data["root_p"], dtype=np.float32)
        self.root_R = np.asarray(data["root_R"], dtype=np.float32)
        self.q = np.asarray(data["q"], dtype=np.float32)
        self.body_p = np.asarray(data["body_p"], dtype=np.float32)
        self.body_R = np.asarray(data["body_R"], dtype=np.float32)
        self.dt = float(data["ctrl_dt"]) if "ctrl_dt" in data else 0.02
        self.num_frames = int(len(self.root_p))

    def get_step_inputs(self, step_idx: int, modality: str = "robot_pose") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f_idx = int(np.clip(step_idx, 0, self.num_frames - 1))
        p_ref_xy = self.root_p[f_idx, :2]
        R_ref = self.root_R[f_idx]
        if modality == "robot_pose":
            m_in = build_robot_pose_590_np(self.root_p, self.root_R, self.q, f_idx, dt=self.dt)
        elif modality == "human_smpl":
            m_in = build_human_smpl_430_np(self.root_p, self.root_R, self.body_p, f_idx, dt=self.dt)
        elif modality == "vr_3pt":
            m_in = build_vr_3pt_360_np(self.body_p, self.body_R, f_idx, dt=self.dt)
        elif modality == "planner_cmd":
            # Static or clip-derived planner command; for static stand use neutral reach=0.04
            row10 = build_planner_cmd_single_10d_np(vx=0.0, vy=0.0, wz=0.0, z_ratio=1.0, reach_x=0.04)
            m_in = np.tile(row10, 10)
        else:
            raise ValueError(f"Unknown modality: {modality!r}")
        return m_in, p_ref_xy, R_ref
