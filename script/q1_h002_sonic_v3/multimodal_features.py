"""SONIC-style 4-Modality 0ms Causal Feature Extractors (RobotPose, HumanSMPL, VR3Pt, PlannerCmd).

All modalities consume purely causal history windows `[f_now - 9, ..., f_now]` (zero future lookahead)
in the current heading-aligned local frame (`ori_mode="heading"`).
- `robot_pose`  (590D = 10 x 59): full retargeted Q1 joint positions & velocities
- `human_smpl`  (430D = 10 x 43): scale-invariant pelvis motion + 9 anatomical unit bone vectors (zero-IK capable)
- `vr_3pt`      (360D = 10 x 36): sparse torso/head + left/right wrist 6DoF poses & relative velocities
- `planner_cmd` (100D = 10 x 10): root velocity (vx, vy, wz) + normalized height + stand/crouch + wrist reach
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    mat_to_6d,
    quat_to_mat,
    so3_log,
    yaw_mat,
)
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import TRACKED_BODIES
from unisupermimic.teleop.realtime_teleop import build_causal_history_goal

Q1_NOMINAL_PELVIS_Z = 0.4213
Q1_NOMINAL_TORSO_Z = 0.5513
Q1_NOMINAL_ARM_LEN = 0.275

IDX = {name: i for i, name in enumerate(TRACKED_BODIES)}

# 9 anatomical unit bone directions in TRACKED_BODIES index space
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

MODALITY_DIMS = {
    "robot_pose": 590,
    "human_smpl": 430,
    "vr_3pt": 360,
    "planner_cmd": 100,
}


def _causal_window_indices(
    motion: Any, frames: torch.Tensor, num_frames: int, latest_offset: int | torch.Tensor = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clip_id = motion.clip_id[frames]
    clip_start = motion.clip_start[clip_id]
    clip_end = motion.clip_end[clip_id]
    if isinstance(latest_offset, torch.Tensor):
        f_now = torch.clamp(frames + latest_offset, min=clip_start, max=clip_end)
    else:
        f_now = torch.clamp(frames + int(latest_offset), min=clip_start, max=clip_end)
    offsets = torch.arange(-(num_frames - 1), 1, device=frames.device, dtype=frames.dtype)
    hist_f = torch.clamp(f_now.unsqueeze(-1) + offsets.unsqueeze(0), min=clip_start.unsqueeze(-1), max=clip_end.unsqueeze(-1))
    prev_f = torch.clamp(hist_f - 1, min=clip_start.unsqueeze(-1), max=clip_end.unsqueeze(-1))
    return f_now, hist_f, prev_f, clip_start


def build_robot_pose_goal(
    motion: Any,
    frames: torch.Tensor,
    num_frames: int = 10,
    *,
    latest_offset: int = 0,
) -> torch.Tensor:
    return build_causal_history_goal(motion, frames, num_frames, latest_offset=latest_offset)


def build_human_smpl_goal(
    motion: Any,
    frames: torch.Tensor,
    num_frames: int = 10,
    *,
    latest_offset: int | torch.Tensor = 0,
    h_nom: float = Q1_NOMINAL_PELVIS_Z,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Build 430D scale-invariant causal human keypoint + 9-bone unit-vector feature."""
    dt = float(getattr(motion, "ctrl_dt", 0.02))
    f_now, hist_f, prev_f, _ = _causal_window_indices(motion, frames, num_frames, latest_offset)

    # Current reference heading (use f_now so strictly causal)
    heading = yaw_mat(motion.root_R[f_now])
    heading_t = heading.transpose(-1, -2).unsqueeze(1)  # (N, 1, 3, 3)

    p_hist = motion.root_p[hist_f]
    p_prev = motion.root_p[prev_f]
    R_hist = motion.root_R[hist_f]
    R_prev = motion.root_R[prev_f]
    bp_hist = motion.body_p[hist_f]  # (N, K, 14, 3)

    v_world = (p_hist - p_prev) / dt
    w_world = so3_log(R_hist @ R_prev.transpose(-1, -2)) / dt
    R_hist_t = R_hist.transpose(-1, -2)

    p_delta = (heading_t @ (p_hist - motion.root_p[f_now].unsqueeze(1)).unsqueeze(-1)).squeeze(-1) / h_nom
    z_norm = p_hist[..., 2:3] / h_nom
    rot_6d = mat_to_6d(heading_t @ R_hist)
    v_local = (R_hist_t @ v_world.unsqueeze(-1)).squeeze(-1) / h_nom
    w_local = (R_hist_t @ w_world.unsqueeze(-1)).squeeze(-1) * 0.2

    bones = []
    for a, b in BONE_PAIRS:
        vec = bp_hist[:, :, b, :] - bp_hist[:, :, a, :]
        vec_h = (heading_t @ vec.unsqueeze(-1)).squeeze(-1)
        unit_v = vec_h / vec_h.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        bones.append(unit_v)
    bone_feat = torch.cat(bones, dim=-1)  # (N, K, 27)
    if noise_std > 0.0:
        bone_feat = bone_feat + torch.randn_like(bone_feat) * noise_std

    feat = torch.cat((p_delta, z_norm, rot_6d, v_local, w_local, bone_feat), dim=-1)
    return feat.reshape(frames.shape[0], -1)


def build_vr_3pt_goal(
    motion: Any,
    frames: torch.Tensor,
    num_frames: int = 10,
    *,
    latest_offset: int | torch.Tensor = 0,
    h_nom: float = Q1_NOMINAL_PELVIS_Z,
    h_torso_nom: float = Q1_NOMINAL_TORSO_Z,
    arm_len: float = Q1_NOMINAL_ARM_LEN,
    pos_noise_m: float = 0.0,
) -> torch.Tensor:
    """Build 360D sparse Torso + Left/Right Wrist 6DoF causal teleop feature."""
    dt = float(getattr(motion, "ctrl_dt", 0.02))
    f_now, hist_f, prev_f, _ = _causal_window_indices(motion, frames, num_frames, latest_offset)

    t_idx = IDX["torso_link"]
    lw_idx = IDX["left_wrist_yaw_link"]
    rw_idx = IDX["right_wrist_yaw_link"]

    R_t_now = motion.body_R[f_now, t_idx]
    heading_t = yaw_mat(R_t_now).transpose(-1, -2).unsqueeze(1)  # (N, 1, 3, 3)

    pt_hist = motion.body_p[hist_f, t_idx]
    pt_prev = motion.body_p[prev_f, t_idx]
    Rt_hist = motion.body_R[hist_f, t_idx]
    pt_now = motion.body_p[f_now, t_idx].unsqueeze(1)

    torso_dxy = (heading_t @ (pt_hist - pt_now).unsqueeze(-1)).squeeze(-1)[..., :2] / h_nom
    torso_z = pt_hist[..., 2:3] / h_torso_nom
    torso_rot6d = mat_to_6d(heading_t @ Rt_hist)
    vt_local = (Rt_hist.transpose(-1, -2) @ ((pt_hist - pt_prev) / dt).unsqueeze(-1)).squeeze(-1) / h_nom

    wrist_feats = []
    for w_idx in (lw_idx, rw_idx):
        pw_hist = motion.body_p[hist_f, w_idx]
        pw_prev = motion.body_p[prev_f, w_idx]
        Rw_hist = motion.body_R[hist_f, w_idx]
        rel_p = (heading_t @ (pw_hist - pt_hist).unsqueeze(-1)).squeeze(-1) / arm_len
        if pos_noise_m > 0.0:
            rel_p = rel_p + torch.randn_like(rel_p) * (pos_noise_m / arm_len)
        w_rot6d = mat_to_6d(heading_t @ Rw_hist)
        vw_rel = (pw_hist - pt_hist) - (pw_prev - pt_prev)
        vw_h = (heading_t @ (vw_rel / dt).unsqueeze(-1)).squeeze(-1) / arm_len * 0.2
        wrist_feats.extend((rel_p, w_rot6d, vw_h))

    feat = torch.cat((torso_dxy, torso_z, torso_rot6d, vt_local, *wrist_feats), dim=-1)
    return feat.reshape(frames.shape[0], -1)


def build_planner_cmd_goal(
    motion: Any,
    frames: torch.Tensor,
    num_frames: int = 10,
    *,
    latest_offset: int | torch.Tensor = 0,
    h_nom: float = Q1_NOMINAL_PELVIS_Z,
    arm_len: float = Q1_NOMINAL_ARM_LEN,
) -> torch.Tensor:
    """Build 100D Gamepad/Keyboard-compatible planner command history feature."""
    dt = float(getattr(motion, "ctrl_dt", 0.02))
    f_now, hist_f, prev_f, _ = _causal_window_indices(motion, frames, num_frames, latest_offset)

    p_hist = motion.root_p[hist_f]
    p_prev = motion.root_p[prev_f]
    R_hist = motion.root_R[hist_f]
    R_prev = motion.root_R[prev_f]
    bp_hist = motion.body_p[hist_f]
    Rt_hist = motion.body_R[hist_f, IDX["torso_link"]]

    v_world = (p_hist - p_prev) / dt
    w_world = so3_log(R_hist @ R_prev.transpose(-1, -2)) / dt
    h_t = yaw_mat(R_hist).transpose(-1, -2)
    v_h = (h_t @ v_world.unsqueeze(-1)).squeeze(-1)
    w_h = (h_t @ w_world.unsqueeze(-1)).squeeze(-1)

    vx = v_h[..., 0:1] / max(h_nom, 1e-4) * Q1_NOMINAL_PELVIS_Z
    vy = v_h[..., 1:2] / max(h_nom, 1e-4) * Q1_NOMINAL_PELVIS_Z
    wz = w_h[..., 2:3]
    z_norm = p_hist[..., 2:3] / h_nom
    torso_pitch = Rt_hist[..., 2, 0:1]
    torso_roll = Rt_hist[..., 2, 1:2]
    speed_tot = torch.sqrt(vx * vx + vy * vy + (0.25 * wz) ** 2)
    stand_flag = torch.sigmoid((0.12 - speed_tot) / 0.03)

    pt_hist = bp_hist[:, :, IDX["torso_link"], :]
    l_rel = (h_t @ (bp_hist[:, :, IDX["left_wrist_yaw_link"], :] - pt_hist).unsqueeze(-1)).squeeze(-1)[..., 0:1] / arm_len
    r_rel = (h_t @ (bp_hist[:, :, IDX["right_wrist_yaw_link"], :] - pt_hist).unsqueeze(-1)).squeeze(-1)[..., 0:1] / arm_len
    crouch_flag = torch.sigmoid((0.88 - z_norm) / 0.04)

    feat = torch.cat((vx, vy, wz, z_norm, torso_pitch, torso_roll, stand_flag, l_rel, r_rel, crouch_flag), dim=-1)
    return feat.reshape(frames.shape[0], -1)


def _mock_motion_from_arrays(
    body_p_50: np.ndarray,
    body_R_50: np.ndarray,
    h_nom: float,
    h_torso_nom: float,
    arm_len: float,
    ctrl_dt: float,
) -> tuple[Any, float, float, float]:
    T50 = len(body_p_50)

    class _MockMotion:
        def __init__(self) -> None:
            self.ctrl_dt = ctrl_dt
            self.body_p = torch.from_numpy(np.ascontiguousarray(body_p_50, dtype=np.float32))
            self.body_R = torch.from_numpy(np.ascontiguousarray(body_R_50, dtype=np.float32))
            self.root_p = self.body_p[:, 0, :]
            self.root_R = self.body_R[:, 0, :, :]
            self.clip_id = torch.zeros(T50, dtype=torch.long)
            self.clip_start = torch.tensor([0], dtype=torch.long)
            self.clip_end = torch.tensor([T50 - 1], dtype=torch.long)
            self.clip_len = torch.tensor([T50], dtype=torch.long)
            self.num_frames = T50

    return _MockMotion(), float(h_nom), float(h_torso_nom), float(arm_len)


def load_bvh_as_mock_motion(
    bvh_path: Path,
    *,
    kind: str,
    soma_canon: Path = Path("/home/yu/datasets/soma_showcase/soma_base_skel_minimal.bvh"),
    gmr_root: Path = Path("/home/yu/models/cache/GMR"),
    ctrl_dt: float = 0.02,
    max_frames_30hz: int = 240,
    save_cache: bool = True,
) -> tuple[Any, float, float, float]:
    """Load raw human SOMA or LAFAN1 BVH (zero Q1 IK!) resampled at 50Hz for Zero-IK HumanSMPL/VR3Pt testing."""
    cache_path = Path(bvh_path).with_suffix(".50hz.npz")
    if cache_path.exists():
        d = np.load(cache_path)
        if abs(float(d["ctrl_dt"]) - ctrl_dt) < 1e-6:
            return _mock_motion_from_arrays(
                d["body_p_50"], d["body_R_50"],
                float(d["h_nom"]), float(d["h_torso_nom"]), float(d["arm_len"]), ctrl_dt
            )

    import sys as _sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    from tools.retarget_q1 import load_bvh
    from tools.soma_source import load_soma

    if kind == "soma":
        src = load_soma(bvh_path, soma_canon, gmr_root, max_frames=max_frames_30hz)
    elif kind == "lafan1":
        src = load_bvh(bvh_path, max_frames=max_frames_30hz, gmr_root=gmr_root)
    else:
        raise ValueError(kind)

    T30 = len(src["p"]["pelvis"])
    dur = (T30 - 1) / 30.0
    T50 = int(np.floor(dur / ctrl_dt)) + 1
    sample = np.arange(T50, dtype=np.float32) * ctrl_dt * 30.0
    lo = np.clip(np.floor(sample).astype(np.int64), 0, T30 - 2)
    alpha = np.clip(sample - lo, 0.0, 1.0)[:, None]

    body_p_50 = np.empty((T50, len(TRACKED_BODIES), 3), dtype=np.float32)
    body_R_50 = np.empty((T50, len(TRACKED_BODIES), 3, 3), dtype=np.float32)

    from scipy.spatial.transform import Rotation, Slerp
    t_30 = np.arange(T30, dtype=np.float64) / 30.0
    t_50 = np.clip(np.arange(T50, dtype=np.float64) * ctrl_dt, 0.0, t_30[-1])

    for i, name in enumerate(TRACKED_BODIES):
        p30 = src["p"][name].astype(np.float32)
        body_p_50[:, i, :] = p30[lo] * (1.0 - alpha) + p30[lo + 1] * alpha
        slerp = Slerp(t_30, Rotation.from_matrix(src["R"][name]))
        body_R_50[:, i, :, :] = slerp(t_50).as_matrix().astype(np.float32)

    # Human natural scale calibration from resting/standing frames
    h_nom = float(np.quantile(body_p_50[:, IDX["pelvis"], 2], 0.85))
    h_torso_nom = float(np.quantile(body_p_50[:, IDX["torso_link"], 2], 0.85))
    arm_l = np.linalg.norm(body_p_50[:, IDX["left_elbow_link"]] - body_p_50[:, IDX["left_shoulder_roll_link"]], axis=-1) + \
            np.linalg.norm(body_p_50[:, IDX["left_wrist_yaw_link"]] - body_p_50[:, IDX["left_elbow_link"]], axis=-1)
    arm_r = np.linalg.norm(body_p_50[:, IDX["right_elbow_link"]] - body_p_50[:, IDX["right_shoulder_roll_link"]], axis=-1) + \
            np.linalg.norm(body_p_50[:, IDX["right_wrist_yaw_link"]] - body_p_50[:, IDX["right_elbow_link"]], axis=-1)
    arm_len = float(0.5 * np.median(arm_l + arm_r))

    if save_cache:
        np.savez(
            cache_path,
            body_p_50=body_p_50,
            body_R_50=body_R_50,
            h_nom=np.float32(h_nom),
            h_torso_nom=np.float32(h_torso_nom),
            arm_len=np.float32(arm_len),
            ctrl_dt=np.float32(ctrl_dt),
        )

    return _mock_motion_from_arrays(body_p_50, body_R_50, h_nom, h_torso_nom, arm_len, ctrl_dt)


def build_synthetic_planner_stream(
    commands: list[tuple[int, float, float, float, float, float]],
    *,
    dt: float = 0.02,
    num_frames: int = 10,
) -> torch.Tensor:
    """Build (T, 100) pure Gamepad/Keyboard PlannerCmd stream from timed commands
    `[(n_steps, vx, vy, wz, z_ratio, reach_x), ...]`, requiring ZERO reference CSV!
    """
    frames_10 = []
    # Neutral torso orientation ~ 0 pitch/roll, stand_flag from speed, crouch_flag from z_ratio
    for n_steps, vx, vy, wz, z_ratio, reach_x in commands:
        sp = np.sqrt(vx * vx + vy * vy + (0.25 * wz) ** 2)
        stand_f = float(1.0 / (1.0 + np.exp(-(0.12 - sp) / 0.03)))
        crouch_f = float(1.0 / (1.0 + np.exp(-(0.88 - z_ratio) / 0.04)))
        row = np.array([vx, vy, wz, z_ratio, 0.10 * crouch_f, 0.0, stand_f, reach_x, reach_x, crouch_f], dtype=np.float32)
        for _ in range(n_steps):
            frames_10.append(row)
    arr = np.stack(frames_10, axis=0)
    T = len(arr)
    out = np.empty((T, num_frames * 10), dtype=np.float32)
    for t in range(T):
        idxs = [max(0, t - (num_frames - 1 - k)) for k in range(num_frames)]
        out[t] = arr[idxs].reshape(-1)
    return torch.from_numpy(out)
