"""Contact-Gated Kinematic Foot-Anchor (CGFA) estimator for mocap-free humanoid teleop.

Purely proprioceptive estimator using (q, qd, R_imu, w_imu) and causal reference increments:
no external motion-capture world position, no LiDAR odometry, and no simulator world XY.
Includes turn-aware yaw-rate gating so in-place pivot turns reset foot slip drift cleanly.
"""

from __future__ import annotations

from typing import Literal

import torch

from unisupermimic.rl.algos.supermimic.geometry import yaw_mat
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import TRACKED_BODIES

LEFT_ANKLE_IDX = TRACKED_BODIES.index("left_ankle_roll_link")
RIGHT_ANKLE_IDX = TRACKED_BODIES.index("right_ankle_roll_link")
LEFT_WRIST_IDX = TRACKED_BODIES.index("left_wrist_yaw_link")
RIGHT_WRIST_IDX = TRACKED_BODIES.index("right_wrist_yaw_link")

# Sole contact point offset in ankle_roll_link local frame (+2.0 cm forward, -2.2 cm down)
SOLE_LOCAL_OFFSET = (0.020, 0.0, -0.022)

EstimatorMode = Literal["cgfa", "cgfa_unbounded", "unfiltered_odom", "zeroed", "oracle"]


class ContactGatedFootAnchor:
    """Proprioceptive sole-stance odometry with locomotion- and yaw-rate-gated forgetting."""

    def __init__(
        self,
        kin: Q1Kinematics,
        batch_size: int,
        dt: float = 0.02,
        *,
        height_temp: float = 50.0,
        angvel_weight: float = 2.5,
        flat_weight: float = 6.0,
        loco_speed_thresh: float = 0.15,
        loco_speed_temp: float = 0.04,
        yaw_rate_equiv_m: float = 0.20,
        gamma_stat: float = 0.998,
        gamma_loco: float = 0.92,
        bound_m: float = 0.25,
        device: str | torch.device = "cpu",
    ):
        self.kin = kin
        self.batch_size = batch_size
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
        self.device = torch.device(device)
        self.sole_offset = torch.tensor(SOLE_LOCAL_OFFSET, dtype=torch.float32, device=self.device)

        self.e_world_xy = torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device)
        self.prev_r_sole = torch.zeros((batch_size, 2, 3), dtype=torch.float32, device=self.device)
        self.prev_p_ref_xy = torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device)
        self.prev_R_ref: torch.Tensor | None = None
        self.v_ref_ema = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        self.initialized = False

    def to(self, device: str | torch.device) -> "ContactGatedFootAnchor":
        self.device = torch.device(device)
        self.sole_offset = self.sole_offset.to(self.device)
        self.e_world_xy = self.e_world_xy.to(self.device)
        self.prev_r_sole = self.prev_r_sole.to(self.device)
        self.prev_p_ref_xy = self.prev_p_ref_xy.to(self.device)
        if self.prev_R_ref is not None:
            self.prev_R_ref = self.prev_R_ref.to(self.device)
        self.v_ref_ema = self.v_ref_ema.to(self.device)
        return self

    def _compute_sole_fk(
        self,
        R_imu: torch.Tensor,
        w_imu: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = R_imu.shape[0]
        zeros3 = torch.zeros((B, 3), dtype=R_imu.dtype, device=R_imu.device)
        body_p, body_R, _, body_w = self.kin.forward(zeros3, R_imu, zeros3, w_imu, q, qd)
        r_ankles = body_p[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
        R_feet = body_R[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :, :]
        w_feet = body_w[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
        sole_off = self.sole_offset.to(device=R_imu.device, dtype=R_imu.dtype).view(1, 1, 3, 1)
        r_sole = r_ankles + (R_feet @ sole_off).squeeze(-1)
        return r_sole, R_feet, w_feet

    def reset(
        self,
        R_imu: torch.Tensor,
        w_imu: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        p_ref_xy: torch.Tensor,
        init_error_xy: torch.Tensor | None = None,
        R_ref: torch.Tensor | None = None,
    ) -> None:
        r_sole, _, _ = self._compute_sole_fk(R_imu, w_imu, q, qd)
        self.prev_r_sole = r_sole.detach().clone()
        self.prev_p_ref_xy = p_ref_xy.detach().clone()
        self.prev_R_ref = R_ref.detach().clone() if R_ref is not None else None
        self.v_ref_ema = torch.zeros((R_imu.shape[0], 1), dtype=torch.float32, device=R_imu.device)
        if init_error_xy is None:
            self.e_world_xy = torch.zeros((R_imu.shape[0], 2), dtype=torch.float32, device=R_imu.device)
        else:
            self.e_world_xy = init_error_xy.detach().clone().to(R_imu.device)
        self.initialized = True

    def step(
        self,
        R_imu: torch.Tensor,
        w_imu: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        p_ref_xy: torch.Tensor,
        *,
        R_ref: torch.Tensor | None = None,
        mode: EstimatorMode = "cgfa",
        true_obs_root_xy: torch.Tensor | None = None,
        slip_noise_std_m: float = 0.0,
        differentiable: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Update anchor state cleanly using ONLY proprioceptive state and return obs[:, 78:80]."""
        if mode == "oracle":
            if true_obs_root_xy is None:
                raise ValueError("oracle mode requires true_obs_root_xy")
            return true_obs_root_xy, {
                "gamma": torch.ones((R_imu.shape[0], 1), device=R_imu.device),
                "w_left": torch.full((R_imu.shape[0],), 0.5, device=R_imu.device),
                "e_world_norm": self.e_world_xy.norm(dim=-1),
                "dp_fk_xy": torch.zeros((R_imu.shape[0], 2), device=R_imu.device),
            }
        if mode == "zeroed":
            zeros2 = torch.zeros((R_imu.shape[0], 2), dtype=R_imu.dtype, device=R_imu.device)
            return zeros2, {
                "gamma": torch.zeros((R_imu.shape[0], 1), device=R_imu.device),
                "w_left": torch.full((R_imu.shape[0],), 0.5, device=R_imu.device),
                "e_world_norm": zeros2.norm(dim=-1),
                "dp_fk_xy": zeros2,
            }

        if not self.initialized or self.e_world_xy.device != R_imu.device:
            self.reset(R_imu, w_imu, q, qd, p_ref_xy, R_ref=R_ref)

        r_sole, R_feet, w_feet = self._compute_sole_fk(R_imu, w_imu, q, qd)
        rz = r_sole[..., 2]
        z_min = rz.min(dim=-1, keepdim=True).values
        dz = (rz - z_min).clamp(min=0.0)
        foot_w_mag = w_feet.norm(dim=-1)
        foot_up_z = R_feet[..., 2, 2].clamp(max=1.0)
        tilt_pen = (1.0 - foot_up_z).clamp(min=0.0)
        logits = -self.height_temp * dz - self.angvel_weight * foot_w_mag - self.flat_weight * tilt_pen
        weights = torch.softmax(logits, dim=-1)

        dr_xy = r_sole[..., :2] - self.prev_r_sole[..., :2]
        dp_fk_xy = -(weights.unsqueeze(-1) * dr_xy).sum(dim=1)

        dp_ref_xy = p_ref_xy - self.prev_p_ref_xy
        v_inst = dp_ref_xy.norm(dim=-1, keepdim=True) / self.dt
        if R_ref is not None:
            if self.prev_R_ref is None:
                self.prev_R_ref = R_ref.detach().clone()
            d_R = R_ref @ self.prev_R_ref.transpose(-1, -2)
            yaw_rate = torch.atan2(d_R[:, 1, 0], d_R[:, 0, 0]).abs().unsqueeze(-1) / self.dt
            v_inst = v_inst + self.yaw_rate_equiv_m * yaw_rate
            self.prev_R_ref = R_ref.detach()

        self.v_ref_ema = 0.7 * self.v_ref_ema.detach() + 0.3 * v_inst.detach()
        alpha_loco = torch.sigmoid((self.v_ref_ema - self.loco_speed_thresh) / self.loco_speed_temp)

        if slip_noise_std_m > 0.0:
            noise = torch.randn_like(dp_fk_xy) * (slip_noise_std_m * alpha_loco)
            dp_fk_xy = dp_fk_xy + noise

        if mode == "unfiltered_odom":
            gamma = torch.ones_like(alpha_loco)
            e_next = self.e_world_xy + dp_ref_xy - dp_fk_xy
        elif mode == "cgfa_unbounded":
            gamma = (1.0 - alpha_loco) * self.gamma_stat + alpha_loco * self.gamma_loco
            e_next = gamma * self.e_world_xy + dp_ref_xy - dp_fk_xy
        elif mode == "cgfa":
            gamma = (1.0 - alpha_loco) * self.gamma_stat + alpha_loco * self.gamma_loco
            raw_e = gamma * self.e_world_xy + dp_ref_xy - dp_fk_xy
            e_next = self.bound_m * torch.tanh(raw_e / self.bound_m)
        else:
            raise ValueError(f"Unknown CGFA mode: {mode!r}")

        self.prev_r_sole = r_sole if differentiable else r_sole.detach()
        self.prev_p_ref_xy = p_ref_xy if differentiable else p_ref_xy.detach()
        self.e_world_xy = e_next if differentiable else e_next.detach()

        heading_t = yaw_mat(R_imu).transpose(-1, -2)
        e3 = torch.cat((e_next, torch.zeros_like(e_next[..., :1])), dim=-1)
        obs_root_xy = (heading_t @ e3.unsqueeze(-1)).squeeze(-1)[..., :2]
        return obs_root_xy, {
            "gamma": gamma.detach(),
            "w_left": weights[:, 0].detach(),
            "e_world_norm": e_next.norm(dim=-1).detach(),
            "dp_fk_xy": dp_fk_xy.detach(),
        }
