"""Upgraded pure-NumPy FGP SMPL24 adapter for Q1 (outage-flicker grace, tilt debiasing, anti-skating conditioning)."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from q1_cgfa_obs_np import BONE_PAIRS, Q1_NOMINAL_PELVIS_Z, build_human_smpl_430_np
from q1_kinematics_np import (
    axis_angle_to_mat_np,
    mat_to_quat_wxyz,
    quat_wxyz_to_mat,
    so3_log_np,
    yaw_mat_np,
)

# FGP SMPL axes are +Z forward, +X left and +Y up. Q1 uses forward/left/up.
FGP_TO_Q1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
# In q1_kinematics_np.TRACKED_BODIES order. Spine3 represents upper torso.
SMPL_TRACKED = [0, 1, 4, 7, 2, 5, 8, 9, 16, 18, 20, 17, 19, 21]

# Tracked body indices inside the 14-body array:
_PELVIS_BI = 0
_L_ANKLE_BI = 3
_R_ANKLE_BI = 6
_TORSO_BI = 7
_UPPER_BODY_BIS = slice(7, 14)  # torso + L/R shoulder, elbow, wrist (legs 1..6 stay gravity-anchored)


@dataclass
class SMPLTick:
    segment: int
    stamp_ns: int
    feature: np.ndarray
    root_xy: np.ndarray
    root_rotation: np.ndarray


def so3_exp_np(rotvec: np.ndarray) -> np.ndarray:
    """Convert single rotation vector (3,) to (3, 3) float32 rotation matrix."""
    rv = np.asarray(rotvec, dtype=np.float64)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-7:
        wx, wy, wz = rv
        return np.array(
            [[1.0, -wz, wy], [wz, 1.0, -wx], [-wy, wx, 1.0]],
            dtype=np.float32,
        )
    return axis_angle_to_mat_np((rv / ang).astype(np.float32), float(ang))


def _vec_tilt_rotvec_xy(vec: np.ndarray) -> np.ndarray:
    """Shortest-arc XY rotation vector (rx, ry) that carries +Z to unit direction `vec`."""
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-6:
        return np.zeros(2, dtype=np.float64)
    u = np.asarray(vec, dtype=np.float64) / nrm
    # cross([0, 0, 1], u) = [-u_y, u_x, 0]
    axis_xy = np.array([-u[1], u[0]], dtype=np.float64)
    sin_a = float(np.linalg.norm(axis_xy))
    if sin_a < 1e-7:
        return np.zeros(2, dtype=np.float64)
    ang = float(np.arctan2(sin_a, float(np.clip(u[2], -1.0, 1.0))))
    return axis_xy * (ang / sin_a)


def interpolate_rotation(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return quat_wxyz_to_mat(a + weight * (b - a))
    angle = np.arccos(np.clip(dot, 0.0, 1.0))
    q = (np.sin((1.0 - weight) * angle) * a + np.sin(weight * angle) * b) / np.sin(angle)
    return quat_wxyz_to_mat(q)


class FGPSMPLStreamV2:
    """Resample FGP SMPL24 at 50 Hz with transient outage grace, tilt debiasing, and anti-skating conditioning."""

    def __init__(
        self,
        dt: float = 0.02,
        *,
        standing_height_m: float | None = None,
        outage_grace_ms: float = 220.0,
        enable_tilt_debias: bool = True,
        enable_vel_conditioning: bool = True,
        tilt_ema_tau_s: float = 5.0,
        max_tilt_bias_deg: float = 24.0,
        max_pelvis_pitch_deg: float = 12.0,
        max_pelvis_roll_deg: float = 10.0,
        vel_stance_beta: float = 0.60,
        vel_deadband_lo_m_s: float = 0.025,
        vel_deadband_hi_m_s: float = 0.060,
        vel_ema_tau_s: float = 0.055,
        vel_max_accel_m_s2: float = 6.5,
        max_interp_steps: int = 8,
    ):
        if standing_height_m is not None and (
            not np.isfinite(standing_height_m) or standing_height_m <= 0
        ):
            raise ValueError("Standing pelvis height must be positive and finite")
        self.dt = float(dt)
        self.period_ns = round(self.dt * 1e9)
        self.timeout_ns = 150_000_000
        self.outage_grace_ns = int(float(outage_grace_ms) * 1e6)
        self.fixed_height = standing_height_m
        self.height = standing_height_m

        self.enable_tilt_debias = bool(enable_tilt_debias)
        self.enable_vel_conditioning = bool(enable_vel_conditioning)
        self.tilt_ema_tau_s = float(tilt_ema_tau_s)
        self.max_tilt_bias_rad = float(np.radians(max_tilt_bias_deg))
        self.max_pelvis_pitch_rad = float(np.radians(max_pelvis_pitch_deg))
        self.max_pelvis_roll_rad = float(np.radians(max_pelvis_roll_deg))
        self.vel_stance_beta = float(vel_stance_beta)
        self.vel_deadband_lo = float(vel_deadband_lo_m_s)
        self.vel_deadband_hi = float(vel_deadband_hi_m_s)
        self.vel_ema_tau_s = float(vel_ema_tau_s)
        self.vel_max_accel = float(vel_max_accel_m_s2)
        self.max_interp_steps = int(max_interp_steps)

        self.identity: tuple[str, int] | None = None
        self.last_sequence = 0
        self.last_source_ns = 0
        self.last_arrival_ns: int | None = None
        self.last_valid_arrival_ns: int | None = None
        self.segment = 0
        self.events: Counter[str] = Counter()
        self.history = deque(maxlen=11)
        self.previous = None
        self._reset_conditioning()

    def _reset_conditioning(self) -> None:
        self._cond_frames = 0
        self._torso_bias_xy = np.zeros(2, dtype=np.float64)
        self._pelvis_rel_bias_xy = np.zeros(2, dtype=np.float64)
        self._prev_pelvis_rot_h = np.zeros(2, dtype=np.float64)
        self._vel_init = False
        self._prev_vel_stamp = 0
        self._raw_pelvis_xy = np.zeros(2, dtype=np.float64)
        self._cond_pelvis_xy = np.zeros(2, dtype=np.float64)
        self._prev_lank_xy = np.zeros(2, dtype=np.float64)
        self._prev_rank_xy = np.zeros(2, dtype=np.float64)
        self._v_xy_filt = np.zeros(2, dtype=np.float64)

    def invalidate(self, reason: str) -> None:
        if self.previous is not None:
            self.events[reason] += 1
        self.previous = None
        self.history.clear()
        self._reset_conditioning()

    def push_response(
        self,
        response: dict,
        arrival_ns: int,
        *,
        allow_sequence_skip: bool = True,
    ) -> list[SMPLTick]:
        if not isinstance(response, dict):
            self.invalidate("invalid_frame")
            return []
        sample = response.get("sample")
        sess = response.get("session_id")
        gen = response.get("calibration_id")
        if response.get("status") != "tracking" or response.get("realtime_valid") is not True:
            # Do not wipe 11-frame history & re-anchor for brief single-tick BLE/skew flickers.
            if (
                self.outage_grace_ns > 0
                and self.previous is not None
                and self.identity == (sess, gen)
                and self.last_valid_arrival_ns is not None
                and 0 <= arrival_ns - self.last_valid_arrival_ns <= self.outage_grace_ns
            ):
                self.events["ignored_transient_outages"] += 1
                return []
            self.invalidate("tracking_lost")
            return []
        if not isinstance(sample, dict) or any(
            response.get(key) != sample.get(key) for key in ("session_id", "calibration_id")
        ):
            self.invalidate("invalid_frame")
            return []
        ticks = self.push_sample(
            sample,
            arrival_ns=arrival_ns,
            allow_sequence_skip=allow_sequence_skip,
        )
        self.last_valid_arrival_ns = int(arrival_ns)
        return ticks

    def _condition_sample(
        self,
        rotation: np.ndarray,
        body: np.ndarray,
        floor: float,
        stamp_ns: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply causal tilt debiasing and contact-aware anti-skating velocity conditioning in Q1 world."""
        rot = rotation.astype(np.float32, copy=True)
        b = body.astype(np.float32, copy=True)

        dt_src = self.dt
        if self._vel_init and stamp_ns > self._prev_vel_stamp:
            dt_src = float(np.clip((stamp_ns - self._prev_vel_stamp) * 1e-9, 0.005, 0.15))

        if self.enable_tilt_debias:
            R_yaw = yaw_mat_np(rot)
            hy = R_yaw.T
            # Estimate slow pelvis & torso heading-frame biases independently during calm upright frames:
            # 1) torso vector bias corrects upper body + pelvis only if spine itself is persistently tilted (cal_id=11)
            # 2) pelvis orientation bias corrects crooked waist IMU mounting without distorting GP gravity legs (cal_id=1/2)
            sp_h = hy @ (b[_TORSO_BI] - b[_PELVIS_BI])
            u_torso_h = _vec_tilt_rotvec_xy(sp_h)
            u_pelvis_h = so3_log_np(hy @ rot)[:2].astype(np.float64)

            if self._cond_frames > 0:
                w_xy_mag = float(np.linalg.norm(u_pelvis_h - self._prev_pelvis_rot_h)) / max(dt_src, 1e-3)
            else:
                w_xy_mag = 0.0
            self._prev_pelvis_rot_h = u_pelvis_h.copy()

            # Gate updates strictly to calm upright standing so real operator bows (e.g. >16 deg) never pollute bias!
            upright_calm = (
                float(np.linalg.norm(u_torso_h)) < np.radians(16.0)
                and float(np.linalg.norm(u_pelvis_h)) < np.radians(36.0)
                and w_xy_mag < 0.7
            )
            if self._cond_frames == 0 or upright_calm:
                alpha = max(1.0 / (self._cond_frames + 1), min(0.25, dt_src / max(self.tilt_ema_tau_s, 0.2)))
                if self._cond_frames >= 18:
                    alpha = min(alpha, dt_src / max(self.tilt_ema_tau_s, 0.2))
                new_t = (1.0 - alpha) * self._torso_bias_xy + alpha * u_torso_h
                new_p = (1.0 - alpha) * self._pelvis_rel_bias_xy + alpha * (u_pelvis_h - new_t)
                t_nrm = float(np.linalg.norm(new_t))
                if t_nrm > self.max_tilt_bias_rad:
                    new_t = new_t * (self.max_tilt_bias_rad / t_nrm)
                p_nrm = float(np.linalg.norm(new_p))
                if p_nrm > self.max_tilt_bias_rad:
                    new_p = new_p * (self.max_tilt_bias_rad / p_nrm)
                self._torso_bias_xy = new_t
                self._pelvis_rel_bias_xy = new_p

            # Rotate ONLY upper body (torso + arms) and pelvis orientation — keep GP's gravity-anchored legs unbent!
            R_un_torso_w = R_yaw @ so3_exp_np([-self._torso_bias_xy[0], -self._torso_bias_xy[1], 0.0]) @ hy
            pel_p = b[_PELVIS_BI : _PELVIS_BI + 1].copy()
            b[_UPPER_BODY_BIS] = pel_p + (b[_UPPER_BODY_BIS] - pel_p) @ R_un_torso_w.T
            rot = R_un_torso_w @ rot

            R_yaw_new = yaw_mat_np(rot)
            R_un_belt_h = so3_exp_np([-self._pelvis_rel_bias_xy[0], -self._pelvis_rel_bias_xy[1], 0.0])
            rot = R_yaw_new @ R_un_belt_h @ (R_yaw_new.T @ rot)

            # Soft-saturate extreme residual reference pelvis tilt so R_ref stays within balanceable envelope
            R_yaw_f = yaw_mat_np(rot)
            res_rv = so3_log_np(R_yaw_f.T @ rot).astype(np.float64)
            clamped_rx = self.max_pelvis_roll_rad * float(np.tanh(res_rv[0] / max(self.max_pelvis_roll_rad, 1e-4)))
            clamped_ry = self.max_pelvis_pitch_rad * float(np.tanh(res_rv[1] / max(self.max_pelvis_pitch_rad, 1e-4)))
            R_soft_h = so3_exp_np([clamped_rx, clamped_ry, float(res_rv[2])])
            rot = R_yaw_f @ R_soft_h

        if self.enable_vel_conditioning:
            raw_xy = b[_PELVIS_BI, :2].astype(np.float64)
            lank_xy = b[_L_ANKLE_BI, :2].astype(np.float64)
            rank_xy = b[_R_ANKLE_BI, :2].astype(np.float64)
            if not self._vel_init:
                self._vel_init = True
                self._prev_vel_stamp = stamp_ns
                self._raw_pelvis_xy = raw_xy.copy()
                self._cond_pelvis_xy = raw_xy.copy()
                self._prev_lank_xy = lank_xy.copy()
                self._prev_rank_xy = rank_xy.copy()
                self._v_xy_filt = np.zeros(2, dtype=np.float64)
            else:
                v_pel = (raw_xy - self._raw_pelvis_xy) / dt_src
                v_lank = (lank_xy - self._prev_lank_xy) / dt_src
                v_rank = (rank_xy - self._prev_rank_xy) / dt_src

                z_l = float(b[_L_ANKLE_BI, 2] - floor)
                z_r = float(b[_R_ANKLE_BI, 2] - floor)
                z_min = min(z_l, z_r)
                w_l = float(np.exp(-40.0 * (z_l - z_min)))
                w_r = float(np.exp(-40.0 * (z_r - z_min)))
                w_sum = w_l + w_r
                v_stance = (w_l * v_lank + w_r * v_rank) / max(w_sum, 1e-6)
                v_anc = v_pel - v_stance

                c_contact = float(np.clip((0.16 - z_min) / 0.08, 0.0, 1.0))
                beta = self.vel_stance_beta * c_contact
                v_mix = (1.0 - beta) * v_pel + beta * v_anc

                # Smooth static deadband removes micro-drift while keeping gentle starts linear
                spd = float(np.linalg.norm(v_mix))
                if spd <= self.vel_deadband_lo:
                    v_db = np.zeros(2, dtype=np.float64)
                elif spd < self.vel_deadband_hi:
                    gain = (spd - self.vel_deadband_lo) / max(self.vel_deadband_hi - self.vel_deadband_lo, 1e-5)
                    v_db = v_mix * gain
                else:
                    v_db = v_mix

                a_vel = 1.0 - float(np.exp(-dt_src / max(self.vel_ema_tau_s, 1e-3)))
                v_target = (1.0 - a_vel) * self._v_xy_filt + a_vel * v_db
                dv = v_target - self._v_xy_filt
                max_dv = self.vel_max_accel * dt_src
                dv_nrm = float(np.linalg.norm(dv))
                if dv_nrm > max_dv:
                    dv = dv * (max_dv / dv_nrm)
                self._v_xy_filt = self._v_xy_filt + dv
                self._cond_pelvis_xy = self._cond_pelvis_xy + self._v_xy_filt * dt_src

                self._prev_vel_stamp = stamp_ns
                self._raw_pelvis_xy = raw_xy.copy()
                self._prev_lank_xy = lank_xy.copy()
                self._prev_rank_xy = rank_xy.copy()

            shift_xy = (self._cond_pelvis_xy - raw_xy).astype(np.float32)
            b[:, 0] += shift_xy[0]
            b[:, 1] += shift_xy[1]

        self._cond_frames += 1
        return rot.astype(np.float32), b.astype(np.float32)

    def push_sample(
        self,
        sample: dict,
        *,
        arrival_ns: int | None = None,
        allow_sequence_skip: bool = False,
    ) -> list[SMPLTick]:
        try:
            session, generation = sample["session_id"], sample["calibration_id"]
            sequence, stamp = sample["sequence"], sample["source_ns"]
            if (
                not isinstance(session, str)
                or not session
                or any(type(value) is not int for value in (generation, sequence, stamp))
                or generation < 0
                or min(sequence, stamp) <= 0
            ):
                raise ValueError("Invalid FGP frame identity or timestamp")
            identity = session, generation
            if arrival_ns is not None and self.last_arrival_ns is not None:
                if arrival_ns - self.last_arrival_ns > self.timeout_ns:
                    self.invalidate("arrival_timeout")
            if identity == self.identity:
                if sequence <= self.last_sequence:
                    return []
                if stamp <= self.last_source_ns:
                    raise ValueError("Non-increasing FGP source timestamp")
            joints = np.asarray(sample["joints_m"], dtype=np.float32)
            pose = np.asarray(sample["pose_mat"], dtype=np.float32)
            floor = float(sample["floor_y_m"])
            if joints.shape != (24, 3) or pose.shape != (24, 3, 3):
                raise ValueError("Unexpected SMPL24 shape")
            root = pose[0]
            if not (np.isfinite(joints).all() and np.isfinite(root).all() and np.isfinite(floor)):
                raise ValueError("Nonfinite SMPL data")
            if np.max(np.abs(root.T @ root - np.eye(3))) > 0.02 or np.linalg.det(root) < 0.98:
                raise ValueError("Invalid pelvis rotation")
            body = joints[SMPL_TRACKED] @ FGP_TO_Q1.T
            if any(np.linalg.norm(body[b] - body[a]) < 1e-4 for a, b in BONE_PAIRS):
                raise ValueError("Degenerate SMPL bone")
            height = float(body[0, 2] - floor)
            if height <= 0:
                raise ValueError("Invalid standing pelvis height")
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            self.invalidate("invalid_frame")
            self.events["rejected_frames"] += 1
            return []

        if identity != self.identity:
            self.invalidate("session_or_calibration_changed")
            self.identity = identity
            self.height = self.fixed_height if self.fixed_height is not None else height
        elif (
            (not allow_sequence_skip and sequence != self.last_sequence + 1)
            or stamp - self.last_source_ns > self.timeout_ns
        ):
            self.invalidate("source_gap")
        self.last_sequence, self.last_source_ns = sequence, stamp
        self.last_arrival_ns = arrival_ns
        self.events["accepted_frames"] += 1

        rotation = FGP_TO_Q1 @ root @ FGP_TO_Q1.T
        if self.enable_tilt_debias or self.enable_vel_conditioning:
            rotation, body = self._condition_sample(rotation, body, floor, stamp)

        if self.previous is None:
            self.segment += 1
            self.next_tick_ns = stamp
            self.origin = np.array([body[0, 0], body[0, 1], floor], dtype=np.float32)
            self.heading = yaw_mat_np(rotation).T
        body = (body - self.origin) @ self.heading.T
        quat = mat_to_quat_wxyz(self.heading @ rotation)
        old_stamp, old_body, old_quat = self.previous or (stamp - self.period_ns, body, quat)
        self.previous = stamp, body, quat

        ticks = []
        steps_emitted = 0
        while self.next_tick_ns <= stamp and steps_emitted < self.max_interp_steps:
            tick_ns = self.next_tick_ns
            self.next_tick_ns += self.period_ns
            weight = (tick_ns - old_stamp) / (stamp - old_stamp)
            body_at_tick = old_body + weight * (body - old_body)
            root_at_tick = interpolate_rotation(old_quat, quat, weight)
            self.history.append((body_at_tick, root_at_tick))
            steps_emitted += 1
            if len(self.history) == self.history.maxlen:
                bodies, rotations = map(np.stack, zip(*self.history))
                feature = build_human_smpl_430_np(
                    bodies[:, 0], rotations, bodies, len(bodies) - 1, dt=self.dt, h_nom=self.height,
                )
                root_xy = body_at_tick[0, :2] * (Q1_NOMINAL_PELVIS_Z / self.height)
                ticks.append(SMPLTick(self.segment, tick_ns, feature, root_xy, root_at_tick))
        if self.next_tick_ns <= stamp:
            self.next_tick_ns = stamp + self.period_ns
        return ticks
