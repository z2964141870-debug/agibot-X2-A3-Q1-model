"""Fall, autonomous rise and stand, then exclusive return of Q1 teleoperation."""

from math import isfinite
import numpy as np
from leg_height import LegHeight
from q1_recovery_runtime import Q1RecoveryBranch
from q1_v2b_runtime import Q1V2bOnnxController
from q1_kinematics_np import Q1KinematicsNP, yaw_mat_np
from q1_cgfa_obs_np import (
    ContactGatedFootAnchorNP,
    SOLE_LOCAL_OFFSET,
    build_obs_80_np,
    build_robot_pose_590_np,
)


class Q1AutonomousRecovery:
    """Call at 50 Hz with monotonic sample and operator-reference timestamps in seconds.

    Outside recovery the base callback retains its normal stream-loss behavior.
    During recovery it is suspended until standing and fresh input are both present.
    """

    def __init__(self, tracker, expert_onnx, q1_xml):
        self.tracker = tracker
        self.stand = Q1V2bOnnxController(tracker.root, ("robot_pose",))
        self.rise = Q1RecoveryBranch(self.stand, expert_onnx, q1_xml)
        self.kin = Q1KinematicsNP(q1_xml)
        self.est = ContactGatedFootAnchorNP(self.kin)
        self.xy = np.zeros(2, np.float32)
        self.zero = np.zeros(3, np.float32)
        self.modal = build_robot_pose_590_np(
            self.zero[None], np.eye(3, dtype=np.float32)[None], self.stand.default_q[None], 0
        )
        self.reset()

    def reset(self):
        self.owner = "teleop"
        self.phase = "teleop"
        self.standing_confirmed = False
        self.since = self.entered = self.last = None
        self.fall_count = 0
        self.rise.reset()
        self.hold_R = None

    def _quiet_feet(self, R, w, q, qd):
        _, rot, vel, omega = self.kin.forward_single(self.zero, R, self.zero, R @ w, q, qd)
        idx = [self.kin.left_ankle_tracked_idx, self.kin.right_ankle_tracked_idx]
        sole_vel = vel[idx] + np.cross(
            omega[idx], (rot[idx] @ SOLE_LOCAL_OFFSET[:, None]).squeeze(-1)
        )
        # With planted soles, root-relative sole velocity bounds pelvis translation.
        return np.linalg.norm(sole_vel[:, :2], axis=1).max() < 0.15

    def step(
        self,
        R_imu,
        omega_body,
        q,
        qd,
        previous_sent_action,
        base_step,
        reset_anchor,
        *,
        sample_time_s,
        teleop_stamp_s,
    ):
        now = sample_time_s
        R, w = R_imu, omega_body
        if (
            not isfinite(now)
            or (self.last is not None and now <= self.last)
            or not all(np.isfinite(x).all() for x in (R, w, q, qd, previous_sent_action))
        ):
            self.since = None
            self.standing_confirmed = False
            self.fall_count = 0
            raise ValueError("expected finite proprioception and increasing sample time")
        if self.last is not None and now - self.last > 0.035:
            self.since = None
            self.fall_count = 0
        self.last = now
        fresh = (
            teleop_stamp_s is not None
            and isfinite(teleop_stamp_s)
            and 0 <= now - teleop_stamp_s <= 0.1 + 1e-9
        )
        height, valid = self.rise.legs.measure(R[2], q)
        down = R[2, 2] < 0.65 or (valid and height < 0.19)
        if self.owner == "teleop":
            self.fall_count = self.fall_count + 1 if down else 0
            if self.fall_count < 3 and not (down and not fresh):
                target, tau, _ = base_step()
                return target, tau, 0.0
            self.owner = "recovery"
            self.entered = now
            self.rise.reset()
            self.rise.expert.gate.mode = "recover"
            self.hold_R = None

        def anchor_stand():
            self.hold_R = yaw_mat_np(R)
            self.est.reset(R, w, q, qd, self.xy, R_ref=self.hold_R)

        def autonomous_stand():
            exy, _ = self.est.step(R, w, q, qd, self.xy, R_ref=self.hold_R)
            obs = build_obs_80_np(
                q=q,
                qd=qd,
                default_q=self.stand.default_q,
                R_imu=R,
                w_body=w,
                a_prev=previous_sent_action,
                R_ref=self.hold_R,
                cgfa_exy=exy,
            )
            return self.stand.step(obs, self.modal, q, qd)

        target, tau, internal = self.rise.step(
            R, w, q, qd, previous_sent_action, autonomous_stand, anchor_stand
        )
        quiet = (
            internal == 0
            and valid
            and height > 0.36
            and R[2, 2] > 0.95
            and np.linalg.norm(w) < 0.7
            and np.sqrt(np.mean(qd**2)) < 0.85
            and self._quiet_feet(R, w, q, qd)
        )
        if not quiet:
            self.since = None
        elif self.since is None:
            self.since = now
        self.standing_confirmed = self.since is not None and now - self.since >= 0.5 - 1e-9
        self.phase = "rise" if internal > 0 else ("ready" if self.standing_confirmed else "settle")
        if self.standing_confirmed and fresh and teleop_stamp_s >= self.entered:
            # Both callbacks must finish before publishing the new owner.
            self.tracker.reset(q)
            reset_anchor()
            target, tau, _ = base_step()
            self.owner = self.phase = "teleop"
            self.since = self.entered = None
            self.fall_count = 0
            self.standing_confirmed = False
            return target, tau, 0.0
        return target, tau, 1.0
