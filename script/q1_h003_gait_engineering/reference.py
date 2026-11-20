"""Rebuild the causal reference window from a standing pose after recovery."""

from collections import deque
import numpy as np
from q1_cgfa_obs_np import build_robot_pose_590_np
from q1_kinematics_np import mat_to_quat_wxyz, quat_wxyz_to_mat


class HandoffReference:
    def __init__(self, anchor, yaw, stand, *, resume_ticks=30, translation_ticks=None):
        self.origin = anchor.copy()
        self.rotation = yaw.copy()
        self._anchor_quaternion = mat_to_quat_wxyz(self.rotation)
        self.stand = stand.copy()
        self.previous_raw = None
        self.position = anchor.copy()
        self.resume_ticks = resume_ticks
        self.translation_ticks = resume_ticks if translation_ticks is None else translation_ticks
        if min(self.resume_ticks, self.translation_ticks) <= 0:
            raise ValueError("reference ramp lengths must be positive")
        self.tick = 0
        self.frames = deque(maxlen=11)
        for _ in range(11):
            self.frames.append((anchor.copy(), yaw.copy(), stand.copy()))

    def step(self, raw_p, raw_R, raw_q):
        # Hold the neutral reference while authority is fading, then resume over the selected ramp.
        u = np.clip((self.tick - 29) / self.resume_ticks, 0.0, 1.0)
        blend = 0.5 - 0.5 * np.cos(np.pi * u)
        position_blend = blend
        if self.translation_ticks != self.resume_ticks:
            v = np.clip((self.tick - 29) / self.translation_ticks, 0.0, 1.0)
            position_blend = 0.5 - 0.5 * np.cos(np.pi * v)
        if self.previous_raw is not None:
            self.position += position_blend * (raw_p - self.previous_raw)
        self.previous_raw = raw_p.copy()
        a = self._anchor_quaternion
        b = mat_to_quat_wxyz(raw_R)
        if a @ b < 0:
            b = -b
        R = quat_wxyz_to_mat((1 - blend) * a + blend * b)
        q = (1 - blend) * self.stand + blend * raw_q
        self.frames.append((self.position.copy(), R, q))
        pstack, rstack, qstack = (np.stack(z) for z in zip(*self.frames))
        modal = build_robot_pose_590_np(pstack, rstack, qstack, len(pstack) - 1)
        self.tick += 1
        return self.position.copy(), R, q, modal
