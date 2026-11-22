"""Convert the FGP SMPL24 stream to the Pure V2b human_smpl input."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from q1_cgfa_obs_np import BONE_PAIRS, Q1_NOMINAL_PELVIS_Z, build_human_smpl_430_np
from q1_kinematics_np import mat_to_quat_wxyz, quat_wxyz_to_mat, yaw_mat_np


# FGP SMPL axes are +Z forward, +X left and +Y up. Q1 uses forward/left/up.
FGP_TO_Q1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
# In q1_kinematics_np.TRACKED_BODIES order. Spine3 represents upper torso.
SMPL_TRACKED = [0, 1, 4, 7, 2, 5, 8, 9, 16, 18, 20, 17, 19, 21]


@dataclass
class SMPLTick:
    segment: int
    stamp_ns: int
    feature: np.ndarray
    root_xy: np.ndarray
    root_rotation: np.ndarray


def interpolate_rotation(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return quat_wxyz_to_mat(a + weight * (b - a))
    angle = np.arccos(np.clip(dot, 0.0, 1.0))
    q = (np.sin((1.0 - weight) * angle) * a + np.sin(weight * angle) * b) / np.sin(angle)
    return quat_wxyz_to_mat(q)


class FGPSMPLStream:
    """Resample at source-time 50 Hz using only frames already received."""

    def __init__(self, dt: float, *, standing_height_m: float | None = None):
        if standing_height_m is not None and (
            not np.isfinite(standing_height_m) or standing_height_m <= 0
        ):
            raise ValueError("Standing pelvis height must be positive and finite")
        self.dt = dt
        self.period_ns = round(dt * 1e9)
        self.timeout_ns = 150_000_000
        self.fixed_height = standing_height_m
        self.height = standing_height_m
        self.identity: tuple[str, int] | None = None
        self.last_sequence = 0
        self.last_source_ns = 0
        self.last_arrival_ns: int | None = None
        self.segment = 0
        self.events: Counter[str] = Counter()
        self.history = deque(maxlen=11)
        self.previous = None

    def invalidate(self, reason: str) -> None:
        if self.previous is not None:
            self.events[reason] += 1
        self.previous = None
        self.history.clear()

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
        if response.get("status") != "tracking" or response.get("realtime_valid") is not True:
            self.invalidate("tracking_lost")
            return []
        if not isinstance(sample, dict) or any(
            response.get(key) != sample.get(key) for key in ("session_id", "calibration_id")
        ):
            self.invalidate("invalid_frame")
            return []
        return self.push_sample(
            sample,
            arrival_ns=arrival_ns,
            allow_sequence_skip=allow_sequence_skip,
        )

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
            if not isinstance(session, str) or not session or any(
                type(value) is not int for value in (generation, sequence, stamp)
            ) or generation < 0 or min(sequence, stamp) <= 0:
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
            height = body[0, 2] - floor
            if height <= 0:
                raise ValueError("Invalid standing pelvis height")
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            self.invalidate("invalid_frame")
            self.events["rejected_frames"] += 1
            return []

        if identity != self.identity:
            self.invalidate("session_or_calibration_changed")
            self.identity = identity
            self.height = self.fixed_height if self.fixed_height is not None else float(height)
        elif (
            (not allow_sequence_skip and sequence != self.last_sequence + 1)
            or stamp - self.last_source_ns > self.timeout_ns
        ):
            self.invalidate("source_gap")
        self.last_sequence, self.last_source_ns = sequence, stamp
        self.last_arrival_ns = arrival_ns
        self.events["accepted_frames"] += 1

        rotation = FGP_TO_Q1 @ root @ FGP_TO_Q1.T
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
        while self.next_tick_ns <= stamp:
            tick_ns = self.next_tick_ns
            self.next_tick_ns += self.period_ns
            weight = (tick_ns - old_stamp) / (stamp - old_stamp)
            body_at_tick = old_body + weight * (body - old_body)
            root_at_tick = interpolate_rotation(old_quat, quat, weight)
            self.history.append((body_at_tick, root_at_tick))
            if len(self.history) == self.history.maxlen:
                bodies, rotations = map(np.stack, zip(*self.history))
                feature = build_human_smpl_430_np(
                    bodies[:, 0], rotations, bodies, len(bodies) - 1, dt=self.dt, h_nom=self.height,
                )
                root_xy = body_at_tick[0, :2] * (Q1_NOMINAL_PELVIS_Z / self.height)
                ticks.append(SMPLTick(self.segment, tick_ns, feature, root_xy, root_at_tick))
        return ticks
