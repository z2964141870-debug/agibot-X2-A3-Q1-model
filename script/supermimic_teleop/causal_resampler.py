"""Arrival-clock resampling for irregular poses, separate from preview studies.

Payload time is preserved, never subtracted from receiver time. Without clock
calibration the target denotes a receiver-arrival timeline, not physical capture.
No result from this module is a measured future slot in the H4/H5 experiment.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from causal_tail import multiply, rotation_quaternion, rotation_vector, unit
from stream_contract import HumanPose, nonnegative_ns


@dataclass(frozen=True)
class IntakeSource:
    source_id: str
    session_id: str
    payload_clock_id: str
    receiver_clock_id: str
    preprocessing: str = "unverified"
    payload_stamp_semantics: str = "payload_time_not_verified_as_capture"

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip() for value in (
                self.source_id, self.session_id, self.payload_clock_id,
                self.receiver_clock_id, self.payload_stamp_semantics)):
            raise ValueError("Source, session, two clock identities and stamp semantics are required")
        if self.preprocessing not in ("unverified", "declared_causal"):
            raise ValueError("Upstream causality must be explicit")


@dataclass(frozen=True)
class ArrivedPose:
    source: IntakeSource
    sequence: int
    payload_ns: int
    arrival_ns: int
    pose: HumanPose
    payload_dependency_end_ns: Optional[int] = None
    receiver_ready_ns: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.source, IntakeSource) or not isinstance(self.pose, HumanPose):
            raise ValueError("Require a validated source and immutable SMPL24 world pose")
        for name in ("sequence", "payload_ns", "arrival_ns"):
            nonnegative_ns(getattr(self, name), name)
        dependency = self.payload_dependency_end_ns
        if dependency is not None:
            nonnegative_ns(dependency, "Payload dependency")
            if dependency < self.payload_ns:
                raise ValueError("Dependency cannot precede the pose stamp on the payload clock")
        if self.source.preprocessing == "declared_causal" and dependency != self.payload_ns:
            raise ValueError("Declared causal preprocessing needs its exact payload dependency")
        if self.receiver_ready_ns is not None:
            nonnegative_ns(self.receiver_ready_ns, "Receiver ready")
            if self.receiver_ready_ns > self.arrival_ns:
                raise ValueError("Receiver-ready time cannot follow arrival")

    def same_sample(self, other):
        return (self.source, self.sequence, self.payload_ns, self.pose,
                self.payload_dependency_end_ns, self.receiver_ready_ns) == (
                other.source, other.sequence, other.payload_ns, other.pose,
                other.payload_dependency_end_ns, other.receiver_ready_ns)


@dataclass(frozen=True)
class ResampledPose:
    source: IntakeSource
    target_ns: int
    control_ns: int
    pose: Optional[HumanPose]
    supports: tuple
    provenance: str
    status: str
    can_step: bool
    generated: bool = False
    speed_cap_applied: bool = False

    def timing(self):
        arrivals = [row.arrival_ns for row in self.supports]
        latest = max(arrivals) if arrivals else None
        return dict(target_clock_id=self.source.receiver_clock_id,
            target_semantics="receiver_arrival_timeline_not_physical_capture",
            target_ns=self.target_ns, control_ns=self.control_ns,
            scheduled_buffer_ns=self.control_ns-self.target_ns,
            used_sequences=[row.sequence for row in self.supports],
            used_arrival_ns=arrivals, used_payload_ns=[row.payload_ns for row in self.supports],
            payload_clock_id=self.source.payload_clock_id,
            payload_stamp_semantics=self.source.payload_stamp_semantics,
            used_payload_dependency_end_ns=[row.payload_dependency_end_ns for row in self.supports],
            used_receiver_ready_ns=[row.receiver_ready_ns for row in self.supports],
            latest_support_arrival_age_ns=None if latest is None else self.control_ns-latest,
            interpolation_right_support_ns=None if latest is None else max(0, latest-self.target_ns),
            locally_causal=all(value <= self.control_ns for value in arrivals),
            upstream_preprocessing=self.source.preprocessing,
            is_exact_measured_capture=False, provenance=self.provenance,
            generated=self.generated, speed_cap_applied=self.speed_cap_applied,
            can_step=self.can_step, status=self.status,
            measured_end_to_end_latency_ns=None)


def pose_arrays(pose):
    return np.asarray(pose.positions), pose.orientations


def relative_rotations(anchor, values):
    return np.asarray([rotation_vector(multiply((q[0], -q[1], -q[2], -q[3]), value))
                       for q, value in zip(anchor, values)])


def rotate(anchor, vectors):
    return [unit(multiply(q, rotation_quaternion(vector))) for q, vector in zip(anchor, vectors)]


def interpolate(left, right, fraction):
    position, rotation = pose_arrays(left)
    next_position, next_rotation = pose_arrays(right)
    delta = relative_rotations(rotation, next_rotation)
    orientation = rotate(rotation, delta*fraction)
    return HumanPose(position+(next_position-position)*fraction, orientation)


def extrapolate(history, horizon_ns, method, tau_s, linear_cap, angular_cap):
    anchor = history[-1]
    position, rotation = pose_arrays(anchor.pose)
    times = np.asarray([(row.arrival_ns-anchor.arrival_ns)/1e9 for row in history])
    denominator = float(times@times)
    if denominator <= 0:
        raise ValueError("Velocity estimation needs distinct receiver arrival times")
    positions = np.asarray([row.pose.positions for row in history])
    relative = np.asarray([relative_rotations(rotation, row.pose.orientations)
                           for row in history])
    linear = np.sum(times[:, None, None]*(positions-position), axis=0)/denominator
    angular = np.sum(times[:, None, None]*relative, axis=0)/denominator
    capped = False
    for velocity, maximum in ((linear, linear_cap), (angular, angular_cap)):
        lengths = np.linalg.norm(velocity, axis=-1, keepdims=True)
        factors = np.minimum(1., maximum/np.maximum(lengths, 1e-15))
        capped = capped or bool(np.any(factors < 1.))
        velocity *= factors
    horizon = horizon_ns/1e9
    travel = tau_s*(-math.expm1(-horizon/tau_s)) if method == "damped_cv" else horizon
    orientation = rotate(rotation, angular*travel)
    return HumanPose(position+linear*travel, orientation), capped


class CausalResampler:
    def __init__(self, source, *, capacity=256, history_frames=5,
                 max_arrival_age_ns=200_000_000, max_extrapolation_ns=80_000_000,
                 max_support_gap_ns=200_000_000, tau_s=.12,
                 max_linear_speed_m_s=3., max_angular_speed_rad_s=8.):
        if not isinstance(source, IntakeSource):
            raise ValueError("Require an intake source with separate clock identities")
        if (type(capacity) is not int or not 2 <= capacity <= 4096
                or type(history_frames) is not int or not 2 <= history_frames <= capacity):
            raise ValueError("Require bounded capacity and a 2..capacity frame history")
        for name, value in (("arrival age", max_arrival_age_ns),
                            ("extrapolation", max_extrapolation_ns), ("support gap", max_support_gap_ns)):
            nonnegative_ns(value, name)
            if value == 0:
                raise ValueError("Timing budgets must be positive")
        for value in (tau_s, max_linear_speed_m_s, max_angular_speed_rad_s):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("Damping and speed caps must be finite and positive")
        self.source, self.capacity, self.history_frames = source, capacity, history_frames
        self.max_age, self.max_extrapolation = max_arrival_age_ns, max_extrapolation_ns
        self.max_gap, self.tau_s = max_support_gap_ns, tau_s
        self.linear_cap, self.angular_cap = max_linear_speed_m_s, max_angular_speed_rad_s
        self.frames, self.by_sequence = [], {}
        self.last_event_ns = -1
        self.counters = dict(received=0, duplicates=0, reordered_drops=0, evicted=0)

    def _advance(self, now_ns):
        nonnegative_ns(now_ns, "Receiver event")
        if now_ns < self.last_event_ns:
            raise ValueError("Cannot rewind receiver time or preload future arrivals")
        self.last_event_ns = now_ns

    def ingest(self, frame):
        if not isinstance(frame, ArrivedPose) or frame.source != self.source:
            raise ValueError("Source/session/clock changed; use a new resampler")
        if frame.arrival_ns < self.last_event_ns:
            raise ValueError("Cannot rewind receiver time or preload future arrivals")
        previous = self.by_sequence.get(frame.sequence)
        if previous is not None and not previous.same_sample(frame):
            raise ValueError("Conflicting payload for one sequence")
        if (previous is None and self.frames and frame.sequence > self.frames[-1].sequence
                and frame.payload_ns <= self.frames[-1].payload_ns):
            raise ValueError("Payload clock discontinuity; require a new source session")
        self._advance(frame.arrival_ns)
        self.counters["received"] += 1
        if previous is not None:
            self.counters["duplicates"] += 1
            return "DUPLICATE"
        if self.frames and frame.sequence <= self.frames[-1].sequence:
            self.counters["reordered_drops"] += 1
            return "REORDERED_DROPPED"
        self.frames.append(frame)
        self.by_sequence[frame.sequence] = frame
        if len(self.frames) > self.capacity:
            removed = self.frames.pop(0)
            del self.by_sequence[removed.sequence]
            self.counters["evicted"] += 1
        return "ACCEPTED"

    def sample(self, *, target_ns, control_ns, receiver_clock_id, method="hold"):
        nonnegative_ns(target_ns, "Receiver target")
        nonnegative_ns(control_ns, "Receiver control")
        if receiver_clock_id != self.source.receiver_clock_id or target_ns > control_ns:
            raise ValueError("Require a same-receiver-clock target no later than control")
        if method not in ("hold", "delayed_linear", "cv", "damped_cv"):
            raise ValueError("Unknown causal resampling method")
        self._advance(control_ns)

        def result(pose, supports, provenance, status="READY", generated=False, capped=False):
            if supports and control_ns-max(row.arrival_ns for row in supports) > self.max_age:
                pose, status, generated = None, "STALE_SUPPORT", False
            return ResampledPose(self.source, target_ns, control_ns, pose, tuple(supports),
                provenance, status, pose is not None, generated, capped)

        arrivals = [row.arrival_ns for row in self.frames]
        index = bisect_right(arrivals, target_ns)-1
        if index < 0:
            return result(None, (), "unavailable", "NO_LEFT_SUPPORT")
        left = self.frames[index]
        if left.arrival_ns == target_ns:
            return result(left.pose, (left,), "receiver_arrival_exact")
        if method == "delayed_linear" and index+1 < len(self.frames):
            right = self.frames[index+1]
            if right.arrival_ns-left.arrival_ns > self.max_gap:
                return result(None, (left, right), "unavailable", "SUPPORT_GAP_LIMIT")
            fraction = (target_ns-left.arrival_ns)/(right.arrival_ns-left.arrival_ns)
            return result(interpolate(left.pose, right.pose, fraction), (left, right),
                          "arrived_linear_slerp")
        if method in ("hold", "delayed_linear"):
            return result(left.pose, (left,), "held_arrival_pose")
        horizon = target_ns-left.arrival_ns
        if horizon > self.max_extrapolation:
            return result(None, (left,), "unavailable", "EXTRAPOLATION_LIMIT")
        history = [left]
        for row in reversed(self.frames[max(0, index-self.history_frames+1):index]):
            if history[-1].arrival_ns-row.arrival_ns > self.max_gap:
                break
            if row.arrival_ns < history[-1].arrival_ns:
                history.append(row)
        history.reverse()
        if len(history) < 2:
            return result(left.pose, (left,), "held_arrival_pose", "INSUFFICIENT_HISTORY_HELD")
        pose, capped = extrapolate(history, horizon, method, self.tau_s,
                                  self.linear_cap, self.angular_cap)
        return result(pose, history, "arrival_clock_forecast:"+method,
                      generated=True, capped=capped)
