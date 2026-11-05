"""Bounded timestamped human-pose input for controlled 50 Hz preview studies.

All times use one explicitly identified receiver/replay clock. This module does
not synchronize sensor clocks or certify that upstream pose estimation is causal.
An unavailable/stale window requests a caller fallback, never an all-zero goal.
"""

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Optional


DT_NS = 20_000_000
SLOTS = 10
POINTS = 24


def nonnegative_ns(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative integer in nanoseconds")
    return value


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    session_id: str
    clock_id: str
    pose_schema: str
    preprocessing: str
    content_sha256: Optional[str] = None

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in
                   (self.source_id, self.session_id, self.clock_id)):
            raise ValueError("Source, session and common clock identities are required")
        if self.pose_schema != "smpl24_world_m_wxyz":
            raise ValueError("Require explicit 24-joint world pose units and quaternion order")
        if self.preprocessing not in ("declared_causal", "offline_unverified"):
            raise ValueError("Preprocessing causality must be declared, not inferred")
        digest = self.content_sha256
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid optional source content SHA256")


@dataclass(frozen=True)
class HumanPose:
    positions: tuple
    orientations: tuple

    def __post_init__(self):
        try:
            positions = tuple(tuple(float(x) for x in row) for row in self.positions)
            orientations = tuple(tuple(float(x) for x in row) for row in self.orientations)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid human pose arrays") from error
        if len(positions) != POINTS or len(orientations) != POINTS:
            raise ValueError("Human poses must contain exactly 24 points")
        if any(len(row) != 3 for row in positions) or any(len(row) != 4 for row in orientations):
            raise ValueError("Human position/quaternion shapes differ from [24,3]/[24,4]")
        if not all(math.isfinite(x) for row in positions + orientations for x in row):
            raise ValueError("Nonfinite human pose")
        if any(abs(sum(x*x for x in row) - 1.) > 1e-4 for row in orientations):
            raise ValueError("Human quaternions must be normalized wxyz values")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "orientations", orientations)


@dataclass(frozen=True)
class PoseFrame:
    source: SourceIdentity
    sequence: int
    capture_ns: int
    dependency_end_ns: Optional[int]
    ready_ns: int
    arrival_ns: int
    pose: HumanPose
    terminal: bool = False

    def __post_init__(self):
        if not isinstance(self.source, SourceIdentity) or not isinstance(self.pose, HumanPose):
            raise ValueError("Frame requires validated source and immutable human pose")
        nonnegative_ns(self.sequence, "Sequence")
        names = ("capture_ns", "dependency_end_ns", "ready_ns", "arrival_ns")
        if self.dependency_end_ns is None:
            if self.source.preprocessing != "offline_unverified":
                raise ValueError("Declared causal preprocessing needs its dependency cutoff")
            names = ("capture_ns", "ready_ns", "arrival_ns")
        times = tuple(nonnegative_ns(getattr(self, name), name) for name in names)
        if times != tuple(sorted(times)):
            raise ValueError("Require capture <= last dependency <= ready <= arrival on one clock")
        if self.source.preprocessing == "declared_causal" and self.dependency_end_ns != self.capture_ns:
            raise ValueError("Declared causal pose cannot depend on a later capture")
        if type(self.terminal) is not bool or (self.terminal and
                self.source.preprocessing != "offline_unverified"):
            raise ValueError("Explicit clip terminal holds are restricted to offline replay")

    def same_sample(self, other):
        return (self.source, self.sequence, self.capture_ns, self.dependency_end_ns,
                self.ready_ns, self.pose, self.terminal) == (other.source, other.sequence, other.capture_ns,
                    other.dependency_end_ns, other.ready_ns, other.pose, other.terminal)


@dataclass(frozen=True)
class PreviewWindow:
    source: SourceIdentity
    origin_ns: int
    control_ns: int
    capture_cutoff_ns: int
    target_ns: tuple
    poses: tuple
    measured_mask: tuple
    policy_frame_mask: tuple
    provenance: tuple
    used_sequences: tuple
    prefix_frames: int
    requested_frames: int
    status: str
    can_step: bool
    newest_capture_ns: Optional[int]
    newest_arrival_ns: Optional[int]
    newest_dependency_ns: Optional[int]
    latest_used_capture_ns: Optional[int]
    max_used_dependency_ns: Optional[int]
    terminal_capture_ns: Optional[int] = None

    def timing(self):
        return dict(
            requested_preview_ns=self.requested_frames * DT_NS,
            scheduled_origin_to_control_ns=self.control_ns - self.origin_ns,
            latest_capture_age_ns=None if self.newest_capture_ns is None
            else self.control_ns - self.newest_capture_ns,
            latest_arrival_age_ns=None if self.newest_arrival_ns is None
            else self.control_ns - self.newest_arrival_ns,
            latest_declared_dependency_ns=self.newest_dependency_ns,
            latest_used_capture_age_ns=None if self.latest_used_capture_ns is None
            else self.control_ns - self.latest_used_capture_ns,
            max_used_dependency_ns=self.max_used_dependency_ns,
            terminal_capture_ns=self.terminal_capture_ns,
            observed_prefix_frames=sum(self.measured_mask),
            policy_prefix_frames=self.prefix_frames,
            source_preprocessing=self.source.preprocessing,
            measured_end_to_end_latency_ns=None,
        )


class PoseTimeline:
    def __init__(self, source, capacity=256):
        if not isinstance(source, SourceIdentity) or type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("Require a source and a bounded 1..4096 frame capacity")
        self.source, self.capacity = source, capacity
        self.frames, self.by_sequence = [], {}
        self.terminal_frame = None
        self.last_event_ns, self.high_sequence = -1, -1
        self.counters = dict(received=0, duplicates=0, reordered=0, overflow_evictions=0,
                             too_old_drops=0)

    def _advance(self, timestamp_ns):
        nonnegative_ns(timestamp_ns, "Event time")
        if timestamp_ns < self.last_event_ns:
            raise ValueError("Cannot rewind receiver time; do not preload future arrivals")
        self.last_event_ns = timestamp_ns

    def ingest(self, frame):
        if not isinstance(frame, PoseFrame) or frame.source != self.source:
            raise ValueError("Wrong source/session/clock; create a new timeline for a new identity")
        if self.terminal_frame and frame.capture_ns > self.terminal_frame.capture_ns:
            raise ValueError("Pose follows a declared end of clip")
        if frame.terminal and self.frames and frame.capture_ns < self.frames[-1].capture_ns:
            raise ValueError("Declared clip end precedes an already received capture")
        self._advance(frame.arrival_ns)
        self.counters["received"] += 1
        previous = self.by_sequence.get(frame.sequence)
        if previous is not None:
            if not previous.same_sample(frame):
                raise ValueError("Conflicting payload or capture for the same source sequence")
            self.counters["duplicates"] += 1
            return "DUPLICATE"
        times = [row.capture_ns for row in self.frames]
        index = bisect_left(times, frame.capture_ns)
        if index < len(times) and times[index] == frame.capture_ns:
            raise ValueError("Two distinct source sequences claim one capture timestamp")
        if len(self.frames) == self.capacity and index == 0:
            self.counters["too_old_drops"] += 1
            return "TOO_OLD"
        if frame.sequence < self.high_sequence:
            self.counters["reordered"] += 1
        self.high_sequence = max(self.high_sequence, frame.sequence)
        self.frames.insert(index, frame)
        self.by_sequence[frame.sequence] = frame
        if frame.terminal:
            self.terminal_frame = frame
        if len(self.frames) > self.capacity:
            removed = self.frames.pop(0)
            del self.by_sequence[removed.sequence]
            self.counters["overflow_evictions"] += 1
        return "ACCEPTED"

    def window(self, *, origin_ns, control_ns, preview_frames, max_capture_age_ns):
        nonnegative_ns(origin_ns, "Reference origin")
        nonnegative_ns(max_capture_age_ns, "Maximum latest-capture age")
        if type(preview_frames) is not int or not 1 <= preview_frames <= SLOTS:
            raise ValueError("Human preview must contain 1..10 future slots")
        if type(control_ns) is not int or control_ns < origin_ns:
            raise ValueError("Control cannot precede its reference origin")
        self._advance(control_ns)
        cutoff = min(control_ns, origin_ns + preview_frames * DT_NS)
        visible = {row.capture_ns: row for row in self.frames if row.capture_ns <= cutoff}
        newest = max(visible.values(), key=lambda row: row.capture_ns) if visible else None
        targets = tuple(origin_ns + (i+1)*DT_NS for i in range(SLOTS))
        prefix, measured_prefix, prefix_provenance = [], [], []
        for target in targets[:preview_frames]:
            row = visible.get(target)
            terminal_hold = (row is None and self.terminal_frame is not None
                             and self.terminal_frame.capture_ns <= cutoff
                             and target > self.terminal_frame.capture_ns)
            if row is None and not terminal_hold:
                break
            prefix.append(self.terminal_frame if terminal_hold else row)
            measured_prefix.append(not terminal_hold)
            prefix_provenance.append("declared_clip_terminal_hold" if terminal_hold else "measured_exact")
        count = len(prefix)
        status = "READY" if count == preview_frames else "GAP_AFTER_PREFIX"
        if not count:
            status = "MISSING_FIRST_TARGET" if newest else "NO_VISIBLE_POSE"
        used_latest = prefix[-1] if prefix else newest
        if used_latest and control_ns - used_latest.capture_ns > max_capture_age_ns:
            status = "STALE"
        can_step = count > 0 and status != "STALE"
        # Held tails are packing placeholders only, always masked from the policy.
        poses = tuple(row.pose for row in prefix)
        poses += ((prefix[-1].pose,) * (SLOTS-count)) if prefix else ((None,) * SLOTS)
        measured = tuple(measured_prefix) + (False,) * (SLOTS-count)
        enabled = tuple(i < count for i in range(SLOTS)) if can_step else (False,) * SLOTS
        provenance = tuple(prefix_provenance) + (("masked_hold_placeholder",) * (SLOTS-count)
                                                if count else ("unavailable",) * SLOTS)
        dependency = None if not prefix or any(row.dependency_end_ns is None for row in prefix) \
            else max(row.dependency_end_ns for row in prefix)
        return PreviewWindow(self.source, origin_ns, control_ns, cutoff, targets, poses,
            measured, enabled, provenance, tuple(row.sequence for row in prefix), count,
            preview_frames, status, can_step,
            None if newest is None else newest.capture_ns,
            None if newest is None else newest.arrival_ns,
            None if newest is None else newest.dependency_end_ns,
            None if not prefix else prefix[-1].capture_ns,
            dependency,
            None if self.terminal_frame is None or self.terminal_frame.capture_ns > cutoff
            else self.terminal_frame.capture_ns)


def command_timing(window, *, policy_start_ns, policy_end_ns, command_sent_ns):
    """Software timing only; actual actuator/sensor measurement is deliberately absent."""
    if not isinstance(window, PreviewWindow) or not window.can_step:
        raise ValueError("Cannot time a policy command from an unavailable/stale goal")
    times = (window.control_ns,) + tuple(nonnegative_ns(value, label) for value, label in
        ((policy_start_ns, "Policy start"), (policy_end_ns, "Policy end"),
         (command_sent_ns, "Command sent")))
    if times != tuple(sorted(times)):
        raise ValueError("Require control <= policy start <= policy end <= command sent")
    return dict(window.timing(), policy_queue_ns=policy_start_ns-window.control_ns,
                policy_compute_ns=policy_end_ns-policy_start_ns,
                send_after_policy_ns=command_sent_ns-policy_end_ns,
                scheduled_origin_to_send_ns=command_sent_ns-window.origin_ns,
                physical_actuation_time_ns=None)
