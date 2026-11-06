"""Frozen-corpus inputs with explicit 120 -> 50 Hz interpolation dependencies.

The synthetic clock models stored source availability, not a measured headset.
The teacher-label method is separate and must never feed the student encoder.
"""

from dataclasses import dataclass

import numpy as np

from forecast_data import ForecastCorpus, source_support
from forecast_geometry import checked_poses
from stream_contract import DT_NS, HumanPose, PoseFrame, SourceIdentity
from stream_events import EventSpec, EventTimeline
from student_inputs import current_history, short_preview


@dataclass(frozen=True)
class TeacherPoseLabels:
    positions: np.ndarray
    quaternions: np.ndarray
    target_ns: tuple
    latest_source_dependency_ns: int
    privileged_training_labels_only: bool = True


class OfflineStudentStream:
    def __init__(self, positions, quaternions, *, spec, sha256, session):
        self.positions, self.quaternions = checked_poses(positions, quaternions)
        self.events = EventTimeline(len(self.positions), spec)
        self.source = SourceIdentity("frozen_offline_student:"+spec.kind, session,
            "synthetic_50hz_with_120hz_support_ns", "smpl24_world_m_wxyz",
            "offline_unverified", sha256)

    def packet_frame(self, packet):
        support = source_support(packet.source_index)
        # A reversed/held old pose is already known at its new synthetic capture.
        dependency = max(packet.capture_ns, support["dependency_end_ns"])
        ready = dependency
        return PoseFrame(self.source, packet.sequence, packet.capture_ns, dependency,
            ready, max(packet.arrival_ns, ready),
            HumanPose(self.positions[packet.source_index], self.quaternions[packet.source_index]))

    def student(self, anchor, *, mode, horizon=None, max_age_frames=10):
        if (type(anchor) is not int or not 0 <= anchor < self.events.frames
                or type(max_age_frames) is not int or not 0 <= max_age_frames <= 50):
            raise ValueError("Require an in-episode anchor and bounded capture-age budget")
        if mode not in ("short_preview", "f0_history"):
            raise ValueError("Unknown student interface")
        if mode == "f0_history" and horizon is not None:
            raise ValueError("f0 has no positive preview horizon")
        extra = 0 if mode == "f0_history" else horizon
        if type(extra) is not int or not 0 <= extra <= 10:
            raise ValueError("Require an explicit bounded preview horizon")
        # Only the local receiver prefix is constructed; full future poses stay out.
        first, last = max(0, anchor-9-max_age_frames), min(self.events.frames-1, anchor+extra)
        rows = [self.packet_frame(packet) for packet in self.events.packets
                if first <= packet.sequence <= last]
        options = dict(anchor_ns=anchor*DT_NS, control_ns=(anchor+extra)*DT_NS,
                       max_capture_age_ns=max_age_frames*DT_NS, episode_start_ns=0)
        if mode == "f0_history":
            return current_history(rows, self.source, **options)
        return short_preview(rows, self.source, horizon=horizon, **options)

    def teacher_labels(self, anchor):
        if type(anchor) is not int or not 0 <= anchor < self.events.frames-10:
            raise ValueError("All teacher future labels must lie inside the same clip")
        sequences = tuple(range(anchor+1, anchor+11))
        indices = [self.events.indices[i] for i in sequences]
        arrays = [values[indices].copy() for values in (self.positions, self.quaternions)]
        for values in arrays:
            values.setflags(write=False)
        dependency = max(max(i*DT_NS, source_support(index)["dependency_end_ns"])
                         for i, index in zip(sequences, indices))
        return TeacherPoseLabels(*arrays, tuple(i*DT_NS for i in sequences), dependency)


class StudentCorpus:
    def __init__(self, plan, human_root):
        self.corpus = ForecastCorpus(plan, human_root)

    def stream(self, relative, *, split, spec, session):
        row = self.corpus.rows[relative]
        if split not in ("train", "validation", "test") or row["split"] != split:
            raise ValueError("The source split owns this complete episode")
        positions, rotations = self.corpus.load(relative)
        return OfflineStudentStream(positions, rotations, spec=spec,
            sha256=row["human"]["sha256"], session=session)
