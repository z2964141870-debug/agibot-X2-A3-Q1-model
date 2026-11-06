"""Independent short-preview and f0/history inputs for teacher distillation.

All timestamps must already share SourceIdentity.clock_id. Selecting available
samples here does not certify an offline pose estimator or calibrate real clocks.
Teacher future labels are deliberately absent from this interface.
"""

from dataclasses import dataclass

import numpy as np

from forecast_geometry import encode_history, inverse, product, rotation_columns
from stream_contract import (DT_NS, PoseFrame, PoseTimeline, SourceIdentity,
                             nonnegative_ns)


HISTORY = 10
GEOMETRY_DIM = 223  # history geometry216, absolute root height1, relative torso6
FRAME_DIM = GEOMETRY_DIM + 3  # valid, exact sample, capture age in seconds
HORIZONS = (1, 2, 4, 5, 10)


def visible_frames(frames, source, *, cutoff_ns, control_ns, episode_start_ns):
    if not isinstance(source, SourceIdentity):
        raise ValueError("Require one explicit source/session/common clock")
    for name, value in (("cutoff", cutoff_ns), ("control", control_ns),
                        ("episode start", episode_start_ns)):
        nonnegative_ns(value, name)
    if not episode_start_ns <= cutoff_ns <= control_ns:
        raise ValueError("Require episode start <= dependency cutoff <= control")
    frames = tuple(frames)
    if len(frames) > 4096:
        raise ValueError("Pass a bounded receiver prefix, not an unbounded recording")
    if any(not isinstance(row, PoseFrame) or row.source != source for row in frames):
        raise ValueError("Mixed source/session/clock; reset at each episode")
    selected, sequences, captures = [], {}, {}
    for row in sorted(frames, key=lambda row: row.arrival_ns):
        if not episode_start_ns <= row.capture_ns <= cutoff_ns or row.arrival_ns > control_ns:
            continue
        if row.terminal:
            raise ValueError("Student training does not invent clip-terminal measurements")
        if row.dependency_end_ns is None:
            raise ValueError("Unknown preprocessing dependency; cannot certify student visibility")
        if row.dependency_end_ns > cutoff_ns:
            continue
        previous = sequences.get(row.sequence)
        if previous is not None:
            if not previous.same_sample(row):
                raise ValueError("Conflicting source sequence")
            continue
        if row.capture_ns in captures:
            raise ValueError("Multiple sequences claim one capture timestamp")
        selected.append(row)
        sequences[row.sequence], captures[row.capture_ns] = row, row
    return tuple(sorted(selected, key=lambda row: (row.arrival_ns, row.capture_ns)))


def short_preview(frames, source, *, anchor_ns, control_ns, horizon,
                  max_capture_age_ns, episode_start_ns):
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("Short preview needs H1/H2/H4/H5/H10; f0 is a separate interface")
    nonnegative_ns(anchor_ns, "Reference anchor")
    cutoff = min(control_ns, anchor_ns+horizon*DT_NS)
    if anchor_ns < episode_start_ns or control_ns < anchor_ns:
        raise ValueError("Reference anchor must be inside this episode before control")
    visible = visible_frames(frames, source, cutoff_ns=cutoff, control_ns=control_ns,
                             episode_start_ns=episode_start_ns)
    timeline = PoseTimeline(source, capacity=max(1, len(visible)))
    for row in visible:
        timeline.ingest(row)
    return timeline.window(origin_ns=anchor_ns, control_ns=control_ns,
        preview_frames=horizon, max_capture_age_ns=max_capture_age_ns)


@dataclass(frozen=True)
class CurrentHistory:
    source: SourceIdentity
    anchor_ns: int
    control_ns: int
    episode_start_ns: int
    targets_ns: tuple
    samples: tuple
    status: str
    can_step: bool

    def __post_init__(self):
        if (len(self.samples) != HISTORY or self.targets_ns != tuple(
                self.anchor_ns-(HISTORY-1-i)*DT_NS for i in range(HISTORY))
                or self.can_step != (self.status == "READY")):
            raise ValueError("Invalid current/history time grid or availability")
        for target, row in zip(self.targets_ns, self.samples):
            if row is not None and (row.source != self.source
                    or not self.episode_start_ns <= row.capture_ns <= target
                    or row.dependency_end_ns is None or row.dependency_end_ns > target
                    or row.arrival_ns > self.control_ns):
                raise ValueError("Current/history support crosses its visibility boundary")
        if self.can_step and self.samples[-1] is None:
            raise ValueError("Ready current/history has no current support")

    def timing(self):
        used = [row for row in self.samples if row is not None]
        return dict(interface="f0_plus_history_oldest_first", future_preview_ns=0,
            scheduled_anchor_to_control_ns=self.control_ns-self.anchor_ns,
            anchor_ns=self.anchor_ns, common_clock_id=self.source.clock_id,
            requested_history_ns=list(self.targets_ns),
            used_sequences=[None if row is None else row.sequence for row in self.samples],
            used_capture_ns=[None if row is None else row.capture_ns for row in self.samples],
            used_arrival_ns=[None if row is None else row.arrival_ns for row in self.samples],
            max_used_dependency_ns=max((row.dependency_end_ns for row in used), default=None),
            current_capture_age_ns=None if not used else self.control_ns-used[-1].capture_ns,
            upstream_preprocessing=self.source.preprocessing, status=self.status,
            measured_end_to_end_latency_ns=None)


def current_history(frames, source, *, anchor_ns, control_ns,
                    max_capture_age_ns, episode_start_ns):
    nonnegative_ns(max_capture_age_ns, "Maximum capture age")
    visible = visible_frames(frames, source, cutoff_ns=anchor_ns, control_ns=control_ns,
                             episode_start_ns=episode_start_ns)
    by_capture = sorted(visible, key=lambda row: row.capture_ns)
    targets = tuple(anchor_ns-(HISTORY-1-i)*DT_NS for i in range(HISTORY))
    samples = []
    for target in targets:
        eligible = [row for row in by_capture if row.capture_ns <= target
                    and row.dependency_end_ns <= target]
        row = eligible[-1] if eligible else None
        if row is not None and target-row.capture_ns > max_capture_age_ns:
            row = None
        samples.append(row)
    latest = samples[-1]
    status = "NO_CURRENT_SUPPORT" if latest is None else "READY"
    if latest is not None and control_ns-latest.capture_ns > max_capture_age_ns:
        status = "STALE_CURRENT_SUPPORT"
    return CurrentHistory(source, anchor_ns, control_ns, episode_start_ns, targets,
                          tuple(samples), status, status == "READY")


def current_features(window, robot_quat):
    if not isinstance(window, CurrentHistory) or not window.can_step:
        raise ValueError("Current student needs fresh support; caller must handle fallback")
    robot_quat = np.asarray(robot_quat, dtype=np.float64)
    if (robot_quat.shape != (4,) or not np.isfinite(robot_quat).all()
            or abs(float(robot_quat@robot_quat)-1.) > 1e-4):
        raise ValueError("Require a current unit robot torso wxyz quaternion")
    anchor = window.samples[-1]
    poses = [anchor.pose if row is None else row.pose for row in window.samples]
    positions = np.array([pose.positions for pose in poses])
    rotations = np.array([pose.orientations for pose in poses])
    # The coordinate anchor uses the last available current pose, never a full clip.
    geometry = encode_history(positions, rotations).features
    relative_torso = rotation_columns(product(inverse(robot_quat), rotations[:, 9]))
    valid = np.array([row is not None for row in window.samples])
    exact = np.array([row is not None and row.capture_ns == target
                      for target, row in zip(window.targets_ns, window.samples)])
    ages = np.array([0. if row is None else (target-row.capture_ns)/1e9
                     for target, row in zip(window.targets_ns, window.samples)])
    features = np.concatenate((geometry, positions[:, 0, 2:3], relative_torso,
        valid[:, None], exact[:, None], ages[:, None]), axis=-1)
    features[~valid] = 0.
    if features.shape != (HISTORY, FRAME_DIM) or not np.isfinite(features).all():
        raise RuntimeError("Current/history feature contract changed")
    features.setflags(write=False)
    return features


def fit_current_statistics(examples, *, split):
    from forecast_data import TrainingMoments

    if split != "train":
        raise ValueError("Student statistics may fit training sources only")
    moments = TrainingMoments()
    for values in examples:
        values = np.asarray(values)
        if values.shape != (HISTORY, FRAME_DIM) or not np.isfinite(values).all():
            raise ValueError("Require a finite current/history example")
        valid = values[:, GEOMETRY_DIM] == 1.
        if valid.any():
            moments.update(values[valid, :GEOMETRY_DIM], split=split)
    result = moments.parameters()
    result["scale"] = np.maximum(result["scale"], .01).tolist()
    return dict(result, geometry_dim=GEOMETRY_DIM, schema="q1_current_geometry_stats_v1",
                minimum_scale=.01, clipping=10.)
