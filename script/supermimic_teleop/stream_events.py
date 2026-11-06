"""Synthetic offline event clocks; no claims about measured human capture timing."""

from dataclasses import dataclass

import numpy as np

from forecast_geometry import checked_poses, inverse, log_rotation, product
from stream_contract import DT_NS


@dataclass(frozen=True)
class EventSpec:
    kind: str
    first_frame: int = 100
    duration_frames: int = 10
    transport_frames: int = 0

    def __post_init__(self):
        if self.kind not in ("nominal", "dropout", "late_burst", "stop", "reverse"):
            raise ValueError("Unknown synthetic offline event")
        if (type(self.first_frame) is not int or self.first_frame < 1
                or type(self.duration_frames) is not int or not 1 <= self.duration_frames <= 50
                or type(self.transport_frames) is not int or not 0 <= self.transport_frames <= 50):
            raise ValueError("Require bounded integer event and transport frames")


@dataclass(frozen=True)
class Packet:
    sequence: int
    source_index: int
    capture_ns: int
    arrival_ns: int
    provenance: str


class EventTimeline:
    """Arrival is on a simulated common clock, separate from the raw source clock."""

    def __init__(self, frames, spec):
        if type(frames) is not int or not 2 <= frames <= 20000 or spec.first_frame >= frames-1:
            raise ValueError("Require an interior event in a bounded recording")
        if spec.kind in ("dropout","late_burst") and spec.first_frame+spec.duration_frames >= frames:
            raise ValueError("Packet event must leave a captured recovery frame")
        self.frames, self.spec = frames, spec
        self.indices = tuple(self.source_index(i) for i in range(frames))
        packets = []
        for sequence in range(frames):
            outage = spec.first_frame <= sequence < spec.first_frame+spec.duration_frames
            if spec.kind == "dropout" and outage:
                continue
            arrival = (spec.first_frame+spec.duration_frames if spec.kind == "late_burst" and outage else sequence)
            packets.append(Packet(sequence, self.indices[sequence], sequence*DT_NS,
                (arrival+spec.transport_frames)*DT_NS, "synthetic_offline:"+spec.kind))
        self.packets = tuple(packets)
        if any(a.arrival_ns > b.arrival_ns for a,b in zip(packets,packets[1:])):
            raise ValueError("This event model requires FIFO packet arrival")

    def source_index(self, capture_sequence):
        if type(capture_sequence) is not int or not 0 <= capture_sequence < self.frames:
            raise ValueError("Capture sequence outside the recording")
        if self.spec.kind == "stop" and capture_sequence >= self.spec.first_frame:
            return self.spec.first_frame
        if self.spec.kind == "reverse" and capture_sequence >= self.spec.first_frame:
            return max(0,2*self.spec.first_frame-capture_sequence)
        return capture_sequence

    def visible(self, control_ns):
        if type(control_ns) is not int or control_ns < 0:
            raise ValueError("Require a nonnegative simulated control timestamp")
        packets = tuple(packet for packet in self.packets if packet.arrival_ns <= control_ns)
        if any(packet.capture_ns > control_ns for packet in packets):
            raise RuntimeError("A future capture entered the visible event prefix")
        return packets

    def visibility(self, origin_frame, horizon):
        if (type(origin_frame) is not int or not 0 <= origin_frame < self.frames-1
                or type(horizon) is not int or horizon not in (1,2,4,5,10)):
            raise ValueError("Invalid origin or positive real-preview horizon")
        control_ns = (origin_frame+horizon)*DT_NS
        visible = self.visible(control_ns)
        sequences = {packet.sequence for packet in visible}
        prefix = []
        for sequence in range(origin_frame+1,min(origin_frame+horizon,self.frames-1)+1):
            if sequence not in sequences:
                break
            prefix.append(sequence)
        latest = max((packet.capture_ns for packet in visible),default=None)
        return dict(simulated_origin_ns=origin_frame*DT_NS,simulated_control_ns=control_ns,
            visible_sequences=tuple(packet.sequence for packet in visible),
            continuous_future_prefix=tuple(prefix),
            newest_visible_capture_ns=latest,
            newest_capture_age_ns=None if latest is None else control_ns-latest,
            physical_fallback_required=not prefix,
            measured_hardware_latency_ns=None,source_preprocessing="synthetic_offline_event",
            terminal_padding_applied=False)


def derived_poses(positions,quaternions,timeline):
    positions,quaternions = checked_poses(positions,quaternions)
    if len(positions) != timeline.frames:
        raise ValueError("Event clock and source recording lengths differ")
    positions = positions[list(timeline.indices)].copy()
    quaternions = quaternions[list(timeline.indices)].copy()
    velocity = np.zeros_like(positions)
    angular = np.zeros_like(positions)
    dt = DT_NS/1e9
    velocity[1:] = np.diff(positions,axis=0)/dt
    angular[1:] = log_rotation(product(quaternions[1:],inverse(quaternions[:-1])))/dt
    return dict(positions=positions,quaternions=quaternions,linear_velocity_w=velocity,
        angular_velocity_w=angular,first_frame_velocity_is_unknown=True,
        first_frame_velocity_placeholder="zero; unavailable earlier sample",
        source_indices=timeline.indices,provenance="synthetic_offline:"+timeline.spec.kind,
        bone_contact_constraints_applied=False,measured_robot_latency_ns=None)
