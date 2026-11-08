"""Dependency-aware completion of frozen offline human preview windows.

This independent receiver shares the R1 raw-support cutoff and the original
causal-tail predictor. It certifies stored resampling support, not sensor clocks,
upstream mocap causality, robot safety or physical tracking quality.
"""

from dataclasses import asdict

from causal_tail import forecast_window, validate_forecast
from event_inputs import EventGoal
from forecast_data import source_support
from stream_contract import DT_NS
from student_data import OfflineStudentStream


METHODS = ("masked", "hold", "damped_cv")
HORIZONS = (1, 2, 4, 5, 10)
MAX_AGE_NS = 10 * DT_NS


def available_packets(timeline, frame, horizon):
    control = (frame + horizon) * DT_NS
    rows = []
    for packet in timeline.packets:
        if packet.capture_ns > control:
            break
        dependency = max(packet.capture_ns,
                         source_support(packet.source_index)["dependency_end_ns"])
        arrival = max(packet.arrival_ns, dependency)
        if dependency <= control and arrival <= control:
            rows.append(dict(asdict(packet), nominal_arrival_ns=packet.arrival_ns,
                             dependency_end_ns=dependency, ready_ns=dependency,
                             arrival_ns=arrival))
    return rows


class DependencyPreviewReceiver:
    def __init__(self, positions, quaternions, event, source_sha256, session):
        self.stream = OfflineStudentStream(positions, quaternions, spec=event,
                                          sha256=source_sha256, session=session)
        self.next_frame = 0

    def step(self, frame, horizon, method):
        if (type(frame) is not int or frame != self.next_frame
                or not 0 <= frame < self.stream.events.frames - 1
                or type(horizon) is not int or horizon not in HORIZONS
                or method not in METHODS):
            raise ValueError("Require sequential in-episode frames and a fixed preview method")
        window = self.stream.student(frame, mode="short_preview", horizon=horizon)
        visible = {row["sequence"]: row for row in
                   available_packets(self.stream.events, frame, horizon)}
        fallback, effective = None, method
        if window.can_step:
            used = [visible[i] for i in window.used_sequences]
            if method == "masked":
                poses, measured, generated, provenance, enabled = (
                    window.poses, window.measured_mask, (False,) * 10,
                    window.provenance, window.prefix_frames)
            else:
                if method == "damped_cv" and window.prefix_frames < 2:
                    effective = "hold"
                goal = validate_forecast(forecast_window(window, effective))
                poses, measured, generated, provenance, enabled = (
                    goal.poses, goal.measured_mask, goal.generated_mask, goal.provenance, 10)
        else:
            eligible = [row for row in visible.values()
                        if row["capture_ns"] <= (frame + 1) * DT_NS
                        and window.control_ns - row["capture_ns"] <= MAX_AGE_NS]
            if not eligible:
                raise ValueError("No fresh dependency-ready past pose; caller must stop or hold action")
            used = [max(eligible, key=lambda row: row["capture_ns"])]
            packet = next(p for p in self.stream.events.packets if p.sequence == used[0]["sequence"])
            poses = (self.stream.packet_frame(packet).pose,) * 10
            measured, generated = (False,) * 10, (True,) * 10
            provenance = ("dependency_ready_past_pose_hold",) * 10
            fallback, effective, enabled = "visible_past_pose_hold", "hold", 10
        timing = dict(window.timing(), event=asdict(self.stream.events.spec), frame=frame,
            horizon_frames=horizon, tail_method=method, effective_tail_method=effective,
            simulated_origin_ns=frame * DT_NS, simulated_control_ns=(frame + horizon) * DT_NS,
            measured_mask=measured, generated_mask=generated, provenance=provenance,
            policy_prefix_frames=enabled, continuous_prefix_frames=window.prefix_frames,
            physical_fallback_required=not window.can_step, fallback=fallback,
            used_packets=used, used_pose_age_ns=window.control_ns-max(row["capture_ns"] for row in used),
            max_used_dependency_ns=max(row["dependency_end_ns"] for row in used),
            raw_support_rule="pinned120to50Hz_right_support_before_control",
            raw_support_verified=True, upstream_capture_causality_verified=False,
            source_preprocessing="offline_unverified", measured_end_to_end_latency_ns=None,
            fallback_safety_validated=False, bone_contact_constraints_applied=False,
            terminal_padding_applied=False, damping_tau_s=.12)
        result = EventGoal(poses, enabled, timing).validate()
        validate_dependency_timing(timing, frame, horizon, method, self.stream.events)
        self.next_frame += 1
        return result


def validate_dependency_timing(value, frame, horizon, method, timeline):
    """Reconstruct support/arrival visibility independently from immutable packets."""
    control = (frame + horizon) * DT_NS
    visible = {}
    for packet in timeline.packets:
        dependency = max(packet.capture_ns,
                         source_support(packet.source_index)["dependency_end_ns"])
        if max(packet.arrival_ns, dependency) <= control:
            visible[packet.sequence] = dict(asdict(packet),
                nominal_arrival_ns=packet.arrival_ns, dependency_end_ns=dependency,
                ready_ns=dependency, arrival_ns=max(packet.arrival_ns, dependency))
    prefix = []
    for index in range(frame + 1, min(frame + horizon + 1, timeline.frames)):
        if index not in visible:
            break
        prefix.append(index)
    fallback = not prefix
    if fallback:
        eligible = [row for row in visible.values()
                    if row["capture_ns"] <= (frame + 1) * DT_NS
                    and control - row["capture_ns"] <= MAX_AGE_NS]
        if not eligible:
            raise ValueError("No fresh raw-supported fallback")
        used = [max(eligible, key=lambda row: row["capture_ns"])]
        measured, generated, enabled = (False,) * 10, (True,) * 10, 10
        effective, provenance = "hold", ("dependency_ready_past_pose_hold",) * 10
    else:
        used = [visible[index] for index in prefix]
        measured = tuple(i < len(prefix) for i in range(10))
        generated = tuple(method != "masked" and i >= len(prefix) for i in range(10))
        enabled = len(prefix) if method == "masked" else 10
        effective = "hold" if method == "damped_cv" and len(prefix) < 2 else method
        provenance = ("measured_exact",) * len(prefix) + (
            "masked_hold_placeholder" if method == "masked" else "causal_forecast:" + effective,
        ) * (10 - len(prefix))
    expected = dict(frame=frame, event=asdict(timeline.spec), horizon_frames=horizon,
        tail_method=method, effective_tail_method=effective,
        simulated_origin_ns=frame * DT_NS, simulated_control_ns=control,
        requested_preview_ns=horizon * DT_NS, scheduled_origin_to_control_ns=horizon * DT_NS,
        continuous_prefix_frames=len(prefix), observed_prefix_frames=len(prefix),
        physical_fallback_required=fallback, fallback="visible_past_pose_hold" if fallback else None,
        policy_prefix_frames=enabled, used_packets=used,
        max_used_dependency_ns=max(row["dependency_end_ns"] for row in used),
        used_pose_age_ns=control-max(row["capture_ns"] for row in used),
        raw_support_rule="pinned120to50Hz_right_support_before_control",
        raw_support_verified=True, upstream_capture_causality_verified=False,
        source_preprocessing="offline_unverified", measured_end_to_end_latency_ns=None,
        terminal_padding_applied=False, damping_tau_s=.12)
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Dependency preview changed its raw-support cutoff, packets or fallback")
    if (tuple(value.get("measured_mask", ())) != measured
            or tuple(value.get("generated_mask", ())) != generated
            or tuple(value.get("provenance", ())) != provenance):
        raise ValueError("Dependency preview changed measured/generated slot identity")
    return value
