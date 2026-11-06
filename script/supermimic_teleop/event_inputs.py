"""Independent event goals and paired Q1 references for offline stress replays."""

from dataclasses import asdict, dataclass

import numpy as np

from causal_tail import forecast_window
from stream_contract import DT_NS, HumanPose, PoseFrame, PoseTimeline, SourceIdentity
from stream_events import EventTimeline


@dataclass(frozen=True)
class EventGoal:
    poses: tuple
    policy_frames: int
    timing: dict

    def validate(self):
        value = self.timing
        if (len(self.poses) != 10 or not all(isinstance(p, HumanPose) for p in self.poses)
                or type(self.policy_frames) is not int or not 1 <= self.policy_frames <= 10):
            raise ValueError("Event policy needs ten poses and an explicit continuous frame mask")
        origin, control = value["simulated_origin_ns"], value["simulated_control_ns"]
        if control-origin != value["horizon_frames"]*DT_NS:
            raise ValueError("Event control and preview clocks differ")
        packets = value["used_packets"]
        if not packets or any(p["capture_ns"] > control or p["arrival_ns"] > control for p in packets):
            raise ValueError("Event policy consumed an unavailable capture or packet")
        if any(p["source_index"] > p["sequence"] for p in packets):
            raise ValueError("Synthetic pose mapping read a future source index")
        measured, generated = value["measured_mask"], value["generated_mask"]
        if (len(measured) != 10 or len(generated) != 10
                or any(type(x) is not bool for x in (*measured, *generated))
                or any(a and b for a, b in zip(measured, generated))):
            raise ValueError("Measured and generated event slots must be distinct")
        if value["fallback"] is not None:
            if (any(measured) or not all(generated) or self.policy_frames != 10
                    or len(packets) != 1 or packets[0]["capture_ns"] > origin+DT_NS
                    or any(p != self.poses[0] for p in self.poses)):
                raise ValueError("Fallback must hold one visible past pose with no measured-slot claim")
        if value["measured_end_to_end_latency_ns"] is not None:
            raise ValueError("Synthetic event clocks are not measured robot latency")
        return self


class EventReceiver:
    def __init__(self, positions, quaternions, event, source_sha256, session):
        from forecast_geometry import checked_poses

        positions, quaternions = checked_poses(positions, quaternions)
        self.events = EventTimeline(len(positions), event)
        self.poses = tuple(HumanPose(positions[i], quaternions[i]) for i in self.events.indices)
        source = SourceIdentity("synthetic_offline:"+event.kind, session, "simulated_common_50hz_ns",
            "smpl24_world_m_wxyz", "offline_unverified", source_sha256)
        self.timeline, self.packet_cursor, self.next_frame = PoseTimeline(source), 0, 0

    def step(self, frame, horizon, method):
        if frame != self.next_frame or type(frame) is not int:
            raise ValueError("Event receiver requires sequential control frames")
        if method not in ("masked", "hold", "damped_cv") or horizon not in (1, 2, 4, 5, 10):
            raise ValueError("Invalid event method or preview")
        control = (frame+horizon)*DT_NS
        packets = self.events.packets
        while self.packet_cursor < len(packets) and packets[self.packet_cursor].arrival_ns <= control:
            packet = packets[self.packet_cursor]
            self.timeline.ingest(PoseFrame(self.timeline.source, packet.sequence, packet.capture_ns,
                None, packet.capture_ns, packet.arrival_ns, self.poses[packet.sequence],
                terminal=packet.sequence == self.events.frames-1))
            self.packet_cursor += 1
        window = self.timeline.window(origin_ns=frame*DT_NS, control_ns=control,
            preview_frames=horizon, max_capture_age_ns=10*DT_NS)
        visible = {p.sequence: p for p in packets[:self.packet_cursor]}
        fallback, effective = None, method
        if window.can_step:
            used = [visible[i] for i in dict.fromkeys(window.used_sequences)]
            count = window.prefix_frames
            if method == "damped_cv" and count < 2:
                effective = "hold"
            if method == "masked":
                poses, measured, generated, provenance = (window.poses, window.measured_mask,
                    (False,)*10, window.provenance)
                enabled = count
            else:
                goal = forecast_window(window, effective)
                poses, measured, generated, provenance = (goal.poses, goal.measured_mask,
                    goal.generated_mask, goal.provenance)
                enabled = 10
        else:
            # Do not skip the missing first slot by substituting a later stored target.
            eligible = [p for p in visible.values() if p.capture_ns <= (frame+1)*DT_NS]
            if not eligible:
                raise ValueError("No visible past pose for the explicit offline fallback")
            used = [max(eligible, key=lambda p: p.capture_ns)]
            poses = (self.poses[used[0].sequence],)*10
            measured, generated, provenance = (False,)*10, (True,)*10, ("visible_past_pose_hold",)*10
            fallback, effective, enabled = "visible_past_pose_hold", "hold", 10
        timing = dict(window.timing(), event=asdict(self.events.spec), frame=frame,
            horizon_frames=horizon, tail_method=method, effective_tail_method=effective,
            simulated_origin_ns=frame*DT_NS, simulated_control_ns=control,
            measured_mask=measured, generated_mask=generated, provenance=provenance,
            policy_prefix_frames=enabled, continuous_prefix_frames=window.prefix_frames,
            physical_fallback_required=not window.can_step, fallback=fallback,
            used_packets=[asdict(p) for p in used],
            used_pose_age_ns=control-max(p.capture_ns for p in used),
            synthetic_mapping="source index is past or current; captures and arrivals simulated",
            source_preprocessing="offline_unverified", measured_end_to_end_latency_ns=None,
            fallback_safety_validated=False, bone_contact_constraints_applied=False)
        self.next_frame += 1
        return EventGoal(poses, enabled, timing).validate()


def validate_event_timing(value, frame, horizon, method, timeline):
    """Parent independently checks received packet identities and fallback semantics."""
    if (value.get("frame") != frame or value.get("horizon_frames") != horizon
            or value.get("tail_method") != method or value.get("event") != asdict(timeline.spec)
            or value.get("simulated_origin_ns") != frame*DT_NS
            or value.get("simulated_control_ns") != (frame+horizon)*DT_NS
            or value.get("requested_preview_ns") != horizon*DT_NS
            or value.get("scheduled_origin_to_control_ns") != horizon*DT_NS
            or value.get("source_preprocessing") != "offline_unverified"
            or value.get("measured_end_to_end_latency_ns") is not None):
        raise ValueError("Event worker changed timing, method, source or latency claims")
    visible = {p.sequence: p for p in timeline.visible((frame+horizon)*DT_NS)}
    used = value.get("used_packets", [])
    if (not used or len({p.get("sequence") for p in used}) != len(used)
            or any(p.get("sequence") not in visible or asdict(visible[p["sequence"]]) != p for p in used)):
        raise ValueError("Event worker used an invisible or changed packet")
    measured, generated = value.get("measured_mask", []), value.get("generated_mask", [])
    if (len(measured) != 10 or len(generated) != 10
            or any(type(x) is not bool for x in (*measured, *generated))
            or any(a and b for a, b in zip(measured, generated))):
        raise ValueError("Invalid event measured/generated masks")
    expected_prefix = []
    for i in range(frame+1, frame+horizon+1):
        source = min(i, timeline.frames-1)
        if source not in visible:
            break
        expected_prefix.append(source)
    fallback = not expected_prefix
    if (value.get("physical_fallback_required") is not fallback
            or value.get("continuous_prefix_frames") != len(expected_prefix)
            or value.get("observed_prefix_frames") != sum(measured)):
        raise ValueError("Event worker changed the available continuous prefix")
    if fallback:
        eligible = [p for p in visible.values() if p.capture_ns <= (frame+1)*DT_NS]
        expected = max(eligible, key=lambda p: p.capture_ns) if eligible else None
        if (expected is None or used != [asdict(expected)]
                or value.get("fallback") != "visible_past_pose_hold" or any(measured)
                or not all(generated) or value.get("policy_prefix_frames") != 10):
            raise ValueError("Event fallback must hold the last visible past pose")
    else:
        packets = [asdict(visible[i]) for i in dict.fromkeys(expected_prefix)]
        effective = "hold" if method == "damped_cv" and len(expected_prefix) < 2 else method
        terminal = expected_prefix[-1] == timeline.frames-1
        expected_measured = tuple(i < len(expected_prefix) and frame+1+i < timeline.frames for i in range(10))
        expected_generated = tuple(method != "masked" and i >= len(expected_prefix) and not terminal for i in range(10))
        if (used != packets or tuple(measured) != expected_measured or tuple(generated) != expected_generated
                or value.get("fallback") is not None or value.get("effective_tail_method") != effective
                or value.get("policy_prefix_frames") != (len(expected_prefix) if method == "masked" else 10)):
            raise ValueError("Event worker misreported its measured prefix or forecast tail")
    if (value.get("policy_device") != "cuda" or "cpu_policy_wall_compute_ns" in value
            or type(value.get("gpu_policy_wall_compute_ns")) is not int
            or value["gpu_policy_wall_compute_ns"] < 0):
        raise ValueError("Event worker must report synchronized GPU software timing")
    return value


def paired_reference(reference, event, model=None):
    """Q1 state: qpos29, world root linear/angular6, joint rates22, action22."""
    reference = np.asarray(reference)
    if (reference.ndim != 2 or reference.shape[1] != 79 or not np.isfinite(reference).all()
            or not np.allclose(np.linalg.norm(reference[:, 3:7], axis=1), 1., atol=1e-5)
            or np.any(reference[:, 57:79])):
        raise ValueError("Require finite Q1 reference79 with unit wxyz and zero action fields")
    timeline = EventTimeline(len(reference), event)
    value = reference[list(timeline.indices)].copy()
    changed = event.kind in ("stop", "reverse")
    if changed:
        import mujoco

        if model is None or model.nq != 29 or model.nv != 28:
            raise ValueError("Paired event velocities require the locked Q1 MuJoCo model")
        velocity = np.empty(28, dtype=np.float64)
        for frame in range(event.first_frame+1, len(value)):
            mujoco.mj_differentiatePos(model, velocity, DT_NS/1e9,
                value[frame-1, :29].astype(np.float64), value[frame, :29].astype(np.float64))
            value[frame, 29:32] = velocity[:3]
            q, v = value[frame, 3:7].astype(np.float64), velocity[3:6]
            # MuJoCo free-joint angular rates are local; the common state stores world rates.
            value[frame, 32:35] = v+2*np.cross(q[1:], np.cross(q[1:], v)+q[0]*v)
            value[frame, 35:57] = velocity[6:]
    return value, dict(event=asdict(event), source_indices=timeline.indices,
        qpos_mapping_shared_with_human=True, input_reference_changed=False,
        velocity_rule=("original before/at event; backward MuJoCo difference after event, world angular rates"
                       if changed else "original frozen velocities retained"),
        state_layout="qpos[0:29], root_v_w[29:32], root_w_w[32:35], dq[35:57], previous_action[57:79]",
        physical_feasibility_validated=False, source_preprocessing="offline_unverified")
