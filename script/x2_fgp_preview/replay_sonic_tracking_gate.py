"""Evaluate a tracking admission sequence on saved planner poses and feedback."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from sonic_token_release import ReleaseAnnouncer
from sonic_tracking_gate import Cycle, JointEnvelope, TrackingGate


def load_cycles(planner, velocity, maximum, max_ref_step, max_speed):
    with planner.open("rb") as source:
        flat = np.fromfile(source, dtype="<f4", count=40 * maximum)
    frames = flat[:len(flat)//40*40].reshape(-1, 40)
    if len(frames) < maximum or (np.diff(frames[:, 0]) <= 0).any() or not np.isfinite(frames).all():
        raise ValueError("missing or unordered planner prefix")
    with velocity.open() as source:
        reader = csv.reader(line for line in source if "\0" not in line)
        next(reader)
        speeds = [max(abs(float(value)) for value in row[1:]) for row in reader if len(row) == 32][:maximum]
    if len(speeds) != maximum or not np.isfinite(speeds).all():
        raise ValueError("incomplete robot feedback prefix")
    times = frames[:, 0].astype(np.float64) - float(frames[0, 0])
    steps = np.max(np.abs(np.diff(frames[:, 9:], axis=0, prepend=frames[:1, 9:])), axis=1)
    cases = [Cycle(epoch="recorded_diagnostic", now=t, planner_sequence=i, planner_received=t,
                   planner_source="upstream", feedback_received=t, planner_head=True,
                   steady=bool(step <= max_ref_step and speed <= max_speed), human_ready=True,
                   operator_requests_tracking=False, target_envelope_ok=True)
             for i, (t, step, speed) in enumerate(zip(times, steps, speeds))]
    return cases, steps, speeds, float(frames[0, 0])


def simple_replay(cases, ready_s):
    gate = TrackingGate(ready_for_s=ready_s, max_input_age_s=.12)
    checked = [gate.update(cycle) for cycle in cases]
    return next((i for i, decision in enumerate(checked) if decision.planner_ready), None)


def replay_events(cases, ready_s, first_ready):
    planned_entry = first_ready + 2
    fallback = planned_entry + 8
    operator_release = fallback + 5
    old_head_return = operator_release + 6
    if old_head_return + 70 >= len(cases):
        return None
    gate = TrackingGate(ready_for_s=ready_s, max_input_age_s=.12)
    announcer = ReleaseAnnouncer()
    events, explicit_release, entries = [], [], []
    rearm = None
    # Saved Stage 3 sign-flip values, with diagnostic-only limits.
    reasons = JointEnvelope(-.45, .45, .5, .45).violations(
        target=.45, measured=-.2382, previous=-.45, dt=.01999)
    for i, raw in enumerate(cases):
        peak = rearm is not None and i == rearm + 3
        request = (i == 2 or planned_entry <= i < operator_release
                   or (rearm is not None and rearm <= i < rearm + 6))
        cycle = replace(raw, operator_requests_tracking=request,
                        planner_source="hold" if i == fallback else "upstream",
                        planner_head=not planned_entry < i < old_head_return,
                        target_envelope_ok=not (peak and reasons))
        decision = gate.update(cycle)
        if decision.event is not None:
            events.append((i, decision.event))
        if decision.event == "enter:planner_ready":
            entries.append(i)
        if i > old_head_return and decision.planner_ready and len(entries) == 1 and rearm is None:
            rearm = i + 1
        if announcer.due(cycle.now, published=decision.tracking_requested,
                         engaged=decision.tracking_requested):
            explicit_release.append(i)
    assert entries[0] == planned_entry and len(entries) == 2
    assert entries[1] >= old_head_return and explicit_release[0] == fallback
    assert (rearm + 3, "release:target_envelope") in events and rearm + 3 in explicit_release
    assert not any(fallback < i < operator_release for i in entries)
    return dict(events=events, first_entry=entries[0], reentry=entries[1],
                injected_fallback=fallback, explicit_release=explicit_release,
                injected_waist_step=rearm + 3, waist_step_violations=reasons,
                injected_planner_head_return=old_head_return)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--ref-step-rad", type=float, required=True)
    parser.add_argument("--joint-speed-rad-s", type=float, required=True)
    parser.add_argument("--dwell-s", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.ref_step_rad < 1 or not 0 < args.joint_speed_rad_s < 5:
        parser.error("supply positive diagnostic limits")
    cases, steps, speeds, offset = load_cycles(args.planner, args.velocity, 330,
                                              args.ref_step_rad, args.joint_speed_rad_s)
    first_ready = simple_replay(cases, args.dwell_s)
    print(json.dumps(dict(ticks=len(cases), first_ready_tick=first_ready,
        planner_log_first_offset_s=offset,
        first_ready_replay_s=None if first_ready is None else cases[first_ready].now,
        max_ref_step_rad=float(max(steps)), max_feedback_speed_rad_s=float(max(speeds)),
        steady_cycle_count=sum(c.steady for c in cases),
        scenario=None if first_ready is None else replay_events(cases, args.dwell_s, first_ready),
        timing="offline aligned prefixes, with injected operator and mode events",
        target_envelope_evidence="one injected saved waist sign flip; other ticks held valid",
        diagnostics_only=True, hardware_commands=0), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
