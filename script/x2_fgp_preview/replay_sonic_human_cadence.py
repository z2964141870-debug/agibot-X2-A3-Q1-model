"""Replay saved garment HumanFrame arrival times through the offline SONIC gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .sonic_tracking_gate import Cycle, TrackingGate

TICK_NS = 20_000_000
WINDOW_NS = 1_000_000_000
REQUEST_RANGES = ((25, 30), (75, 280), (283, 300), (375, 625))
DROP_NS = (5_000_000_000, 5_300_000_000)
HEAD_FIXTURE_TICKS = 4
END_TICK = 675


def capture(path):
    source = path.read_bytes()
    arrivals, stamps, ids = [], [], []
    identity = None
    for line in source.splitlines():
        row = json.loads(line)
        current = tuple(row[k] for k in ("arrival_clock", "capture_clock", "calibration_id", "skeleton_id"))
        if (row["kind"] != "morph_human_frame" or row["version"] != 1
                or current[0] != "replay_monotonic" or not all(current)):
            raise ValueError("expected saved v1 garment packets on the replay clock")
        if identity is not None and current != identity:
            raise ValueError("capture changed clock or calibration")
        identity = current
        ids.append(row["sequence"])
        arrivals.append(row["arrival_ns"])
        stamps.append(row["capture_ns"])
    for sequence in (ids, arrivals, stamps):
        if not sequence or any(type(x) is not int for x in sequence) or any(
                b <= a for a, b in zip(sequence, sequence[1:])):
            raise ValueError("packet identities and timestamps must strictly increase")
    received = [(i, t - arrivals[0]) for i, t in zip(ids, arrivals)]
    if received[-1][1] < END_TICK * TICK_NS:
        raise ValueError("capture shorter than the fixture")
    peak = max(b - a for a, b in zip(arrivals, arrivals[1:]))
    return received, dict(frames=len(received), max_original_gap_ms=peak / 1e6,
                          sha256=hashlib.sha256(source).hexdigest())


def replay(arrivals, threshold_ms, dropout):
    drop = [seq for seq, t in arrivals if dropout and DROP_NS[0] <= t < DROP_NS[1]]
    drop_ids = set(drop)
    received = [(seq, t) for seq, t in arrivals if seq not in drop_ids]
    gate = TrackingGate(ready_for_s=WINDOW_NS / 1e9, max_input_age_s=threshold_ms / 1000)
    cursor, fixture_tracking = 0, False
    pending_head = None
    events, head_events, max_age = [], [], 0.0
    active_during_asserted_gap_recovery = 0
    counters = Counter()
    for tick in range(END_TICK + 1):
        if pending_head is not None and pending_head[0] == tick:
            _, fixture_tracking = pending_head
            head_events.append(dict(tick=tick, fixture_mode="tracking" if fixture_tracking else "planner"))
            pending_head = None
        now = tick * TICK_NS
        while cursor < len(received) and received[cursor][1] <= now:
            cursor += 1
        last_seq, last_arrival = received[cursor - 1]
        age_ms = (now - last_arrival) / 1e6
        human_ready = age_ms <= threshold_ms
        requested = any(start <= tick < end for start, end in REQUEST_RANGES)
        wall = now / 1e9
        decision = gate.update(Cycle(
            epoch="offline_cadence_fixture", now=wall, planner_sequence=tick, planner_received=wall,
            planner_source="upstream", feedback_received=wall, planner_head=not fixture_tracking,
            steady=True, human_ready=human_ready, operator_requests_tracking=requested,
            target_envelope_ok=True))
        if decision.tracking_requested:
            if not human_ready:
                raise RuntimeError("gate requested tracking on a stale HumanFrame")
            max_age = max(max_age, age_ms)
            counters["tracking_requested_ticks"] += 1
            if dropout and now >= DROP_NS[1] and tick < REQUEST_RANGES[1][1]:
                active_during_asserted_gap_recovery += 1
        if decision.tracking_requested != fixture_tracking:
            if pending_head is None or pending_head[1] != decision.tracking_requested:
                pending_head = (tick + HEAD_FIXTURE_TICKS, decision.tracking_requested)
        else:
            pending_head = None
        if decision.event:
            events.append(dict(tick=tick, time_ms=now / 1e6, gate_event=decision.event,
                               latest_human_sequence=last_seq, age_ms=age_ms))
    if threshold_ms == 120:
        entries = [item["tick"] for item in events if item["gate_event"] == "enter:planner_ready"]
        rejects = [e["tick"] for e in events if e["gate_event"] == "reject:planner_or_human_unready"]
        failures = [e for e in events if e["gate_event"] == "release:input_unavailable"]
        if (entries != [75, 375] or rejects != [25, 283] or len(failures) != int(dropout)
                or active_during_asserted_gap_recovery):
            raise ValueError("120 ms archival replay changed the expected admission/release sequence")
    return dict(dropout_injected=dropout, threshold_ms=threshold_ms, dropped_frame_ids=drop,
                gate_events=events, simulated_heads=head_events,
                tracking_requested_ticks=counters["tracking_requested_ticks"],
                max_age_ms_while_tracking_requested=max_age,
                reengaged_during_asserted_gap_recovery_ticks=active_during_asserted_gap_recovery)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="+")
    args = parser.parse_args()
    evidence = {}
    for path in args.capture:
        arrivals, info = capture(path)
        evidence[path.parent.name] = dict(**info, cases=[replay(arrivals, age, dropout)
                                                        for age in (80, 100, 120)
                                                        for dropout in (False, True)])
    print(json.dumps(dict(scope="recorded FGP packet arrivals; no tokenizer or robot; synthetic planner, feedback, "
                         "operator schedule, head acknowledgements, and valid target envelope", actual_head_samples=0,
                         fixture_head_delay_ms=HEAD_FIXTURE_TICKS*TICK_NS/1e6, results=evidence,
                         hardware_commands=0), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
