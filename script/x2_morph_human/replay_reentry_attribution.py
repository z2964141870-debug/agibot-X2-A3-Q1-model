"""Attribute post-gap target changes to reference versus carried policy state."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import sha
from .core import DT_NS
from .replay_pacing_holdout import CLOCK, DROPOUT_FRAMES
from .review_saved_trace import rows
from .runtime import Runtime, synthetic_state

FOLLOW_TICKS = 5


class Capture:
    def __init__(self, session):
        self.session = session
        self.feeds = None

    def run(self, names, feeds):
        self.feeds = {name: value.copy() for name, value in feeds.items()}
        return self.session.run(names, feeds)


def run_target(session, robot, feeds):
    action = session.run(["action_human"], feeds)[0][0].astype(np.float64)
    return robot.targets(action)


def attribute(runtime, session, previous, current, gap_reasons):
    from deploy.sim2sim.obs import PROPRIO_DIM

    old_tick, old_target, old_feed, old_timing = previous
    new_tick, new_target, new_feed, new_timing = current
    names = list(runtime.robot.native_joint_names)
    joint = int(np.argmax(np.abs(new_target - old_target)))
    old_history = old_feed["obs_history"].reshape(-1, PROPRIO_DIM)
    new_history = new_feed["obs_history"].reshape(-1, PROPRIO_DIM)
    action_width = len(runtime.runner.last_action)
    split = PROPRIO_DIM - action_width
    if not np.array_equal(old_history[:, :split], new_history[:, :split]):
        raise ValueError("fixed synthetic feedback unexpectedly changed")
    if old_feed.keys() != new_feed.keys():
        raise ValueError("ONNX inputs changed across the gap")
    if any(not np.array_equal(old_feed[name], new_feed[name])
           for name in old_feed if name != "obs_history" and not name.startswith("human_")
           and name != "goal_human"):
        raise ValueError("fixed ONNX input changed across the gap")
    cases = {}
    targets = {}
    for window, reference in (("old", old_feed), ("new", new_feed)):
        for history, obs in (("old", old_feed), ("new", new_feed)):
            feed = dict(reference, obs_history=obs["obs_history"])
            target = run_target(session, runtime.robot, feed)
            targets[window, history] = target
            cases[f"{window}_window_{history}_history"] = {
                "same_joint_target_deg": float(np.rad2deg(target[joint])),
                "same_joint_delta_from_pre_gap_deg": float(np.rad2deg((target-old_target)[joint])),
                "max_joint_delta_from_pre_gap_deg": float(np.rad2deg(np.max(np.abs(target-old_target)))),
            }
    if not np.array_equal(targets["old", "old"], old_target) or not np.array_equal(targets["new", "new"], new_target):
        raise ValueError("replayed ONNX targets did not reproduce live benchmark targets exactly")
    # Two-factor interaction is separately reported rather than attributing it to either input.
    both = cases["new_window_new_history"]["same_joint_delta_from_pre_gap_deg"]
    only_window = cases["new_window_old_history"]["same_joint_delta_from_pre_gap_deg"]
    only_history = cases["old_window_new_history"]["same_joint_delta_from_pre_gap_deg"]
    return dict(
        last_ready_tick=old_tick, reentry_tick=new_tick, no_target_ticks=new_tick-old_tick-1,
        not_ready_reasons=sorted(gap_reasons), largest_jump_joint=names[joint],
        largest_jump_deg=abs(both),
        old_target_deg=float(np.rad2deg(old_target[joint])),
        new_target_deg=float(np.rad2deg(new_target[joint])),
        reference_advance_ms=(new_timing["slot_capture_ns"][-1]-old_timing["slot_capture_ns"][-1])/1e6,
        newest_human_arrival_age_ms=new_timing["newest_arrival_age_ms"],
        fixed_robot_joint_deg=float(np.rad2deg(runtime.robot.default_joint_pos[joint])),
        interaction_deg=both-only_window-only_history, cases=cases, follow=[],
        joint_index=joint, pre_gap_target=old_target, prior_target=new_target,
    )


def replay(root, path, dropout):
    packets = list(rows(path))
    if len(packets) < 100 or any(p["arrival_clock"] != CLOCK for p in packets):
        raise ValueError("expected recorded HumanFrames on replay clock")
    middle = len(packets)//2
    dropped = packets[middle:middle+DROPOUT_FRAMES] if dropout else []
    retained = packets[:middle]+packets[middle+len(dropped):]
    runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    original = runtime.runner.session
    capture = Capture(original)
    runtime.runner.session = capture
    position = gap_outputs = 0
    events, gap_reasons = [], set()
    previous = None
    first_tick = None
    start, end = retained[0]["arrival_ns"], retained[-1]["arrival_ns"]
    for seq, now in enumerate(range(start, end+1, DT_NS)):
        while position < len(retained) and retained[position]["arrival_ns"] <= now:
            runtime.push(retained[position])
            position += 1
        capture.feeds = None
        there = runtime.handle(synthetic_state(runtime, seq, now), now, CLOCK)
        if there.get("reset") or there.get("publish_allowed"):
            raise ValueError(f"unexpected reply at {seq}")
        if not there["ready"]:
            if capture.feeds is not None or "q_target_rad" in there:
                gap_outputs += 1
            if first_tick is not None:
                gap_reasons.add(there["reason"])
            continue
        current = seq, np.asarray(there["q_target_rad"]), capture.feeds, there["timing"]
        if capture.feeds is None:
            raise ValueError("ready reply without ONNX invocation")
        if previous is not None and seq > previous[0]+1:
            events.append(attribute(runtime, original, previous, current, gap_reasons))
        if events and 0 < seq-events[-1]["reentry_tick"] <= FOLLOW_TICKS:
            event = events[-1]
            joint = event["joint_index"]
            target = current[1]
            event["follow"].append(dict(
                tick=seq, same_joint_delta_from_pre_gap_deg=float(np.rad2deg((target-event["pre_gap_target"])[joint])),
                maximum_joint_delta_from_preceding_ready_deg=float(np.rad2deg(np.max(np.abs(target-event["prior_target"])))),
            ))
            event["prior_target"] = target
        first_tick = seq if first_tick is None else first_tick
        previous = current
        gap_reasons.clear()
    if gap_outputs:
        raise ValueError("policy produced output within a no-target gap")
    for event in events:
        for key in ("joint_index", "pre_gap_target", "prior_target"):
            del event[key]
    return dict(source=str(path), source_sha256=sha(path), packets=len(retained),
                dropped_sequences=[p["sequence"] for p in dropped], gap_outputs=gap_outputs, events=events)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--policy-root", type=Path, required=True)
    cli.add_argument("captures", nargs="+", type=Path)
    args = cli.parse_args()
    print(json.dumps(dict(
        scope="recorded human input, fixed synthetic robot feedback, advisory output, offline two-factor ONNX replay",
        hardware_commands=0,
        cases=[replay(args.policy_root, path, dropout) for path in args.captures for dropout in (False, True)],
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
