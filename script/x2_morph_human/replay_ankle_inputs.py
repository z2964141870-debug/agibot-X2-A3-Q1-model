"""Replay alternate human references and ankle feedback without opening robot I/O."""

import argparse
from itertools import islice
import json
import math
from pathlib import Path

from .artifacts import sha
from .core import DT_NS
from .runtime import Runtime

CLOCK = "offline_ankle_inputs"


def read_prefix(path, count):
    with path.open() as stream:
        return [json.loads(line) for line in islice(stream, count)]


def replay(runtime, frames, feedback, *, frozen, neutral_ankles):
    runtime.reset()
    ankle_indices = [list(runtime.robot.native_joint_names).index(name)
                     for name in ("left_ankle_roll_joint", "right_ankle_roll_joint")]
    first_capture = frames[0]["capture_ns"]
    arrivals = [frame["capture_ns"]-first_capture for frame in frames]
    outcomes = {}
    cursor = 0
    for sequence, now in enumerate(range(0, arrivals[-1]+1, DT_NS)):
        while cursor < len(frames) and arrivals[cursor] <= now:
            frame = frames[cursor]
            if frozen:
                frame = dict(frame, positions=frames[0]["positions"], quaternions=frames[0]["quaternions"])
            runtime.handle(frame, arrivals[cursor], CLOCK)
            cursor += 1
        state = dict(feedback[sequence]["state"], sequence=sequence)
        if neutral_ankles:
            state["q_rad"] = state["q_rad"].copy()
            state["dq_rad_s"] = state["dq_rad_s"].copy()
            for i in ankle_indices:
                state["q_rad"][i] = float(runtime.robot.default_joint_pos[i])
                state["dq_rad_s"][i] = 0.
        reply = runtime.handle(state, now, CLOCK)
        if reply.get("publish_allowed") or reply.get("reset"):
            raise ValueError("Replay is advisory only")
        if reply["ready"]:
            outcomes[sequence] = reply["q_target_rad"][ankle_indices[0]]
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("human", "feedback", "profile", "policy-root", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    frames = read_prefix(args.human, 120)
    if len(frames) != 120 or any(a["capture_ns"] >= b["capture_ns"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Expected 120 ordered recorded HumanFrames")
    feedback = read_prefix(args.feedback, 260)
    if (frames[-1]["capture_ns"]-frames[0]["capture_ns"])//DT_NS >= len(feedback):
        raise ValueError("Recorded input is longer than the saved feedback prefix")
    profile = json.loads(args.profile.read_text())
    left = profile["joint_names"].index("left_ankle_roll_joint")
    upper = profile["upper"][left]
    runtime = Runtime(args.policy_root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    if tuple(profile["joint_names"]) != runtime.robot.native_joint_names:
        raise ValueError("Deployment and replay joint order differ")
    results = {}
    for frozen in (False, True):
        for neutral in (False, True):
            results[f"{'frozen' if frozen else 'moving'}_human_{'neutral' if neutral else 'measured'}_ankles"] = replay(
                runtime, frames, feedback, frozen=frozen, neutral_ankles=neutral)
    common = sorted(set.intersection(*(set(states) for states in results.values())))
    if not common:
        raise ValueError("No common ready replay ticks")
    baseline = results["moving_human_measured_ankles"]
    summary = {}
    for key, targets in results.items():
        values = [targets[tick] for tick in common]
        summary[key] = dict(ready=len(targets), above_upper=sum(value > upper for value in values),
                            min_deg=math.degrees(min(values)), max_deg=math.degrees(max(values)),
                            max_abs_change_from_baseline_deg=math.degrees(
                                max(abs(targets[tick]-baseline[tick]) for tick in common)))
    data = dict(scope="old recorded human poses and separately saved robot feedback on a virtual clock",
                source_frames=len(frames), compared_ready_ticks=len(common), upper_rad=upper, variants=summary,
                human_sha256=sha(args.human), feedback_sha256=sha(args.feedback), hardware_commands=0)
    with args.output.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
