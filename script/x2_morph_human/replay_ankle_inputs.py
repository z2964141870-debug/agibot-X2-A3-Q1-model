"""Replay alternate human references and ankle feedback without opening robot I/O."""

import argparse
from itertools import islice
import json
import math
from pathlib import Path

from .artifacts import MODEL_SHA, sha
from .core import DT_NS
from .runtime import Runtime

CLOCK = "offline_ankle_inputs"
LEFT_ANKLE = "left_ankle_roll_joint"


def read_prefix(path, count):
    with path.open() as stream:
        return [json.loads(line) for line in islice(stream, count)]


def replay(runtime, frames, feedback, *, static_reference, neutral_ankles):
    runtime.reset()
    ankle_indices = [runtime.robot.native_joint_names.index(name)
                     for name in (LEFT_ANKLE, "right_ankle_roll_joint")]
    first_capture = frames[0]["capture_ns"]
    arrivals = [frame["capture_ns"]-first_capture for frame in frames]
    outcomes = {}
    cursor = 0
    for sequence, now in enumerate(range(0, arrivals[-1]+1, DT_NS)):
        while cursor < len(frames) and arrivals[cursor] <= now:
            frame = frames[cursor]
            if static_reference is not None:
                frame = dict(frame, **static_reference)
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


def replace_joint_feedback(rows, robot, indices):
    replaced = []
    for row in rows:
        state = row["state"]
        q, dq = state["q_rad"].copy(), state["dq_rad_s"].copy()
        for i in indices:
            q[i], dq[i] = float(robot.default_joint_pos[i]), 0.
        replaced.append(dict(state=dict(state, q_rad=q, dq_rad_s=dq)))
    return replaced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("human", "feedback", "profile", "policy-root", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compare-exported-stand", action="store_true",
                       help="compare recorded motion to the exported static SMPL stand with measured feedback")
    modes.add_argument("--compare-stand-feedback", action="store_true",
                       help="hold exported stand and the saved IMU, varying groups of robot joint feedback")
    args = parser.parse_args()
    frames = read_prefix(args.human, 120)
    if len(frames) != 120 or any(a["capture_ns"] >= b["capture_ns"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Expected 120 ordered recorded HumanFrames")
    feedback = read_prefix(args.feedback, 260)
    if (frames[-1]["capture_ns"]-frames[0]["capture_ns"])//DT_NS >= len(feedback):
        raise ValueError("Recorded input is longer than the saved feedback prefix")
    profile = json.loads(args.profile.read_text())
    left = profile["joint_names"].index(LEFT_ANKLE)
    upper = profile["upper"][left]
    lower = profile["lower"][left]
    runtime = Runtime(args.policy_root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    if tuple(profile["joint_names"]) != runtime.robot.native_joint_names:
        raise ValueError("Deployment and replay joint order differ")
    skeleton_path = runtime.root / "upstream/deploy/vr_teleop/human_skeleton.json"
    frozen = {key: frames[0][key] for key in ("positions", "quaternions")}
    if args.compare_exported_stand or args.compare_stand_feedback:
        stand = json.loads(skeleton_path.read_text())["stand"]
        stand = dict(positions=stand["pos"], quaternions=stand["quat"])
    if args.compare_stand_feedback:
        names = runtime.robot.native_joint_names
        groups = {"legs": ("hip_", "knee_", "ankle_"), "waist": ("waist_",),
                  "arms": ("shoulder_", "elbow_", "wrist_"), "head": ("head_",)}
        baseline_key = "saved_feedback"
        cases = [(baseline_key, stand, False, feedback)]
        for group, parts in groups.items():
            indices = [i for i, name in enumerate(names) if any(part in name for part in parts)]
            cases.append((f"default_{group}", stand, False, replace_joint_feedback(feedback, runtime.robot, indices)))
        for axis in ("yaw", "pitch", "roll"):
            index = names.index(f"waist_{axis}_joint")
            cases.append((f"default_waist_{axis}", stand, False,
                          replace_joint_feedback(feedback, runtime.robot, [index])))
        cases.append(("default_all_joints", stand, False, replace_joint_feedback(feedback, runtime.robot, range(31))))
    elif args.compare_exported_stand:
        baseline_key = "moving_human_measured_ankles"
        cases = [(baseline_key, None, False, feedback),
                 ("frozen_human_measured_ankles", frozen, False, feedback),
                 ("exported_stand_measured_ankles", stand, False, feedback)]
    else:
        baseline_key = "moving_human_measured_ankles"
        cases = [(f"{label}_human_{'neutral' if neutral else 'measured'}_ankles", reference, neutral, feedback)
                 for label, reference in (("moving", None), ("frozen", frozen)) for neutral in (False, True)]
    results = {name: replay(runtime, frames, states, static_reference=reference, neutral_ankles=neutral)
               for name, reference, neutral, states in cases}
    common = sorted(set.intersection(*(set(states) for states in results.values())))
    if not common:
        raise ValueError("No common ready replay ticks")
    baseline = results[baseline_key]
    summary = {}
    for key, targets in results.items():
        values = [targets[tick] for tick in common]
        summary[key] = dict(ready=len(targets), above_upper=sum(value > upper for value in values),
                            below_lower=sum(value < lower for value in values),
                            min_deg=math.degrees(min(values)), max_deg=math.degrees(max(values)),
                            max_abs_change_from_baseline_deg=math.degrees(
                                max(abs(targets[tick]-baseline[tick]) for tick in common)))
    data = dict(scope="offline reference and feedback replay on a virtual clock",
                source_frames=len(frames), compared_ready_ticks=len(common), lower_rad=lower,
                upper_rad=upper, variants=summary,
                human_sha256=sha(args.human), feedback_sha256=sha(args.feedback), policy_sha256=MODEL_SHA,
                skeleton_sha256=sha(skeleton_path), profile_sha256=sha(args.profile), hardware_commands=0)
    with args.output.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
