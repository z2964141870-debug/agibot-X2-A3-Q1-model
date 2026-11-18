"""Isolate waist proprioception and torso IMU reconstruction during offline replay."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from unittest.mock import patch

from . import runtime as runtime_module
from .artifacts import MODEL_SHA, sha
from .replay_ankle_inputs import LEFT_ANKLE, read_prefix, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-root", "profile", "human", "feedback", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    frames, feedback = read_prefix(args.human, 120), read_prefix(args.feedback, 260)
    if len(frames) != 120 or len(feedback) != 260:
        raise ValueError("Expected complete saved prefixes")
    runtime = runtime_module.Runtime(args.policy_root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    from deploy.sim2sim import policy_node

    robot = runtime.robot
    index = robot.native_joint_names.index("waist_pitch_joint")
    profile = json.loads(args.profile.read_text())
    if tuple(profile["joint_names"]) != robot.native_joint_names:
        raise ValueError("Profile joint order differs")
    left = robot.native_joint_names.index(LEFT_ANKLE)
    lo, hi = profile["lower"][left], profile["upper"][left]
    skeleton = runtime.root / "upstream/deploy/vr_teleop/human_skeleton.json"
    stand = json.loads(skeleton.read_text())["stand"]
    reference = dict(positions=stand["pos"], quaternions=stand["quat"])
    original_obs = policy_node.proprio_frame
    original_imu = runtime_module.pelvis_from_sensor

    def replace(q, dq):
        q, dq = q.copy(), dq.copy()
        q[index], dq[index] = robot.default_joint_pos[index], 0.
        return q, dq

    def observation(robot_, q, dq, quat, gyro, action):
        q, dq = replace(q, dq)
        return original_obs(robot_, q, dq, quat, gyro, action)

    def imu(robot_, q, dq, *sensor):
        q, dq = replace(q, dq)
        return original_imu(robot_, q, dq, *sensor)

    outcomes = {}
    for label, change_obs, change_imu in (
        ("saved", False, False), ("proprio_only", True, False),
        ("pelvis_only", False, True), ("proprio_and_pelvis", True, True),
    ):
        with ExitStack() as stack:
            if change_obs:
                stack.enter_context(patch.object(policy_node, "proprio_frame", observation))
            if change_imu:
                stack.enter_context(patch.object(runtime_module, "pelvis_from_sensor", imu))
            outcomes[label] = replay(runtime, frames, feedback, static_reference=reference, neutral_ankles=False)
    common = sorted(set.intersection(*(set(rows) for rows in outcomes.values())))
    if not common or any(len(rows) != len(common) for rows in outcomes.values()):
        raise ValueError("Replay readiness differs")
    baseline = outcomes["saved"]
    variants = {}
    for name, rows in outcomes.items():
        values = [rows[i] for i in common]
        variants[name] = dict(above_upper=sum(v > hi for v in values), below_lower=sum(v < lo for v in values),
                              min_deg=math.degrees(min(values)), max_deg=math.degrees(max(values)),
                              max_delta_deg=math.degrees(max(abs(rows[i]-baseline[i]) for i in common)))
    result = dict(compared_ready_ticks=len(common), variants=variants,
                  feedback_sha256=sha(args.feedback), profile_sha256=sha(args.profile),
                  human_clock_source_sha256=sha(args.human), skeleton_sha256=sha(skeleton),
                  policy_sha256=MODEL_SHA, hardware_commands=0)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
