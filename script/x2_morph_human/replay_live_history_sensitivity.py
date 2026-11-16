"""Offline sensitivity of live X2 targets to the policy's previous-action input."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import sha
from .runtime import Runtime


def replay(runtime, trace, alpha):
    runtime.reset()
    feedback_action = runtime.runner.last_action.copy()
    replies = []
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "trace_end":
                if not row["complete"]:
                    raise ValueError("Incomplete saved trace")
                return replies
            runtime.runner.last_action = feedback_action
            reply = runtime.handle(row["request"], row["received_ns"], "offline_history_sensitivity")
            if reply.get("publish_allowed") or reply.get("reset") or reply["ready"] != row["reply"]["ready"]:
                raise ValueError("Unexpected advice or different readiness")
            if not reply["ready"]:
                continue
            raw_action = runtime.runner.last_action
            feedback_action = feedback_action + alpha*(raw_action-feedback_action)
            replies.append((reply["state_sequence"], reply["q_target_rad"], row["reply"]["q_target_rad"]))
    raise ValueError("Saved trace has no footer")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("policy-root", "trace", "profile", "output"):
        p.add_argument(f"--{key}", required=True, type=Path)
    args = p.parse_args()
    runtime = Runtime(args.policy_root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    names = runtime.robot.native_joint_names
    profile = json.loads(args.profile.read_text())
    if list(names) != profile["joint_names"]:
        raise ValueError("Profile joint order differs")
    lower, upper = (np.asarray(profile[k]) for k in ("lower", "upper"))
    knee = names.index("right_knee_joint")
    arm = [names.index(f"{side}_elbow_joint") for side in ("left", "right")]
    variants = {}
    for alpha in (1., .75, .5, .25):
        rows = replay(runtime, args.trace, alpha)
        sequence = np.asarray([r[0] for r in rows])
        target = np.asarray([r[1] for r in rows])
        expected = np.asarray([r[2] for r in rows])
        unit = lambda value: float(np.rad2deg(value))
        error = target-expected
        if alpha == 1 and unit(np.max(np.abs(error))) > .02:
            raise ValueError("Unchanged Mac replay failed diagnostic parity")
        outside = (target < lower) | (target > upper)
        jumps = np.diff(target, axis=0)[np.diff(sequence) == 1]
        event = np.flatnonzero(sequence == 1011).item()
        if sequence[event-1] != 1010:
            raise ValueError("Saved event is not adjacent")
        variants[str(alpha)] = dict(samples=len(target), any_outside=int(outside.any(axis=1).sum()),
                                    knee_below_lower=int((target[:, knee] < lower[knee]).sum()),
                                    knee_outside=int(outside[:, knee].sum()),
                                    knee_jump_max_deg=unit(np.max(np.abs(jumps[:, knee]))),
                                    all_joint_jump_max_deg=unit(np.max(np.abs(jumps))),
                                    event_knee_step_deg=unit(target[event, knee]-target[event-1, knee]),
                                    elbows_p95_abs_change_deg=unit(np.percentile(np.abs(error[:, arm]), 95)))
    data = dict(variants=variants, trace_sha256=sha(args.trace), profile_sha256=sha(args.profile),
                hardware_commands=0)
    with args.output.open("x") as out:
        out.write(json.dumps(data, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
