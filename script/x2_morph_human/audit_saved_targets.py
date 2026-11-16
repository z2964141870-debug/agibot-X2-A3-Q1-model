"""Audit saved X2 advisory targets against policy export and native bounds."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(path, model, profile):
    robot = model["robot"]
    names = robot["native_joint_names"]
    if names != profile["joint_names"] or len(set(names)) != 31:
        raise ValueError("Configuration and exported joint order differ")
    defaults = robot["default_joint_pos"]
    upper, lower = profile["upper"], profile["lower"]
    translator = robot["translator"]
    permutation = translator["perm"]
    if len(permutation) != 29 or len(set(permutation)) != 29 or any(i not in range(31) for i in permutation):
        raise ValueError("Exported policy permutation is invalid")

    focus = [names.index(joint) for joint in ("left_ankle_roll_joint", "right_ankle_roll_joint")]
    tracked = {index: dict(action_range=[math.inf, -math.inf], target_rad_range=[math.inf, -math.inf],
                           feedback_rad_range=[math.inf, -math.inf]) for index in focus}
    violations = Counter()
    checked = accepted = 0
    range_disagreement = 0
    max_mapping_error = 0.
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            reply = row["reply"]
            if not reply["ready"]:
                continue
            checked += 1
            if (reply["publish_allowed"] or reply["joint_names"] != names or row["state"]["joint_names"] != names
                    or len(reply["canonical_action"]) != len(permutation)
                    or len(reply["q_target_rad"]) != len(names)
                    or len(row["state"]["q_rad"]) != len(names)):
                raise ValueError("Saved joint or advisory contract differs from export")
            action = reply["canonical_action"]
            expected = defaults.copy()
            for canonical, native in enumerate(permutation):
                expected[native] += translator["sign"][canonical] * action[canonical] * robot["action_scale"][native]
            target = reply["q_target_rad"]
            feedback = row["state"]["q_rad"]
            max_mapping_error = max(max_mapping_error, *(abs(q-estimate) for q, estimate in zip(target, expected)))
            outside = [name for name, value, lo, hi in zip(names, target, lower, upper) if not lo <= value <= hi]
            range_disagreement += set(outside) != set(row["pd_targets_outside_qpos_ranges"])
            if not row["accepted"]:
                continue
            accepted += 1
            violations.update(outside)
            for native, metrics in tracked.items():
                samples = dict(action_range=action[permutation.index(native)], target_rad_range=target[native],
                               feedback_rad_range=feedback[native])
                for key, value in samples.items():
                    old_min, old_max = metrics[key]
                    metrics[key] = [min(old_min, value), max(old_max, value)]
    if not accepted:
        raise ValueError("No accepted advice to audit")
    info = {}
    for native, values in tracked.items():
        canonical = permutation.index(native)
        gain = translator["sign"][canonical] * robot["action_scale"][native]
        info[names[native]] = dict(
            native_index=native, canonical_index=canonical, sign=translator["sign"][canonical],
            default_target_rad=defaults[native], export_state_offset_rad=translator["offset"][canonical],
            gain_rad_per_action=gain, profile_rad_range=[lower[native], upper[native]],
            allowable_action_range=sorted((lo-defaults[native])/gain for lo in (lower[native], upper[native])),
            **values,
        )
    return dict(ready_checked=checked, accepted=accepted, accepted_violations=dict(violations),
                max_export_mapping_error_rad=max_mapping_error, range_flag_disagreements=range_disagreement,
                ankles=info, log_sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "profile", "live", "replay", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    model, profile = (json.loads(path.read_text()) for path in (args.model, args.profile))
    data = dict(live=audit(args.live, model, profile), replay=audit(args.replay, model, profile),
                model_config_sha256=digest(args.model), deployment_profile_sha256=digest(args.profile),
                hardware_commands=0, new_policy_calls=0)
    with args.output.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
