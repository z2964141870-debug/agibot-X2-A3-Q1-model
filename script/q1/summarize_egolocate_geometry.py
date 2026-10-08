#!/usr/bin/env python3
"""Report recorded skeleton geometry without assuming it matches human truth."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path):
    with np.load(path, allow_pickle=False) as data:
        if data["usage"].item() != "TEST_ONLY" or data["training_allowed"].item() is not False:
            raise ValueError("Expected TEST_ONLY input")
        joints = data["joints"].astype(float)
    report = {"recording": path.parent.name, "usage": "TEST_ONLY",
              "training_allowed": False, "ground_truth_verified": False, "joints": {}}
    chains = {"left_knee": (1, 4, 7), "right_knee": (2, 5, 8),
              "left_elbow": (16, 18, 20), "right_elbow": (17, 19, 21)}
    for name, ids in chains.items():
        a = joints[:, ids[1]] - joints[:, ids[0]]
        b = joints[:, ids[2]] - joints[:, ids[1]]
        lengths = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        if np.any(lengths <= 0) or not np.isfinite(lengths).all():
            raise ValueError("Invalid bone lengths")
        angles = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1) / lengths, -1, 1)))
        report["joints"][name] = {
            "definition": "turn_angle_between_bone_segments_0_is_straight",
            "first_deg": float(angles[0]), "min_deg": float(angles.min()),
            "max_deg": float(angles.max()),
        }
    report["max_root_relative_joint_displacement_m"] = {
        name: float(np.linalg.norm((joints[:, idx] - joints[:, 0])
                                  - (joints[0, idx] - joints[0, 0]), axis=1).max())
        for name, idx in {"left_ankle": 7, "right_ankle": 8,
                          "left_wrist": 20, "right_wrist": 21}.items()
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.inputs:
        print(json.dumps(summarize(path), sort_keys=True))


if __name__ == "__main__":
    main()
