#!/usr/bin/env python3
"""Render saved TEST_ONLY kinematics independently of inference and PD replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from infer_clotho_fgp import identity
from q1_retarget import BodyReference, Q1Retargeter
from recover_q1_validation import load_trace
from validate_egolocate import render_overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "egl"), default="cpu")
    args = parser.parse_args()
    directory = args.output_dir
    if (directory / "body_q1_overview.png").exists() or (directory / "render_summary.json").exists():
        parser.error("Preserve existing overview/render evidence")
    recovered = json.loads((directory / "recovered_summary.json").read_text())
    if not recovered["full_replay_timeline_verified"] or recovered["usage"] != "TEST_ONLY":
        raise ValueError("Numerical recovery must pass before rendering saved output")
    for artifact in recovered["artifacts"]:
        if identity(directory / Path(artifact["path"]).name)["sha256"] != artifact["sha256"]:
            raise ValueError("Saved traces changed after numerical recovery")
    retargeter = Q1Retargeter()
    source = BodyReference.load(directory / "body_reference.npz", retargeter.names)
    trace = load_trace(directory / "kinematic_trace.npz")
    if (trace["qpos"].shape != (len(source.times), retargeter.sim.model.nq)
            or not np.array_equal(trace["time_s"], source.times)):
        raise ValueError("Saved qpos and body-reference frames disagree")
    print(f"Rendering saved {len(source.times)} frames overview using {args.backend}", flush=True)
    render_overview(retargeter, source, trace["qpos"], directory, args.backend)
    report = {"usage": "TEST_ONLY", "training_allowed": False,
              "scope": "SAVED_KINEMATIC_POSE_OVERVIEW_NOT_DYNAMIC_BALANCE",
              "render_backend": args.backend, "render_completed": True,
              "source_files": [identity(Path(__file__).resolve()),
                               identity(Path(__file__).with_name("validate_egolocate.py"))],
              "numerical_recovery": identity(directory / "recovered_summary.json"),
              "image": identity(directory / "body_q1_overview.png"),
              "hardware_control": False, "policy_trained": False,
              "baidu_backup_status": "LOCAL_ONLY"}
    with (directory / "render_summary.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
