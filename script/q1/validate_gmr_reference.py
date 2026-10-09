#!/usr/bin/env python3
"""Normalize tiny bound roundoff and validate saved GMR references with supported PD."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from dataset_input import identity
from q1_retarget import Q1Retargeter
from q1_sim import JointReference, Q1Sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserve existing validated references")
    args.output_dir.mkdir(parents=True)
    original = json.loads(args.summary.read_text())
    if original.get("frame_speed_limit_rad_s") != 2 or original["usage"] != "TEST_ONLY":
        raise ValueError("Require the completed TEST_ONLY 2 rad/s GMR comparison")
    r = Q1Retargeter()
    report = {"usage": "TEST_ONLY", "training_allowed": False, "source_summary": identity(args.summary),
              "validator": identity(__file__), "roundoff_budget_rad": 1e-8, "cases": [],
              "scope": "TINY_BOUND_NORMALIZATION_AND_FIXED_PELVIS_PD", "balance_verified": False,
              "policy_trained": False, "hardware_control": False, "backup_status": "LOCAL_ONLY"}
    for case in original["cases"]:
        if case["status"] != "COMPLETE":
            raise ValueError("Incomplete source GMR case")
        for artifact in case["artifacts"]:
            if identity(artifact["path"])["sha256"] != artifact["sha256"]:
                raise ValueError("Saved GMR artifact identity changed")
        trace = next(a for a in case["artifacts"] if Path(a["path"]).name == "kinematic_trace.npz")
        with np.load(trace["path"], allow_pickle=False) as data:
            times, qpos = data["time_s"].copy(), data["qpos"].copy()
        raw = qpos[:, r.sim.q_indices]
        violation = float(max(0., (r.sim.lower - raw).max(), (raw - r.sim.upper).max()))
        if violation > 1e-8:
            raise ValueError(f"{case['id']} has real reference-margin violation: {violation}")
        clipped = np.clip(raw, r.sim.lower, r.sim.upper)
        row = {"id": case["id"], "source_trace": trace, "raw_bound_violation_max_rad": violation,
               "normalization_max_abs_change_rad": float(np.max(np.abs(clipped - raw)))}
        reference = JointReference(times, clipped, r.sim.names, r.sim.names)
        reference.validate_limits(r.sim)
        directory = args.output_dir / case["id"]
        directory.mkdir()
        reference.save(directory / "joint_reference.npz")
        if case["supported_replay_run"] and row["normalization_max_abs_change_rad"] == 0:
            row["supported_replay"] = next(a for a in case["artifacts"] if Path(a["path"]).name == "supported_replay.npz")
            row["supported_max_joint_rmse_rad"] = case["supported_max_joint_rmse_rad"]
            row["replay_reused_verified_unchanged_reference"] = True
        else:
            sim = Q1Sim(r.sim.config_path, supported=True)
            sim.data.qpos[sim.q_indices] = clipped[0]
            mujoco.mj_forward(sim.model, sim.data)
            replay_times, actual, targets = [], [], []
            while sim.data.time < times[-1] - sim.dt / 2:
                target, velocity = reference.sample(sim.data.time)
                observation = sim.advance(target, velocity)
                replay_times.append(observation["time_s"])
                actual.append(observation["joint_position_rad"])
                targets.append(target)
            actual, targets = map(np.asarray, (actual, targets))
            output = directory / "supported_replay.npz"
            np.savez_compressed(output, time_s=replay_times, actual_rad=actual, target_rad=targets,
                                usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
            row["supported_replay"] = identity(output)
            row["supported_max_joint_rmse_rad"] = float(np.sqrt(np.mean((actual - targets) ** 2, axis=0)).max())
            row["replay_reused_verified_unchanged_reference"] = False
        row.update({"status": "COMPLETE", "reference": identity(directory / "joint_reference.npz"),
                    "joint_limit_violations": 0,
                    "max_joint_speed_rad_s": float(np.abs(reference.slopes).max()),
                    "supported_initialization": "FITTED_FIRST_FRAME_JOINTS_FIXED_PELVIS"})
        report["cases"].append(row)
        (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{case['id']}: bound roundoff {violation:.3e} rad, supported RMSE {row['supported_max_joint_rmse_rad']:.4f} rad", flush=True)
    return 0


if __name__ == "__main__":
    main()
