#!/usr/bin/env python3
"""FK round trip and supported PD replay, explicitly without human SMPL input."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from q1_retarget import BodyReference, Q1Retargeter
from q1_sim import JointReference, Q1Sim
from validate_control import synthetic_reference


def make_synthetic_source(retargeter):
    sim = retargeter.sim
    reference, _ = synthetic_reference(sim)
    calibration_q = sim.data.qpos.copy()
    positions, quaternions, qposes = [], [], []
    for frame, timestamp in enumerate(reference.times):
        sim.data.qpos[:] = calibration_q
        sim.data.qpos[sim.q_indices] = reference.positions[frame]
        sim.data.qpos[0] += 0.04 * timestamp / reference.times[-1]
        sim.data.qpos[3:7] = Rotation.from_euler("z", 0.1 * np.sin(2 * np.pi * timestamp / 24)).as_quat(scalar_first=True)
        mujoco.mj_forward(sim.model, sim.data)
        positions.append(sim.data.xpos[retargeter.bodies].copy())
        quaternions.append(Rotation.from_matrix(
            sim.data.xmat[retargeter.bodies].reshape(-1, 3, 3)).as_quat(scalar_first=True))
        qposes.append(sim.data.qpos.copy())
    sim.reset()
    source = BodyReference(reference.times, retargeter.names, positions, quaternions,
                           retargeter.robot_positions,
                           Rotation.from_matrix(retargeter.robot_rotations).as_quat(scalar_first=True),
                           retargeter.names, "SYNTHETIC_Q1_FK_NOT_HUMAN_SMPL")
    return source, reference, np.asarray(qposes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new experiment directory")
    args.output_dir.mkdir(parents=True)
    retargeter = Q1Retargeter()
    source, expected, expected_qpos = make_synthetic_source(retargeter)
    source.save(args.output_dir / "synthetic_body_reference.npz")
    source = BodyReference.load(args.output_dir / "synthetic_body_reference.npz", retargeter.names)
    started = time.perf_counter()
    reference, qposes, targets, rotations, body_errors = retargeter.retarget(source, scale=1.0)
    wall_seconds = time.perf_counter() - started
    reference.save(args.output_dir / "joint_reference.npz")
    np.savez_compressed(args.output_dir / "kinematic_trace.npz", time_s=source.times,
                        expected_qpos=expected_qpos, retargeted_qpos=qposes,
                        target_position_m=targets, target_rotation_matrix=rotations,
                        body_position_error_m=body_errors)
    joint_errors = reference.positions - expected.positions
    rmse = np.sqrt(np.mean(joint_errors ** 2, axis=0))
    root_error = float(np.abs(qposes[:, :3] - expected_qpos[:, :3]).max())
    root_orientation_error = float((Rotation.from_quat(qposes[:, 3:7], scalar_first=True)
                                    * Rotation.from_quat(expected_qpos[:, 3:7], scalar_first=True).inv()).magnitude().max())
    excursions = np.ptp(reference.positions, axis=0)
    sim = Q1Sim(retargeter.sim.config_path, supported=True)
    reference = JointReference.load(args.output_dir / "joint_reference.npz", sim.names)
    reference.validate_limits(sim)
    replay_times, replay_positions, replay_targets = [], [], []
    while sim.data.time < reference.times[-1] - sim.dt / 2:
        target, velocity = reference.sample(sim.data.time)
        obs = sim.advance(target, velocity)
        replay_times.append(obs["time_s"])
        replay_positions.append(obs["joint_position_rad"])
        replay_targets.append(target)
    replay_positions, replay_targets = map(np.asarray, (replay_positions, replay_targets))
    replay_rmse = np.sqrt(np.mean((replay_positions - replay_targets) ** 2, axis=0))
    np.savez_compressed(args.output_dir / "supported_replay.npz", time_s=replay_times,
                        position_rad=replay_positions, target_rad=replay_targets)
    max_body_error = float(np.linalg.norm(body_errors, axis=-1).max())
    passed = (max_body_error < 0.01 and rmse.max() < 0.02 and root_error < 1e-6
              and root_orientation_error < 1e-6 and np.all(excursions > 0.015)
              and replay_rmse.max() < 0.08)
    root = Path(__file__).resolve().parents[2]
    report = {
        "stage": args.output_dir.name,
        "scope": "SYNTHETIC_KINEMATIC_ROUNDTRIP_AND_SUPPORTED_REPLAY",
        "source_kind": source.source_kind, "source_is_smpl": False,
        "mink_version": importlib.metadata.version("mink"), "mujoco_version": mujoco.__version__,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_files": [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
                         + sorted(Path(__file__).resolve().parent.glob("*.json"))],
        "frames": len(source.times), "duration_s": float(source.times[-1]), "ik_wall_s": wall_seconds,
        "max_body_position_error_m": max_body_error,
        "max_joint_roundtrip_rmse_rad": float(rmse.max()),
        "max_root_position_error_m": root_error,
        "max_root_orientation_error_rad": root_orientation_error,
        "all_joint_pulses_detected": bool(np.all(excursions > 0.015)),
        "supported_replay_max_joint_rmse_rad": float(replay_rmse.max()),
        "supported_replay_saturation_fraction": sim.torque_saturated / sim.torque_samples,
        "pipeline_passed": bool(passed), "balance_verified": False,
        "policy_trained": False, "hardware_control": False,
        "joints": [{"name": name, "roundtrip_rmse_rad": float(rmse[i]),
                    "excursion_rad": float(excursions[i])} for i, name in enumerate(sim.names)],
        "artifacts": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(args.output_dir.iterdir())],
        "baidu_backup_status": "LOCAL_ONLY",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
