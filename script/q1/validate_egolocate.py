#!/usr/bin/env python3
"""Evaluate real recorded body input using kinematic IK and supported PD only."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egolocate_input import convert_bundle, save_test_only
from q1_retarget import Q1Retargeter
from q1_sim import JointReference, Q1Sim


def leg_scale(source, retargeter):
    robot_lengths, human_lengths = [], []
    for side in ("left", "right"):
        ids = [source.names.index(name) for name in ("pelvis", f"{side}_hip", f"{side}_knee", f"{side}_ankle")]
        robot_lengths.append(np.linalg.norm(np.diff(retargeter.robot_positions[ids], axis=0), axis=-1).sum())
        human_lengths.append(np.median(np.linalg.norm(np.diff(source.positions[:, ids], axis=1), axis=-1).sum(axis=-1)))
    scale = float(np.mean(robot_lengths) / np.mean(human_lengths))
    if not np.isfinite(scale) or not 0.1 < scale < 2:
        raise ValueError("Implausible metre-scale anthropometry")
    return scale, {"method": "MEAN_MAPPED_PELVIS_HIP_KNEE_ANKLE_CHAIN_LENGTH_RATIO",
                   "robot_leg_chain_lengths_m": list(map(float, robot_lengths)),
                   "human_median_leg_chain_lengths_m": list(map(float, human_lengths)), "scale": scale}


def kinematic_orientation_errors(retargeter, qposes, targets):
    data = mujoco.MjData(retargeter.sim.model)
    errors = []
    for q, target in zip(qposes, targets):
        data.qpos[:] = q
        mujoco.mj_forward(retargeter.sim.model, data)
        actual = Rotation.from_matrix(data.xmat[retargeter.bodies].reshape(-1, 3, 3))
        errors.append((actual * Rotation.from_matrix(target).inv()).magnitude())
    return np.asarray(errors)


def render_overview(retargeter, source, qposes, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from io import BytesIO

    left = source.names.index("left_wrist")
    right = source.names.index("right_wrist")
    picks = [0, int(np.argmax(np.linalg.norm(source.positions[:, left] - source.positions[0, left], axis=-1))),
             int(np.argmax(np.linalg.norm(source.positions[:, right] - source.positions[0, right], axis=-1))),
             len(source.times) - 1]
    links = [("pelvis", "spine3"), ("pelvis", "left_hip"), ("pelvis", "right_hip"),
             ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
             ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
             ("spine3", "left_shoulder"), ("spine3", "right_shoulder"),
             ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
             ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]
    montage = Image.new("RGB", (960, len(picks) * 540), "white")
    data = mujoco.MjData(retargeter.sim.model)
    with mujoco.Renderer(retargeter.sim.model, height=500, width=480) as renderer:
        for row, frame in enumerate(picks):
            points = source.positions[frame] - source.positions[frame, source.names.index("pelvis")]
            fig = plt.figure(figsize=(4.8, 5), dpi=100)
            ax = fig.add_subplot(projection="3d")
            for a, b in links:
                ids = [source.names.index(a), source.names.index(b)]
                color = "#2471a3" if "knee" in (a + b) or "ankle" in (a + b) else "#a93226"
                ax.plot(*points[ids].T, color=color, marker="o", markersize=3)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1.1, 0.9), xlabel="Forward (m)", ylabel="Left (m)", zlabel="Up (m)")
            ax.set_box_aspect((2, 2, 2))
            ax.view_init(elev=12, azim=130)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            montage.paste(Image.open(buf).convert("RGB"), (0, row * 540 + 40))
            data.qpos[:] = qposes[frame]
            mujoco.mj_forward(retargeter.sim.model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.xpos[retargeter.sim.pelvis_id] + [0, 0, 0.08]
            camera.distance, camera.azimuth, camera.elevation = 1.9, 130, -12
            renderer.update_scene(data, camera=camera)
            montage.paste(Image.fromarray(renderer.render()), (480, row * 540 + 40))
            draw = ImageDraw.Draw(montage)
            draw.text((10, row * 540 + 10), f"Recorded SMPL body: t={source.times[frame]:.2f}s", fill="black")
            draw.text((490, row * 540 + 10), "Q1 kinematic IK (not dynamic balance)", fill="black")
    montage.save(directory / "body_q1_overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    retargeter = Q1Retargeter()
    source, conversion = convert_bundle(args.input, retargeter.names, retargeter.sim.config["reference_hz"])
    save_test_only(source, args.output_dir / "body_reference.npz")
    scale, scaling = leg_scale(source, retargeter)
    started = time.perf_counter()
    reference, qposes, targets, target_rotations, body_errors = retargeter.retarget(source, scale)
    ik_wall_seconds = time.perf_counter() - started
    save_test_only(reference, args.output_dir / "joint_reference.npz")
    orientation_errors = kinematic_orientation_errors(retargeter, qposes, target_rotations)
    np.savez_compressed(args.output_dir / "kinematic_trace.npz", time_s=source.times, qpos=qposes,
                        target_position_m=targets, target_rotation_matrix=target_rotations,
                        body_position_error_m=body_errors, body_orientation_error_rad=orientation_errors,
                        usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
    sim = Q1Sim(retargeter.sim.config_path, supported=True)
    reference = JointReference.load(args.output_dir / "joint_reference.npz", sim.names)
    reference.validate_limits(sim)
    times, actual_positions, target_positions, torques = [], [], [], []
    while sim.data.time < reference.times[-1] - sim.dt / 2:
        target, velocity = reference.sample(sim.data.time)
        obs = sim.advance(target, velocity)
        times.append(obs["time_s"])
        actual_positions.append(obs["joint_position_rad"])
        target_positions.append(target)
        torques.append(sim.last_torque.copy())
    actual_positions, target_positions = map(np.asarray, (actual_positions, target_positions))
    replay_errors = actual_positions - target_positions
    np.savez_compressed(args.output_dir / "supported_replay.npz", time_s=times,
                        position_rad=actual_positions, target_rad=target_positions, torque_nm=torques,
                        usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
    if args.render:
        render_overview(retargeter, source, qposes, args.output_dir)
    position_norms = np.linalg.norm(body_errors, axis=-1)
    speeds = np.abs(np.diff(reference.positions, axis=0) / np.diff(reference.times)[:, None])
    margin = np.minimum(reference.positions - sim.lower, sim.upper - reference.positions)
    root = Path(__file__).resolve().parents[2]
    report = {
        "stage": args.output_dir.name, "usage": "TEST_ONLY", "training_allowed": False,
        "scope": "REAL_RECORDED_POSE_INPUT_KINEMATIC_IK_SUPPORTED_PD_REPLAY",
        "conversion": conversion, "scaling": scaling,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ["mujoco", "mink", "qpsolvers", "quadprog", "scipy", "numpy"]},
        "source_files": [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in [Path(__file__).resolve(), root / "script/q1/egolocate_input.py",
                                   root / "script/q1/inspect_egolocate.py", root / "script/q1/q1_retarget.py",
                                   root / "script/q1/q1_sim.py", root / "script/q1/retarget_config.json", root / "script/q1/sim_config.json"]],
        "input_decoded": True, "finite_kinematic_output": bool(np.isfinite(qposes).all()),
        "kinematic_max_body_position_error_m": float(position_norms.max()),
        "kinematic_body_position_error_p95_m": float(np.quantile(position_norms, 0.95)),
        "kinematic_max_body_orientation_error_rad": float(orientation_errors.max()),
        "kinematic_body_orientation_error_p95_rad": float(np.quantile(orientation_errors, 0.95)),
        "reference_max_joint_speed_rad_s": float(speeds.max()),
        "reference_joint_limit_violations": int(np.count_nonzero(margin < -1e-8)),
        "reference_fraction_joint_samples_within_0_001_rad_of_limit": float(np.mean(margin < 0.001)),
        "supported_replay_duration_s": float(sim.data.time),
        "supported_replay_max_joint_rmse_rad": float(np.sqrt(np.mean(replay_errors ** 2, axis=0)).max()),
        "supported_replay_max_abs_joint_error_rad": float(np.abs(replay_errors).max()),
        "supported_replay_torque_saturation_fraction": sim.torque_saturated / sim.torque_samples,
        "supported_replay_finite": bool(np.isfinite(actual_positions).all()),
        "pose_fidelity_validated": False, "root_translation_tracking_tested": False,
        "balance_verified": False, "policy_trained": False, "hardware_control": False,
        "body_residuals": [{"name": name, "position_p95_m": float(np.quantile(position_norms[:, i], 0.95)),
                            "orientation_p95_rad": float(np.quantile(orientation_errors[:, i], 0.95))}
                           for i, name in enumerate(source.names)],
        "artifacts": [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(args.output_dir.iterdir())], "baidu_backup_status": "LOCAL_ONLY",
    }
    report["ik_wall_s"] = ik_wall_seconds
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
