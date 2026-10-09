#!/usr/bin/env python3
"""Independently inspect a local UMR pilot without promoting it to a control reference."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dataset_input import identity, knee_geometry, smplx_stageii
from egolocate_input import SMPL_BODY_IDS
from q1_retarget import Q1Retargeter
from validate_egolocate import render_robot_cpu


ROOT = Path(__file__).resolve().parents[2]


def flexion(positions, names, side="right"):
    hip, knee, ankle = [names.index(f"{side}_{part}") for part in ("hip", "knee", "ankle")]
    upper = positions[:, hip] - positions[:, knee]
    lower = positions[:, ankle] - positions[:, knee]
    cosine = np.sum(upper * lower, axis=-1) / (np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1))
    return np.rad2deg(np.pi - np.arccos(np.clip(cosine, -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=ROOT / "data/third_party/umr_q1_20261009")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=3)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        parser.error("Preserve existing inspection")
    status_path = args.experiment / f"pilot_status_attempt{args.attempt}.json"
    status = json.loads(status_path.read_text())
    if [s["stage"] for s in status["stages"]] != ["build", "train", "retarget"] or any(
            s["status"] != "COMPLETE" for s in status["stages"]):
        raise ValueError("Pilot did not finish all stages")
    for item in [status["config"], status["retarget_artifact"], status["upstream_license"]] + [
            s["log_identity"] for s in status["stages"]]:
        if identity(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Pilot evidence changed: " + item["path"])
    config = json.loads(Path(status["config"]["path"]).read_text())
    retargeter = Q1Retargeter()
    sim = retargeter.sim
    with np.load(status["retarget_artifact"]["path"], allow_pickle=False) as result:
        qpos = result["qpos"].astype(float)
        fps = float(result["fps"].item())
        frame_ids = result["frame_ids"].copy()
        output_xml = Path(result["robot_xml"].item())
        scale = float(result["smpl_scale"].item())
        ground_z = float(result["ground_z"].item())
    if qpos.shape != (len(frame_ids), sim.model.nq) or len(frame_ids) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Invalid qpos shape or values")
    if not np.isfinite(fps) or fps <= 0 or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-5):
        raise ValueError("Invalid output FPS or root quaternion")
    derived = mujoco.MjModel.from_xml_path(str(output_xml))
    if derived.nq != sim.model.nq or derived.njnt != sim.model.njnt:
        raise ValueError("Export model differs from Q1 topology")
    for joint in range(derived.njnt):
        if (mujoco.mj_id2name(derived, mujoco.mjtObj.mjOBJ_JOINT, joint) !=
                mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or
                derived.jnt_qposadr[joint] != sim.model.jnt_qposadr[joint]):
            raise ValueError("Export joint order differs from Q1")
    original_data, derived_data = mujoco.MjData(sim.model), mujoco.MjData(derived)
    actual, mesh_min_z, fk_error = [], [], 0.0
    for pose in qpos:
        original_data.qpos[:], derived_data.qpos[:] = pose, pose
        mujoco.mj_forward(sim.model, original_data)
        mujoco.mj_forward(derived, derived_data)
        fk_error = max(fk_error, float(np.max(np.abs(original_data.xpos - derived_data.xpos))))
        actual.append(original_data.xpos[retargeter.bodies].copy())
        lowest = np.inf
        for geom in range(sim.model.ngeom):
            if sim.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = sim.model.geom_dataid[geom]
            begin, count = sim.model.mesh_vertadr[mesh], sim.model.mesh_vertnum[mesh]
            vertices = sim.model.mesh_vert[begin:begin + count]
            z = vertices @ original_data.geom_xmat[geom].reshape(3, 3)[2] + original_data.geom_xpos[geom, 2]
            lowest = min(lowest, float(z.min()))
        mesh_min_z.append(lowest)
    if fk_error > 1e-9:
        raise ValueError("Derived model FK differs from pinned Q1")
    actual = np.asarray(actual)
    source_path = Path(config["motion"]["data"]) / (config["motion"]["seq_key"] + ".npz")
    sys.path[:0] = [str(args.vendor), str(args.vendor / "scripts")]
    import smpl_surface_retarget_common as upstream
    sequence = upstream.load_smplx_npz_motion(source_path)
    expected_ids = upstream.slice_frames(len(sequence["pose_aa"]), config["motion"]["start"], config["motion"].get("end", -1),
                                         config["motion"]["stride"], config["motion"]["max_frames"])
    if not np.array_equal(frame_ids, expected_ids) or not np.isclose(fps, sequence["fps"] / config["motion"]["stride"]):
        raise ValueError("Frame selection or output timebase differs from source")
    _, source_joints, _ = upstream.smplx_motion_vertices_joints(
        sequence, frame_ids, Path(config["smplx_model_dir"]), device="cpu")
    source = source_joints[:, [SMPL_BODY_IDS[name] for name in retargeter.names]]
    _, original_source, _, original_details = smplx_stageii(
        source_path, Path(config["smplx_model_dir"]).parent,
        float(frame_ids[-1] / sequence["fps"] + 1e-8), retargeter.names)
    original_source = original_source[frame_ids]
    times = (frame_ids - frame_ids[0]) / sequence["fps"]
    joints = qpos[:, sim.q_indices]
    speed = np.abs(np.diff(joints, axis=0) / np.diff(times)[:, None])
    raw_limits = sim.model.jnt_range[sim.joint_ids]
    raw_violation = np.maximum(raw_limits[:, 0] - joints, joints - raw_limits[:, 1])
    margin_violation = np.maximum(sim.lower - joints, joints - sim.upper)
    peak = np.unravel_index(speed.argmax(), speed.shape)
    robot_knee, human_knee = flexion(actual, retargeter.names), flexion(source, retargeter.names)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, layout="constrained")
    axes[0].plot(times, human_knee, label="UMR human (10 betas)", color="#26776f")
    axes[0].plot(times, robot_knee, label="Q1 UMR", color="#b34242")
    axes[0].set(xlabel="Source time (s)", ylabel="Right knee flexion (deg)", title="Same sampled frames")
    axes[0].legend(fontsize=9)
    axes[1].plot(times[1:], speed.max(axis=1), color="#615385")
    axes[1].axhline(sim.config["reference_velocity_limit_rad_s"], color="#b34242", linestyle="--", label="Existing PD reference limit")
    axes[1].set(xlabel="Source time (s)", ylabel="Peak joint speed (rad/s)", title="Raw UMR trajectory")
    axes[1].legend(fontsize=8)
    fig.suptitle("UMR Q1 pilot: 256 points / 50 epochs / no dynamic balance test", fontsize=12)
    plot = args.experiment / "pilot_diagnostics.png"
    if plot.exists():
        raise ValueError("Preserve existing diagnostic plot")
    fig.savefig(plot)
    plt.close(fig)
    pick = int(np.argmax(robot_knee))
    original_data.qpos[:] = qpos[pick]
    mujoco.mj_forward(sim.model, original_data)
    preview = args.experiment / "peak_knee_q1.png"
    if preview.exists():
        raise ValueError("Preserve existing preview")
    render_robot_cpu(sim.model, original_data, sim.pelvis_id).save(preview)
    report = {
        "usage": "TEST_ONLY", "training_allowed": False, "backup_status": "LOCAL_ONLY",
        "scope": "SMALL_SURFACE_CORRESPONDENCE_AND_KINEMATIC_PILOT", "policy_trained": False,
        "hardware_control": False, "dynamic_balance_verified": False,
        "inspector": identity(__file__), "status": identity(status_path), "run": status,
        "source": identity(source_path), "source_model": original_details["model"],
        "umr_source_model": identity(Path(config["smplx_model_dir"]) / f"SMPLX_{sequence['gender'].upper()}.pkl"),
        "source_num_betas": len(sequence["beta"]), "umr_used_num_betas": 10,
        "shape_preserving_comparison": False,
        "original_vs_umr_source_joint_max_abs_error_m": float(np.max(np.abs(original_source - source))),
        "q1_model": identity(sim.asset_path), "derived_model": identity(output_xml),
        "derived_vs_original_model_fk_max_abs_error_m": fk_error,
        "frames": len(qpos), "source_fps": sequence["fps"], "output_fps": fps,
        "first_source_frame": int(frame_ids[0]), "last_source_frame": int(frame_ids[-1]),
        "duration_s": float(times[-1]), "qpos_shape": list(qpos.shape), "finite": True,
        "raw_joint_limit_violations_over_1e_6_rad": int((raw_violation > 1e-6).sum()),
        "raw_joint_limit_max_excess_rad": float(max(0, raw_violation.max())),
        "configured_margin_violations_over_1e_6_rad": int((margin_violation > 1e-6).sum()),
        "configured_margin_max_excess_rad": float(max(0, margin_violation.max())),
        "peak_joint_speed_rad_s": float(speed.max()), "peak_speed_joint": sim.names[peak[1]],
        "joint_intervals_over_2_rad_s": int((speed > 2).sum()),
        "within_pd_conversion_numeric_budget": bool(raw_violation.max() <= 1e-8 and margin_violation.max() <= 1e-8 and speed.max() <= 2),
        "strict_pd_reference_limits_passed": bool(raw_violation.max() <= 0 and margin_violation.max() <= 0 and speed.max() <= 2),
        "pd_reference_exported": False,
        "requested_per_step_joint_delta": config["robot"].get("dof_max_dq_box"),
        "post_filter_mode": config["solver"].get("trajectory_filter_mode", "UPSTREAM_DEFAULT"),
        "umr_input_leg_geometry": knee_geometry(source, retargeter.names),
        "original_source_leg_geometry": knee_geometry(original_source, retargeter.names),
        "q1_leg_geometry": knee_geometry(actual, retargeter.names), "uniform_source_scale": scale,
        "root_xyz_range_m": np.ptp(qpos[:, :3], axis=0).tolist(),
        "visual_mesh_min_z_m": float(min(mesh_min_z)), "ground_z_m": ground_z,
        "frames_with_visual_mesh_below_ground_over_1mm": int((np.asarray(mesh_min_z) < ground_z - .001).sum()),
        "floor_scope": "ALL_VISUAL_MESH_VERTICES_ONLY_NOT_CONTACT_OR_COLLISION_ACCEPTANCE",
        "upstream_worktree_status": subprocess.check_output(["git", "status", "--short"], cwd=args.vendor, text=True),
        "upstream_files": [identity(path) for path in sorted((args.vendor / "scripts").glob("*.py"))] + [
            identity(args.vendor / name) for name in ("LICENSE", "humanoid_retarget_defaults.json", "view_smpl_mujoco.py", "assets/smplx_parts_segm.pkl")],
        "versions": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "torch", "smplx")},
        "artifacts": [identity(path) for path in sorted(args.experiment.rglob("*")) if path.is_file()],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("frames", "duration_s", "peak_joint_speed_rad_s", "raw_joint_limit_violations_over_1e_6_rad", "configured_margin_max_excess_rad", "visual_mesh_min_z_m", "q1_leg_geometry", "umr_input_leg_geometry", "original_vs_umr_source_joint_max_abs_error_m")}, indent=2))


if __name__ == "__main__":
    main()
