#!/usr/bin/env python3
"""Prepare a bounded UMR Q1 pilot with a fitted canonical T-pose and existing assets."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from dataset_input import identity
from q1_retarget import Q1Retargeter
from validate_egolocate import render_robot_cpu


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=ROOT / "data/third_party/umr_q1_20261009")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-speed-limit", type=float)
    args = parser.parse_args()
    args.vendor = args.vendor.resolve()
    args.output_dir = args.output_dir.resolve()
    required = ("LICENSE", "humanoid_retarget_defaults.json", "view_smpl_mujoco.py",
                "scripts/humanoid_retarget_pipeline.py", "assets/smplx_parts_segm.pkl")
    missing = [str(args.vendor / name) for name in required if not (args.vendor / name).is_file()]
    if missing:
        parser.error("Incomplete upstream checkout: " + ", ".join(missing))
    if args.reference_speed_limit is not None and (
            not np.isfinite(args.reference_speed_limit) or args.reference_speed_limit <= 0):
        parser.error("Reference speed must be positive and finite")
    if args.output_dir.exists():
        parser.error("Preserve existing UMR preparation")
    args.output_dir.mkdir(parents=True)
    r = Q1Retargeter()
    arms = [i for i, name in enumerate(r.sim.names) if "shoulder" in name or "elbow" in name]
    bodies, targets = [], []
    for side, sign in (("left", 1), ("right", -1)):
        ids = [r.names.index(f"{side}_{part}") for part in ("shoulder", "elbow", "wrist")]
        positions = r.robot_positions[ids]
        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        for part, distance in (("elbow", lengths[0]), ("wrist", lengths.sum())):
            bodies.append(r.bodies[r.names.index(f"{side}_{part}")])
            targets.append(positions[0] + [0, sign * distance, 0])
    targets = np.asarray(targets)
    data = mujoco.MjData(r.sim.model)
    data.qpos[:] = r.sim.data.qpos.copy()

    def residual(values):
        data.qpos[r.sim.q_indices[arms]] = values
        mujoco.mj_forward(r.sim.model, data)
        return np.concatenate(((data.xpos[bodies] - targets).ravel(), 1e-5 * values))

    fitted = least_squares(residual, r.sim.neutral[arms],
                           bounds=(r.sim.lower[arms], r.sim.upper[arms]),
                           max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    residual(fitted.x)
    tpose_error = float(np.linalg.norm(data.xpos[bodies] - targets, axis=-1).max())
    if not fitted.success or tpose_error > .05:
        raise ValueError(f"Q1 T-pose fitting did not meet pilot budget: {tpose_error}")
    qpos = data.qpos.copy()
    import matplotlib
    matplotlib.use("Agg")
    render_robot_cpu(r.sim.model, data, r.sim.pelvis_id).save(args.output_dir / "q1_tpose.png")
    config = {
        "smplx_model_dir": "/media/yu/FAFF-E9771/data/humanplus-phuma/assets/body_models/smplx",
        "smpl_template": {"source": "motion", "type": "smplx", "use_betas": True, "use_gender": True},
        "robot": {"name": "q1_pilot", "xml": str(r.sim.asset_path), "point_cloud_center": "body:torso_link",
                  "tpose_qpos": {name: float(qpos[index]) for name, index in zip(r.sim.names, r.sim.q_indices)},
                  "joint_limits": {name: [float(lo), float(hi)] for name, lo, hi in zip(r.sim.names, r.sim.lower, r.sim.upper)}},
        "correspondence": {
            "dataset": {"out": str(args.output_dir / "correspondence.npz"), "num_points": 256,
                        "surface_oversample_ratio": 4, "exterior_method": "first_hit"},
            "train": {"out_dir": str(args.output_dir / "correspondence_training"),
                      "epochs": 50, "device": "cuda", "log_every": 10, "save_every": 50,
                      "edge_k": 8, "repulsion_k": 4}},
        "motion": {"data": "/media/yu/FAFF-E9771/data/project_snapshots/HUMAN_PLUS_A3/2026-08-09/datasets/amass_sources",
                   "seq_key": "G3_-_front_kick_stageii", "start": 0, "stride": 4, "max_frames": 32},
        "solver": {"batch_size": 8},
        "retarget": {"out": str(args.output_dir / "q1_retarget.npz"),
                     "smplx_batch_size": 32, "smplx_batch_size_max": 64},
        "view": {"enabled": False},
    }
    speed_recipe = None
    if args.reference_speed_limit is not None:
        source_path = Path(config["motion"]["data"]) / (config["motion"]["seq_key"] + ".npz")
        with np.load(source_path, allow_pickle=False) as source:
            source_fps = float(source["mocap_frame_rate"])
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Invalid source FPS for speed bound")
        fps = source_fps / config["motion"]["stride"]
        delta = args.reference_speed_limit / fps * (1 - 1e-5)
        config["robot"]["dof_max_dq_box"] = {name: delta for name in r.sim.names}
        config["solver"].update(iters=1, pose_init_iters=15, trajectory_filter_mode="off")
        speed_recipe = {"requested_rad_s": args.reference_speed_limit, "output_fps": fps,
                        "per_step_joint_delta_rad": delta, "iterations_per_moving_frame": 1,
                        "initial_pose_iterations": 15, "post_trajectory_filter": "off",
                        "bound_verified_only_by_independent_output_inspection": True}
    config_path = args.output_dir / "q1_pilot_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"usage": "TEST_ONLY", "training_allowed": False,
                "scope": "UMR_DEPENDENCY_AND_MAPPING_PILOT_NOT_POLICY_TRAINING",
                "preparer": identity(__file__), "q1_model": identity(r.sim.asset_path),
                "config": identity(config_path), "tpose_image": identity(args.output_dir / "q1_tpose.png"),
                "tpose_max_endpoint_error_m": tpose_error, "q1_tpose_joint_limit_violations": 0,
                "correspondence_points": 256, "correspondence_epochs": 50, "motion_max_frames": 32,
                "full_quality_configuration": False, "policy_trained": False, "hardware_control": False,
                "speed_recipe": speed_recipe,
                "upstream_license": identity(args.vendor / "LICENSE")}
    (args.output_dir / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
