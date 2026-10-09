#!/usr/bin/env python3
"""Run pinned upstream GMR with an experimental Q1 mapping, kinematics only."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dataset_input import identity, knee_geometry, load_dataset_reference, paired_smplx_standing
from q1_retarget import Q1Retargeter, heading_rotation
from validate_egolocate import leg_scale, render_overview
from q1_sim import JointReference, Q1Sim


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
UPSTREAM_HASHES = {
    "LICENSE": "c5a85c0b0012230739a0ab8f30eafba7f8ee7a1b29e025d01e27fa330298e522",
    "motion_retarget.py": "3a4f9fb0d13b204c8dcbb2ab62db59bce9de81b1f5911167aa73f079e0c90663",
    "params.py": "3506cdd277375139770b0ced7108719145620a389a8dee85bc6c94c6bf135bef",
}


def load_gmr(vendor, model_path, config_path):
    for name, digest in UPSTREAM_HASHES.items():
        if identity(vendor / name)["sha256"] != digest:
            raise ValueError(f"Pinned upstream {name} identity changed")
    package = types.ModuleType("q1_pinned_gmr")
    package.__path__ = [str(vendor)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("q1_pinned_gmr.motion_retarget", vendor / "motion_retarget.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROBOT_XML_DICT["q1_experimental"] = model_path
    module.IK_CONFIG_DICT.setdefault("q1_reference", {})["q1_experimental"] = config_path

    # Upstream passes limits as argument 6; installed Mink 1.2 uses that slot for safety_break.
    def compatible_solve(configuration, tasks, dt, solver, damping, limits):
        return mink.solve_ik(configuration, tasks, dt, solver, damping=damping,
                             safety_break=True, limits=limits)

    module.mink = types.SimpleNamespace(Configuration=mink.Configuration, FrameTask=mink.FrameTask,
                                        ConfigurationLimit=mink.ConfigurationLimit, VelocityLimit=mink.VelocityLimit,
                                        SE3=mink.SE3, SO3=mink.SO3, solve_ik=compatible_solve)
    return module.GeneralMotionRetargeting("q1_reference", "q1_experimental", solver="quadprog", verbose=False)


def configuration(source, conversion, retargeter, scale):
    standing, recipe = paired_smplx_standing(source, conversion)
    human_cal = Rotation.from_quat(standing.calibration_orientations, scalar_first=True).as_matrix()
    align = heading_rotation(retargeter.robot_rotations[retargeter.root]) @ heading_rotation(human_cal[retargeter.root]).T
    human_cal = align @ human_cal
    root = retargeter.root
    mapped_rest = (standing.calibration_positions - standing.calibration_positions[root]) @ align.T
    table = {}
    for i, (human, robot) in enumerate(retargeter.config["body_map"].items()):
        offset_rotation = human_cal[i].T @ retargeter.robot_rotations[i]
        global_offset = retargeter.robot_positions[i] - retargeter.robot_positions[root] - scale * mapped_rest[i]
        local_offset = retargeter.robot_rotations[i].T @ global_offset
        table[robot] = [human, 20.0, 1.0, local_offset.tolist(),
                        Rotation.from_matrix(offset_rotation).as_quat(scalar_first=True).tolist()]
    config = {"human_height_assumption": 1.0, "human_root_name": "pelvis", "robot_root_name": "pelvis",
              "ground_height": 0.0, "human_scale_table": {name: scale for name in source.names},
              "use_ik_match_table1": True, "use_ik_match_table2": False,
              "ik_match_table1": table, "ik_match_table2": table}
    positions = (source.positions - source.positions[0, root]) @ align.T + retargeter.robot_positions[root] / scale
    quats = Rotation.from_matrix(align @ Rotation.from_quat(source.orientations.reshape(-1, 4),
                                                           scalar_first=True).as_matrix()).as_quat(scalar_first=True)
    quats = quats.reshape(source.orientations.shape)
    return config, positions, quats, recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=ROOT / "data/third_party/gmr_bb1bbe407747")
    parser.add_argument("--source-suite", type=Path, default=ROOT / "data/experiments/q1_datasets_20261009")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--frame-speed-limit", type=float)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserve existing experimental GMR results")
    if args.frame_speed_limit is not None and (not np.isfinite(args.frame_speed_limit) or args.frame_speed_limit <= 0):
        parser.error("Frame speed limit must be positive")
    args.output_dir.mkdir(parents=True)
    report = {"usage": "TEST_ONLY", "training_allowed": False, "upstream_commit": UPSTREAM_COMMIT,
              "upstream": [identity(args.vendor / name) for name in UPSTREAM_HASHES],
              "driver": identity(__file__), "cases": [],
              "q1_source_files": [identity(HERE / name) for name in (
                  "q1_retarget.py", "dataset_input.py", "validate_egolocate.py", "q1_sim.py",
                  "retarget_config.json", "sim_config.json")],
              "mapping": "EXPERIMENTAL_Q1_SYNTHETIC_STANDING_OFFSETS_UNIFORM_LEG_SCALE",
              "solver_behavior": "UPSTREAM_ITERATIVE_GMR_ONE_MATCH_TABLE_NO_FRAME_VELOCITY_LIMIT",
              "compatibility": "ONLY_ADAPT_POSITIONAL_LIMITS_TO_MINK_1_2_KEYWORD",
              "frame_speed_limit_rad_s": args.frame_speed_limit,
              "frame_speed_method": "INTERSECT_JOINT_POSITION_LIMITS_WITH_PREVIOUS_OUTPUT_PLUS_MINUS_SPEED_TIMES_FRAME_DT",
              "balance_verified": False, "policy_trained": False, "hardware_control": False,
              "backup_status": "LOCAL_ONLY"}
    for case in ("amass_stand", "amass_walk_turn", "amass_kick", "amass_swing_arms"):
        row = {"id": case, "status": "RUNNING"}
        report["cases"].append(row)
        try:
            directory = args.output_dir / case
            directory.mkdir()
            retargeter = Q1Retargeter()
            body = args.source_suite / case / "body_reference.npz"
            source, conversion = load_dataset_reference(body, retargeter.names)
            scale, scaling = leg_scale(source, retargeter)
            config, positions, quats, recipe = configuration(source, conversion, retargeter, scale)
            config_path = directory / "gmr_q1_config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            solver = load_gmr(args.vendor, retargeter.sim.asset_path, config_path)
            solver.configuration.update(retargeter.sim.data.qpos.copy())
            if args.frame_speed_limit is not None:
                solver.ik_limits = [mink.ConfigurationLimit(solver.model,
                    min_distance_from_limits=retargeter.sim.config["joint_limit_margin_rad"])]

            def frame_data(frame):
                return {name: [positions[frame, i].copy(), quats[frame, i].copy()] for i, name in enumerate(source.names)}

            # Record initial-pose fitting separately from the moving clip's 50 Hz timestamps.
            for _ in range(30):
                solver.retarget(frame_data(0))
            frame_limit = None
            if args.frame_speed_limit is not None:
                frame_limit = solver.ik_limits[0]
            qposes, actual, targets = [], [], []
            for frame in range(len(source.times)):
                if frame_limit is not None:
                    previous = solver.configuration.data.qpos[retargeter.sim.q_indices].copy()
                    delta = args.frame_speed_limit * retargeter.sim.reference_dt * (1 - 1e-6)
                    frame_limit.lower[retargeter.sim.q_indices] = np.maximum(retargeter.sim.lower, previous - delta)
                    frame_limit.upper[retargeter.sim.q_indices] = np.minimum(retargeter.sim.upper, previous + delta)
                q = solver.retarget(frame_data(frame))
                if not np.isfinite(q).all():
                    raise ValueError("Non-finite GMR state")
                qposes.append(q)
                actual.append(solver.configuration.data.xpos[retargeter.bodies].copy())
                targets.append([solver.scaled_human_data[n][0] for n in source.names])
            qposes, actual, targets = map(np.asarray, (qposes, actual, targets))
            np.savez_compressed(directory / "kinematic_trace.npz", time_s=source.times, qpos=qposes,
                                target_position_m=targets, actual_body_position_m=actual,
                                usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
            joints = qposes[:, retargeter.sim.q_indices]
            speeds = np.abs(np.diff(joints, axis=0) / np.diff(source.times)[:, None])
            limits = retargeter.sim.model.jnt_range[retargeter.sim.joint_ids]
            if args.render and case in ("amass_walk_turn", "amass_kick"):
                render_overview(retargeter, source, qposes, directory, "cpu")
            row.update({"status": "COMPLETE", "frames": len(qposes), "input": identity(body),
                        "calibration_recipe": recipe, "scaling": scaling, "initial_pose_warmup_calls": 30,
                        "q1_leg_geometry": knee_geometry(actual, source.names),
                        "source_leg_geometry": conversion["source_leg_geometry"],
                        "ik_position_p95_m": float(np.quantile(np.linalg.norm(actual - targets, axis=-1), .95)),
                        "max_joint_speed_rad_s": float(speeds.max()),
                        "joint_speed_fraction_above_2_rad_s": float(np.mean(speeds > 2 + 1e-6)),
                        "model_joint_limit_violations": int(np.count_nonzero((joints < limits[:, 0] - 1e-6) | (joints > limits[:, 1] + 1e-6))),
                        "reference_margin_limit_violations": int(np.count_nonzero((joints < retargeter.sim.lower) | (joints > retargeter.sim.upper))),
                        "root_translation_range_m": np.ptp(qposes[:, :3], axis=0).tolist(),
                        "fixed_pd_contract_passed": bool(speeds.max() <= 2 and np.all(joints >= retargeter.sim.lower) and np.all(joints <= retargeter.sim.upper)),
                        "supported_replay_run": False,
                        "artifacts": [identity(p) for p in sorted(directory.iterdir())]})
            if args.frame_speed_limit is not None and row["fixed_pd_contract_passed"]:
                sim = Q1Sim(retargeter.sim.config_path, supported=True)
                reference = JointReference(source.times, joints, sim.names, sim.names)
                reference.validate_limits(sim)
                sim.data.qpos[sim.q_indices] = joints[0]
                mujoco.mj_forward(sim.model, sim.data)
                replay_times, replay_actual, replay_targets = [], [], []
                while sim.data.time < reference.times[-1] - sim.dt / 2:
                    target, velocity = reference.sample(sim.data.time)
                    obs = sim.advance(target, velocity)
                    replay_times.append(obs["time_s"])
                    replay_actual.append(obs["joint_position_rad"])
                    replay_targets.append(target)
                replay_actual, replay_targets = map(np.asarray, (replay_actual, replay_targets))
                np.savez_compressed(directory / "supported_replay.npz", time_s=replay_times,
                                    actual_rad=replay_actual, target_rad=replay_targets,
                                    usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
                row["supported_replay_run"] = True
                row["supported_max_joint_rmse_rad"] = float(np.sqrt(np.mean((replay_actual - replay_targets) ** 2, axis=0)).max())
                row["supported_initialization"] = "START_FROM_FITTED_FIRST_FRAME_JOINTS_FIXED_PELVIS_NOT_NEUTRAL_START"
                row["artifacts"] = [identity(p) for p in sorted(directory.iterdir())]
            print(f"{case}: P95 {row['ik_position_p95_m']:.4f} m, max speed {row['max_joint_speed_rad_s']:.2f} rad/s, right knee range {row['q1_leg_geometry']['right']['knee_flexion_range_deg']:.2f} deg", flush=True)
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            print(f"Failed {case}: {row['error']}", flush=True)
        (args.output_dir / "gmr_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["complete_count"] = sum(row["status"] == "COMPLETE" for row in report["cases"])
    report["failed_count"] = len(report["cases"]) - report["complete_count"]
    (args.output_dir / "gmr_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return bool(report["failed_count"])


if __name__ == "__main__":
    sys.exit(main())
