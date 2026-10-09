"""Calibrated body-pose targets to Q1 kinematics using the installed Mink solver."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from q1_sim import JointReference, Q1Sim


HERE = Path(__file__).resolve().parent


def heading_rotation(rotation):
    """Extract world-Z heading without rotating the world gravity axis."""
    forward = rotation[:2, 0]
    if np.linalg.norm(forward) > 1e-8:
        yaw = np.arctan2(forward[1], forward[0])
    else:
        left = rotation[:2, 1]
        yaw = np.arctan2(-left[0], left[1])
    return Rotation.from_euler("z", yaw).as_matrix()


class BodyReference:
    """Explicit world-space skeleton contract, not a raw SMPL pose decoder."""

    def __init__(self, times, names, positions, orientations, calibration_positions,
                 calibration_orientations, required_names, source_kind,
                 convention="x_forward_y_left_z_up_m_wxyz"):
        if convention != "x_forward_y_left_z_up_m_wxyz":
            raise ValueError("Convert axes, units and quaternion order before retargeting")
        if not isinstance(source_kind, str) or not source_kind.strip():
            raise ValueError("Record the actual source kind")
        names = list(names)
        if (len(names) != len(required_names) or len(set(names)) != len(names)
                or set(names) != set(required_names)):
            raise ValueError("Body names must match the configured skeleton exactly")
        self.names = list(required_names)
        self.source_kind = source_kind
        self.times = np.asarray(times, dtype=float).copy()
        if (self.times.ndim != 1 or len(self.times) < 2 or self.times[0] != 0
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Times must start at zero and strictly increase")
        order = [names.index(name) for name in required_names]
        frames, bodies = len(self.times), len(names)

        def values(value, shape, quaternion=False):
            array = np.asarray(value, dtype=float)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Expected finite body data with shape {shape}")
            if quaternion and not np.allclose(np.linalg.norm(array, axis=-1), 1, atol=1e-5, rtol=0):
                raise ValueError("Body quaternions must be unit wxyz quaternions")
            return array.copy()

        self.positions = values(positions, (frames, bodies, 3))[:, order]
        self.orientations = values(orientations, (frames, bodies, 4), True)[:, order]
        self.calibration_positions = values(calibration_positions, (bodies, 3))[order]
        self.calibration_orientations = values(calibration_orientations, (bodies, 4), True)[order]

    @classmethod
    def load(cls, path, required_names):
        with np.load(path, allow_pickle=False) as source:
            if source["schema_version"].item() != 1:
                raise ValueError("Unsupported body reference schema")
            return cls(source["time_s"], source["body_names"].tolist(),
                       source["position_m_world"], source["orientation_wxyz_world"],
                       source["calibration_position_m_world"],
                       source["calibration_orientation_wxyz_world"], required_names,
                       source["source_kind"].item(), source["convention"].item())

    def save(self, path):
        np.savez_compressed(path, schema_version=np.asarray(1), time_s=self.times,
                            body_names=np.asarray(self.names),
                            position_m_world=self.positions,
                            orientation_wxyz_world=self.orientations,
                            calibration_position_m_world=self.calibration_positions,
                            calibration_orientation_wxyz_world=self.calibration_orientations,
                            source_kind=np.asarray(self.source_kind),
                            convention=np.asarray("x_forward_y_left_z_up_m_wxyz"))


class Q1Retargeter:
    def __init__(self, sim_config=HERE / "sim_config.json", config=HERE / "retarget_config.json"):
        self.config_path = Path(config).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.translation_alignment = self.config.get("root_translation_alignment", "gravity_preserving_heading")
        if self.translation_alignment not in ("gravity_preserving_heading", "full_rotation_legacy_diagnostic"):
            raise ValueError("Unknown root translation alignment")
        self.sim = Q1Sim(sim_config)
        self.names = list(self.config["body_map"])
        self.root = self.names.index(self.config["root_name"])
        self.bodies = [mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
                       for name in self.config["body_map"].values()]
        if any(body < 0 for body in self.bodies):
            raise ValueError("Mapped robot body missing from model")
        root_joint = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT,
                                     "floating_base_joint")
        if (root_joint < 0 or self.sim.model.jnt_qposadr[root_joint] != 0
                or self.sim.model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Expected named Q1 floating base at the start of qpos")
        self.robot_positions = self.sim.data.xpos[self.bodies].copy()
        self.robot_rotations = self.sim.data.xmat[self.bodies].reshape(-1, 3, 3).copy()
        self.configuration = mink.Configuration(self.sim.model, q=self.sim.data.qpos.copy())
        self.root_task = mink.FrameTask("pelvis", "body", 1.0, 1.0)
        self.tasks = []
        for index, name in enumerate(self.config["body_map"].values()):
            if index != self.root:
                self.tasks.append((index, mink.FrameTask(
                    name, "body", self.config["position_cost"],
                    self.config["orientation_cost"], lm_damping=1.0)))
        cost = np.full(self.sim.model.nv, self.config["posture_cost"])
        cost[:6] = 0
        self.posture = mink.PostureTask(self.sim.model, cost=cost)
        self.posture.set_target(self.sim.data.qpos)
        velocities = {name: self.sim.config["reference_velocity_limit_rad_s"] * (1 - 1e-6)
                      for name in self.sim.names}
        self.limits = [mink.ConfigurationLimit(
            self.sim.model, min_distance_from_limits=self.sim.config["joint_limit_margin_rad"]),
            mink.VelocityLimit(self.sim.model, velocities)]

    def retarget(self, source, scale):
        if source.names != self.names:
            raise ValueError("Body reference must use this retargeter's contract order")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Human-to-robot displacement scale must be positive and explicit")
        if not np.allclose(np.diff(source.times), self.sim.reference_dt, atol=1e-8, rtol=0):
            raise ValueError("Resample skeleton to configured reference Hz before retargeting")
        robot_root_rotation = self.robot_rotations[self.root]
        human_cal_rotation = Rotation.from_quat(source.calibration_orientations,
                                               scalar_first=True).as_matrix()
        alignment = robot_root_rotation @ human_cal_rotation[self.root].T
        world_alignment = (heading_rotation(robot_root_rotation)
                           @ heading_rotation(human_cal_rotation[self.root]).T)
        if self.translation_alignment == "full_rotation_legacy_diagnostic":
            world_alignment = alignment
        human_cal_local = ((source.calibration_positions - source.calibration_positions[self.root])
                           @ human_cal_rotation[self.root])
        robot_cal_local = ((self.robot_positions - self.robot_positions[self.root])
                           @ robot_root_rotation)
        self.configuration.update(self.sim.data.qpos.copy())
        qposes, target_positions, target_rotations, errors = [], [], [], []
        iterations = self.config["iterations_per_frame"]
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations_per_frame must be a positive integer")
        dt = self.sim.reference_dt / iterations
        for frame in range(len(source.times)):
            human_rotations = Rotation.from_quat(source.orientations[frame], scalar_first=True).as_matrix()
            root_rotation = alignment @ human_rotations[self.root]
            root_position = (self.robot_positions[self.root] + scale * world_alignment
                             @ (source.positions[frame, self.root] - source.calibration_positions[self.root]))
            human_local = ((source.positions[frame] - source.positions[frame, self.root])
                           @ human_rotations[self.root])
            positions = root_position + (robot_cal_local + scale * (human_local - human_cal_local)) @ root_rotation.T
            rotations = (alignment @ human_rotations @ human_cal_rotation.transpose(0, 2, 1)
                         @ alignment.T @ self.robot_rotations)
            q = self.configuration.data.qpos.copy()
            q[:3] = root_position
            q[3:7] = Rotation.from_matrix(root_rotation).as_quat(scalar_first=True)
            self.configuration.update(q)
            self.root_task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(root_rotation), root_position))
            for index, task in self.tasks:
                task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(rotations[index]), positions[index]))
            for _ in range(iterations):
                velocity = mink.solve_ik(self.configuration, [t for _, t in self.tasks] + [self.posture],
                                         dt, self.config["solver"], damping=self.config["damping"],
                                         safety_break=True, limits=self.limits, constraints=[self.root_task])
                self.configuration.integrate_inplace(velocity, dt)
            q = self.configuration.data.qpos.copy()
            joint_values = q[self.sim.q_indices]
            if (not np.isfinite(q).all() or np.any(joint_values < self.sim.lower - 1e-6)
                    or np.any(joint_values > self.sim.upper + 1e-6)):
                raise RuntimeError("IK solver violated the finite-state/joint-limit contract")
            q[self.sim.q_indices] = np.clip(joint_values, self.sim.lower, self.sim.upper)
            self.configuration.update(q)
            qposes.append(q)
            target_positions.append(positions)
            target_rotations.append(rotations)
            errors.append(self.configuration.data.xpos[self.bodies] - positions)
        qposes = np.asarray(qposes)
        reference = JointReference(source.times, qposes[:, self.sim.q_indices], self.sim.names, self.sim.names)
        reference.validate_limits(self.sim)
        return reference, qposes, np.asarray(target_positions), np.asarray(target_rotations), np.asarray(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; preserve existing artifacts")
    retargeter = Q1Retargeter()
    source = BodyReference.load(args.input, retargeter.names)
    reference, qposes, targets, rotations, errors = retargeter.retarget(source, args.scale)
    args.output_dir.mkdir(parents=True)
    reference.save(args.output_dir / "joint_reference.npz")
    np.savez_compressed(args.output_dir / "kinematic_trace.npz", time_s=source.times,
                        qpos=qposes, target_position_m=targets,
                        target_rotation_matrix=rotations, body_position_error_m=errors)
    root = HERE.parents[1]
    report = {
        "scope": "KINEMATIC_RETARGETING_NOT_DYNAMIC_CONTROL",
        "source_kind": source.source_kind, "scale": args.scale,
        "frames": len(source.times), "mink_version": importlib.metadata.version("mink"),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model_sha256": retargeter.sim.config["model_sha256"],
        "source_files": [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in [Path(__file__).resolve(), HERE / "q1_sim.py", HERE / "sim_config.json",
                                   retargeter.config_path]],
        "max_body_position_error_m": float(np.linalg.norm(errors, axis=-1).max()),
        "policy_trained": False, "hardware_control": False,
        "baidu_backup_status": "LOCAL_ONLY",
        "artifacts": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(args.output_dir.iterdir())],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
