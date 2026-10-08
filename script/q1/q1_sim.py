"""Simulation-only Q1 reference/PD adapter; contains no robot transport."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


class JointReference:
    """Offline robot-space radians, separate from the unknown SMPL input."""

    def __init__(self, times, positions, joint_names, contract_names):
        self.times = np.asarray(times, dtype=float).copy()
        names = list(joint_names)
        values = np.asarray(positions, dtype=float)
        if (len(names) != len(contract_names) or len(set(names)) != len(names)
                or set(names) != set(contract_names)):
            raise ValueError("Reference must name every robot joint exactly once")
        if (self.times.ndim != 1 or len(self.times) < 2
                or not np.isfinite(self.times).all() or self.times[0] != 0
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Times must be finite, start at zero and strictly increase")
        if values.shape != (len(self.times), len(names)) or not np.isfinite(values).all():
            raise ValueError("Joint positions must be a finite [frames, joints] array")
        self.positions = values[:, [names.index(name) for name in contract_names]].copy()
        self.joint_names = list(contract_names)
        self.slopes = np.diff(self.positions, axis=0) / np.diff(self.times)[:, None]

    @classmethod
    def load(cls, path, contract_names):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["time_s"], data["joint_position_rad"],
                       data["joint_names"].tolist(), contract_names)

    def save(self, path):
        np.savez_compressed(path, time_s=self.times,
                            joint_position_rad=self.positions,
                            joint_names=np.asarray(self.joint_names))

    def sample(self, time_s):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Reference time must be finite and nonnegative")
        if time_s >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(len(self.joint_names))
        index = np.searchsorted(self.times, time_s, side="right") - 1
        return (self.positions[index] + self.slopes[index] * (time_s - self.times[index]),
                self.slopes[index].copy())

    def validate_limits(self, sim):
        if (np.any(self.positions < sim.lower) or np.any(self.positions > sim.upper)
                or np.any(np.abs(self.slopes) > sim.config["reference_velocity_limit_rad_s"])):
            raise ValueError("Reference exceeds configured simulation position/velocity limits")


class Q1Sim:
    def __init__(self, config_path, supported=False):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        root_dir = self.config_path.parents[2]
        self.asset_path = root_dir / self.config["model"]
        raw = self.asset_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.config["model_sha256"]:
            raise ValueError("Q1 source XML identity differs from pinned configuration")
        self.supported = bool(supported)
        if supported:
            tree = ET.fromstring(raw)
            compiler = tree.find("compiler")
            compiler.set("meshdir", str((self.asset_path.parent / compiler.get("meshdir", "")).resolve()))
            pelvis = tree.find("./worldbody/body[@name='pelvis']")
            pelvis.set("pos", f"0 0 {self.config['supported_pelvis_height_m']}")
            equality = ET.SubElement(tree, "equality")
            ET.SubElement(equality, "weld", name="diagnostic_support", body1="pelvis")
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(self.asset_path))
        self.data = mujoco.MjData(self.model)
        self.names = self.config["joint_names"]
        if len(self.names) != 22 or len(set(self.names)) != 22:
            raise ValueError("Expected 22 unique named Q1 joints")
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in self.names]
        if any(i < 0 for i in joint_ids):
            raise ValueError("Configured joint missing from model")
        self.joint_ids = np.asarray(joint_ids)
        if np.any(self.model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Q1 contract expects scalar hinge joints")
        self.q_indices = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dq_indices = self.model.jnt_dofadr[self.joint_ids].copy()
        self.motor_indices = []
        for joint in joint_ids:
            motors = [i for i in range(self.model.nu)
                      if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                      and self.model.actuator_trnid[i, 0] == joint]
            if len(motors) != 1:
                raise ValueError("Each Q1 joint must map to exactly one joint motor")
            self.motor_indices.append(motors[0])
        self.motor_indices = np.asarray(self.motor_indices)
        if self.model.nu != 22 or not np.allclose(self.model.actuator_gear[self.motor_indices, 0], 1):
            raise ValueError("Expected 22 unit-gear direct-torque motors")
        if not np.all(self.model.actuator_ctrllimited[self.motor_indices]):
            raise ValueError("Every motor must declare its simulation torque limits")
        margin = self.config["joint_limit_margin_rad"]
        self.lower = self.model.jnt_range[self.joint_ids, 0] + margin
        self.upper = self.model.jnt_range[self.joint_ids, 1] - margin
        self.neutral = np.clip(self.model.qpos0[self.q_indices], self.lower, self.upper)
        self.torque_bounds = (self.model.actuator_ctrlrange[self.motor_indices]
                              * self.config["torque_fraction_of_model"])
        self.dt = float(self.model.opt.timestep)
        if self.config["reference_hz"] <= 0:
            raise ValueError("Reference frequency must be positive")
        ratio = 1 / self.config["reference_hz"] / self.dt
        if not np.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError("Reference period must be an integer number of physics steps")
        self.substeps = int(round(ratio))
        self.reference_dt = self.substeps * self.dt
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.reset()

    def vector(self, value, label):
        result = np.asarray(value, dtype=float)
        if result.shape != (22,) or not np.isfinite(result).all():
            raise ValueError(f"{label} must contain 22 finite values in contract order")
        return result

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.q_indices] = self.neutral
        mujoco.mj_forward(self.model, self.data)
        self.last_torque = np.zeros(22)
        self.torque_saturated = 0
        self.torque_samples = 0
        self.initial_height = float(self.data.xpos[self.pelvis_id, 2])
        return self.observe()

    def decode_action(self, action):
        value = self.vector(action, "Action")
        requested = self.neutral + np.clip(value, -1, 1) * self.config["normalized_action_scale_rad"]
        return np.clip(requested, self.lower, self.upper)

    def advance(self, target, target_velocity=None):
        target = self.vector(target, "Target")
        velocity = self.vector(np.zeros(22) if target_velocity is None else target_velocity,
                               "Target velocity")
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise ValueError("Target exceeds simulation joint limits")
        if np.any(np.abs(velocity) > self.config["reference_velocity_limit_rad_s"]):
            raise ValueError("Target velocity exceeds configured limit")
        for _ in range(self.substeps):
            torque = (self.config["kp"] * (target - self.data.qpos[self.q_indices])
                      + self.config["kd"] * (velocity - self.data.qvel[self.dq_indices]))
            self.last_torque = np.clip(torque, self.torque_bounds[:, 0], self.torque_bounds[:, 1])
            self.torque_saturated += int(np.count_nonzero(torque != self.last_torque))
            self.torque_samples += 22
            self.data.ctrl[self.motor_indices] = self.last_torque
            mujoco.mj_step(self.model, self.data)
            if not all(np.isfinite(v).all() for v in (self.data.qpos, self.data.qvel,
                                                       self.data.qacc, self.data.sensordata)):
                raise RuntimeError("Non-finite simulated state")
            if any(w.number for w in self.data.warning):
                raise RuntimeError("MuJoCo emitted a physics warning")
        # Refresh derived positions/sensors at the state returned by mj_step.
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def sensor(self, name):
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if index < 0:
            raise ValueError(f"Missing simulation sensor: {name}")
        start = self.model.sensor_adr[index]
        return self.data.sensordata[start:start + self.model.sensor_dim[index]].copy()

    def observe(self):
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        return {
            "time_s": float(self.data.time),
            "joint_position_rad": self.data.qpos[self.q_indices].copy(),
            "joint_velocity_rad_s": self.data.qvel[self.dq_indices].copy(),
            "orientation_wxyz_world": self.sensor("body-orientation"),
            "angular_velocity_rad_s_body": self.sensor("body-angular-velocity"),
            "projected_gravity_body": rotation.T @ np.asarray([0., 0., -1.]),
            "root_position_m_world_privileged": self.data.xpos[self.pelvis_id].copy(),
        }

    def fallen(self):
        obs = self.observe()
        upright_cosine = -obs["projected_gravity_body"][2]
        return (obs["root_position_m_world_privileged"][2]
                < self.initial_height * self.config["fall_height_fraction"]
                or upright_cosine < np.cos(np.deg2rad(self.config["fall_tilt_deg"])))

    def contract(self):
        return {
            "scope": "SIMULATION_ONLY",
            "model_sha256": self.config["model_sha256"],
            "supported": self.supported,
            "physics_hz": 1 / self.dt,
            "reference_hz": self.config["reference_hz"],
            "servo_hz": 1 / self.dt,
            "action": "neutral_rad + clipped_normalized_action * scale_rad",
            "action_scale_rad": self.config["normalized_action_scale_rad"],
            "torque_fraction_of_model": self.config["torque_fraction_of_model"],
            "joints": [{"name": name, "qpos_index": int(self.q_indices[i]),
                        "qvel_index": int(self.dq_indices[i]),
                        "actuator_index": int(self.motor_indices[i]),
                        "target_range_rad": [float(self.lower[i]), float(self.upper[i])],
                        "torque_range_nm": self.torque_bounds[i].tolist()}
                       for i, name in enumerate(self.names)],
        }
