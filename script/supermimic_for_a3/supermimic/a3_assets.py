"""The explicit A3 state, actuator, and input identity contract."""

from dataclasses import asdict
import hashlib
import importlib
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads


MJCF_SHA = "3915053acd573f0cc093a080d64197574fdc146ad4cf91ec1b0ceff71983b75f"
TRACKED = ("pelvis_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
           "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link", "torso_Link",
           "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
           "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link")


def identity(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


class A3Assets:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.vendor = self.root / "script/vendor/sonic_for_a3"
        self.xml = self.vendor / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml"
        if identity(self.xml)["sha256"] != MJCF_SHA:
            raise ValueError("The registered A3 MJCF has changed")
        sys.path.insert(0, str(self.vendor))
        self.control_module = importlib.import_module("gear_sonic.scripts.sim2sim_a3_mujoco")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        m = self.model
        if (m.nq, m.nv, m.nu) != (42, 41, 31):
            raise ValueError("Unexpected A3 generalized state or actuator dimensions")
        self.names = [m.joint(j).name for j in range(1, m.njnt)]
        self.active_names = [m.actuator(i).name for i in range(m.nu)
                             if not m.actuator(i).name.startswith("head_")]
        if len(self.active_names) != 29 or len(set(self.active_names)) != 29:
            raise ValueError("Expected 29 distinct policy joints")
        self.jids = np.array([m.joint(name).id for name in self.active_names])
        self.qadr = m.jnt_qposadr[self.jids].copy()
        self.vadr = m.jnt_dofadr[self.jids].copy()
        self.active_indices = (self.qadr - 7).tolist()
        self.aids = np.array([m.actuator(name).id for name in self.active_names])
        if not np.array_equal(m.actuator_trnid[self.aids, 0], self.jids):
            raise ValueError("A3 actuator-to-joint contract differs")
        controls = [self.control_module.get_a3_joint_control(n) for n in self.active_names]
        self.controls = controls
        self.kp = np.array([c.kp for c in controls])
        self.kd = np.array([c.kd for c in controls])
        self.effort = np.array([c.effort for c in controls])
        self.q0 = np.array([self.control_module.DEFAULT_Q_BY_NAME[n] for n in self.active_names])
        self.scale = 0.25 * self.effort / self.kp
        limits = m.jnt_range[self.jids]
        center = limits.mean(-1)
        self.soft_limits = np.stack([center - .45 * np.diff(limits)[:, 0],
                                     center + .45 * np.diff(limits)[:, 0]], axis=-1)
        self.body_ids = np.array([m.body(name).id for name in TRACKED])
        self.head_jids = np.array([m.joint(n).id for n in ("head_yaw_joint", "head_pitch_joint")])
        self.head_aids = np.array([m.actuator(n).id for n in ("head_yaw_joint", "head_pitch_joint")])
        self.foot_geoms = np.array([i for i in range(m.ngeom) if
            ("foot" in m.body(int(m.geom_bodyid[i])).name or "ankle_roll" in m.body(int(m.geom_bodyid[i])).name)
            and (m.geom_contype[i] or m.geom_conaffinity[i])], dtype=int)
        if not len(self.foot_geoms):
            raise ValueError("No real A3 foot collision geometry found")
        self.configure_model(m)

    def configure_model(self, model):
        model.opt.timestep = .001
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for i, c in enumerate(self.controls):
            model.dof_armature[self.vadr[i]] = c.armature
            model.dof_frictionloss[self.vadr[i]] = c.friction
            # Damping is supplied by the explicit PD torque each physics substep.
            model.dof_damping[self.vadr[i]] = 0.
        model.geom_friction[self.foot_geoms, 0] = 1.

    def heads(self, device="cpu", dtype=torch.float32):
        fk = G1Kinematics(self.model, device=device, dtype=dtype)
        t = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
        return AnalyticHeads(fk, TRACKED, "pelvis_link", "pelvis_link", t(self.soft_limits),
                             t(self.q0), torch.tensor(self.active_indices, device=device))

    def manifest(self):
        return {"mjcf": identity(self.xml), "control_source": identity(self.control_module.__file__),
                "nq": 42, "nv": 41, "full_joint_names": self.names,
                "active_joint_names": self.active_names, "active_indices": self.active_indices,
                "controls": [asdict(c) for c in self.controls], "q0": self.q0.tolist(),
                "action_scale": self.scale.tolist(), "action_clip": 20., "dt": .02,
                "physics_dt": .001, "substeps": 20, "tracked_bodies": list(TRACKED),
                "reward_anchor": "pelvis_link", "self_collision": "retained from input model",
                "head_control": "PD hold zero, kp=40 kd=2, original torque limit",
                "foot_geoms": [self.model.geom(int(i)).name for i in self.foot_geoms]}
