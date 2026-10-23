"""Q1 v3 asset and simulation-control contract; no hardware transport."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robot_spec import ACT_DIM, NQ, NV
from supermimic.world_model.heads import AnalyticHeads


MJCF_SHA = "d9442f0de18071b100d927e310bc58b04e7be305fea8f126893b2700e8c4bd3b"
TRACKED = ("pelvis", "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
           "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
           "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link")


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class Q1Assets:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.xml = self.root / "data/references/q1/model_20261023_P01/q1_v3/xml/q1_v3_mc.xml"
        if identity(self.xml)["sha256"] != MJCF_SHA:
            raise ValueError("Registered Q1 MJCF identity changed")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        m = self.model
        if (m.nq, m.nv, m.nu) != (NQ, NV, ACT_DIM):
            raise ValueError("Q1 generalized state or actuator dimensions changed")
        self.jids = np.array([j for j in range(m.njnt)
                              if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
        self.names = self.active_names = [m.joint(int(j)).name for j in self.jids]
        self.qadr = m.jnt_qposadr[self.jids].copy()
        self.vadr = m.jnt_dofadr[self.jids].copy()
        if len(self.jids) != ACT_DIM or not np.array_equal(self.qadr, np.arange(7, NQ)):
            raise ValueError("Q1 hinge layout differs from the registered contract")
        self.active_indices = (self.qadr - 7).tolist()
        self.aids = []
        for j in self.jids:
            matches = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == j
                       and m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]
            if len(matches) != 1:
                raise ValueError("Q1 joint must have exactly one actuator")
            self.aids.append(matches[0])
        self.aids = np.asarray(self.aids)
        if not np.allclose(m.actuator_gear[self.aids, 0], 1):
            raise ValueError("Q1 expects unit transmission gears")
        bounds = m.actuator_ctrlrange[self.aids]
        if not np.all(m.actuator_ctrllimited[self.aids]) or not np.allclose(bounds[:, 0], -bounds[:, 1]):
            raise ValueError("Q1 requires explicit symmetric model torque bounds")
        self.kp, self.kd = np.full(ACT_DIM, 50.), np.full(ACT_DIM, 2.)
        self.effort = bounds[:, 1].copy() * .25
        limits = m.jnt_range[self.jids]
        self.q0 = np.clip(m.qpos0[self.qadr], limits[:, 0] + .01, limits[:, 1] - .01)
        self.scale = .25 * self.effort / self.kp
        center = limits.mean(-1)
        width = np.diff(limits)[:, 0]
        self.soft_limits = np.stack([center - .45 * width, center + .45 * width], axis=-1)
        self.body_ids = np.array([m.body(name).id for name in TRACKED])
        self.foot_geoms = np.array([i for i in range(m.ngeom)
                                    if m.body(int(m.geom_bodyid[i])).name in
                                    ("left_ankle_roll_link", "right_ankle_roll_link")
                                    and (m.geom_contype[i] or m.geom_conaffinity[i])])
        if len(self.foot_geoms) == 0:
            raise ValueError("Q1 model has no real foot collision geometry")
        self.configure_model(m)

    def configure_model(self, model):
        model.opt.timestep = .001
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.dof_damping[self.vadr] = 0.
        model.geom_friction[self.foot_geoms, 0] = 1.

    def heads(self, device="cpu", dtype=torch.float32):
        fk = G1Kinematics(self.model, device=device, dtype=dtype)
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
        return AnalyticHeads(fk, TRACKED, "pelvis", "pelvis", tensor(self.soft_limits),
                             tensor(self.q0), torch.tensor(self.active_indices, device=device))

    def manifest(self):
        return {"robot": "Q1 v3", "mjcf": identity(self.xml), "adapter": identity(__file__),
                "nq": NQ, "nv": NV, "nu": ACT_DIM, "full_joint_names": self.names,
                "active_joint_names": self.active_names, "active_indices": self.active_indices,
                "q0": self.q0.tolist(), "kp": self.kp.tolist(), "kd": self.kd.tolist(),
                "torque_fraction_of_model": .25, "effective_effort_nm": self.effort.tolist(),
                "action_scale": self.scale.tolist(), "action_clip": 20.,
                "dt": .02, "physics_dt": .001, "substeps": 20,
                "tracked_bodies": list(TRACKED), "support": "free base; no weld",
                "control_scope": "simulation only; not manufacturer-verified hardware gains",
                "self_collision": "retained from input model"}
