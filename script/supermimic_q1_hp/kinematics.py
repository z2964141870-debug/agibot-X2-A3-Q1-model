"""Adapt the verified full-tree Q1 FK to HP's differentiable state contract."""

import copy
import os
from pathlib import Path
import sys

import mujoco
import torch

sys.path.insert(0, str(Path(os.environ.get("YUANQI_ROOT", "/opt/data/private/ZHY")) / "script/supermimic_for_q1"))
from supermimic.kinematics.g1_fk import G1Kinematics
from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat, quat_to_mat
from .robot import Q1_JOINT_NAMES, TRACKED_BODIES


class Q1Kinematics:
    def __init__(self, mjcf_path, device, tracked=TRACKED_BODIES):
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        natural = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if natural != Q1_JOINT_NAMES:
            raise ValueError("Q1 joint order changed")
        self.device = torch.device(device)
        self.fk = G1Kinematics(model, self.device)
        self.body_names = self.fk.body_names
        self.tracked_idx = [self.body_names.index(name) for name in tracked]
        self.num_bodies = len(self.body_names)
        ranges = [model.joint(name).range.copy() for name in natural]
        self.q_lo = torch.tensor([v[0] for v in ranges], dtype=torch.float32, device=device)
        self.q_hi = torch.tensor([v[1] for v in ranges], dtype=torch.float32, device=device)

    def to(self, device):
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        new.q_lo, new.q_hi = self.q_lo.to(device), self.q_hi.to(device)
        new.fk = copy.copy(self.fk)
        new.fk.device = device
        for name, value in vars(self.fk).items():
            if isinstance(value, torch.Tensor): setattr(new.fk, name, value.to(device))
        return new

    def forward(self, p, rotation, v, w, q, qd, all_bodies=False):
        qpos = torch.cat((p, mat_to_quat(rotation), q), -1)
        w_local = (rotation.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        output = self.fk.forward(qpos, torch.cat((v, w_local, qd), -1))
        ids = list(range(self.num_bodies)) if all_bodies else self.tracked_idx
        return output["pos"][..., ids, :], quat_to_mat(output["quat"][..., ids, :]), output["lin_vel"][..., ids, :], output["ang_vel"][..., ids, :]
