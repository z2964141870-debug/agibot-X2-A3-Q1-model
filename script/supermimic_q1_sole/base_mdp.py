"""Isolated native SuperMimic reward candidate using collision-sole foot Z."""

import copy
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import yaw_mat
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import CONTROL_DT

from sole_geometry import FEET, SoleGeometry
from tensor_geometry import TensorSole


class SoleMDP(MDP):
    def __init__(self, kin, motion, default_q, cfg=None, *, scene, observation_mode="legacy"):
        super().__init__(kin, motion, default_q, cfg, observation_mode=observation_mode)
        self.sole = TensorSole(SoleGeometry.from_scene(scene), kin.body_names, motion.device)
        self.foot_slots = [kin.tracked_idx.index(kin.body_names.index(name)) for name in FEET]
        heights = []
        with torch.no_grad():
            for start in range(0, motion.num_frames, 1024):
                frames = torch.arange(start, min(start + 1024, motion.num_frames), device=motion.device)
                full = kin.forward(*motion.ref_state(frames), all_bodies=True)
                heights.append(self.sole(full[0], full[1]))
        self.reference_sole = torch.cat(heights)
        if not bool(torch.isfinite(self.reference_sole).all()):
            raise ValueError("Nonfinite reference collision-sole heights")
        self.sole_contract = dict(scene=str(Path(scene)), geometry=self.sole.contract,
                                  tracked_body_count=len(kin.tracked_idx),
                                  foot_slots=self.foot_slots,
                                  change="Only two foot Z errors within native mean body-position error")

    def to(self, device):
        new = super().to(device)
        if new is self:
            return self
        new.sole = copy.deepcopy(self.sole).to(device)
        new.reference_sole = self.reference_sole.to(device)
        return new

    def reward(self, state, frames, action, previous_action, terms=False):
        original, parts = super().reward(state, frames, action, previous_action, terms=True)
        full = self.kin.forward(*state, all_bodies=True)
        body_p = full[0][..., self.kin.tracked_idx, :]
        ref_root_p, ref_root_r = self.motion.root_p[frames], self.motion.root_R[frames]
        delta = yaw_mat(state[1] @ ref_root_r.transpose(-1, -2))
        anchor = torch.cat((state[0][..., :2], ref_root_p[..., 2:3]), -1)
        ref_body = anchor.unsqueeze(-2) + (delta.unsqueeze(-3) @
            (self.motion.body_p[frames] - ref_root_p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        errors = (ref_body - body_p).square()
        errors[..., self.foot_slots, 2] = (self.sole(full[0], full[1]) - self.reference_sole[frames]).square()
        error = errors.sum(-1).mean(-1)
        adjustment = (torch.exp(-error / self.cfg.body_pos_std**2) -
                      torch.exp(-parts["e_body_pos"] / self.cfg.body_pos_std**2)) * CONTROL_DT
        value = original + adjustment
        if terms:
            return value, {**parts, "track": parts["track"] + adjustment,
                           "e_body_pos": error, "e_body_pos_ankle": parts["e_body_pos"],
                           "sole_position_adjustment": adjustment}
        return value
