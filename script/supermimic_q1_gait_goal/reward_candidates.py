"""Independent phase/sole rewards for the base learner and residual PPO."""

import copy
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "supermimic_q1_sole"),
               str(HERE.parent / "supermimic_q1_contact_comp")]
from contact_reference import configure
from sole_reference import SoleReference
from support_geometry import BoundedTensorSole
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import CONTROL_DT

HEIGHT_STD_M = .02
STANCE_SPEED_STD_MPS = .15
BASE_SWING_WEIGHT = .25
BASE_STANCE_WEIGHT = .25
PPO_Z_WEIGHT = .01


def phase_mean(values, mask):
    if values.shape != mask.shape or mask.dtype != torch.bool or values.shape[-1] != 2:
        raise ValueError("Expected aligned two-foot values and boolean phase")
    return (values * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def lowest_material_velocity(sole, full_fk, offsets=None):
    positions, rotations, velocities, angular = full_fk
    offsets = sole.offsets(rotations) if offsets is None else offsets
    return velocities[..., sole.body_ids, :] + torch.linalg.cross(
        angular[..., sole.body_ids, :], offsets, dim=-1)


class PhaseSoleReference(SoleReference):
    def __init__(self, mdp, scene):
        super().__init__(mdp, scene)
        self.sole = BoundedTensorSole.from_existing(self.sole)
        self.contract = {**self.contract,
            "reward_change": "RF16 ankle-Z4cm to collision-sole-Z2cm, same maximum .01 per20ms",
            "height_std_m": HEIGHT_STD_M, "ppo_z_weight": PPO_Z_WEIGHT,
            "features_phase_xy_stance_stationary_smooth_residual": "Unchanged RF16"}

    def reward(self, original, state, frames, targets, previous_targets, residual, force, slip):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        total, parts = super().reward(original, state, frames, targets, previous_targets,
                                      residual, force, slip)
        full = self.mdp.kin.forward(*unflatten_state(state), all_bodies=True)
        error = self.sole(full[0], full[1]) - self.reference_sole[frames]
        swing = self.swing[frames]
        correction = PPO_Z_WEIGHT * phase_mean(
            torch.exp(-(error / HEIGHT_STD_M).square()) - torch.exp(-(error / .04).square()), swing)
        return total + correction, {**parts, "swing_bonus": parts["swing_bonus"] + correction}


class PhaseSoleMDP(MDP):
    max_step_reward = MDP.max_step_reward + CONTROL_DT * (BASE_SWING_WEIGHT + BASE_STANCE_WEIGHT)

    def __init__(self, kin, motion, default_q, cfg=None, *, scene, library, observation_mode="root_xy"):
        super().__init__(kin, motion, default_q, cfg, observation_mode=observation_mode)
        configure(library)
        self.foot_reference = PhaseSoleReference(self, scene)
        self.phase_contract = dict(height_std_m=HEIGHT_STD_M, stance_speed_std_mps=STANCE_SPEED_STD_MPS,
            swing_weight=BASE_SWING_WEIGHT, stance_weight=BASE_STANCE_WEIGHT,
            maximum_phase_bonus_per20ms=CONTROL_DT * (BASE_SWING_WEIGHT + BASE_STANCE_WEIGHT),
            original_reward_terms_preserved=True, reference=self.foot_reference.contract,
            stance_velocity="Instantaneous material velocity at the lowest collision point; intent-conditioned proxy, not measured load")

    def to(self, device):
        new = super().to(device)
        if new is self:
            return self
        reference = copy.copy(self.foot_reference)
        reference.mdp, reference.motion = new, new.motion
        reference.sole = copy.deepcopy(reference.sole).to(device)
        for name in ("positions", "velocities", "floor", "stance", "swing", "reference_sole"):
            setattr(reference, name, getattr(reference, name).to(device))
        new.foot_reference = reference
        return new

    def reward(self, state, frames, action, previous_action, terms=False):
        original, parts = super().reward(state, frames, action, previous_action, terms=True)
        reference = self.foot_reference
        full = self.kin.forward(*state, all_bodies=True)
        offsets = reference.sole.offsets(full[1])
        height_error = full[0][..., reference.sole.body_ids, 2] + offsets[..., 2] - reference.reference_sole[frames]
        height_score = torch.exp(-(height_error / HEIGHT_STD_M).square())
        velocity = lowest_material_velocity(reference.sole, full, offsets)
        speed_score = torch.exp(-velocity[..., :2].square().sum(-1) / STANCE_SPEED_STD_MPS**2)
        swing = phase_mean(height_score, reference.swing[frames])
        stance = phase_mean(height_score * speed_score, reference.stance[frames])
        bonus = CONTROL_DT * (BASE_SWING_WEIGHT * swing + BASE_STANCE_WEIGHT * stance)
        value = original + bonus
        if terms:
            return value, {**parts, "track": parts["track"] + bonus,
                           "track_original": parts["track"], "phase_bonus": bonus,
                           "phase_swing_score": swing, "phase_stance_score": stance}
        return value
