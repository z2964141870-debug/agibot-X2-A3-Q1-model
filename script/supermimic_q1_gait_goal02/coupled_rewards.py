"""Independent bounded reward variant; frozen GOAL01 sources stay unchanged."""

import torch

from coupling import coupling_factors
from goal02_config import contract
from reward_candidates import (BASE_STANCE_WEIGHT, BASE_SWING_WEIGHT, CONTROL_DT,
                               HEIGHT_STD_M, MDP, PhaseSoleMDP, PhaseSoleReference,
                               STANCE_SPEED_STD_MPS, lowest_material_velocity, phase_mean)


class CoupledMDP(PhaseSoleMDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_contract = {**self.phase_contract, "goal02": contract()}

    def reward(self, state, frames, action, previous_action, terms=False):
        original, parts = MDP.reward(self, state, frames, action, previous_action, terms=True)
        reference = self.foot_reference
        full = self.kin.forward(*state, all_bodies=True)
        offsets = reference.sole.offsets(full[1])
        sole = full[0][..., reference.sole.body_ids, 2] + offsets[..., 2]
        expected = reference.reference_sole[frames]
        height_score = torch.exp(-((sole - expected) / HEIGHT_STD_M).square())
        velocity = lowest_material_velocity(reference.sole, full, offsets)
        speed_score = torch.exp(-velocity[..., :2].square().sum(-1) / STANCE_SPEED_STD_MPS**2)
        swing = phase_mean(height_score, reference.swing[frames])
        stance = phase_mean(height_score * speed_score, reference.stance[frames])
        uncoupled = CONTROL_DT * (BASE_SWING_WEIGHT * swing + BASE_STANCE_WEIGHT * stance)
        factors = coupling_factors(state[1], self.motion.root_R[frames], sole, expected,
                                   reference.stance[frames], reference.swing[frames])
        bonus = uncoupled * factors["factor"]
        value = original + bonus
        if terms:
            return value, {**parts, "track": parts["track"] + bonus,
                           "track_original": parts["track"], "phase_bonus": bonus,
                           "phase_bonus_uncoupled": uncoupled,
                           "phase_swing_score": swing, "phase_stance_score": stance,
                           "posture_factor": factors["posture_factor"],
                           "pair_factor": factors["pair_factor"]}
        return value


class CoupledReference(PhaseSoleReference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = {**self.contract, "goal02": contract()}
        self.record_diagnostics = False
        self.latest_diagnostics = None

    def reward(self, original, state, frames, targets, previous_targets, residual, force, slip):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        total, parts = super().reward(original, state, frames, targets, previous_targets,
                                      residual, force, slip)
        unflat = unflatten_state(state)
        full = self.mdp.kin.forward(*unflat, all_bodies=True)
        sole = self.sole(full[0], full[1])
        expected = self.reference_sole[frames]
        factors = coupling_factors(unflat[1], self.motion.root_R[frames], sole, expected,
                                   self.stance[frames], self.swing[frames])
        before = parts["stance_bonus"] + parts["swing_bonus"]
        stance_bonus = parts["stance_bonus"] * factors["factor"]
        swing_bonus = parts["swing_bonus"] * factors["factor"]
        total = total - before + stance_bonus + swing_bonus
        if self.record_diagnostics:
            # Force belongs to the last pre-integration solve, pose to post-step.
            self.latest_diagnostics = dict(
                frames=frames.detach().clone(), swing=self.swing[frames].detach().clone(),
                stance=self.stance[frames].detach().clone(),
                reference_sole=expected.detach().clone(), sole=sole.detach().clone(),
                last_solve_force=force.detach().clone(),
                residual_rms=residual.square().mean(-1).sqrt().detach().clone(),
                posture_factor=factors["posture_factor"].detach().clone(),
                pair_factor=factors["pair_factor"].detach().clone(),
            )
        return total, {**parts, "stance_bonus": stance_bonus, "swing_bonus": swing_bonus,
                       "uncoupled_foot_bonus": before,
                       "posture_factor": factors["posture_factor"],
                       "pair_factor": factors["pair_factor"]}
