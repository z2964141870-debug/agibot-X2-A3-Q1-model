"""Frozen Q1 teacher with reference arms and a separate lower-body residual."""

from __future__ import annotations

import math

import torch
from torch import nn

from q1_upper_reference import reference_centered_actions


UPPER_OBSERVATIONS = {
    "position": slice(14, 22),
    "velocity": slice(36, 44),
    "previous_action": slice(64, 72),
}


def swap_upper_history(history, donor, fields=tuple(UPPER_OBSERVATIONS)):
    """Counterfactual attribution only; do not substitute actual feedback in use."""
    if history.shape != donor.shape or history.shape[-2:] != (10, 80):
        raise ValueError("Expected matching Q1 histories with shape (..., 10, 80)")
    result = history.clone()
    for field in fields:
        result[..., UPPER_OBSERVATIONS[field]] = donor[..., UPPER_OBSERVATIONS[field]]
    return result


def lower_gaussian_kl(candidate, teacher, std=.1):
    """KL for the 14 lower action coordinates, with equal fixed action-unit std."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and positive")
    if candidate.shape != teacher.shape or candidate.shape[-1] != 22:
        raise ValueError("Expected matching 22-coordinate Q1 action means")
    return .5 * ((candidate[..., :14] - teacher[..., :14]) / std).square().sum(-1)


class ReferenceBalance(nn.Module):
    """Only the 14-output residual head is trainable; upper targets are explicit.

    This is a candidate architecture, not a trained controller or PPO runner.
    A future stochastic learner must model only the 14 lower stochastic inputs;
    the eight deterministic arm coordinates are not a 22-D Gaussian action.
    """

    def __init__(self, teacher, action_scale, residual_radius=.15):
        super().__init__()
        if not math.isfinite(residual_radius) or residual_radius <= 0:
            raise ValueError("residual_radius must be finite and positive")
        scale = torch.as_tensor(action_scale, dtype=torch.float32)
        if scale.shape != (22,) or not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("Expected 22 finite, positive action scales")
        self.teacher = teacher.requires_grad_(False).eval()
        self.register_buffer("action_scale", scale.clone())
        self.residual_radius = residual_radius
        self.head = nn.Sequential(nn.Linear(10 * 80 + 590 + 8, 256), nn.ELU(),
                                  nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 14))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def teacher_actions(self, history, goal):
        with torch.no_grad():
            return self.teacher(history, goal).float().clamp(-20, 20)

    def forward(self, history, goal, reference, base):
        if history.shape[1:] != (10, 80) or goal.shape != (len(history), 590):
            raise ValueError("Expected Q1 history (N, 10, 80) and goal (N, 590)")
        if reference.shape != (len(history), 22) or base.shape != reference.shape:
            raise ValueError("Expected current physical base and next-frame reference (N, 22)")
        with torch.no_grad():
            normalized_history = self.teacher.obs_norm(history).flatten(1)
            normalized_goal = self.teacher.goal_norm(goal)
            reference_action = (reference - base) / self.action_scale
            features = torch.cat((normalized_history, normalized_goal,
                                  reference_action[:, 14:]), -1).float()
        nominal = self.teacher_actions(history, goal)
        residual = self.residual_radius * torch.tanh(self.head(features))
        result = reference_centered_actions(nominal, reference, base,
                                             self.action_scale, 0.)
        result[:, :14] = (nominal[:, :14] + residual / self.action_scale[:14]).clamp(-20, 20)
        return result


class ReferenceBalancePolicy:
    """Bind explicit physical reference/base inputs to a no-noise evaluator."""

    def __init__(self, model, env):
        self.model, self.env = model, env

    def __call__(self, history, goal):
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

        current = self.env.state.info[SUPERMIMIC_STATE_KEY]
        motion = self.env.supermimic_mdp.motion
        frames = current["f"]
        next_frames = (frames + 1).minimum(motion.clip_end[motion.clip_id[frames]])
        base = self.env._supermimic_command.default_q + current["xi"][:, 1:23]
        return self.model(history, goal, motion.q[next_frames], base)
