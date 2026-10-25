"""Ensure open-loop reward gradients include the perturbed action-rate history."""

import unittest

import numpy as np
import torch

from probe_reward_directions import model_rewards
from supermimic.world_model.heads import ReferenceFrame


class RateOnlyHeads:
    def reward(self, state, action, previous, reference):
        return {"total": -((action - previous) ** 2).sum(-1)}


class ToyModel:
    heads = RateOnlyHeads()

    def step(self, state, action, context):
        return state + torch.nn.functional.pad(action, (0, state.shape[-1] - action.shape[-1]))


class RewardDirectionTests(unittest.TestCase):
    def test_first_action_also_changes_next_rate_penalty(self):
        dtype = torch.float64
        state = torch.zeros(1, 57, dtype=dtype)
        previous = torch.zeros(1, 22, dtype=dtype)
        actions = torch.zeros(32, 22, dtype=dtype)
        refs = ReferenceFrame(*[torch.zeros(32, 1, dtype=dtype) for _ in range(6)])
        delta = torch.zeros(1, 22, dtype=dtype, requires_grad=True)
        with torch.no_grad():
            delta[0, 0] = .1
        rewards, _ = model_rewards(ToyModel(), state, previous, actions, refs, delta)
        np.testing.assert_allclose(rewards[0, :3].detach().numpy(), [-.01, -.01, 0.], atol=1e-15)
        gradient = torch.autograd.grad(rewards.sum(), delta)[0]
        self.assertAlmostEqual(float(gradient[0, 0]), -.4)
        direction = torch.zeros_like(delta)
        direction[0, 0] = 1.
        with torch.no_grad():
            plus, _ = model_rewards(ToyModel(), state, previous, actions, refs, delta + direction * 1e-5)
            minus, _ = model_rewards(ToyModel(), state, previous, actions, refs, delta - direction * 1e-5)
        self.assertAlmostEqual(float((plus.sum() - minus.sum()) / 2e-5), -.4, places=10)


if __name__ == "__main__":
    unittest.main()
