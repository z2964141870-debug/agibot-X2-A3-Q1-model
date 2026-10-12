"""Captured environment ordering must survive chunked and temporal replay."""

import unittest

import torch

from script.a3.zhy_batch_probe import replay_layout


class IdentityActor:
    def forward(self, obs, **kwargs):
        return obs['actor_obs'][..., :29]


class LayoutTests(unittest.TestCase):
    def test_permuted_environment_groups_preserve_positions(self):
        values = torch.arange(16*24*29).reshape(16,24,29).float()
        groups = [torch.tensor([5,2,0,7]), torch.tensor([1,3,4,6]),
                  torch.arange(8,12), torch.arange(12,16)]
        for frames in (False, True):
            actual = replay_layout(IdentityActor(), {'actor_obs':values}, groups, frames, False)
            self.assertTrue(torch.equal(actual, values))

    def test_missing_or_repeated_environment_is_rejected(self):
        values = torch.zeros(16,24,29)
        for groups in ([torch.arange(15)], [torch.arange(16), torch.tensor([0])]):
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                replay_layout(IdentityActor(), {'actor_obs':values}, groups, False, False)


if __name__ == '__main__':
    unittest.main()
