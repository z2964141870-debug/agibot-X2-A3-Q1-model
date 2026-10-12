"""Sampling snapshots must survive mutable environment storage."""

import unittest

import torch
from tensordict import TensorDict

from script.a3.zhy_likelihood_trainer import sampling_identity, snapshot


class SamplingIdentityTests(unittest.TestCase):
    def test_snapshot_survives_in_place_environment_step(self):
        value = torch.tensor([[1., 2.]])
        stored = snapshot({'obs':value})
        value.add_(10.)
        self.assertTrue(torch.equal(stored['obs'], torch.tensor([[1., 2.]])))

    def test_snapshot_copies_actual_tensordict_container(self):
        value = TensorDict({'obs':torch.tensor([[1., 2.]])}, batch_size=[1])
        stored = snapshot(value)
        self.assertIsInstance(stored, dict)
        value['obs'].add_(10.)
        self.assertTrue(torch.equal(stored['obs'], torch.tensor([[1., 2.]])))

    def test_detects_observation_shift_even_when_means_match(self):
        records = [dict(inputs={'actor_obs':torch.full((2,1,3), float(i))},
                        mean=torch.zeros(2,1,29)) for i in range(3)]
        rollout = dict(all_obs_dict={'actor_obs':torch.ones(2,3,3)}, old_mu_batch=torch.zeros(2,3,29))
        result = sampling_identity(records, rollout)
        self.assertFalse(result['all_inputs_exact'])
        self.assertTrue(result['sampled_means_match_storage'])

    def test_refuses_missing_sampling_call(self):
        record = dict(inputs={'obs':torch.zeros(2,1,3)}, mean=torch.zeros(2,1,29))
        rollout = dict(all_obs_dict={'obs':torch.zeros(2,3,3)}, old_mu_batch=torch.zeros(2,3,29))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            sampling_identity([record, record], rollout)


if __name__ == '__main__':
    unittest.main()
