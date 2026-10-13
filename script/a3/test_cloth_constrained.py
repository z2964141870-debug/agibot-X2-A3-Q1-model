"""Exercise freeze ownership and a bad likelihood before any real update."""

import unittest

import torch

from script.a3.cloth_constrained_trainer import restrict_policy, initial_probability


class ClothContractTests(unittest.TestCase):
    def test_optimizer_cannot_change_encoder_or_std(self):
        policy = torch.nn.Module()
        policy.actor_module = torch.nn.Module()
        policy.actor_module.encoders = torch.nn.ModuleDict({'a3_fast': torch.nn.Linear(2, 2)})
        policy.actor_module.decoders = torch.nn.ModuleDict({
            'g1_dyn': torch.nn.Linear(2, 1), 'g1_kin': torch.nn.Linear(2, 2)})
        policy.std = torch.nn.Parameter(torch.ones(1))
        scope = restrict_policy(policy)
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=.01)
        output = policy.actor_module.decoders['g1_dyn'](
            policy.actor_module.encoders['a3_fast'](torch.ones(3, 2)))
        output.square().mean().backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in policy.named_parameters()
                            if n in scope['trainable']))
        for n, p in policy.named_parameters():
            if n in scope['frozen']:
                self.assertTrue(torch.equal(before[n], p))
                self.assertIsNone(p.grad)

    def test_missing_codec_rejected(self):
        with self.assertRaises(ValueError):
            restrict_policy(torch.nn.Linear(2, 1))

    def test_masked_likelihood_and_shift_rejected(self):
        mean, std, actions = torch.zeros(2, 3, 29), torch.ones(2, 3, 29), torch.zeros(2, 3, 29)
        old = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
        batch = dict(mb_padding_mask=torch.zeros(2, 3, dtype=torch.bool),
            mb_old_mu=mean.clone(), mb_old_sigma=std,
            mb_actions=actions, mb_logprobs=old.clone())
        batch['mb_padding_mask'][0, 0] = True
        batch['mb_logprobs'][0, 0] = float('nan')
        result = dict(action_mean=mean, action_std=std)
        self.assertEqual(initial_probability(result, batch)['max_abs_ratio_minus_one'], 0.)
        result['action_mean'] = mean + .01
        with self.assertRaises(ValueError):
            initial_probability(result, batch)
        batch['mb_padding_mask'].fill_(True)
        with self.assertRaises(ValueError):
            initial_probability(dict(action_mean=mean, action_std=std), batch)


if __name__ == '__main__':
    unittest.main()
