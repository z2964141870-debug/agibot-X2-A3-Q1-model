"""Cached tokens preserve decoder gradients and reject unfrozen encoders."""

import unittest
import torch

from script.a3.cloth_cached_tokens import cached_distribution


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_module = torch.nn.Module()
        self.actor_module.encoders = torch.nn.ModuleDict({'a3_fast':torch.nn.Linear(2,2)})
        self.actor_module.decoders = torch.nn.ModuleDict({'g1_dyn':torch.nn.Linear(64,29)})
        self.std = torch.nn.Parameter(torch.ones(29)*.3, requires_grad=False)
        self.actor_module.encoders.requires_grad_(False)
        self.actor_module.forward_with_external_tokens = lambda obs,tokens: self.actor_module.decoders['g1_dyn'](tokens.flatten(2))

    @property
    def get_std(self):
        return self.std

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)


class CachedTokenTests(unittest.TestCase):
    def test_decoder_has_gradient_and_codec_stays_frozen(self):
        policy = ToyPolicy()
        tokens, actions = torch.randn(4,24,2,32), torch.randn(4,24,29)
        before = {n:p.detach().clone() for n,p in policy.named_parameters()}
        result = cached_distribution(policy, {}, tokens, actions)
        loss = -result['logprobs'].mean()
        loss.backward()
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=.001)
        optimizer.step()
        self.assertIsNone(policy.std.grad)
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in policy.named_parameters() if 'decoders' in n))
        self.assertTrue(all(torch.equal(before[n],p) and p.grad is None for n,p in policy.named_parameters() if 'decoders' not in n))
        self.assertTrue(torch.equal(result['logprobs'], torch.distributions.Normal(
            result['action_mean'],result['action_std']).log_prob(actions).sum(-1)))

    def test_unfrozen_codec_rejected(self):
        policy = ToyPolicy()
        policy.actor_module.encoders.requires_grad_(True)
        with self.assertRaises(ValueError):
            cached_distribution(policy, {}, torch.zeros(4,24,2,32), torch.zeros(4,24,29))

    def test_bad_token_shape_gradient_and_nonfinite_rejected(self):
        policy, actions = ToyPolicy(), torch.zeros(4,24,29)
        for tokens in (torch.zeros(4,24,64), torch.zeros(4,24,2,32, requires_grad=True),
                       torch.full((4,24,2,32), float('nan'))):
            with self.assertRaises(ValueError):
                cached_distribution(policy, {}, tokens, actions)
