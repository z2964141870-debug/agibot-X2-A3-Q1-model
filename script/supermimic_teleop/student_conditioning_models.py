"""Isolate direct causal human conditioning of the existing robot residual."""

import copy
from itertools import chain

import torch
from torch import nn

from student_onpolicy_pair_fit import warm_student


ARMS = ('zero_human', 'real_human')


class ConditionedResidualStudent(nn.Module):
    def __init__(self, reference, variant, latent_dim):
        super().__init__()
        if variant not in ARMS or type(latent_dim) is not int or latent_dim < 1:
            raise ValueError('Use the registered conditioning arm and actual latent width')
        self.variant = variant
        self.encoder = copy.deepcopy(reference.encoder).eval().requires_grad_(False)
        self.decoder = copy.deepcopy(reference.decoder).eval().requires_grad_(False)
        self.adapter = copy.deepcopy(reference.adapter).requires_grad_(True)
        self.human_adapter = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(), nn.Linear(128, latent_dim))
        nn.init.zeros_(self.human_adapter[-1].weight)
        nn.init.zeros_(self.human_adapter[-1].bias)
        parameter = next(self.decoder.parameters())
        self.human_adapter.to(device=parameter.device, dtype=parameter.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.decoder.eval()
        return self

    def conditioning(self, base):
        return base if self.variant == 'real_human' else torch.zeros_like(base)

    def latent(self, normalized_history, goal):
        if (normalized_history.ndim != 2 or normalized_history.shape[1] != 930
                or goal.shape != (len(normalized_history), 10, 226)):
            raise ValueError('Require paired causal human history and930 robot-history features')
        base = self.encoder(goal)
        delta = torch.tanh(self.adapter(normalized_history)+self.human_adapter(self.conditioning(base)))
        return base+(1.-base.abs())*delta

    def forward(self, normalized_history, goal):
        return self.decoder(self.latent(normalized_history, goal), normalized_history)

    def trainable_parameters(self):
        return chain(self.adapter.parameters(), self.human_adapter.parameters())


def conditioned_student(native, payload, variant):
    return ConditionedResidualStudent(warm_student(native, payload), variant, native.policy.cfg.latent)


def zero_human_branch(state):
    return (set(state) == {'0.weight', '0.bias', '2.weight', '2.bias'}
        and not bool(torch.count_nonzero(state['2.weight']))
        and not bool(torch.count_nonzero(state['2.bias'])))
