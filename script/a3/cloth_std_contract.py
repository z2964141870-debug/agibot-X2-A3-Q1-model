"""Make the official runtime std clamp explicit, without altering action means."""

import torch


def canonical_policy_state(state):
    value = state['std']
    if not torch.isfinite(value).all():
        raise ValueError('Official std is nonfinite')
    result = dict(state)
    result['std'] = value.clamp(.001, .5)
    return result
