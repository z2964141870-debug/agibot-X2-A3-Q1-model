"""Scalar-tested, non-mutating E47 AdamW formula and fixed diagnostic limits."""

import math
from script.a3.retention_direction_protocol import Budget as BaseBudget


class Budget(BaseBudget):
    LIMITS = dict(policy_states=1, actor_forwards=12, critic_forwards=1, autograd_grad_calls=3)


def candidate(parameter, gradient, moment, variance, step, lr, beta1, beta2, eps,
              weight_decay, sqrt=math.sqrt):
    """Pure AdamW expression: no optimizer object, parameter/moment mutation, or save."""
    if (not isinstance(step, int) or step < 0 or not all(math.isfinite(v) for v in
            (lr, beta1, beta2, eps, weight_decay)) or lr <= 0 or eps <= 0 or
            weight_decay < 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1):
        raise ValueError('Unsupported AdamW scalar configuration')
    next_m = beta1 * moment + (1 - beta1) * gradient
    next_v = beta2 * variance + (1 - beta2) * gradient * gradient
    denominator = sqrt(next_v) / math.sqrt(1 - beta2 ** (step + 1)) + eps
    value = parameter * (1 - lr * weight_decay) - (
        lr / (1 - beta1 ** (step + 1))) * next_m / denominator
    return value, next_m, next_v


def clip_factor(total_norm, maximum=.1):
    if not math.isfinite(total_norm) or total_norm < 0 or maximum <= 0:
        raise ValueError('Invalid full gradient norm')
    return min(1., maximum / (total_norm + 1e-6))


def require_group_ids(ids, count, start):
    if ids != list(range(start, start + count)):
        raise ValueError('Saved optimizer group IDs differ from full model order')
