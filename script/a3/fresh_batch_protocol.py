"""E48 fixed budgets and small, runtime-independent contract checks."""

from script.a3.retention_direction_protocol import Budget as BaseBudget


class Budget(BaseBudget):
    LIMITS = dict(policy_states=1, actor_forwards=40, critic_forwards=8,
                  autograd_grad_calls=3, fresh_rollouts=1, environment_steps=24,
                  mathematical_candidates=1)


def require_batch_contract(environments, steps, batch_size, local_batch_size):
    if (environments, steps, batch_size, local_batch_size) != (16, 24, 4, 16):
        raise ValueError('Registered rollout or first microbatch dimensions changed')


def classify(raw_descent, adam_first_order, finite_change):
    if raw_descent < 0 and adam_first_order > 0:
        return 'RAW_DESCENT_INWARD_ADAM_OUTWARD'
    if adam_first_order < 0 and finite_change > 0:
        return 'FIRST_ORDER_INWARD_FINITE_OUTWARD'
    if raw_descent > 0:
        return 'FRESH_BATCH_RAW_DESCENT_OUTWARD'
    if raw_descent < 0 and adam_first_order < 0 and finite_change < 0:
        return 'FRESH_BATCH_ALL_INWARD_FAILURE_NOT_REPRODUCED'
    return 'OTHER_OR_DEGENERATE_DIRECTION'
