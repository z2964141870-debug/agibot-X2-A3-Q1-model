"""E49 fixed budgets, controls, and finite acceptance; no tensor dependencies."""

import math

from script.a3.retention_direction_protocol import Budget as BaseBudget


class Budget(BaseBudget):
    LIMITS = dict(policy_states=1, actor_forwards=16, critic_forwards=1,
                  autograd_grad_calls=3, original_candidates=1, projected_candidates=1)


TOLERANCES = dict(control_atol=1e-7, control_rtol=1e-5,
                  direction_control_atol=1e-8, direction_control_rtol=1e-4,
                  rounded_direction_max=1e-8, finite_mse_increase_max=1e-9,
                  actor_objective_increase_max=1e-7)


def projection_scale(directional, squared_norm):
    if not math.isfinite(directional) or not math.isfinite(squared_norm) or squared_norm <= 0:
        raise ValueError('Nonfinite geometry or zero teacher gradient; no candidate')
    return max(0., directional) / squared_norm


def match_control(actual, expected, tolerances, directional=False):
    prefix = 'direction_control' if directional else 'control'
    if (not math.isfinite(actual) or not math.isfinite(expected) or
            not math.isclose(actual, expected, abs_tol=tolerances[prefix + '_atol'],
                             rel_tol=tolerances[prefix + '_rtol'])):
        raise ValueError('Original E48 scalar/direction control mismatch: ' + repr((actual, expected)))
    return dict(actual=actual, expected=expected, difference=actual-expected, passed=True)


def acceptance(rounded_direction, mse_change, objective_change, rmse, maxabs, tolerances):
    if not all(math.isfinite(v) for v in (rounded_direction, mse_change, objective_change, rmse, maxabs)):
        raise ValueError('Nonfinite finite-candidate evidence')
    gates = dict(rounded_first_order=rounded_direction <= tolerances['rounded_direction_max'],
                 finite_teacher=mse_change <= tolerances['finite_mse_increase_max'],
                 fixed_actor_objective=objective_change <= tolerances['actor_objective_increase_max'],
                 original_teacher_guard=rmse <= .05 and maxabs <= .5)
    accepted = all(gates.values())
    return dict(accepted=accepted, gates=gates,
                classification='LOCAL_PROJECTION_ACCEPTED' if accepted else 'LOCAL_PROJECTION_REJECTED',
                rejection_reasons=[n for n, passed in gates.items() if not passed],
                baseline_improvement_proven=False)
