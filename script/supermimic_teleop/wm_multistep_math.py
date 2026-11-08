"""Open-loop saved-action rollouts and numerical directional gradient checks."""

import numpy as np
import torch

HORIZONS = (1, 2, 4, 8, 16, 32, 40)


def require(value, message):
    if not value:
        raise ValueError(message)


def common_support(rows, arrays):
    require(len(rows) == len(arrays) == 2
        and [r['arm'] for r in rows] == ['student_f0', 'privileged_teacher_h10'],
        'Require both original ordered recovery arms')
    for row, data in zip(rows, arrays):
        n = row['attempted_steps']
        require(type(n) is int and n > 0
            and data['actual'].shape == data['reference'].shape == (n+1, 79)
            and data['canonical_action'].shape == (n, 29) and data['load'].shape == (n+1, 2)
            and all(np.isfinite(data[k]).all() for k in ('actual', 'reference', 'canonical_action', 'load')),
            'Invalid original saved physical trace')
        valid = n if row['failure_frame'] is None else n-1
        require(row['valid_prefix_steps'] == valid
            and (row['failure_frame'] is None or row['failure_frame'] == row['anchor']+n),
            'Exclude the terminating transition, not only its label')
    n = min(r['valid_prefix_steps'] for r in rows)
    require(n > 0 and rows[0]['clip'] == rows[1]['clip'] and rows[0]['anchor'] == rows[1]['anchor']
        and rows[0]['split'] == rows[1]['split'] and rows[0]['split'] in ('train', 'validation')
        and np.array_equal(arrays[0]['actual'][0], arrays[1]['actual'][0])
        and np.array_equal(arrays[0]['reference'][:n+1], arrays[1]['reference'][:n+1]),
        'Require the identical original anchor and common nonterminal target')
    return n


def rollout(step, initial, actions, context):
    require(initial.ndim == 2 and initial.shape[0] == 1 and actions.ndim == 3
        and actions.shape[1] == 1 and context.ndim == 2 and context.shape[0] == 1
        and len(actions) > 0, 'Require one fixed initial state and an ordered action prefix')
    state, predictions = initial, []
    for action in actions:
        state = step(state, action, context)
        require(state.shape == initial.shape and torch.isfinite(state).all(), 'Invalid open-loop prediction')
        predictions.append(state)
    return torch.cat(predictions, dim=0)


def directional_gradient(step, initial, actions, context, present, physical_scalar,
                         epsilons=(.01, .001), relative=.05, absolute=.002):
    require(not initial.requires_grad and not actions.requires_grad and present.dtype == torch.bool
        and present.shape == actions.shape[-1:] and present.any(), 'Fixed detached states/actions required')
    first = actions[0].detach().clone().requires_grad_(True)
    sequence = torch.cat((first[None], actions[1:]), dim=0)
    value = physical_scalar(rollout(step, initial, sequence, context)[-1:])
    require(value.numel() == 1 and value.requires_grad and torch.isfinite(value), 'Finite physical-only scalar required')
    gradient = torch.autograd.grad(value, first)[0]
    require(torch.isfinite(gradient).all(), 'Nonfinite multi-step gradient')
    direction = torch.cos(torch.arange(first.shape[-1], device=first.device, dtype=first.dtype)+.5)
    direction = torch.where(present, direction, 0.)
    direction = direction/direction.norm()
    analytic = float((gradient*direction).sum())
    checks = []
    with torch.no_grad():
        for epsilon in epsilons:
            require(np.isfinite(epsilon) and epsilon > 0, 'Positive finite difference size required')
            values = []
            for sign in (1., -1.):
                changed = actions.clone()
                changed[0] += sign*epsilon*direction
                values.append(physical_scalar(rollout(step, initial, changed, context)[-1:]))
            numeric = float((values[0]-values[1])/(2*epsilon))
            error = abs(numeric-analytic)
            checks.append(dict(epsilon=epsilon, finite_difference=numeric, absolute_error=error,
                passed=bool(np.isfinite(numeric) and error <= absolute+relative*max(abs(analytic), abs(numeric)))))
    absent_zero = bool(torch.equal(gradient[:, ~present], torch.zeros_like(gradient[:, ~present])))
    return dict(steps=len(actions), autograd_directional_derivative=analytic, finite_differences=checks,
        gradient_norm=float(gradient.norm()), absent_joint_gradient_exact_zero=absent_zero,
        nonzero_response=bool(float(gradient[:, present].abs().max()) > 1e-7),
        numerical_directional_gradient_passed=absent_zero and any(c['passed'] for c in checks)
            and bool(float(gradient[:, present].abs().max()) > 1e-7),
        differentiated_action_index=0, later_actions_fixed=True, command_memory_excluded=True,
        physical_action_gradient_validated=False,
        scope='One deterministic present-joint direction through model states; not a full Jacobian or physical gradient test')


def prefix_errors(errors, steps):
    require(type(steps) is int and steps > 0
        and all(len(v) >= steps and np.isfinite(v).all() for v in errors.values()), 'Exact finite support required')
    return dict(endpoint={k:float(v[steps-1]) for k,v in errors.items()},
        prefix_rms={k:float(np.sqrt(np.mean(np.square(v[:steps])))) for k,v in errors.items()})


def translation_decomposition(prediction, target, root_slot):
    require(prediction.shape == target.shape and prediction.ndim == 3 and prediction.shape[2] == 3
        and 0 <= root_slot < prediction.shape[1], 'Require paired native body translations')
    error = prediction-target
    root = error[:, root_slot]
    relative = error-root[:, None]
    values = dict(world=np.square(error).sum(-1).mean(-1), root=np.square(root).sum(-1),
        relative=np.square(relative).sum(-1).mean(-1),
        signed_cross=2*np.einsum('nbi,ni->nb', relative, root).mean(-1))
    require(np.allclose(values['world'], values['root']+values['relative']+values['signed_cross'],
        rtol=1e-5, atol=1e-8), 'Signed translation decomposition does not close')
    return values
