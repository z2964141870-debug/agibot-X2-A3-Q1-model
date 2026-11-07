"""Physical-only input gradients and central finite differences; no fitting."""

import time

import torch


class StepCounter:
    def __init__(self, model, deadline_ns, maximum=512):
        self.model, self.deadline_ns, self.maximum = model, deadline_ns, maximum
        self.calls = 0

    def __call__(self, state, action, context):
        if self.calls >= self.maximum or time.monotonic_ns() >= self.deadline_ns:
            raise RuntimeError("Bounded world-model forward call/deadline reached")
        self.calls += 1
        result = self.model.step(state, action, context)
        if result.shape != state.shape or not torch.isfinite(result).all():
            raise ValueError("Nonfinite or incorrectly shaped world-model prediction")
        return result


def gradient_probe(step, state, action, context, present, feature_fn, *, epsilons=(.01, .001),
                   relative_tolerance=.05, absolute_tolerance=.002):
    if state.shape != (1, 269) or action.shape != (1, 29) or context.shape != (1, 62):
        raise ValueError("Gradient probe uses one exact native transition")
    action = action.detach().clone().requires_grad_(True)
    predicted = step(state, action, context)
    features = feature_fn(predicted[..., :240])
    if features.ndim != 2 or features.shape[0] != 1 or not torch.isfinite(features).all():
        raise ValueError("Physical probe features must exclude the deterministic command tail")
    weights = torch.cos(torch.arange(features.shape[-1], device=features.device, dtype=features.dtype)+.5)
    scalar = (features*weights).sum()
    gradient = torch.autograd.grad(scalar, action, allow_unused=True)[0] if scalar.requires_grad else None
    gradient = torch.zeros_like(action) if gradient is None else gradient
    finite = bool(torch.isfinite(gradient).all())
    absent_zero = bool(torch.equal(gradient[:, ~present], torch.zeros_like(gradient[:, ~present])))
    values = []
    with torch.no_grad():
        for joint in present.nonzero().flatten().tolist():
            analytic = float(gradient[0, joint])
            checks = []
            for epsilon in epsilons:
                direction = torch.zeros_like(action)
                direction[0, joint] = epsilon
                high = (feature_fn(step(state, action+direction, context)[..., :240])*weights).sum()
                low = (feature_fn(step(state, action-direction, context)[..., :240])*weights).sum()
                numeric = float((high-low)/(2*epsilon))
                error = abs(numeric-analytic)
                checks.append(dict(epsilon=epsilon, finite_difference=numeric, absolute_error=error,
                    passed=error <= absolute_tolerance+relative_tolerance*max(abs(analytic), abs(numeric))))
            values.append(dict(joint=joint, autograd=analytic, finite_differences=checks,
                finite_difference_passed=any(c["passed"] for c in checks)))
        direction = (~present).to(action.dtype).unsqueeze(0)*epsilons[0]
        high = step(state, action+direction, context)[..., :240]
        low = step(state, action-direction, context)[..., :240]
        absent_response = float((high-low).abs().max())
    return dict(physical_gradient_finite=finite, absent_joint_gradient_exact_zero=absent_zero,
        absent_joint_physical_response_max=absent_response, present_joints=values,
        finite_difference_passed=finite and absent_zero and absent_response == 0.
            and all(row["finite_difference_passed"] for row in values),
        nonzero_physical_action_response=finite and float(gradient[:, present].abs().max()) > 1e-7,
        physical_gradient_max=float(gradient[:, present].abs().max()) if finite else None,
        command_memory_excluded=True, scalar_probe="cosine projection of fixed-frame normalized physical features",
        jacobian_scope="one VJP, not a complete physical Jacobian or closed-loop guarantee")
