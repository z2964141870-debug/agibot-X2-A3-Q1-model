"""Bounded lower-body residual and a measured, heuristic upper-arm fallback."""

from __future__ import annotations

import math

import torch
from torch import nn

DT = .02
MAX_ALPHA = .75
RADIUS = .15
FEATURES = 800 + 590 + 8 + 1 + 4 + 1 + 1


def physical_base(default_q, context, indices=None):
    if indices is not None and default_q.ndim == 2 and default_q.shape[0] != 1:
        default_q = default_q[indices]
    result = default_q + context[:, 1:23]
    if result.shape != (len(context), 22):
        raise ValueError("Default joint angles must match the selected physical environments")
    return result


def risk_features(state, reference_p, reference_quaternion, force, slip):
    """Actual root error, reference-relative tilt, angular speed and loaded slip."""
    def gravity(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                            1 - 2 * (x.square() + y.square())), -1)

    actual_g = torch.nn.functional.normalize(gravity(state[:, 3:7]), dim=-1)
    reference_g = torch.nn.functional.normalize(gravity(reference_quaternion), dim=-1)
    tilt = torch.acos((actual_g * reference_g).sum(-1).clamp(-1, 1))
    loaded_slip = torch.where(force > 5., slip, 0.).amax(-1)
    return torch.stack(((state[:, :2] - reference_p[:, :2]).norm(dim=-1), tilt,
                        state[:, 10:12].norm(dim=-1), loaded_slip), -1)


class Fallback:
    """Hysteresis/rate limiter; this is not a dynamics constraint or safety proof."""

    def __init__(self, count, thresholds, device):
        thresholds = torch.as_tensor(thresholds, device=device, dtype=torch.float32)
        if thresholds.shape != (4,) or not bool(torch.isfinite(thresholds).all() & (thresholds > 0).all()):
            raise ValueError("Four finite positive calibration thresholds required")
        self.thresholds = thresholds
        self.alpha = torch.zeros(count, device=device)
        self.filtered = torch.zeros(count, 4, device=device)
        self.cooldown = torch.zeros(count, dtype=torch.long, device=device)

    def reset(self, mask, cap):
        self.alpha[mask] = cap[mask]
        self.filtered[mask] = 0
        self.cooldown[mask] = 0

    def preview(self, risk, cap):
        if risk.shape != self.filtered.shape or cap.shape != self.alpha.shape:
            raise ValueError("Gate inputs must match its environment count")
        filtered = self.filtered + (1 - math.exp(-DT / .1)) * (risk / self.thresholds - self.filtered)
        score = filtered.amax(-1)
        cooldown = torch.where(score > 1., 10, (self.cooldown - 1).clamp(min=0))
        alpha = torch.where(score > 1., self.alpha - 1.5 * DT, self.alpha)
        recover = (score < .7) & (cooldown == 0)
        alpha = torch.where(recover, alpha + .25 * DT, alpha).clamp(min=0)
        return torch.minimum(alpha, cap), filtered, cooldown

    def advance(self, risk, cap):
        self.alpha, self.filtered, self.cooldown = self.preview(risk, cap)

    def state_tensors(self):
        return self.alpha, self.filtered, self.cooldown


def fixed_blocks(function, *inputs):
    outputs = []
    for start in range(0, len(inputs[0]), 4096):
        block = [value[start:start + 4096] for value in inputs]
        count = len(block[0])
        if count < 4096:
            block = [torch.cat((value, value.new_zeros((4096 - count,) + value.shape[1:])))
                     for value in block]
        outputs.append(function(*block)[:count])
    return torch.cat(outputs)


class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES, 256), nn.ELU(), nn.Linear(256, 128),
                                 nn.ELU(), nn.Linear(128, 14))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return fixed_blocks(self.net, features.float())


@torch.no_grad()
def inputs(teacher, history, goal, reference, base, scale, alpha, risk, cooldown, cap):
    normalized_h = teacher.obs_norm(history).flatten(1)
    normalized_g = teacher.goal_norm(goal)
    reference_action = (reference - base) / scale
    features = torch.cat((normalized_h, normalized_g, reference_action[:, 14:],
                          alpha[:, None], risk.clamp(0, 4), cooldown[:, None] / 10., cap[:, None]), -1).float()
    nominal = fixed_blocks(teacher, history, goal).float().clamp(-20, 20)
    return features, nominal, reference_action


def compose(nominal, reference_action, latent, alpha, scale):
    if nominal.shape[-1] != 22 or reference_action.shape != nominal.shape or latent.shape != (len(nominal), 14):
        raise ValueError("Expected physical22 actions, latent14 and per-environment alpha")
    if alpha.shape != (len(nominal),) or scale.shape != (22,):
        raise ValueError("Expected alpha(N) and scale(22)")
    if not bool(torch.isfinite(latent).all()):
        raise FloatingPointError("Nonfinite latent action")
    residual = RADIUS * torch.tanh(latent)
    lower = nominal[:, :14] + residual / scale[:14]
    upper = (1 - alpha[:, None]) * nominal[:, 14:] + alpha[:, None] * reference_action[:, 14:]
    action = torch.cat((lower, upper), -1).clamp(-20, 20)
    return action, residual


def balance_rewards(original, state, reference_p, reference_q, residual):
    root = torch.exp(-((state[:, :2] - reference_p[:, :2]) / .15).square().sum(-1))
    leg = torch.exp(-((state[:, 13:25] - reference_q[:, :12]) / .3).square()).mean(-1)
    cost = (residual / RADIUS).square().mean(-1)
    total = original + .02 * root + .01 * leg - .002 * cost
    return total, root, leg, cost
