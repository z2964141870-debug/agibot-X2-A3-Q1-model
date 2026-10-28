"""Physical-radian leg target filters; arms, waist, gains and torque limits stay fixed."""

import math

import torch

CANDIDATES = {
    "baseline": ("none", 0.),
    "ema010": ("ema", .010),
    "ema020": ("ema", .020),
    "ema040": ("ema", .040),
    "rate010": ("rate", .10),
    "rate015": ("rate", .15),
}


class LegTargetFilter:
    def __init__(self, count, candidate, device):
        self.mode, self.parameter = CANDIDATES[candidate]
        self.value = torch.zeros(count, 12, device=device)
        self.ready = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, mask):
        self.ready[mask] = False

    def filter(self, target):
        if target.shape != self.value.shape:
            raise ValueError("Expected one physical-radian target for each of12 leg joints")
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError("Nonfinite physical target")
        previous = torch.where(self.ready[:, None], self.value, target)
        if self.mode == "none":
            value = target
        elif self.mode == "ema":
            value = previous + (1-math.exp(-.02/self.parameter)) * (target-previous)
        else:
            value = previous + (target-previous).clamp(-self.parameter, self.parameter)
        self.value = value.clone()
        self.ready[:] = True
        return value
