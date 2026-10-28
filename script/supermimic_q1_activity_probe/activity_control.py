"""Reference-motion activity schedules for upper-body composition."""

import math


CANDIDATES = {
    "baseline": (.5, .5),
    "static025": (.25, .25),
    "static075": (.75, .75),
    "activity025075": (.25, .75),
    "activity050075": (.5, .75),
}


class UpperActivityBlend:
    def __init__(self, count, candidate, device):
        import torch
        self.torch = torch
        self.low, self.high = CANDIDATES[candidate]
        self.speed = torch.zeros(count, device=device)
        self.yaw = torch.zeros(count, device=device)
        self.alpha = torch.full((count,), .5, device=device)
        self.initialized = torch.zeros(count, device=device, dtype=torch.bool)
        self.factor = 1-math.exp(-.02/.20)

    def reset(self, mask):
        self.initialized[mask] = False

    def update(self, speed, yaw_rate):
        torch = self.torch
        if speed.shape != self.speed.shape or yaw_rate.shape != self.yaw.shape:
            raise ValueError("Expected per-environment reference activity")
        if not bool(torch.isfinite(speed).all() and torch.isfinite(yaw_rate).all()) or bool((speed < 0).any()):
            raise ValueError("Finite nonnegative speed and finite yaw rate required")
        self.speed = torch.where(self.initialized, self.speed+self.factor*(speed-self.speed), speed)
        self.yaw = torch.where(self.initialized, self.yaw+self.factor*(yaw_rate.abs()-self.yaw), yaw_rate.abs())
        activity = torch.maximum(((self.speed-.10)/.25).clamp(0, 1), ((self.yaw-.5)/1.).clamp(0, 1))
        desired = self.high-(self.high-self.low)*activity
        self.alpha = torch.where(self.initialized, self.alpha+self.factor*(desired-self.alpha), desired)
        self.initialized.fill_(True)
        return self.alpha.clone()
