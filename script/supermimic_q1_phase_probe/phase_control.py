"""Bounded, filtered leg-reference feedforward with no learned parameters."""

import math

import torch


CANDIDATES = {
    "baseline": (0., 0., "all"),
    "uniform010": (.10, .10, "all"),
    "uniform025": (.25, .25, "all"),
    "swing025": (.25, 0., "phase"),
    "swing050": (.50, 0., "phase"),
    "mixed050010": (.50, .10, "phase"),
}


class LegReference:
    def __init__(self, count, candidate, device):
        if candidate not in CANDIDATES:
            raise ValueError(candidate)
        self.swing_gain, self.stance_gain, self.mode = CANDIDATES[candidate]
        self.value = torch.zeros(count, 12, device=device)

    def reset(self, mask):
        self.value[mask] = 0.

    def correction(self, nominal_target, reference, stance, swing):
        if nominal_target.shape != self.value.shape or reference.shape != self.value.shape:
            raise ValueError("Expected physical12 leg targets")
        if stance.shape != (len(reference), 2) or swing.shape != stance.shape:
            raise ValueError("Expected two reference foot phase flags")
        if bool((stance & swing).any()):
            raise ValueError("Reference stance and swing must not overlap")
        if not bool(torch.isfinite(nominal_target).all() & torch.isfinite(reference).all()):
            raise FloatingPointError("Nonfinite leg target")
        if self.mode == "all":
            gain = torch.full_like(stance, self.swing_gain, dtype=reference.dtype)
        else:
            gain = self.swing_gain * swing + self.stance_gain * stance
        desired = (gain.repeat_interleave(6, -1) * (reference - nominal_target)).clamp(-.15, .15)
        delta = (1. - math.exp(-.02 / .1)) * (desired - self.value)
        self.value = (self.value + delta.clamp(-.02, .02)).clamp(-.15, .15)
        return self.value.clone()
