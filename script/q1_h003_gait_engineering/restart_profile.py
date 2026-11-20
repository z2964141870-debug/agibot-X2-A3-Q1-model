"""Optional Q1 per-recovery restart settings; angular-rate correction precedes safety PD."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RestartProfile:
    resume_ticks: int
    roll_rate_gain: float = 0.0

    def __post_init__(self):
        if type(self.resume_ticks) is not int or self.resume_ticks <= 0:
            raise ValueError("resume_ticks must be a positive integer")
        if not math.isfinite(self.roll_rate_gain) or self.roll_rate_gain < 0:
            raise ValueError("roll_rate_gain must be finite and nonnegative")

    def ankle_offset(self, reference_tick, roll_rate, sole_heights):
        # The reference clock is zeroed at each reanchor and starts walking at tick 30.
        age = reference_tick - 30
        if not self.roll_rate_gain or age <= 0 or age >= 60:
            return None
        heights = np.asarray(sole_heights)
        if not math.isfinite(roll_rate) or heights.shape != (2,) or not np.isfinite(heights).all():
            return None
        fade = min(age / 8.0, (60 - age) / 12.0, 1.0)
        envelope = 0.5 - 0.5 * math.cos(math.pi * fade)
        # The lower sole receives the full correction. Unload the other ankle
        # gradually as its FK height rises by up to 30 mm. No contact labels.
        support = np.clip(1.0 - (heights - heights.min()) / 0.03, 0.0, 1.0)
        rate_error = math.copysign(max(abs(roll_rate) - 0.12, 0.0), roll_rate)
        bounded = np.clip(self.roll_rate_gain * rate_error, -0.03, 0.03) * envelope
        offset = np.zeros(22, np.float32)
        offset[[5, 11]] = support * bounded
        return offset


PROFILES = {
    "published": RestartProfile(30),
    "fast": RestartProfile(15),
    "moderate": RestartProfile(20),
    "damped_fast": RestartProfile(15, 0.024),
    "damped_moderate": RestartProfile(20, 0.024),
}
