"""Stop only at complete reporting boundaries for a declared training budget."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CollectionBudget:
    maximum_steps: int
    training_seconds: float | None = None

    def __post_init__(self):
        if self.maximum_steps < 300:
            raise ValueError("At least 300 collection rounds are required")
        if self.training_seconds is not None and (
                not math.isfinite(self.training_seconds) or self.training_seconds <= 0
                or self.maximum_steps % 25):
            raise ValueError("Timed collection needs positive seconds and a reporting-aligned step cap")

    def finished(self, step, elapsed):
        if self.training_seconds is None:
            return step >= self.maximum_steps
        if step % 25 == 0 and elapsed >= self.training_seconds:
            return True
        if step >= self.maximum_steps:
            raise RuntimeError("Collection step cap reached before the requested training duration")
        return False
