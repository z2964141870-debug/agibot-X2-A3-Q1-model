"""Cumulative physical PPO budget within the explicit one-night authority."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_gait"))
from clock import TrainingClock
from resources import save
from overnight import MAXIMUM_TRAINING_SECONDS, OPTIMIZATION_DEADLINE, DELIVERY_DEADLINE


class ConditionalClock(TrainingClock):
    def __init__(self, path, synchronize):
        self.cap = MAXIMUM_TRAINING_SECONDS
        super().__init__(path, 0., self.cap, synchronize)

    def flush(self):
        save(self.path, {"status": "ACTIVE" if self.active is not None else "PAUSED_FOR_NONTRAINING",
            "accumulated_training_seconds": self.total, "active_started_monotonic": self.active,
            "absolute_maximum_seconds": MAXIMUM_TRAINING_SECONDS, "current_cap_seconds": self.cap,
            "soft_stop_seconds": self.cap-100., "overnight_authorization": "2026-10-29 user freedom until09:00 Beijing",
            "optimization_wall_deadline_unix": OPTIMIZATION_DEADLINE,
            "delivery_wall_deadline_unix": DELIVERY_DEADLINE,
            "definition": "Synchronized actual collection/GAE/RSL-RL updates; initialization/evaluation/checkpoint/media excluded."})
