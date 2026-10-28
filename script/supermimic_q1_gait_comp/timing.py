"""First-hour cap is lifted only after the saved physical extension decision."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_gait"))
from clock import TrainingClock
from resources import save


class ConditionalClock(TrainingClock):
    def __init__(self, path, synchronize):
        self.cap = 3600.
        super().__init__(path, 0., 14400., synchronize)

    def flush(self):
        save(self.path, {"status": "ACTIVE" if self.active is not None else "PAUSED_FOR_NONTRAINING",
            "accumulated_training_seconds": self.total, "active_started_monotonic": self.active,
            "absolute_maximum_seconds": 14400., "current_cap_seconds": self.cap,
            "soft_stop_seconds": self.cap-100., "extension_authorized_by_physical_gate": self.cap > 3600,
            "definition": "Synchronized actual collection/GAE/RSL-RL updates; initialization/evaluation/checkpoint/media excluded."})
