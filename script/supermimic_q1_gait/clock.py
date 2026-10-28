"""Cumulative training time with explicit pauses for evaluation and persistence."""

from __future__ import annotations

import json
from pathlib import Path
import time


class TrainingClock:
    def __init__(self, path, inherited_seconds=0., maximum_seconds=14400., synchronize=lambda: None):
        self.path = Path(path)
        self.inherited = float(inherited_seconds)
        self.total = self.inherited
        self.maximum = float(maximum_seconds)
        self.active = None
        self.synchronize = synchronize
        self.flush()

    def flush(self):
        value = {"status": "ACTIVE" if self.active is not None else "PAUSED_FOR_NONTRAINING",
                 "accumulated_training_seconds": self.total,
                 "inherited_training_seconds": self.inherited,
                 "new_training_seconds": self.total - self.inherited,
                 "active_started_monotonic": self.active,
                 "absolute_maximum_seconds": self.maximum,
                 "current_cap_seconds": self.maximum,
                 "soft_stop_seconds": self.maximum - 100.,
                 "definition": "Synchronized collection, model/actor/critic optimization and update diagnostics; evaluation, initialization, checkpoint and media excluded."}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def begin(self):
        if self.active is not None:
            raise RuntimeError("Training clock already active")
        self.synchronize()
        self.active = time.monotonic()
        self.flush()

    def end(self):
        if self.active is not None:
            self.synchronize()
            self.total += time.monotonic() - self.active
            self.active = None
            self.flush()
