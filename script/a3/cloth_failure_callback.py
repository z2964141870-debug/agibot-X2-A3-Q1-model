"""E45 changes only durable checkpoint coverage, not the training optimizer."""

from script.a3.cloth_curve_trainer import CurveCheckpointCallback
from script.a3.cloth_failure_protocol import CHECKPOINTS, UPDATES


class FailureCheckpointCallback(CurveCheckpointCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in CHECKPOINTS:
            self._save(args, state, **kwargs)
        if state.global_step >= UPDATES:
            control.should_training_stop = True
        return control
