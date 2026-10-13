"""E45 checkpoint behavior; run inside guarded ZHY preparation only."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.a3.cloth_failure_callback import FailureCheckpointCallback
from script.a3.resumable_sonic import DurableModelSaveCallback


class FailureRuntimeTests(unittest.TestCase):
    def test_early_and_late_checkpoints_survive_and_stop_only_at_budget(self):
        saved = []
        def record(callback, args, state, **kwargs):
            if callback.saved_step != state.global_step:
                saved.append((str(callback.save_dir), state.global_step))
                callback.saved_step = state.global_step
        callback = FailureCheckpointCallback(save_dir='/unused/failure', target_global_step=32)
        state = SimpleNamespace(global_step=0)
        control = SimpleNamespace(should_training_stop=False)
        with patch.object(DurableModelSaveCallback, '_save', record):
            callback.on_train_begin(None, state, control)
            for step in range(1, 33):
                state.global_step = step
                callback.on_step_end(None, state, control)
                self.assertEqual(control.should_training_stop, step == 32)
            callback.on_train_end(None, state, control)
        self.assertEqual(saved, [(f'/unused/failure/milestone_{s}', s) for s in (0, 2, 4, 8, 16, 32)])
        self.assertEqual(callback.save_dir, Path('/unused/failure'))

    def test_budget_cannot_be_extended(self):
        with self.assertRaises(ValueError):
            FailureCheckpointCallback(save_dir='/unused/failure', target_global_step=64)
