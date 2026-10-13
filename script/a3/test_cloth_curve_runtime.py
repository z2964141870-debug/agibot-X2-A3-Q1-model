"""Run only inside the guarded ZHY preparation stage, before real training."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from script.a3.cloth_curve_trainer import ClothCurveTrainer, CurveCheckpointCallback
from script.a3.resumable_sonic import DurableModelSaveCallback


class CurveRuntimeTests(unittest.TestCase):
    def test_sparse_checkpoint_paths_keep_all_declared_milestones(self):
        saved = []
        def record(callback, args, state, **kwargs):
            if callback.saved_step != state.global_step:
                saved.append((str(callback.save_dir), state.global_step))
                callback.saved_step = state.global_step
        callback = CurveCheckpointCallback(save_dir='/unused/curve', target_global_step=32)
        state = SimpleNamespace(global_step=0)
        control = SimpleNamespace(should_training_stop=False)
        with patch.object(DurableModelSaveCallback, '_save', record):
            callback.on_train_begin(None, state, control)
            for step in range(1, 33):
                state.global_step = step
                callback.on_step_end(None, state, control)
                self.assertEqual(control.should_training_stop, step == 32)
            callback.on_train_end(None, state, control)
        self.assertEqual(saved, [(f'/unused/curve/milestone_{s}', s) for s in (0, 8, 16, 32)])
        self.assertEqual(callback.save_dir, Path('/unused/curve'))

    def test_route_recording_keeps_native_counts_and_sampled_tokens(self):
        trainer = object.__new__(ClothCurveTrainer)
        trainer._collect_sample_tokens = True
        trainer._sample_token_records = []
        trainer._curve_routes = Counter()
        indices = torch.tensor([[1., 0., 0., 0.]] * 5 + [[1., 0., 0., 1.]] * 11).reshape(16, 1, 4)
        tokens = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 1, 64)
        module = SimpleNamespace(_last_full_latent_flat=tokens,
            parse_tokenizer_obs=lambda _: {'encoder_index':indices})
        trainer._capture_tokens(module, ({},), None)
        self.assertEqual(trainer._curve_routes, {'g1':5, 'a3_fast':11})
        self.assertTrue(torch.equal(trainer._sample_token_records[0].flatten(2), tokens))
        self.assertNotEqual(trainer._sample_token_records[0].data_ptr(), tokens.data_ptr())
        indices[0, 0, 1] = 1.
        with self.assertRaises(ValueError):
            trainer._capture_tokens(module, ({},), None)
