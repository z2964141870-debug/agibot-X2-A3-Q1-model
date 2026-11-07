"""CPU fixtures for causal-only fitting and preserved-data split ownership."""

import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from student_causal_fit import causal_records, fit_r2, sample_sources, source_groups
from student_fit import module_sha
from student_inputs import GEOMETRY_DIM


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(934, 29)

    def forward(self, goal, history):
        return self.net(torch.cat((goal, history), -1))


class FuturePoison(dict):
    def __getitem__(self, key):
        if key in ("full_goal", "prefix", "teacher_future_poses"):
            raise AssertionError("Future geometry is not a student input")
        return super().__getitem__(key)


def dataset(offset=0):
    rng = np.random.default_rng(11)
    values = rng.normal(size=(8, 10, 226)).astype(np.float32)
    values[..., GEOMETRY_DIM:] = [1., 0., .02]
    return FuturePoison(f0=values, history=rng.normal(size=(8, 930)).astype(np.float32),
        teacher_action=rng.normal(size=(8, 29)).astype(np.float32),
        clip_id=np.arange(8) % 2+offset, full_goal=object(), teacher_future_poses=object())


class StudentCausalFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_input_selection_never_reads_preserved_teacher_future_arrays(self):
        value = causal_records(dataset())
        self.assertEqual(set(value), {"f0", "history", "teacher_action", "clip_id"})

    def test_invalid_current_support_and_future_age_are_rejected(self):
        for change in (lambda a: a.__setitem__((0, -1, GEOMETRY_DIM), 0.),
                lambda a: a.__setitem__((0, -1, -1), -.02),
                lambda a: a.__setitem__((0, -1, -1), .22),
                lambda a: a.__setitem__((0, 0, GEOMETRY_DIM+1), .5),
                lambda a: a.__setitem__((0, -1, GEOMETRY_DIM+1), 1.)):
            value = dataset()
            change(value["f0"])
            with self.assertRaisesRegex(ValueError, "current support"):
                causal_records(value)
        value = dataset()
        value["f0"][0, 0] = 0.
        self.assertEqual(causal_records(value)["f0"].shape, (8, 10, 226))

    def test_sampling_is_source_balanced_without_replacing_original_row_ids(self):
        data = causal_records(dataset())
        data["clip_id"][:7] = 0
        data["clip_id"][7] = 1
        indices = sample_sources(source_groups(data), np.random.default_rng(3), 128, 1)
        self.assertEqual(np.bincount(data["clip_id"][indices]).tolist(), [64, 64])

    def test_fit_preserves_teacher_decoder_and_cpu_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = nn.Module()
            policy.cfg = SimpleNamespace(latent=4)
            policy.human_encoder = nn.Linear(2254, 4)
            policy.decoder = Decoder()
            native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
            before = module_sha(policy)
            train, val = dataset(), dataset(2)
            val["f0"][..., :GEOMETRY_DIM] += 100.
            settings = dict(learning_rate=.001, maximum_updates=3, batch_size=8,
                gradient_norm_limit=1., validation_every_updates=1, early_stop_stale_validations=6)
            result = fit_r2(native, dict(train=train, validation=val), settings, root,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            self.assertEqual(result["actual_updates"], 3)
            self.assertEqual(set(result["actual_adam_steps"]), {3})
            self.assertEqual(len(result["actual_adam_steps"]), 6)
            self.assertEqual(module_sha(policy), before)
            self.assertEqual(result["human_real_future_frames"], 0)
            statistics = json.loads((root/"current_statistics.json").read_text())
            self.assertLess(max(statistics["mean"]), 2.)
            self.assertEqual(statistics["fitted_split"], "train")
            trace = json.loads((root/"training_trace.json").read_text())
            self.assertEqual(result["selected_update"], min(trace, key=lambda row: row["validation_risk"])["update"])
            for name in ("selected.pt", "last.pt"):
                saved = torch.load(root/name, map_location="cpu", weights_only=True)
                self.assertEqual(saved["schema"], "q1_low_latency_f0_history_encoder_v1")
                self.assertTrue(all(t.device.type == "cpu" and torch.isfinite(t).all()
                    for t in saved["encoder"].values()))
            with self.assertRaises(FileExistsError):
                fit_r2(native, dict(train=train, validation=val), settings, root,
                    time.monotonic_ns()+60_000_000_000, {})

    def test_test_split_and_cross_split_source_ownership_fail_before_fitting(self):
        data = dataset()
        with tempfile.TemporaryDirectory() as temp:
            for values in (dict(train=data, validation=dataset(2), test=data),
                    dict(train=data, validation=copy.deepcopy(data))):
                with self.assertRaisesRegex(ValueError, "never test|cross train"):
                    fit_r2(None, values, {}, Path(temp), 0, {})


if __name__ == "__main__":
    unittest.main()
