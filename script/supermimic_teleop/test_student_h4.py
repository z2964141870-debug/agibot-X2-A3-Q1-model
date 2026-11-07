"""Focused CPU contracts for H4 visibility, source ownership and selected checkpoints."""

import copy
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from student_fit import batch_goal, cpu_state, module_sha
from student_h4_fit import H4_COLUMN, fit, sample_indices, source_groups


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(934, 29)

    def forward(self, goal, history):
        return self.net(torch.cat((goal, history), -1))


def fixture(offset=0):
    rng = np.random.default_rng(8)
    return dict(full_goal=rng.normal(size=(8, 2254)).astype(np.float32),
        history=rng.normal(size=(8, 930)).astype(np.float32),
        teacher_action=rng.normal(size=(8, 29)).astype(np.float32),
        prefix=np.tile((10, 5, 4, 2, 1), (8, 1)), clip_id=np.arange(8) % 2+offset)


def native_fixture():
    torch.manual_seed(4)
    policy = nn.Module()
    policy.human_encoder = nn.Sequential(nn.Linear(2254, 4), nn.Tanh())
    policy.decoder = Decoder()
    native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
    payload = dict(schema="q1_low_latency_student_encoder_v1", interface="human_prefix_adapter",
        selection_split="validation", test_used_for_selection=False, selected_update=5000,
        encoder=cpu_state(policy.human_encoder), frozen_teacher_policy_sha256=module_sha(policy),
        decoder_sha256=module_sha(policy.decoder))
    return native, payload


class StudentH4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_requested_h4_never_reads_fifth_or_later_pose_and_heading(self):
        data = fixture()
        altered = copy.deepcopy(data)
        altered["full_goal"][:, 4*216:2160] = 1e6
        altered["full_goal"][:, -60+4*6:] = -1e6
        indices = np.arange(8)
        columns = np.full(8, H4_COLUMN)
        self.assertTrue(torch.equal(batch_goal(data, indices, columns, "cpu")[0],
                                    batch_goal(altered, indices, columns, "cpu")[0]))

    def test_missing_first_target_is_excluded_without_filling_future_holes(self):
        data = fixture()
        data["prefix"][0, H4_COLUMN] = 0
        data["prefix"][2, H4_COLUMN] = 2
        groups = source_groups(data)
        self.assertNotIn(0, groups[0])
        goal, _, _, counts = batch_goal(data, np.array((0, 2)), np.full(2, H4_COLUMN), "cpu")
        self.assertEqual(counts.tolist(), [2])
        self.assertEqual(goal.shape, (1, 2254))
        self.assertEqual(torch.count_nonzero(goal[:, 2*216:2160]).item(), 0)
        data["prefix"][:, H4_COLUMN] = 5
        with self.assertRaisesRegex(ValueError, "H4 prefix"):
            source_groups(data)

    def test_h4_sampling_preserves_ids_and_balances_unequal_source_sizes(self):
        data = fixture()
        data["clip_id"][:7], data["clip_id"][7] = 0, 1
        indices = sample_indices(source_groups(data), np.random.default_rng(2), 128, 0)
        self.assertEqual(np.bincount(data["clip_id"][indices]).tolist(), [64, 64])

    def test_test_split_or_cross_source_ownership_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            for data in (dict(train=fixture(), validation=fixture(2), test=fixture(4)),
                         dict(train=fixture(), validation=fixture())):
                with self.assertRaisesRegex(ValueError, "never test|cross train"):
                    fit(None, data, {}, {}, Path(temp), 0, {})

    def test_update0_selection_preserves_original_and_last_is_only_diagnostics(self):
        native, payload = native_fixture()
        train, val = fixture(), fixture(2)
        for data in (train, val):
            goal, history, _, _ = batch_goal(data, np.arange(8), np.full(8, H4_COLUMN), "cpu")
            with torch.no_grad():
                data["teacher_action"] = native.policy.decoder(native.policy.human_encoder(goal), history).numpy()
        train["teacher_action"] += .2
        before = module_sha(native.policy)
        settings = dict(seed=1, learning_rate=.001, maximum_updates=3, batch_size=8,
            gradient_norm_limit=1., validation_every_updates=1, early_stop_stale_validations=6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = fit(native, dict(train=train, validation=val), payload, settings, root,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            selected = torch.load(root/"selected.pt", map_location="cpu", weights_only=True)
            last = torch.load(root/"last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(result["selected_update"], 0)
            self.assertEqual(result["actual_updates"], 3)
            self.assertEqual(set(result["actual_adam_steps"]), {3})
            self.assertEqual(result["allowed_request_horizons"], [4])
            self.assertEqual(result["requested_h4_samples"], 24)
            self.assertEqual(result["visible_prefix_counts"], [0, 0, 0, 0, 24])
            self.assertEqual(module_sha(native.policy), before)
            self.assertTrue(all(torch.equal(v, payload["encoder"][k]) for k, v in selected["encoder"].items()))
            self.assertTrue(any(not torch.equal(v, selected["encoder"][k]) for k, v in last["encoder"].items()))
            self.assertFalse(last["used_for_selection"])
            self.assertTrue(all(v.device.type == "cpu" and torch.isfinite(v).all()
                                for v in selected["encoder"].values()))
            with self.assertRaises(FileExistsError):
                fit(native, dict(train=train, validation=val), payload, settings, root, 0, {})

    def test_wrong_warm_start_checkpoint_is_rejected(self):
        native, payload = native_fixture()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "update5000"):
                fit(native, dict(train=fixture(), validation=fixture(2)), dict(payload, selected_update=8000),
                    {}, Path(temp), 0, {})


if __name__ == "__main__":
    unittest.main()
