"""Check label/behavior identities, causal startup coverage and bounded warm fit."""

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

from student_aggregate import validate_records
from student_aggregate_fit import balanced_indices, checked_splits, fit
from student_aggregate_worker import mixed_action
from student_causal_fit import causal_records
from student_fit import cpu_state, module_sha
from student_inputs import fit_current_statistics
from student_models import current_student
from test_student_causal_fit import Decoder, dataset


def physical_fixture(first=0):
    jobs = [dict(split="train", maximum_steps=4)]
    teacher = np.ones((4, 29), dtype=np.float32)
    student = np.zeros_like(teacher)
    behavior = .5*(teacher+student)
    actual = np.arange(5*79, dtype=np.float64).reshape(5, 79)
    ready = np.arange(4) >= first
    frames = np.arange(first, 4, 2)
    values = np.zeros((len(frames), 10, 226), dtype=np.float32)
    values[:, -1, -3:] = [1., 1., 0.]
    data = dict(f0=values, history=np.zeros((len(frames), 930), dtype=np.float32),
        teacher_action=teacher[frames], clip_id=np.zeros(len(frames), dtype=np.int64), frame=frames,
        actual_state=actual[frames], behavior_action=behavior[frames], student_action=student[frames])
    traces = {0:dict(actual=actual, canonical_action=behavior, teacher_action=teacher,
                    student_action=student, input_ready=ready)}
    return data, jobs, traces


class StudentAggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_state_teacher_queries_are_distinct_from_executed_mixture(self):
        data, jobs, traces = physical_fixture()
        result = validate_records(data, "train", jobs, traces)
        self.assertFalse(result["teacher_labels_are_executed_actions"])
        self.assertTrue(result["state_and_behavior_match"])
        bad = {key:value.copy() for key, value in data.items()}
        bad["teacher_action"][:] = bad["behavior_action"]
        with self.assertRaisesRegex(RuntimeError, "queried teacher label"):
            validate_records(bad, "train", jobs, traces)
        bad = {key:value.copy() for key, value in data.items()}
        bad["actual_state"][0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "physical trace"):
            validate_records(bad, "train", jobs, traces)

    def test_initial_history_is_included_only_when_causally_available(self):
        for first in (0, 2):
            data, jobs, traces = physical_fixture(first)
            value = validate_records(data, "train", jobs, traces)
            self.assertEqual(value["first_causally_available_sampled_frames"], {"0":first})
        data, jobs, traces = physical_fixture()
        data = {key:value[1:] for key, value in data.items()}
        with self.assertRaisesRegex(RuntimeError, "first causally available"):
            validate_records(data, "train", jobs, traces)

    def test_source_splits_remain_owned_and_test_never_enters_fit(self):
        original = dict(train=dataset(), validation=dataset(2))
        aggregate = copy.deepcopy(original)
        checked_splits(original, aggregate)
        aggregate["validation"]["clip_id"][0] = 0
        with self.assertRaisesRegex(ValueError, "ownership crosses"):
            checked_splits(original, aggregate)
        with self.assertRaisesRegex(ValueError, "train and validation only"):
            checked_splits(dict(original, test=dataset(4)), original)
        changed = copy.deepcopy(original)
        changed["train"]["clip_id"][:] = 0
        with self.assertRaisesRegex(ValueError, "every original source"):
            checked_splits(original, changed)

    def test_balancing_covers_both_origins_and_source_groups(self):
        old, new = causal_records(dataset()), causal_records(dataset())
        old["clip_id"][:7], old["clip_id"][7] = 0, 1
        choices = balanced_indices(old, new, np.random.default_rng(8), 128, 1)
        for origin, data in enumerate((old, new)):
            indices = [index for key, index in choices if key == origin]
            self.assertEqual(len(indices), 64)
            self.assertEqual(np.bincount(data["clip_id"][indices]).tolist(), [32, 32])

    def test_only_registered_mixture_and_finite_actions_are_accepted(self):
        teacher, student = np.ones((1, 29), dtype=np.float32), np.zeros((1, 29), dtype=np.float32)
        self.assertTrue(np.array_equal(mixed_action(teacher, student, .5), teacher*.5))
        for fraction in (0., .4, 1.):
            with self.assertRaises(ValueError):
                mixed_action(teacher, student, fraction)

    def test_warm_fit_keeps_statistics_teacher_decoder_and_actual_adam_counts(self):
        policy = nn.Module()
        policy.cfg = SimpleNamespace(latent=4)
        policy.human_encoder = nn.Linear(2254, 4)
        policy.decoder = Decoder()
        native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
        original = dict(train=dataset(), validation=dataset(2))
        aggregate = copy.deepcopy(original)
        statistics = fit_current_statistics(original["train"]["f0"], split="train")
        initial = current_student(native, statistics)
        payload = dict(encoder=cpu_state(initial.encoder), current_statistics=statistics,
            frozen_teacher_policy_sha256=module_sha(policy), decoder_sha256=module_sha(policy.decoder))
        before = module_sha(policy)
        settings = dict(seed=3, learning_rate=.0001, maximum_updates=3, batch_size=8,
            validation_every_updates=1, early_stop_stale_validations=6, gradient_norm_limit=1.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit(native, original, aggregate, payload, settings, root,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            self.assertEqual(result["actual_updates"], 3)
            self.assertEqual(result["actual_adam_steps"], [3]*6)
            self.assertEqual(module_sha(policy), before)
            self.assertEqual(result["current_statistics"], statistics)
            trace = json.loads((root/"training_trace.json").read_text())
            self.assertEqual(result["selected_update"], min(trace, key=lambda row:row["validation_risk"])["update"])
            saved = torch.load(root/"selected.pt", map_location="cpu", weights_only=True)
            self.assertTrue(all(value.device.type == "cpu" and torch.isfinite(value).all()
                                for value in saved["encoder"].values()))
            with self.assertRaises(FileExistsError):
                fit(native, original, aggregate, payload, settings, root,
                    time.monotonic_ns()+60_000_000_000, {})


if __name__ == "__main__":
    unittest.main()
