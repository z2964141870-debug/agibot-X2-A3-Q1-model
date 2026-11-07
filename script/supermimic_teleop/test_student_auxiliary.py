"""Check recorded-behavior labels, episode boundaries and matched ablation exposure."""

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

from student_auxiliary_data import target_indices, transition_records
from student_auxiliary_fit import TransitionHead, fit_pair
from student_fit import cpu_state, module_sha
from student_inputs import fit_current_statistics
from student_models import current_student
from test_student_aggregate import physical_fixture
from test_student_causal_fit import Decoder, dataset


class StudentAuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        data, _, traces = physical_fixture()
        present = np.arange(29) < 22
        obs = np.arange(5*93, dtype=np.float32).reshape(5, 93)/100
        for i, frame in enumerate(data["frame"]):
            data["history"][i] = np.tile(obs[frame], 10)
        return data, traces, present, obs

    def test_morphology_mask_retains_gravity_and_excludes_previous_action(self):
        mask = np.arange(29) < 22
        indices = target_indices(mask)
        self.assertEqual(len(indices), 50)
        self.assertEqual(indices[-6:].tolist(), list(range(58, 64)))
        self.assertFalse(np.isin([22, 51, 64, 92], indices).any())
        with self.assertRaises(ValueError):
            target_indices(np.ones(29, dtype=bool))

    def test_next_state_is_frame_plus_one_under_executed_behavior(self):
        data, traces, present, obs = self.fixture()
        values, audit = transition_records(data, traces, {0:None}, lambda _:obs, present, aggregate=True)
        indices = target_indices(present)
        self.assertTrue(np.array_equal(values["next_state"], obs[data["frame"]+1][:, indices]))
        self.assertTrue(np.array_equal(values["executed_action"], data["behavior_action"][:, present]))
        self.assertFalse(np.array_equal(values["executed_action"], data["teacher_action"][:, present]))
        self.assertTrue(audit["next_state_is_training_label_only"])
        for key in ("f0", "history", "teacher_action"):
            self.assertTrue(np.array_equal(values[key], data[key]))

    def test_behavior_or_current_state_mismatch_is_rejected(self):
        data, traces, present, obs = self.fixture()
        for key in ("behavior_action", "actual_state"):
            bad = copy.deepcopy(data)
            bad[key][0, 0] += 1
            with self.assertRaises(ValueError):
                transition_records(bad, traces, {0:None}, lambda _:obs, present, aggregate=True)
        altered = obs.copy()
        altered[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "newest observation"):
            transition_records(data, traces, {0:None}, lambda _:altered, present, aggregate=True)

    def test_terminal_transition_is_not_a_cross_episode_target(self):
        data, traces, present, obs = self.fixture()
        result, audit = transition_records(data, traces, {0:3}, lambda _:obs, present, aggregate=True)
        self.assertEqual(result["frame"].tolist(), [0])
        self.assertEqual(audit["terminal_rows_dropped"], [dict(clip_id=0, frame=2)])
        data["frame"][0] = len(traces[0]["canonical_action"])
        with self.assertRaisesRegex(ValueError, "source boundary"):
            transition_records(data, traces, {0:None}, lambda _:obs, present, aggregate=True)

    def test_auxiliary_has_encoder_gradient_and_conditions_on_real_action(self):
        torch.manual_seed(6)
        head = TransitionHead(4)
        latent = torch.zeros(2, 4, requires_grad=True)
        current = torch.zeros(2, 50)
        executed = torch.ones(2, 22)
        prediction = head(latent, current, executed)
        self.assertFalse(torch.equal(prediction, head(latent, current, executed*0)))
        prediction.square().mean().backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0.)

    def test_paired_fit_keeps_source_batches_teacher_decoder_and_deployment_contract(self):
        policy = nn.Module()
        policy.cfg = SimpleNamespace(latent=4)
        policy.human_encoder = nn.Linear(2254, 4)
        policy.decoder = Decoder()
        native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
        original = dict(train=dataset(), validation=dataset(2))
        for values in original.values():
            n = len(values["clip_id"])
            values.update(current_state=np.zeros((n, 50), dtype=np.float32),
                next_state=np.ones((n, 50), dtype=np.float32),
                executed_action=np.ones((n, 22), dtype=np.float32))
        mixed = copy.deepcopy(original)
        stats = fit_current_statistics(original["train"]["f0"], split="train")
        student = current_student(native, stats)
        payload = dict(encoder=cpu_state(student.encoder), current_statistics=stats,
            frozen_teacher_policy_sha256=module_sha(policy), decoder_sha256=module_sha(policy.decoder))
        before = module_sha(policy)
        settings = dict(seed=3, learning_rate=.0001, maximum_updates=3, batch_size=8,
            validation_every_updates=1, gradient_norm_limit=1., auxiliary_weight=.1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit_pair(native, [original, mixed], payload, settings, root,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            self.assertEqual(module_sha(policy), before)
            self.assertEqual(result["actual_optimizer_updates"], 6)
            arms = result["arms"]
            self.assertEqual(arms["action_only"]["source_sample_counts"], arms["next_state"]["source_sample_counts"])
            self.assertEqual(arms["action_only"]["batch_identity_sha256"], arms["next_state"]["batch_identity_sha256"])
            self.assertEqual(arms["action_only"]["actual_adam_steps"], [3]*6)
            self.assertEqual(arms["next_state"]["auxiliary_adam_steps"], [3]*4)
            for arm in arms:
                saved = torch.load(root/arm/"selected.pt", map_location="cpu", weights_only=True)
                self.assertNotIn("auxiliary_head", saved)
                self.assertEqual(saved["current_statistics"], stats)
                self.assertFalse(saved["auxiliary_head_deployed"])
                trace = json.loads((root/arm/"training_trace.json").read_text())
                self.assertEqual(saved["selected_update"], min(trace, key=lambda r:r["validation_risk"])["update"])
            with self.assertRaises(FileExistsError):
                fit_pair(native, [original, mixed], payload, settings, root,
                    time.monotonic_ns()+60_000_000_000, {})


if __name__ == "__main__":
    unittest.main()
