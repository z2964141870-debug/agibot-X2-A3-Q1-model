"""Teacher-label separation, exact historic-control exposure and frozen decoder."""

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

from student_auxiliary_fit import fit_pair
from student_fit import cpu_state, module_sha
from student_inputs import fit_current_statistics
from student_latent_fit import fit_latent, latent_records
from student_latent_worker import attach_goals
from student_models import current_student
from test_student_causal_fit import Decoder, dataset


class StudentLatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(8)
        policy = nn.Module()
        policy.cfg = SimpleNamespace(latent=4)
        policy.human_encoder = nn.Linear(2254, 4)
        policy.decoder = Decoder()
        policy.eval()
        native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
        origins = []
        for origin in range(2):
            splits = {}
            for split, offset in (("train", 0), ("validation", 2)):
                fixture = dataset(offset)
                data = {key:fixture[key] for key in ("f0", "history", "teacher_action", "clip_id")}
                n = len(data["clip_id"])
                full = np.zeros((n, 2254), dtype=np.float32)
                full[:, 2160:2194] = 1.
                full[:, 0] = np.arange(n)/10+origin*.01
                with torch.no_grad():
                    target = policy.decoder(policy.human_encoder(torch.tensor(full)),
                                            torch.tensor(data["history"])).numpy()
                data.update(full_goal=full, teacher_action=target, frame=np.arange(n, dtype=np.int64),
                    current_state=np.zeros((n, 50), dtype=np.float32),
                    next_state=np.ones((n, 50), dtype=np.float32),
                    executed_action=np.ones((n, 22), dtype=np.float32))
                splits[split] = data
            origins.append(splits)
        statistics = fit_current_statistics(origins[0]["train"]["f0"], split="train")
        student = current_student(native, statistics)
        payload = dict(encoder=cpu_state(student.encoder), current_statistics=statistics,
            frozen_teacher_policy_sha256=module_sha(policy), decoder_sha256=module_sha(policy.decoder))
        settings = dict(seed=3, learning_rate=.0001, maximum_updates=3, batch_size=8,
            validation_every_updates=1, gradient_norm_limit=1., auxiliary_weight=.1)
        return native, origins, payload, settings

    def test_future_teacher_is_a_label_and_reproduces_the_saved_action(self):
        native, origins, _, _ = self.fixture()
        data = origins[0]["train"]
        before = {key:data[key].copy() for key in ("f0", "history")}
        records, audit = latent_records(data, native)
        self.assertTrue(audit["future_goal_is_offline_label_only"])
        self.assertFalse(audit["labels_read_by_deployed_student"])
        for key in before:
            self.assertTrue(np.array_equal(records[key], before[key]))
        self.assertNotIn("full_goal", records)
        changed = copy.deepcopy(data)
        changed["full_goal"][:, 0] += 1.
        with torch.no_grad():
            changed["teacher_action"] = native.policy.decoder(
                native.policy.human_encoder(torch.tensor(changed["full_goal"])),
                torch.tensor(changed["history"])).numpy()
        other, _ = latent_records(changed, native)
        self.assertFalse(np.array_equal(records["teacher_latent"], other["teacher_latent"]))
        self.assertTrue(np.array_equal(records["f0"], other["f0"]))
        self.assertTrue(np.array_equal(records["history"], other["history"]))

    def test_wrong_same_state_action_and_incomplete_teacher_mask_are_rejected(self):
        native, origins, _, _ = self.fixture()
        for key, column in (("teacher_action", 0), ("full_goal", 2160)):
            data = copy.deepcopy(origins[0]["train"])
            data[key][0, column] += 1.
            with self.assertRaises(ValueError):
                latent_records(data, native)

    def test_preserved_goal_join_requires_every_original_source_frame(self):
        _, origins, _, _ = self.fixture()
        values = {split:{key:data[key].copy() for key in ("f0", "history", "teacher_action", "clip_id", "frame")}
                  for split, data in origins[0].items()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, data in origins[0].items():
                np.savez(root/(split+"_data.npz"), **data)
            result = attach_goals(copy.deepcopy(values), root)
            self.assertTrue(np.array_equal(result["train"]["full_goal"], origins[0]["train"]["full_goal"]))
            values["train"]["history"][0, 0] += 1.
            with self.assertRaisesRegex(RuntimeError, "control rows"):
                attach_goals(values, root)

    def test_three_real_fixture_updates_match_preserved_control_without_retraining_it(self):
        native, origins, payload, settings = self.fixture()
        before = module_sha(native.policy)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control_root = root/"control"
            control_root.mkdir()
            result = fit_pair(native, origins, payload, settings, control_root,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            control = result["arms"]["action_only"]
            control_path = control_root/"action_only/training_trace.json"
            raw_control = control_path.read_bytes()
            trace = json.loads(raw_control)
            labeled = [{split:latent_records(data, native)[0] for split, data in origin.items()}
                       for origin in origins]
            config = {key:value for key, value in settings.items() if key != "auxiliary_weight"}
            config["latent_weight"] = .1
            output = root/"latent"
            output.mkdir()
            fit = fit_latent(native, labeled, payload, config, control, trace, output,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            self.assertEqual(fit["actual_adam_steps"], [3]*6)
            self.assertEqual(fit["batch_identity_sha256"], control["batch_identity_sha256"])
            self.assertEqual(fit["source_sample_counts"], control["source_sample_counts"])
            self.assertFalse(fit["control_retrained"])
            self.assertEqual(control_path.read_bytes(), raw_control)
            self.assertEqual(module_sha(native.policy), before)
            saved = torch.load(output/"selected.pt", map_location="cpu", weights_only=True)
            self.assertFalse(saved["teacher_latent_label_deployed"])
            self.assertNotIn("teacher_latent", saved)
            self.assertEqual(saved["current_statistics"], payload["current_statistics"])
            saved_trace = json.loads((output/"training_trace.json").read_text())
            self.assertEqual(saved["selected_update"], min(saved_trace, key=lambda r:r["validation_risk"])["update"])
            bad = copy.deepcopy(trace)
            bad[0]["source_sample_counts"]["0:0"] = 1
            failed = root/"bad_control"
            failed.mkdir()
            with self.assertRaisesRegex(ValueError, "batch exposures"):
                fit_latent(native, labeled, payload, config, control, bad, failed,
                    time.monotonic_ns()+60_000_000_000, {})

    def test_aggregate_rows_without_saved_full_goal_use_explicit_offline_reconstruction(self):
        _, origins, _, _ = self.fixture()
        values = {split:{key:data[key].copy() for key in ("f0", "history", "teacher_action", "clip_id", "frame")}
                  for split, data in origins[1].items()}
        preserved = copy.deepcopy(values)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, data in origins[1].items():
                np.savez(root/(split+"_data.npz"), **{k:v for k, v in data.items() if k != "full_goal"})
            with self.assertRaisesRegex(RuntimeError, "Reconstruct omitted"):
                attach_goals(copy.deepcopy(values), root)
            calls = []
            def reconstruct(data):
                self.assertNotIn("full_goal", data)
                split = "train" if int(data["clip_id"].min()) < 2 else "validation"
                calls.append(data["history"].copy())
                return origins[1][split]["full_goal"].copy()
            result = attach_goals(copy.deepcopy(values), root, reconstruct=reconstruct)
            self.assertEqual(len(calls), 2)
            for split in result:
                for key in preserved[split]:
                    self.assertTrue(np.array_equal(result[split][key], preserved[split][key]))
                self.assertTrue(np.array_equal(result[split]["full_goal"], origins[1][split]["full_goal"]))
            with self.assertRaisesRegex(RuntimeError, "invalid rows"):
                attach_goals(copy.deepcopy(values), root, reconstruct=lambda data:np.zeros((1, 2254)))


if __name__ == "__main__":
    unittest.main()
