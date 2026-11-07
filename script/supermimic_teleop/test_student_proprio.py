"""CPU contracts for causal robot-history residuals and paired fit selection."""

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

from student_fit import cpu_state, module_sha
from student_inputs import fit_current_statistics
from student_models import current_student
from student_proprio_fit import fit_pair
from student_proprio_models import ARMS, ProprioResidualStudent, residual_student, zero_adapter


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(934, 29)

    def forward(self, latent, history):
        return self.net(torch.cat((latent, history), -1))


def fixtures():
    rng = np.random.default_rng(83)
    def data(clip):
        goal = np.zeros((8, 10, 226), dtype=np.float32)
        goal[..., :223] = rng.normal(size=(8, 10, 223))
        goal[..., 223:225] = 1.
        return dict(f0=goal, history=rng.normal(size=(8, 930)).astype(np.float32),
            teacher_action=rng.normal(size=(8, 29)).astype(np.float32),
            clip_id=np.full(8, clip, dtype=np.int64))
    rows = [dict(relative=name, group_id=name, split=split)
        for name, split in (("old_train", "train"), ("old_val", "validation"),
            ("new_train", "train"), ("new_val", "validation"))]
    candidate = dict(selected_sources=rows, original_jobs=rows[:2], new_collection_jobs=rows[2:])
    origins = [dict(train=data(0), validation=data(1)) for _ in range(3)]
    torch.manual_seed(14)
    policy = nn.Module()
    policy.cfg = SimpleNamespace(latent=4)
    policy.human_encoder = nn.Linear(2254, 4)
    policy.decoder = Decoder()
    native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
    stats = fit_current_statistics(origins[0]["train"]["f0"], split="train")
    base = current_student(native, stats)
    payload = dict(encoder=cpu_state(base.encoder), current_statistics=stats,
        frozen_teacher_policy_sha256=module_sha(policy), decoder_sha256=module_sha(policy.decoder))
    return native, payload, origins, candidate


class ProprioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_residual_is_exact_original_for_both_arms_with_identical_capacity(self):
        native, payload, origins, _ = fixtures()
        base = current_student(native, payload["current_statistics"])
        base.encoder.load_state_dict(payload["encoder"])
        goal = torch.from_numpy(origins[0]["train"]["f0"])
        history = torch.from_numpy(origins[0]["train"]["history"])
        hashes, counts = [], []
        for arm in ARMS:
            torch.manual_seed(7)
            student = residual_student(native, payload, arm).train()
            self.assertTrue(torch.equal(student(history, goal), base(history, goal)))
            self.assertTrue(zero_adapter(cpu_state(student.adapter)))
            self.assertFalse(student.encoder.training or student.decoder.training)
            self.assertTrue(all(p.requires_grad == name.startswith("adapter.")
                for name, p in student.named_parameters()))
            hashes.append(module_sha(student.adapter))
            counts.append(sum(p.numel() for p in student.trainable_parameters()))
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(counts[0], counts[1])

    def test_only_causal_arm_conditions_latent_on_real_robot_history(self):
        native, payload, origins, _ = fixtures()
        goal = torch.from_numpy(origins[0]["train"]["f0"])
        history = torch.from_numpy(origins[0]["train"]["history"])
        for arm in ARMS:
            student = residual_student(native, payload, arm)
            with torch.no_grad():
                student.adapter[-1].weight.fill_(.01)
            same = torch.equal(student.latent(history, goal), student.latent(history+3., goal))
            self.assertEqual(same, arm == "zero_history")

    def test_real_adam_changes_only_adapter_and_retains_tanh_latent_bounds(self):
        native, payload, origins, _ = fixtures()
        student = residual_student(native, payload, "causal_history").train()
        before = [module_sha(m) for m in (native.policy, student.encoder, student.decoder, student.adapter)]
        history = torch.from_numpy(origins[0]["train"]["history"])
        goal = torch.from_numpy(origins[0]["train"]["f0"])
        optimizer = torch.optim.Adam(student.trainable_parameters(), lr=.001)
        for _ in range(3):
            optimizer.zero_grad()
            student(history, goal).square().mean().backward()
            optimizer.step()
        after = [module_sha(m) for m in (native.policy, student.encoder, student.decoder, student.adapter)]
        self.assertEqual(before[:3], after[:3])
        self.assertNotEqual(before[3], after[3])
        with torch.no_grad():
            student.adapter[-1].weight.fill_(1e4)
            latent = student.latent(history, goal)
        self.assertTrue(torch.isfinite(latent).all())
        self.assertLessEqual(float(latent.abs().max()), 1.)

    def test_native_future_window_and_wrong_shape_or_arm_are_rejected(self):
        native, payload, origins, _ = fixtures()
        student = residual_student(native, payload, "causal_history")
        goal = torch.from_numpy(origins[0]["train"]["f0"])
        history = torch.from_numpy(origins[0]["train"]["history"])
        for bad_h, bad_g in ((history[:, :-1], goal), (history, goal[:, :4]),
                (history, torch.zeros(8, 2254))):
            with self.assertRaisesRegex(ValueError, "paired current human"):
                student(bad_h, bad_g)
        with self.assertRaises(ValueError):
            ProprioResidualStudent(student, "future_history", 4)

    def test_three_update_pair_preserves_batch_exposure_and_validation_selection(self):
        native, payload, origins, candidate = fixtures()
        before = module_sha(native.policy)
        settings = dict(seed=24, maximum_updates=3, batch_size=8, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit_pair(native, origins, payload, candidate, settings, root,
                time.monotonic_ns()+120_000_000_000, dict(scope="synthetic_cpu_only"))
            a, b = result["arms"]
            self.assertEqual(result["new_optimizer_updates"], 6)
            self.assertEqual(a["batch_sha256"], b["batch_sha256"])
            self.assertEqual(a["source_sample_counts"], b["source_sample_counts"])
            self.assertEqual(a["initial_validation_risk"], b["initial_validation_risk"])
            for arm in ARMS:
                saved = torch.load(root/arm/"selected.pt", map_location="cpu", weights_only=True)
                last = torch.load(root/arm/"last.pt", map_location="cpu", weights_only=True)
                trace = json.loads((root/arm/"training_trace.json").read_text())
                eligible = [r for r in trace if r["fixed_anchor_non_regression"]]
                self.assertEqual(saved["selected_update"],
                    min(eligible, key=lambda r:r["validation_risk"])["update"])
                self.assertEqual(last["actual_adam_steps"], [3]*4)
                self.assertEqual(saved["human_real_future_frames"], 0)
                self.assertTrue(all(torch.equal(v, payload["encoder"][k])
                    for k, v in saved["base_encoder"].items()))
                self.assertTrue(all(v.device.type == "cpu" and torch.isfinite(v).all()
                    for v in saved["adapter"].values()))
            self.assertEqual(before, module_sha(native.policy))
            with self.assertRaises(FileExistsError):
                fit_pair(native, origins, payload, candidate, settings, root, 0, {})

    def test_pair_rejects_test_sources_before_loading_model(self):
        _, _, origins, candidate = fixtures()
        origins[0]["test"] = copy.deepcopy(origins[0]["train"])
        settings = dict(seed=24, maximum_updates=3, batch_size=8, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "two preserved origins"):
                fit_pair(None, origins, {}, candidate, settings, Path(directory), 0, {})


if __name__ == "__main__":
    unittest.main()
