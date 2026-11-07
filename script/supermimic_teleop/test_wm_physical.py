"""Physical-loss causality, absent joints and exact historic-control exposure."""

import copy
import json
from collections import namedtuple
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from student_auxiliary_fit import fit_pair
from student_fit import module_sha
from test_student_latent import StudentLatentTests
from wm_physical_fit import fit_physical
from wm_physical_loss import PhysicalLoss, matched_settings
from wm_physical_worker import attach_states


Frame = namedtuple("Frame", "pos_xy quat")


class FrozenWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(bf16=False)
        self.register_buffer("joint_presence", torch.arange(29) < 22)
        self.register_buffer("feature_presence", torch.cat((self.joint_presence, torch.ones(239, dtype=torch.bool))))
        self.normalizer = nn.Module()
        self.normalizer.register_buffer("std", torch.ones(268))
        self.gain = nn.Parameter(torch.tensor(.2), requires_grad=False)

    def frame(self, state):
        return Frame(state[:, 58:60], torch.cat((torch.ones_like(state[:, :1]), state[:, :3]*0), -1))

    def step(self, state, action, context):
        return torch.cat((state[:, :29]+self.gain*action, state[:, 29:240], action), -1)


def canonicalize(physical, anchor):
    return torch.cat((physical, physical[:, :28]*0), -1)


def physical_loss():
    return PhysicalLoss(FrozenWM().eval(), canonicalize, time.monotonic_ns()+60_000_000_000)


class PhysicalDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_settings_reject_the_draft_learning_rate(self):
        keys = dict(maximum_updates=3000, batch_size=128, learning_rate=.0001,
            seed=2026110703, gradient_norm_limit=1., validation_every_updates=500)
        self.assertEqual(matched_settings(keys, {"fit":keys})["physical_weight"], .1)
        with self.assertRaisesRegex(ValueError, "control settings"):
            matched_settings(dict(keys, learning_rate=.0003), {"fit":keys})

    def test_frozen_target_and_physical_gradient_exclude_absent_actions_and_memory(self):
        physical = physical_loss()
        state, context = torch.zeros(2, 269), torch.zeros(2, 62)
        teacher = torch.ones(2, 29, requires_grad=True)
        target = physical.targets(state, teacher, context)
        self.assertFalse(target.requires_grad)
        candidate = torch.zeros(2, 29, requires_grad=True)
        loss = physical.loss(state, candidate, context, target)
        loss.backward()
        self.assertGreater(float(candidate.grad[:, :22].abs().max()), 0.)
        self.assertTrue(torch.equal(candidate.grad[:, 22:], torch.zeros(2, 7)))
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is None for p in physical.model.parameters()))
        other = candidate.detach().clone()
        other[:, 22:] = 999.
        self.assertEqual(float(loss), float(physical.loss(state, other, context, target)))
        altered = state.clone()
        altered[:, 240:] = 99.
        self.assertTrue(torch.equal(target, physical.targets(altered, teacher, context)))

    def test_reject_trainable_wm_attached_state_and_attached_targets(self):
        model = FrozenWM()
        model.gain.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen FP32"):
            PhysicalLoss(model, canonicalize, time.monotonic_ns()+60_000_000_000)
        physical = physical_loss()
        state, action, context = torch.zeros(1, 269), torch.zeros(1, 29), torch.zeros(1, 62)
        with self.assertRaisesRegex(ValueError, "saved detached state"):
            physical.targets(state.requires_grad_(True), action, context)
        state = state.detach()
        target = physical.targets(state, action, context).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached same-state"):
            physical.loss(state, action, context, target)

    def test_row_join_preserves_causal_inputs_and_rejects_state_misalignment(self):
        _, origins, _, _ = StudentLatentTests().fixture()
        values = copy.deepcopy(origins[0])
        originals = copy.deepcopy(values)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, data in values.items():
                data["actual_state"] = np.zeros((len(data["clip_id"]), 79), dtype=np.float32)
                np.savez(root/(split+"_data.npz"), **data)
            convert = lambda states:np.zeros((len(states), 269), dtype=np.float32)
            result, _ = attach_states(values, root, convert, np.zeros(62, dtype=np.float32))
            for split in result:
                for key in ("f0", "history", "teacher_action", "frame"):
                    self.assertTrue(np.array_equal(result[split][key], originals[split][key]))
            values["train"]["history"][0, 0] += 1.
            with self.assertRaisesRegex(RuntimeError, "exact control rows"):
                attach_states(values, root, convert, np.zeros(62, dtype=np.float32))

    def test_three_fixture_updates_match_control_exposure_without_retraining_it(self):
        native, origins, payload, settings = StudentLatentTests().fixture()
        before = module_sha(native.policy)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control_root = root/"control"
            control_root.mkdir()
            control = fit_pair(native, origins, payload, settings, control_root,
                time.monotonic_ns()+60_000_000_000, {"scope":"synthetic_cpu_fixture_not32k"})["arms"]["action_only"]
            path = control_root/"action_only/training_trace.json"
            raw = path.read_bytes()
            trace = json.loads(raw)
            for origin in origins:
                for data in origin.values():
                    n = len(data["clip_id"])
                    data.update(physical_state=np.zeros((n, 269), dtype=np.float32),
                                wm_context=np.zeros((n, 62), dtype=np.float32))
            config = {k:v for k, v in settings.items() if k != "auxiliary_weight"}
            config["physical_weight"] = .1
            output = root/"physical"
            output.mkdir()
            fit = fit_physical(native, physical_loss(), origins, payload, config, control, trace,
                output, time.monotonic_ns()+60_000_000_000, {"scope":"synthetic_cpu_fixture_not32k"})
            self.assertEqual(fit["actual_adam_steps"], [3]*6)
            self.assertEqual(fit["batch_identity_sha256"], control["batch_identity_sha256"])
            self.assertEqual(fit["source_sample_counts"], control["source_sample_counts"])
            self.assertFalse(fit["control_retrained"])
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(module_sha(native.policy), before)
            saved = torch.load(output/"selected.pt", map_location="cpu", weights_only=True)
            self.assertFalse(saved["world_model_deployed"])
            self.assertNotIn("teacher_physical", saved)
            self.assertEqual(saved["current_statistics"], payload["current_statistics"])
            bad = copy.deepcopy(trace)
            bad[0]["source_sample_counts"]["0:0"] = 1
            failed = root/"bad_control"
            failed.mkdir()
            with self.assertRaisesRegex(ValueError, "batch exposures"):
                fit_physical(native, physical_loss(), origins, payload, config, control, bad, failed,
                    time.monotonic_ns()+60_000_000_000, {})


if __name__ == "__main__":
    unittest.main()
