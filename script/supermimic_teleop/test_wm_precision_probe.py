"""Check precision isolation, physical VJP parity and source-level regression gates."""

from dataclasses import dataclass
import time
import unittest

import torch

from wm_precision_math import anchor_vjp, fp32_shadow, native_reproduction, prediction_preservation
from wm_probe_math import StepCounter, gradient_probe


@dataclass(frozen=True)
class Config:
    bf16: bool = True
    frame: str = "root"


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.register_buffer("present", torch.arange(29) < 22)
        self.weight = torch.nn.Parameter(torch.tensor(.2), requires_grad=False)

    def step(self, state, action, context):
        projected = torch.where(self.present, action, 0.)
        return torch.cat((state[:, :240]+torch.nn.functional.pad(projected*self.weight, (0, 211)), projected), -1)


class PrecisionProbeTests(unittest.TestCase):
    def test_shadow_configuration_and_parameters_are_independent(self):
        native = ToyModel()
        shadow = fp32_shadow(native)
        self.assertTrue(native.cfg.bf16)
        self.assertFalse(shadow.cfg.bf16)
        self.assertEqual(native.cfg.frame, shadow.cfg.frame)
        self.assertNotEqual(native.weight.data_ptr(), shadow.weight.data_ptr())
        self.assertTrue(torch.equal(native.weight, shadow.weight))
        shadow.weight.fill_(.3)
        self.assertNotEqual(float(native.weight), float(shadow.weight))

    def test_non_native_or_trainable_shadow_is_rejected(self):
        native = ToyModel()
        native.cfg = Config(bf16=False)
        with self.assertRaises(ValueError):
            fp32_shadow(native)
        native.cfg = Config()
        native.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            fp32_shadow(native)

    def test_one_anchor_vjp_reproduces_saved_probe_without_repeating_fd(self):
        model = ToyModel()
        state, action, context = torch.zeros(1, 269), torch.ones(1, 29), torch.zeros(1, 62)
        counter = StepCounter(model, time.monotonic_ns()+30_000_000_000)
        saved = gradient_probe(counter, state, action, context, model.present, lambda x:x)
        before = counter.calls
        actual = anchor_vjp(counter, state, action, context, model.present, lambda x:x)
        self.assertEqual(counter.calls-before, 1)
        self.assertTrue(native_reproduction(actual, saved)["passed"])
        actual["present_joints"][0]["autograd"] += .001
        self.assertFalse(native_reproduction(actual, saved)["passed"])

    def summary(self, bad_source=False, missing=False):
        metrics = ("joint_position_rad", "body_position_m")
        groups = [dict(origin=o, split="validation", wm={m:1. for m in metrics}) for o in ("original", "mixed")]
        sources = [dict(origin=o, split="validation", clip_id=c,
            wm={m:(1.3 if bad_source and o == "mixed" and c == 1 else 1.) for m in metrics})
            for o in ("original", "mixed") for c in (0, 1)]
        return dict(groups=groups, per_source=sources[:-1] if missing else sources)

    def test_good_means_cannot_hide_one_source_regression(self):
        self.assertTrue(prediction_preservation(self.summary(), self.summary())["passed"])
        self.assertFalse(prediction_preservation(self.summary(), self.summary(bad_source=True))["passed"])

    def test_missing_or_duplicate_validation_source_is_rejected(self):
        with self.assertRaises(ValueError):
            prediction_preservation(self.summary(), self.summary(missing=True))
        duplicate = self.summary()
        duplicate["per_source"].append(duplicate["per_source"][0])
        with self.assertRaises(ValueError):
            prediction_preservation(self.summary(), duplicate)


if __name__ == "__main__":
    unittest.main()
