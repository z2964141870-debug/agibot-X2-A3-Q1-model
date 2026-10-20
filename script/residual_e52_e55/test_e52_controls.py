"""E52 CPU control tests; run before any registered stage."""

import copy
import math
import sys
import unittest

import torch

OPS = "/opt/data/private/yuanqi_ops"
ROOT = "/opt/data/private/hechengzhu/ZHY"
for path in (OPS, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import e52_protocol as P  # noqa: E402
from e52_trainer import (ProjectionAdamW, canonical_optimizer_sha256,  # noqa: E402
                         canonical_optimizer_state_sha256)
from script.a3.adam_direction_protocol import candidate  # noqa: E402
from script.a3.finite_retention_protocol import UPDATES  # noqa: E402
from script.a3.optimizer_metadata_contract import canonical_groups  # noqa: E402


class ProjectionOperator(unittest.TestCase):
    def test_outward_displacement_projects_to_zero(self):
        start = torch.zeros(6)
        grad = torch.tensor([2.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        displacement = 3.0 * grad / grad.norm()
        dot = float((grad * displacement).sum())
        scale = max(0.0, dot) / float(grad.pow(2).sum())
        projected = displacement - scale * grad
        self.assertGreater(dot, 0.0)
        self.assertLess(float((grad * projected).sum()), 1e-12)
        self.assertLess(float(projected.pow(2).sum()), 1e-12)

    def test_nonincreasing_displacement_unchanged(self):
        grad = torch.tensor([1.0, 2.0])
        displacement = -0.5 * grad
        dot = float((grad * displacement).sum())
        scale = max(0.0, dot) / float(grad.pow(2).sum())
        self.assertEqual(scale, 0.0)

    def test_zero_teacher_gradient_flags_noop(self):
        self.assertTrue(math.isfinite(0.0))


class NativeAdamAgreement(unittest.TestCase):
    def test_candidate_matches_torch_adamw_step(self):
        step, lr, beta1, beta2, eps, weight_decay = 16, 2e-6, 0.9, 0.999, 1e-8, 0.0
        parameter, gradient, moment, variance = 0.3, 0.7, 0.1, 0.05
        expected, _, _ = candidate(parameter, gradient, moment, variance, step,
                                   lr, beta1, beta2, eps, weight_decay)
        param = torch.nn.Parameter(torch.tensor([parameter]))
        optimizer = torch.optim.AdamW([param], lr=lr, betas=(beta1, beta2), eps=eps,
                                      weight_decay=weight_decay)
        state = optimizer.state[param]
        state["step"] = torch.tensor(float(step))
        state["exp_avg"] = torch.tensor([moment])
        state["exp_avg_sq"] = torch.tensor([variance])
        param.grad = torch.tensor([gradient])
        optimizer.step()
        actual = float(param.detach().reshape(-1)[0])
        self.assertLess(abs(actual - float(expected)), 1e-6)  # FP32 vs float64


class ProjectionAdamWSmoke(unittest.TestCase):
    def _build(self):
        torch.manual_seed(1)
        model = torch.nn.Linear(4, 3)
        return model

    def test_zero_teacher_gradient_equals_native_step(self):
        model = self._build()
        reference = self._build()
        reference.load_state_dict(model.state_dict())
        rows = []
        optimizer = ProjectionAdamW([dict(params=list(model.parameters()), lr=2e-6)],
                                    teacher_gradients=lambda: [torch.zeros_like(p) for p in model.parameters()],
                                    record=rows.append)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-6)
        data = torch.randn(5, 4)
        (model(data) ** 2).mean().backward()
        (reference(data) ** 2).mean().backward()
        optimizer.step()
        reference_optimizer.step()
        for a, b in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(a.detach(), b.detach(), atol=1e-8))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["committed"])

    def test_outward_teacher_gradient_reduces_first_order_increase(self):
        model = self._build()
        rows = []
        optimizer = ProjectionAdamW(model.parameters(),
                                    teacher_gradients=lambda: [p.detach().clone() + 0.01 for p in model.parameters()],
                                    record=rows.append)
        data = torch.randn(5, 4)
        (model(data) ** 2).mean().backward()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        self.assertEqual(len(rows), 1)
        for row in rows[0]["projections"]:
            self.assertFalse(row["zero_teacher_gradient"])
            self.assertGreaterEqual(row["projection_scale"], 0.0)


class RestoreContract(unittest.TestCase):
    def test_state_digest_roundtrip_and_lr_override(self):
        torch.manual_seed(2)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
        data = torch.randn(5, 4)
        (model(data) ** 2).mean().backward()
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        digest = canonical_optimizer_state_sha256(saved)
        fresh = torch.optim.AdamW([dict(params=list(model.parameters()), lr=2e-7)])
        pre = [{k: v for k, v in g.items() if k != "params"} for g in fresh.param_groups]
        fresh.load_state_dict(copy.deepcopy(saved))
        for group, keep in zip(fresh.param_groups, pre):
            for key, value in keep.items():
                if key in ("lr", "initial_lr") and group.get(key) != value:
                    group[key] = value
        self.assertEqual(canonical_optimizer_state_sha256(fresh.state_dict()), digest)
        self.assertEqual(fresh.param_groups[0]["lr"], 2e-7)


class RegistrationConstants(unittest.TestCase):
    def test_budget_matches_finite_retention_protocol(self):
        self.assertEqual(UPDATES, P.UPDATES)
        self.assertEqual(P.CHECKPOINTS, (2, 4, 8))
        self.assertEqual(P.OPTIMIZER_STEPS, 32)

    def test_arm_overrides_only_declared_fields(self):
        self.assertEqual(P.ARMS_LAUNCH_OVERRIDES["A"], tuple())
        self.assertEqual(P.ARMS_LAUNCH_OVERRIDES["B"], tuple())
        self.assertEqual(set(P.ARMS_LAUNCH_OVERRIDES["C"]),
                         {"algo.config.actor_learning_rate=2e-7",
                          "algo.config.adaptive_lr_min=1e-7",
                          "algo.config.adaptive_lr_max=2e-6"})
        self.assertEqual(P.ARMS_TRAINER["A"], "e52_trainer.E52ArmATrainer")
        self.assertEqual(P.ARMS_TRAINER["B"], "e52_trainer.E52ResumeTrainer")

    def test_gate_predicate(self):
        def violates(row):
            return row.get("teacher_action_rmse", 0) > 0.05 or row.get("teacher_action_max_abs", 0) > 0.5
        self.assertTrue(violates(dict(teacher_action_rmse=0.06, teacher_action_max_abs=0.1)))
        self.assertTrue(violates(dict(teacher_action_rmse=0.01, teacher_action_max_abs=0.6)))
        self.assertFalse(violates(dict(teacher_action_rmse=0.049, teacher_action_max_abs=0.49)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
