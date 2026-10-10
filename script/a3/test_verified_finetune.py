"""CPU-only contracts for the opt-in trainer, using scalar losses/optimizers."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import sys

import torch

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from script.a3.finetune_trainer import FineTuneTrainer
from script.a3.resumable_sonic import ResumableTrainer
from script.a3.checkpoint_store import atomic_json
from script.a3.fullchain_support import ROOT
from script.a3.verified_finetune import VerifiedFineTuneTrainer, restore_adaptive_lr


def controller():
    trainer = object.__new__(VerifiedFineTuneTrainer)
    trainer.args = SimpleNamespace(learning_rate=2e-5, lr_scheduler_type="constant")
    trainer.config = {"desired_kl": 0.01}
    trainer.desired_kl = 0.01
    trainer.adaptive_lr_min = 1e-5
    trainer.adaptive_lr_max = 2e-4
    trainer.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=2e-5)
    trainer.lr_scheduler = None
    trainer.state = SimpleNamespace(global_step=2)
    return trainer


class VerifiedFineTuneTests(unittest.TestCase):
    def test_official_loss_and_stats_are_used(self):
        self.assertIs(VerifiedFineTuneTrainer._compute_loss, TRLAuxLossPPOTrainer._compute_loss)
        self.assertIs(VerifiedFineTuneTrainer._register_stats_buffer, TRLAuxLossPPOTrainer._register_stats_buffer)
        self.assertIs(VerifiedFineTuneTrainer._update_stats_buffer, TRLAuxLossPPOTrainer._update_stats_buffer)
        self.assertIs(VerifiedFineTuneTrainer._get_train_metric_stats, TRLAuxLossPPOTrainer._get_train_metric_stats)
        self.assertIs(FineTuneTrainer._compute_loss, TRLPPOTrainer._compute_loss)
        self.assertLess(VerifiedFineTuneTrainer.__mro__.index(ResumableTrainer),
                        VerifiedFineTuneTrainer.__mro__.index(TRLAuxLossPPOTrainer))

    def test_auxiliary_coefficients_and_gradient(self):
        trainer = controller()
        trainer.config = {"compute_aux_loss": True}
        trainer.compute_aux_loss = True
        trainer.compute_imgaug_bc_loss = False
        trainer.aux_loss_scale = 2.0
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        main = torch.tensor(3.0, requires_grad=True)
        auxiliary = torch.tensor(5.0, requires_grad=True)
        zero_weight = torch.tensor(9.0, requires_grad=True)
        trainer._compute_ppo_loss = lambda *_: {"ppo_loss": main.square()}
        result = trainer._compute_loss({"policy_results": {
            "aux_losses": {"a3_fast_g1_latent": auxiliary.square(), "unused": zero_weight.square()},
            "aux_loss_coef": {"a3_fast_g1_latent": 0.5, "unused": 0.0},
        }}, {})
        self.assertEqual(result["loss"].item(), 34.0)
        grads = torch.autograd.grad(result["loss"], (main, auxiliary, zero_weight))
        self.assertEqual([gradient.item() for gradient in grads], [6.0, 10.0, 0.0])

    def test_disabled_auxiliary_branch(self):
        trainer = controller()
        trainer.compute_aux_loss = False
        trainer.compute_imgaug_bc_loss = False
        trainer._compute_ppo_loss = lambda *_: {"ppo_loss": torch.tensor(9.0)}
        self.assertEqual(trainer._compute_loss({"policy_results": {}}, {})["loss"].item(), 9.0)

    def test_scheduler_does_not_reset_adaptive_rate(self):
        trainer = controller()
        scheduler = trainer.create_scheduler(20)
        trainer._adjust_learning_rate_based_on_kl(0.03, trainer.optimizer)
        rate = trainer.args.learning_rate
        parameter = trainer.optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.ones_like(parameter)
        trainer.optimizer.step()
        scheduler.step()
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], rate)
        self.assertEqual(scheduler.get_last_lr(), [rate])
        self.assertIs(trainer.create_scheduler(20), scheduler)

    def test_restore_legacy_discrepancy_uses_saved_controller(self):
        trainer = controller()
        trainer.create_scheduler(20)
        receipt = restore_adaptive_lr(trainer, {"args": SimpleNamespace(learning_rate=1e-5)})
        self.assertEqual(receipt["before_repair_optimizer_lrs"], [2e-5])
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(trainer.lr_scheduler.get_last_lr(), [1e-5])
        trainer._adjust_learning_rate_based_on_kl(0.01, trainer.optimizer)
        self.assertEqual(trainer.args.learning_rate, 1e-5)

    def test_continuous_and_resume_match_all_kl_branches(self):
        continuous = controller()
        continuous.create_scheduler(20)
        for kl in [0.03, 0.03, 0.002]:
            continuous._adjust_learning_rate_based_on_kl(kl, continuous.optimizer)
            for group in continuous.optimizer.param_groups:
                group["params"][0].grad = torch.ones_like(group["params"][0])
            continuous.optimizer.step()
            continuous.optimizer.zero_grad()
            continuous.lr_scheduler.step()
        saved_optimizer = copy.deepcopy(continuous.optimizer.state_dict())
        saved_scheduler = copy.deepcopy(continuous.lr_scheduler.state_dict())
        parameter_value = continuous.optimizer.param_groups[0]["params"][0].detach().clone()
        for kl in [0.0, 0.002, 0.01, 0.03]:
            resumed = controller()
            resumed.create_scheduler(20)
            resumed.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            resumed.lr_scheduler.load_state_dict(copy.deepcopy(saved_scheduler))
            restore_adaptive_lr(resumed, {"args": SimpleNamespace(learning_rate=continuous.args.learning_rate)})
            with torch.no_grad():
                resumed.optimizer.param_groups[0]["params"][0].copy_(parameter_value)
            expected = controller()
            expected.create_scheduler(20)
            expected.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            expected.lr_scheduler.load_state_dict(copy.deepcopy(saved_scheduler))
            expected.args.learning_rate = continuous.args.learning_rate
            with torch.no_grad():
                expected.optimizer.param_groups[0]["params"][0].copy_(parameter_value)
            for item in (expected, resumed):
                item._adjust_learning_rate_based_on_kl(kl, item.optimizer)
                parameter = item.optimizer.param_groups[0]["params"][0]
                parameter.grad = torch.ones_like(parameter)
                item.optimizer.step()
                item.lr_scheduler.step()
            self.assertEqual(expected.args.learning_rate, resumed.args.learning_rate)
            self.assertEqual(expected.lr_scheduler.get_last_lr(), resumed.lr_scheduler.get_last_lr())
            self.assertTrue(torch.equal(expected.optimizer.param_groups[0]["params"][0],
                                        resumed.optimizer.param_groups[0]["params"][0]))

    def test_nonfinite_and_out_of_bounds_restore_rejected(self):
        trainer = controller()
        trainer.create_scheduler(20)
        for rate in [float("nan"), float("inf"), 0.0, 1e-7, 0.1]:
            with self.assertRaises(ValueError):
                restore_adaptive_lr(trainer, {"args": SimpleNamespace(learning_rate=rate)})

    def test_unsupported_scheduler_rejected(self):
        trainer = controller()
        trainer.args.lr_scheduler_type = "linear"
        with self.assertRaises(ValueError):
            trainer.create_scheduler(20)

    def test_actual_optimizer_step_logged_and_mismatch_stops(self):
        trainer = controller()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            trainer._install_optimizer_audit(path)
            parameter = trainer.optimizer.param_groups[0]["params"][0]
            parameter.grad = torch.ones_like(parameter)
            trainer.optimizer.step()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["applied_group_lrs"], [2e-5])
            self.assertEqual(rows[0]["attempt_optimizer_step"], 1)
            trainer.optimizer.param_groups[0]["lr"] = 1e-5
            before = parameter.detach().clone()
            with self.assertRaises(ValueError):
                trainer.optimizer.step()
            self.assertTrue(torch.equal(before, parameter))
            self.assertEqual(len(path.read_text().splitlines()), 1)


if __name__ == "__main__":
    program = unittest.main(exit=False, verbosity=2)
    result = program.result
    atomic_json(ROOT / "data/experiments/a3_optimizer_20261010_E02/contracts.json",
                dict(tests_run=result.testsRun, successful=result.wasSuccessful(),
                     failures=[test.id() for test, _ in result.failures],
                     errors=[test.id() for test, _ in result.errors],
                     skipped=[test.id() for test, _ in result.skipped],
                     scope="CPU_SCALAR_LOSS_OPTIMIZER_AND_MRO_NO_POLICY_TRAINING"))
    sys.exit(0 if result.wasSuccessful() else 1)
