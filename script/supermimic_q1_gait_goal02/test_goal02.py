"""CPU tests for reward geometry and passive observation of the real RSL PPO."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_comp")]
from coupling import coupling_factors
from ppo_diagnostics import PPODiagnostics, summary


class CouplingTests(unittest.TestCase):
    def setUp(self):
        self.rotation = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
        self.sole = torch.tensor([[0., .025], [.025, 0.]], dtype=torch.float64)
        self.stance = torch.tensor([[True, False], [False, True]])
        self.swing = ~self.stance

    def factors(self, rotation=None, sole=None, stance=None, swing=None, reference=None):
        return coupling_factors(self.rotation if rotation is None else rotation, self.rotation,
            self.sole if sole is None else sole, self.sole if reference is None else reference,
            self.stance if stance is None else stance, self.swing if swing is None else swing)

    def test_reference_is_full_score(self):
        result = self.factors()
        torch.testing.assert_close(result["factor"], torch.ones(2, dtype=torch.float64))
        self.assertTrue(bool(result["pair_active"].all()))

    def test_common_height_does_not_fix_wrong_foot_difference(self):
        equal = self.factors(sole=self.sole + .12)
        wrong = self.factors(sole=self.sole.flip(-1))
        torch.testing.assert_close(equal["pair_factor"], torch.ones(2, dtype=torch.float64))
        self.assertTrue(bool((wrong["pair_factor"] < .01).all()))

    def test_posture_penalized_without_height_change(self):
        angle = torch.tensor(.15, dtype=torch.float64)
        rotation = self.rotation.clone()
        rotation[:, 1, 1] = rotation[:, 2, 2] = angle.cos()
        rotation[:, 1, 2], rotation[:, 2, 1] = -angle.sin(), angle.sin()
        result = self.factors(rotation=rotation)
        self.assertTrue(bool((result["posture_factor"] < .38).all()))
        torch.testing.assert_close(result["pair_factor"], torch.ones(2, dtype=torch.float64))

    def test_world_yaw_invariance(self):
        angle = torch.tensor(.8, dtype=torch.float64)
        yaw = torch.tensor([[angle.cos(), -angle.sin(), 0.],
                            [angle.sin(), angle.cos(), 0.], [0., 0., 1.]], dtype=torch.float64)
        changed = self.factors(rotation=yaw @ self.rotation)
        torch.testing.assert_close(changed["factor"], self.factors()["factor"])

    def test_neutral_and_double_support_do_not_invent_swing(self):
        empty = torch.zeros_like(self.stance)
        for stance, swing in ((empty, empty), (~empty, empty)):
            result = self.factors(sole=self.sole.flip(-1), stance=stance, swing=swing)
            self.assertFalse(bool(result["pair_active"].any()))
            torch.testing.assert_close(result["pair_factor"], torch.ones(2, dtype=torch.float64))

    def test_low_reference_clearance_does_not_activate_pair(self):
        result = self.factors(reference=self.sole * .1)
        self.assertFalse(bool(result["pair_active"].any()))

    def test_bounded_and_reference_not_mutated(self):
        before = self.sole.clone()
        result = self.factors(sole=self.sole.flip(-1) + .01)
        self.assertTrue(bool(((result["factor"] >= 0) & (result["factor"] <= 1)).all()))
        torch.testing.assert_close(before, self.sole, atol=0, rtol=0)

    def test_gradient_matches_finite_difference(self):
        sole = (self.sole + torch.tensor([[.003, -.004], [-.002, .003]])).requires_grad_()
        objective = self.factors(sole=sole)["factor"].sum()
        gradient, = torch.autograd.grad(objective, sole)
        numerical = torch.empty_like(gradient)
        for row in range(2):
            for foot in range(2):
                delta = torch.zeros_like(sole)
                delta[row, foot] = 1e-6
                numerical[row, foot] = ((self.factors(sole=sole.detach() + delta)["factor"].sum()
                    - self.factors(sole=sole.detach() - delta)["factor"].sum()) / 2e-6)
        torch.testing.assert_close(gradient, numerical, atol=1e-7, rtol=1e-6)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0.)

    def test_horizon_batch_shape_and_invalid_shape(self):
        result = coupling_factors(self.rotation.repeat(3, 1, 1, 1),
            self.rotation.repeat(3, 1, 1, 1), self.sole.repeat(3, 1, 1),
            self.sole.repeat(3, 1, 1), self.stance.repeat(3, 1, 1), self.swing.repeat(3, 1, 1))
        self.assertEqual(result["factor"].shape, (3, 2))
        with self.assertRaises(ValueError):
            self.factors(sole=self.sole[:, :1])
        with self.assertRaises(ValueError):
            self.factors(stance=self.stance.float())


class ObserverTests(unittest.TestCase):
    def make(self):
        from tensordict import TensorDict
        from gait_adapter import Actor, make_ppo
        torch.set_num_threads(1)
        torch.manual_seed(9182)
        obs = TensorDict(dict(features=torch.randn(4, 1420), privileged=torch.randn(4, 96)),
                         batch_size=[4], device="cpu")
        actor = Actor()
        algorithm = make_ppo(actor, obs, 4)
        algorithm.optimizer.step = lambda *args, **kwargs: None
        adapter = SimpleNamespace(diagnostic_steps=[],
            foot_reference=SimpleNamespace(latest_diagnostics=None))
        return algorithm, adapter, obs

    def run_update(self, output=None):
        algorithm, adapter, obs = self.make()
        if output is not None:
            PPODiagnostics(algorithm, adapter, output).attach()
        before = {k: v.clone() for k, v in algorithm.actor.state_dict().items()}
        with torch.no_grad():
            for step in range(24):
                algorithm.act(obs)
                if output is not None:
                    adapter.foot_reference.latest_diagnostics = dict(
                        frames=torch.full((4,), step),
                        swing=torch.tensor([[True, False]]).repeat(4, 1),
                        stance=torch.tensor([[False, True]]).repeat(4, 1),
                        reference_sole=torch.tensor([[.025, 0.]]).repeat(4, 1),
                        sole=torch.tensor([[.015, 0.], [.001, 0.], [.015, 0.], [.001, 0.]]),
                        last_solve_force=torch.tensor([[0., 20.]]).repeat(4, 1),
                        residual_rms=torch.zeros(4), posture_factor=torch.ones(4), pair_factor=torch.ones(4))
                algorithm.process_env_step(obs, torch.arange(4).float() / 20,
                                            torch.zeros(4, dtype=torch.bool), {})
            algorithm.compute_returns(obs)
        advantages = algorithm.storage.advantages.clone()
        original_clip = torch.nn.utils.clip_grad_norm_
        losses = algorithm.update()
        self.assertIs(torch.nn.utils.clip_grad_norm_, original_clip)
        for key, value in algorithm.actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        gradients = [p.grad.clone() if p.grad is not None else None
                     for p in (*algorithm.actor.parameters(), *algorithm.critic.parameters())]
        return losses, gradients, advantages, torch.get_rng_state().clone()

    def test_real_rsl_observer_preserves_loss_gradients_parameters_rng(self):
        raw = self.run_update()
        with tempfile.TemporaryDirectory() as output:
            observed = self.run_update(output)
            self.assertEqual(raw[0], observed[0])
            for before, after in zip(raw[1], observed[1]):
                if before is None:
                    self.assertIsNone(after)
                else:
                    torch.testing.assert_close(before, after, atol=0, rtol=0)
            torch.testing.assert_close(raw[2], observed[2], atol=0, rtol=0)
            self.assertTrue(torch.equal(raw[3], observed[3]))
            row = json.loads((Path(output) / "ppo_update_diagnostics.jsonl").read_text())
            self.assertEqual(len(row["minibatches"]), 20)
            self.assertEqual(len(row["gradient_clips"]), 40)
            phase = json.loads((Path(output) / "ppo_phase_advantages.jsonl").read_text())
            self.assertEqual(phase["transition_count"], 96)
            self.assertEqual(phase["force_pose_time_offset_seconds"], -.005)
            self.assertEqual(len(phase["samples"]), 32)
            self.assertIn("last_solve_force", phase["samples"][0])

    def test_empty_condition_preserves_missing_values(self):
        self.assertEqual(summary(torch.ones(3), torch.zeros(3, dtype=torch.bool)),
                         dict(count=0, mean=None, positive_fraction=None))

    def test_missing_transition_rejected_before_storage_changes(self):
        algorithm, adapter, _ = self.make()
        with tempfile.TemporaryDirectory() as output:
            PPODiagnostics(algorithm, adapter, output).attach()
            with self.assertRaises(AssertionError):
                algorithm.process_env_step(None, None, None, {})
            self.assertEqual(algorithm.storage.step, 0)

    def test_clip_function_restored_on_update_exception(self):
        algorithm, adapter, _ = self.make()
        def fail():
            raise RuntimeError("synthetic failure")
        algorithm.update = fail
        original = torch.nn.utils.clip_grad_norm_
        with tempfile.TemporaryDirectory() as output:
            PPODiagnostics(algorithm, adapter, output).attach()
            with self.assertRaises(RuntimeError):
                algorithm.update()
        self.assertIs(torch.nn.utils.clip_grad_norm_, original)


if __name__ == "__main__":
    unittest.main()
