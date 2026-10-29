"""Reject misleading checkpoint mappings, counter claims and log summaries."""

from copy import deepcopy
import math
import unittest

import torch

from saved_ppo_chain import ACTOR, CRITIC, audit_checkpoint, audit_logs, counts, zero_mean_residual_reference


def fixture(iteration=0):
    actor = {name: torch.full(shape, .1 if name == "distribution.std_param" else 0.)
             for name, shape in ACTOR.items()}
    critic = {name: torch.zeros(shape) for name, shape in CRITIC.items()}
    actor_steps, critic_steps = counts(iteration)
    states = {}
    for index, value in enumerate([*actor.values(), *critic.values()]):
        step = 0 if index == 6 else actor_steps if index < 7 else critic_steps
        if step:
            states[index] = dict(step=torch.tensor(float(step)), exp_avg=torch.ones_like(value),
                                 exp_avg_sq=torch.ones_like(value))
    return dict(format="Q1_GAIT_GOAL01_PPO_V1", actor=actor, critic=critic,
        optimizer=dict(state=states, param_groups=[dict(params=list(range(13)), lr=5e-5,
            weight_decay=0, amsgrad=False, maximize=False, betas=(.9, .999), eps=1e-8)]),
        source_base={}, training_seconds=0., iteration=iteration, upper_reference_alpha=.5,
        user_accepted_checkpoint=None, maximum_training_seconds=1200., code_commit="fixture")


def log_fixture():
    return dict(step=1, transitions=98304, training_seconds=1., actor_optimizer_steps=0,
        critic_optimizer_steps=20, wm_optimizer_steps=0, learning_rate=5e-5,
        ppo_entropy=14 * (.5 * math.log(2 * math.pi * math.e) + math.log(.1)),
        alpha_sum=98304 * .375, residual_squared_sum=98304 * 14 * .015 ** 2,
        policy_update_kl_mean=0., policy_update_kl_p99=0., latent_mean_rms=0.,
        original_reward_sum=100., stance_bonus_sum=20., swing_bonus_sum=10.,
        stationary_bonus_sum=3., smooth_cost_sum=5., residual_cost_sum=2., reward_sum=126.,
        artificial_timeouts=0, hard_terminations=0, completed_episodes=1, ppo_value=.2, ppo_surrogate=0.)


class SavedPpoChainTests(unittest.TestCase):
    def test_optimizer_steps_respect_warmup_and_fixed_std_without_mutating(self):
        initial, trained = fixture(), fixture(58)
        trained["actor"]["head.net.4.bias"] += .01
        before = deepcopy(trained)
        rows = audit_checkpoint(trained, initial, "stage_01")
        self.assertEqual([row["adam_step"] for row in rows], [960] * 6 + [0] + [1160] * 6)
        self.assertEqual(rows[5]["changed_elements"], 14)
        for kind in ("actor", "critic"):
            for key, value in trained[kind].items():
                self.assertTrue(torch.equal(value, before[kind][key]))

    def test_mapping_and_missing_actor_update_are_rejected(self):
        for mutation in ("mapping", "missing", "counter", "std"):
            saved = fixture(113)
            if mutation == "mapping":
                saved["optimizer"]["param_groups"][0]["params"][0:2] = [1, 0]
            elif mutation == "missing":
                del saved["optimizer"]["state"][0]
            elif mutation == "counter":
                saved["optimizer"]["state"][0]["step"] += 1
            else:
                saved["actor"]["distribution.std_param"][0] = .2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_checkpoint(saved, fixture(), "stage_02")

    def test_nonfinite_and_negative_second_moment_are_rejected(self):
        for target in ("parameter", "moment"):
            saved = fixture(11)
            if target == "parameter":
                saved["actor"]["head.net.4.bias"][0] = float("nan")
            else:
                saved["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1
            with self.subTest(target=target), self.assertRaises(ValueError):
                audit_checkpoint(saved, fixture(), "bad")

    def test_log_components_and_binary_mixture_keep_noise_during_warmup(self):
        row = audit_logs([log_fixture()])[0]
        self.assertEqual(row["component_sum_discrepancy"], 0.)
        self.assertEqual(row["alpha0_fraction_under_binary_cap_contract"], .25)
        self.assertAlmostEqual(row["sampled_residual_rms_rad"], .015)
        for key, value in (("actor_optimizer_steps", 20), ("step", 2),
                ("policy_update_kl_mean", .01), ("reward_sum", 100.), ("learning_rate", .0001),
                ("residual_squared_sum", float("nan"))):
            record = log_fixture()
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_logs([record])

    def test_distribution_reference_is_bounded_and_has_small_noise_limit(self):
        self.assertEqual(zero_mean_residual_reference(0., .15), 0.)
        self.assertAlmostEqual(zero_mean_residual_reference(1e-6, .15), .15e-6, places=15)
        value = zero_mean_residual_reference(.1, .15)
        self.assertGreater(value, .014)
        self.assertLess(value, .015)
        for std, radius in ((-1., .15), (float("nan"), .15), (.1, 0.)):
            with self.assertRaises(ValueError):
                zero_mean_residual_reference(std, radius)


if __name__ == "__main__":
    unittest.main()
