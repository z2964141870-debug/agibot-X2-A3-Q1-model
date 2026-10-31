"""Check exposure accounting, warm-up boundaries, and pinned-source protection."""

from pathlib import Path
import tempfile
import unittest

from scale_plan import make_plan, official_defaults, volumes


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "data/references/morphmimic_review_20261029/source/src/morphmimic/train/trainer.py"


class ScalePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defaults = official_defaults(TRAINER)

    def test_actual_smoke_warmup_left_only100_policy_updates(self):
        cfg = dict(self.defaults, iterations=600, num_envs=64,
                   wm_batch=128, imag_batch=128, kp_imag_batch=128)
        result = volumes(cfg)
        self.assertEqual(result["policy_optimizer_updates"], 100)
        self.assertEqual(result["collection_transitions"], 96000)

    def test_paper_experiment_is_not320k_default(self):
        plan = make_plan(self.defaults, 4096)
        self.assertEqual(plan["original"]["iterations"], 40000)
        self.assertEqual(plan["upstream_traincfg_default_iterations"], 320000)
        self.assertEqual(plan["volumes_per_robot"]["original"]["collection_transitions"], 409600000)

    def test_quarter_scale_matches_samples_but_not_adam_update_count(self):
        plan = make_plan(self.defaults, 1024)
        original, scaled = plan["volumes_per_robot"].values()
        self.assertEqual(plan["scaled"]["iterations"], 160000)
        self.assertEqual(plan["scaled"]["wm_only_iters"], 2000)
        self.assertEqual(plan["scaled"]["wm_batch"], 512)
        self.assertEqual(plan["scaled"]["imag_batch"], 1024)
        for key in plan["exactly_matched_counted_exposures"]:
            self.assertEqual(original[key], scaled[key], key)
        self.assertEqual(scaled["policy_optimizer_updates"], 158000)
        self.assertEqual(original["policy_optimizer_updates"], 39500)
        self.assertEqual(scaled["replay_transition_capacity"] * 4,
                         original["replay_transition_capacity"])

    def test_eighth_scale_has_same_postwarmup_exposure(self):
        plan = make_plan(self.defaults, 512)
        self.assertEqual(plan["scaled"]["iterations"], 320000)
        self.assertEqual(plan["scaled"]["wm_only_iters"], 4000)
        self.assertEqual(plan["scaled"]["lr_decay_iters"], 2000000)
        self.assertEqual(plan["scaled"]["horizon"], 32)

    def test_reject_nonintegral_scaling_and_invalid_env_counts(self):
        for envs in (0, -1, 1000, 8192, True, 1.0):
            with self.subTest(envs=envs), self.assertRaises(ValueError):
                make_plan(self.defaults, envs)

    def test_source_change_is_rejected_before_import_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trainer.py"
            source.write_text("raise AssertionError('must not be imported')\n")
            with self.assertRaisesRegex(ValueError, "identity"):
                official_defaults(source)

    def test_no_policy_updates_during_wm_only_phase(self):
        result = volumes(dict(self.defaults, iterations=500))
        self.assertEqual(result["policy_optimizer_updates"], 0)
        self.assertEqual(result["motion_imagined_rollouts"], 0)


if __name__ == "__main__":
    unittest.main()
