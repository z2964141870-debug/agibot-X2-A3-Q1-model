"""Check that the observed upstream run cannot silently become a short trial."""

import copy
from pathlib import Path
import unittest

from scale_plan import make_plan, official_defaults
from worker import validate_train


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        source = root / "data/references/morphmimic_review_20261029/source"
        plan = make_plan(official_defaults(source / "src/morphmimic/train/trainer.py"), 1024)
        self.config = dict(source=str(source), train_cfg=dict(plan["scaled"], robot="g1", wandb=False))

    def test_full_scaled_run_is_accepted(self):
        self.assertEqual(validate_train(self.config, "train")["iterations"], 160000)

    def test_short_or_changed_training_is_rejected(self):
        for name, value in (("iterations", 600), ("wm_only_iters", 500),
                            ("horizon", 16), ("policy_lr", .001)):
            config = copy.deepcopy(self.config)
            config["train_cfg"][name] = value
            with self.assertRaisesRegex(ValueError, "Deviation"):
                validate_train(config, "train")

    def test_eager_mode_requires_a_materialized_reason(self):
        self.config["train_cfg"]["compile"] = "none"
        with self.assertRaisesRegex(ValueError, "resource reason"):
            validate_train(self.config, "train")
        self.config["eager_execution_resource_reason"] = "Recorded profile exceeded compilation memory reserve"
        validate_train(self.config, "train")

    def test_profile_requires_full_update_phase_and_is_bounded(self):
        self.config["train_cfg"].update(iterations=8, wm_only_iters=0, ckpt_every=8)
        validate_train(self.config, "profile")
        self.config["train_cfg"]["wm_only_iters"] = 500
        with self.assertRaisesRegex(ValueError, "complete updates"):
            validate_train(self.config, "profile")

    def test_single_baseline_cannot_be_relabeled_multirobot(self):
        self.config["train_cfg"]["robots"] = ["g1", "x2", "t2"]
        with self.assertRaisesRegex(ValueError, "independent G1"):
            validate_train(self.config, "train")


if __name__ == "__main__":
    unittest.main()
