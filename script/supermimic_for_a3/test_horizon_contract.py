"""Keep warm-start contracts strict while admitting the explicit H32-to-H8 trial."""

import copy
import unittest

from initialization import validate_contract


class HorizonContractControls(unittest.TestCase):
    def setUp(self):
        self.source = {key: {"fixed": "same"} for key in
                       ("assets", "sim", "world_model", "policy", "critic", "wm_trainer",
                        "train_clips", "heldout_clips", "backend_receipt")}
        self.source["policy_trainer"] = {"horizon": 32, "batch_size": 64, "gamma": .99}

    def candidate(self):
        target = copy.deepcopy(self.source)
        target["policy_trainer"]["horizon"] = 8
        return target

    def test_default_requires_exact_contract(self):
        self.assertEqual(validate_contract(self.source, self.source), {})
        with self.assertRaisesRegex(ValueError, "policy_trainer"):
            validate_contract(self.source, self.candidate())

    def test_only_explicit_32_to_8_change_is_admitted(self):
        self.assertEqual(validate_contract(self.source, self.candidate(), True),
                         {"policy_trainer.horizon": {"source": 32, "target": 8}})
        with self.assertRaisesRegex(ValueError, "requires a 32-to-8"):
            validate_contract(self.source, self.source, True)

    def test_other_fields_stay_protected(self):
        for field in self.source:
            target = self.candidate()
            target[field]["unregistered_change"] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contract differs"):
                validate_contract(self.source, target, True)

    def test_no_arbitrary_horizon_override(self):
        for horizon in (1, 16, 64):
            target = self.candidate()
            target["policy_trainer"]["horizon"] = horizon
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "policy_trainer"):
                validate_contract(self.source, target, True)


if __name__ == "__main__":
    unittest.main()
