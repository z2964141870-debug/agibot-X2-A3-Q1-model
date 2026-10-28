"""Nonfinite/partial evidence and first-hour extension regression tests."""

import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from decisions import gate


def evidence():
    return {"ratios":{"walking_slip":.85,"swing_error":.85,"walking_tilt":1.,"target_change":1.,
                      "upper_rmse":1.,"wrist_rmse":1.,"raised_peak_error":1.},
            "stationary_drift_m":.04,"baseline_stationary_drift_m":.04,
            "validation_failures":0,"validation_completion":1.,
            "fixed_demos_complete":True,"walking_progress_preserved":True}


class Gates(unittest.TestCase):
    def test_two_full_repeats_can_authorize_extension(self):
        rows = [evidence() for _ in range(3)]
        rows[2]["ratios"]["walking_slip"] = 1.01
        self.assertTrue(gate(rows)["improvement_confirmed"])

    def test_one_fallen_fixed_demo_blocks_extension_even_with_two_passes(self):
        rows = [evidence() for _ in range(3)]
        rows[2]["fixed_demos_complete"] = False
        self.assertFalse(gate(rows)["improvement_confirmed"])

    def test_nonfinite_cannot_pass(self):
        rows = [evidence() for _ in range(3)]
        for row in rows:
            row["ratios"]["walking_slip"] = float("nan")
        self.assertFalse(gate(rows)["improvement_confirmed"])
        self.assertTrue(gate(rows)["physical_regression"])

    def test_not_moving_cannot_win_anti_slip_reward(self):
        rows = [evidence() for _ in range(3)]
        for row in rows:
            row["walking_progress_preserved"] = False
        self.assertFalse(gate(rows)["improvement_confirmed"])

    def test_reward_like_gain_does_not_override_physical_slip(self):
        rows = [evidence() for _ in range(3)]
        for row in rows[:2]:
            row["ratios"]["walking_slip"] = 1.6
        self.assertTrue(gate(rows)["physical_regression"])
        self.assertFalse(gate(rows)["improvement_confirmed"])


if __name__ == "__main__":
    unittest.main()
