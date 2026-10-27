"""Test objective gates, missing evidence, regressions and cumulative limits."""

import copy
import unittest

from gates import classify_repeats, next_phase, ratio


def passing():
    return {"ratios": {"upper_rmse": .8, "wrist_relative_rmse": .8, "raised_peak_error": .7,
                       "root_xy_rmse": 1.05, "leg_rmse": 1.05, "target_oscillation": 1.1, "loaded_slip": 1.1},
            "completion": .99, "failures": 1, "fixed_demos_complete": True}


class GatesTest(unittest.TestCase):
    def test_two_complete_repeats_required(self):
        bad = passing()
        bad["ratios"]["upper_rmse"] = 1.
        self.assertTrue(classify_repeats([passing(), passing(), bad])["extension_qualified"])
        self.assertFalse(classify_repeats([passing(), bad, bad])["extension_qualified"])

    def test_three_demo_completions_required(self):
        bad = passing()
        bad["fixed_demos_complete"] = False
        self.assertFalse(classify_repeats([passing(), passing(), bad])["extension_qualified"])

    def test_missing_and_nonfinite_contact_never_pass(self):
        for value in (None, float("nan")):
            bad = passing()
            bad["ratios"]["loaded_slip"] = value
            self.assertFalse(classify_repeats([bad] * 3)["extension_qualified"])

    def test_severe_root_or_upper_error_and_failure(self):
        for key in ("root_xy_rmse", "upper_rmse"):
            bad = passing()
            bad["ratios"][key] = 1.25
            self.assertTrue(classify_repeats([bad, bad, passing()])["severe_regression"])
        bad = passing()
        bad["failures"] = 6
        self.assertTrue(classify_repeats([bad, bad, passing()])["severe_regression"])

    def test_extension_five_additional_hours(self):
        gate = classify_repeats([passing()] * 3)
        result = next_phase("A", 3502., gate)
        self.assertEqual(result["deadline"], 21502.)
        self.assertEqual(result["phase"], "C")

    def test_no_gate_stops_at_three_hours(self):
        gate = dict(extension_qualified=False, severe_regression=False)
        self.assertEqual(next_phase("B", 10701., gate)["action"], "stop")
        self.assertEqual(next_phase("A", 3501., gate)["phase"], "B")

    def test_extension_requires_consecutive_regressions(self):
        bad = dict(extension_qualified=False, severe_regression=True)
        first = next_phase("C", 7100., bad, 21500., 0)
        self.assertEqual(first["action"], "continue")
        self.assertEqual(next_phase("C", 10700., bad, 21500., first["severe_streak"])["action"], "stop")
        good = dict(extension_qualified=False, severe_regression=False)
        self.assertEqual(next_phase("C", 10000., good, 21500., 1)["severe_streak"], 0)

    def test_absolute_cap_and_conservative_exit_reserve(self):
        gate = dict(extension_qualified=True, severe_regression=False)
        self.assertEqual(next_phase("B", 10701., gate)["deadline"], 28701.)
        self.assertEqual(next_phase("C", 28800., gate, 28800.)["action"], "stop")
        self.assertEqual(next_phase("C", 21440., gate, 21500.)["action"], "stop")

    def test_invalid_metric(self):
        self.assertIsNone(ratio(None, 1.))
        self.assertIsNone(ratio(float("nan"), 1.))
        self.assertEqual(ratio(0., 0.), 0.)


if __name__ == "__main__":
    unittest.main()
