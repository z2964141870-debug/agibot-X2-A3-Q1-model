"""Counterexamples for per-clip gains, per-repeat guards and malformed data."""

import copy
import unittest

from physical_gate import FIXED, RAISE, WALKS, evaluate


def rows(gain=1.):
    return [dict(clip=clip, repeat=repeat, steps=200, reference_steps=200, failed="False",
                 swing_sole_z_rmse_m=.02 * gain, swing_sole_foot_frames=100,
                 swing_sole_phase_sha256="a" * 64,
                 loaded_slip_mps=.1 * gain, torso_tilt_rms_rad=.1,
                 upper_rmse=.3, wrist_relative_rmse=.05, raised_peak_error=.01,
                 forward_displacement_m=2., reference_displacement_m=2.1,
                 sustained_single_support_alternations=5)
            for clip in FIXED for repeat in range(3)]


class PhysicalGateTests(unittest.TestCase):
    def test_pass_is_not_goal_completion_or_user_adoption(self):
        result = evaluate(rows(), rows(.75))
        self.assertTrue(result["fixed_numeric_passed"])
        self.assertFalse(result["overall_goal_achieved"])
        self.assertFalse(result["original_speed_video_reviewed"])
        self.assertIsNone(result["user_accepted_checkpoint"])

    def test_two_walks_cannot_be_pooled(self):
        candidate = rows(.2)
        for row in candidate:
            if row["clip"] == WALKS[1]:
                row["loaded_slip_mps"] = .101
        result = evaluate(rows(), candidate)
        self.assertTrue(result["per_walk"][0]["passed"])
        self.assertFalse(result["per_walk"][1]["passed"])
        self.assertFalse(result["fixed_numeric_passed"])

    def test_median_does_not_hide_one_bad_posture_or_upper_repeat(self):
        for name in ("torso_tilt_rms_rad", "upper_rmse", "wrist_relative_rmse"):
            candidate = rows(.75)
            candidate[2][name] *= 1.11
            result = evaluate(rows(), candidate)
            self.assertTrue(all(row["passed"] for row in result["per_walk"]))
            self.assertFalse(result["fixed_numeric_passed"])

    def test_raise_amplitude_and_all_completions_are_guarded(self):
        for field, value in (("failed", "true"), ("steps", 199), ("raised_peak_error", .012)):
            candidate = rows(.75)
            next(row for row in candidate if row["clip"] == RAISE)[field] = value
            self.assertFalse(evaluate(rows(), candidate)["fixed_numeric_passed"])

    def test_reference_completion_is_also_required(self):
        baseline = rows()
        baseline[0]["steps"] = 199
        self.assertFalse(evaluate(baseline, rows(.75))["fixed_numeric_passed"])

    def test_progress_and_continuous_switching(self):
        for field, value in (("forward_displacement_m", -1.), ("forward_displacement_m", 1.79),
                             ("sustained_single_support_alternations", 1)):
            candidate = rows(.75)
            candidate[0][field] = value
            self.assertFalse(evaluate(rows(), candidate)["fixed_numeric_passed"])

    def test_zero_baseline_cannot_claim_relative_gain(self):
        baseline, candidate = rows(), rows(.75)
        for row in baseline + candidate:
            row["loaded_slip_mps"] = 0.
        self.assertFalse(evaluate(baseline, candidate)["fixed_numeric_passed"])
        baseline, candidate = rows(), rows(.75)
        for row in baseline + candidate:
            row["upper_rmse"] = 0.
        candidate[0]["upper_rmse"] = 1e-5
        self.assertFalse(evaluate(baseline, candidate)["fixed_numeric_passed"])

    def test_missing_duplicate_or_unknown_clip_is_invalid(self):
        invalid = [rows(.75)[:-1], rows(.75) + [copy.deepcopy(rows(.75)[0])]]
        changed = rows(.75)
        changed[0]["clip"] = "replacement_walk"
        invalid.append(changed)
        for candidate in invalid:
            with self.assertRaises(ValueError):
                evaluate(rows(), candidate)

    def test_nonfinite_or_malformed_numbers_fail_closed(self):
        for field, value in (("loaded_slip_mps", "nan"), ("upper_rmse", "inf"),
                             ("torso_tilt_rms_rad", -.1), ("repeat", True),
                             ("steps", "200.5"), ("swing_sole_z_rmse_m", None),
                             ("swing_sole_foot_frames", 0), ("failed", "maybe")):
            candidate = rows(.75)
            candidate[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                evaluate(rows(), candidate)

    def test_reference_identity_and_phase_masks_must_match(self):
        for field, value in (("reference_steps", 201), ("swing_sole_foot_frames", 101),
                             ("swing_sole_phase_sha256", "b" * 64),
                             ("reference_displacement_m", 2.2)):
            candidate = rows(.75)
            candidate[0][field] = value
            with self.assertRaises(ValueError):
                evaluate(rows(), candidate)


if __name__ == "__main__":
    unittest.main()
