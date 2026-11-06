"""Guard against interpreting failed or missing physical cases as convergence."""

import unittest

from analyze import compare_group


def case(name, complete=True, error=.1):
    value = dict(clip=name, complete=complete, valid_steps=100 if complete else 10, reference_steps=100)
    if complete:
        value.update(full_clip_metrics=dict(joint_rmse_rad=error, body_relative_rmse_m=.02), diagnostics=dict(
            upper_joint_rmse_rad=.12, root_xy_final_error_m=.03, torso_tilt_rms_rad=.04,
            alternating_lift_transitions=5, sole_p95_m=[.01, .02], reference_sole_p95_m=[.02, .02],
            loaded_slip_mean_m_s=[.05, .06]))
    return value


class AnalysisTests(unittest.TestCase):
    def test_failed_case_has_no_full_set_mean_or_full_motion_difference(self):
        result = compare_group([
            dict(iteration=32000, cases=[case("forward"), case("start")]),
            dict(iteration=48000, cases=[case("forward", error=.09), case("start", complete=False)])])
        self.assertNotIn("joint_rmse_rad", result["full_fixed_set_means"][48000])
        self.assertEqual(result["mean_changes"], [])
        failed = next(value for value in result["changes"] if value["clip"] == "start")
        self.assertFalse(failed["both_complete"])
        self.assertNotIn("relative_change", failed)

    def test_changes_preserve_regression_direction(self):
        result = compare_group([
            dict(iteration=32000, cases=[case("forward", error=.1)]),
            dict(iteration=48000, cases=[case("forward", error=.11)])])
        self.assertAlmostEqual(result["mean_changes"][0]["relative_change"]["joint_rmse_rad"], .1)
        self.assertEqual(result["missing_iterations"], [16000])

    def test_missing_group_remains_explicit(self):
        result = compare_group([])
        self.assertEqual(result["available_iterations"], [])
        self.assertEqual(result["missing_iterations"], [16000, 32000, 48000])
        self.assertEqual(result["full_fixed_set_means"], {})

    def test_changed_fixed_clip_set_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Changed fixed case set"):
            compare_group([dict(iteration=32000, cases=[case("forward")]),
                           dict(iteration=48000, cases=[case("start")])])


if __name__ == "__main__":
    unittest.main()
