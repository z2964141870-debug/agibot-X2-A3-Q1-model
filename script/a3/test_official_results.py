"""Aggregation uses frame weights rather than weighting clips equally."""

import unittest

from script.a3.official_results import aggregate


class AggregationTests(unittest.TestCase):
    def row(self, n, error, fall=False):
        metrics = {"policy_steps": n, "joint_rmse_rad": error,
                   "root_pos_error_m_mean": error, "root_quat_error_deg_mean": error}
        return {"fall": fall, "motion_completed_without_fall": not fall,
                "windows": {"full_rollout": metrics, "before_first_fall": metrics}}

    def test_weighted_joint_rms_and_position_mean(self):
        result = aggregate([self.row(1, 1), self.row(3, 3, True)])
        self.assertAlmostEqual(result["full_rollout"]["joint_rmse_rad"], 7 ** .5)
        self.assertEqual(result["full_rollout"]["root_pos_error_m_mean"], 2.5)
        self.assertEqual(result["falls"], 1)
        self.assertEqual(result["completed_without_fall"], 1)

    def test_empty_before_fall_has_no_error_value(self):
        result = aggregate([self.row(0, None, True)])
        self.assertIsNone(result["before_first_fall"]["joint_rmse_rad"])


if __name__ == "__main__":
    unittest.main()
