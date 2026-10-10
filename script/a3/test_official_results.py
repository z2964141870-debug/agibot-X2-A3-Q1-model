"""Aggregation uses frame weights rather than weighting clips equally."""

import unittest

from script.a3.official_results import aggregate
from script.a3.evaluate_mujoco import rollout_windows
import numpy as np


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

    def test_body_errors_use_same_before_fall_mask(self):
        names = ["torso_Link", "left_wrist_yaw_Link", "right_wrist_yaw_Link", "left_knee_Link"]
        points = np.zeros((2, 4, 3))
        points[0, :, 0] = .1
        points[1, :, 0] = 10
        trace = {"policy_tick": [0, 1], "q_state_29": np.zeros((2, 29)),
                 "reference_q_29": np.zeros((2, 29)), "root_pos_error_m": [0, 0],
                 "root_quat_error_deg": [0, 0], "anchor_pos_error_m": [0, 0], "body_names": names,
                 "sim_body_pos_w": points, "ref_body_pos_w": np.zeros_like(points),
                 "sim_body_pos_anchor": points, "ref_body_pos_anchor": np.zeros_like(points)}
        summary = {"num_policy_steps": 2, "fall": True, "fall_tick": 1,
                   "tracking": {"all_29_rmse": 0}}
        windows = rollout_windows(summary, trace)
        self.assertAlmostEqual(windows["before_first_fall"]["wrists_world_mean_m"], .1)
        self.assertAlmostEqual(windows["full_rollout"]["legs_anchor_local_mean_m"], 5.05)


if __name__ == "__main__":
    unittest.main()
