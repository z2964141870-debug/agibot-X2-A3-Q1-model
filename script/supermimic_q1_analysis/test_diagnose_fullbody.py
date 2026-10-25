"""Accounting, rotation invariance and reference-only selection controls."""

import unittest

import numpy as np

from diagnose_fullbody import arm_profile, local_points, normalized_terms, select_arm_clips


class FullBodyTests(unittest.TestCase):
    def test_torso_frame_removes_common_rigid_motion(self):
        positions = np.zeros((2, 14, 3))
        positions[:, 8] = [0, 0, .2]
        positions[:, 10] = [1, 0, .5]
        quat = np.zeros((2, 14, 4))
        quat[:, :, 0] = 1
        target = local_points(positions, quat)
        transformed = positions @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [5, 4, 2]
        rotated = np.tile([np.sqrt(.5), 0, 0, np.sqrt(.5)], (2, 14, 1))
        np.testing.assert_allclose(local_points(transformed, rotated), target, atol=1e-12)
        self.assertAlmostEqual(arm_profile(transformed, rotated)["left"]["wrist_above_shoulder_peak_m"], .3)

    def test_early_failure_denominator_and_negative_penalty(self):
        result = normalized_terms({"positive": [1., 1.], "penalty": [-.2, -.3]}, 10, 2.)
        self.assertEqual(result["positive"], .1)
        self.assertEqual(result["penalty"], -.025)

    def test_selection_is_distinct_deterministic_and_score_free(self):
        rows = [{"clip": name, "arms": {side: {"wrist_height_span_m": value,
                    "wrist_above_shoulder_peak_m": value / 2} for side in ("left", "right")}}
                for name, value in (("b", 1.), ("a", 1.), ("c", 2.), ("d", 0.))]
        selected = select_arm_clips(rows)
        self.assertEqual([r["clip"] for r in selected], ["c", "a", "b", "d"])


if __name__ == "__main__":
    unittest.main()
