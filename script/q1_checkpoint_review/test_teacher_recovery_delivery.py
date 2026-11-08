"""Checks for unequal failed horizons and identical restoration support."""

import unittest

import numpy as np

from deliver_teacher_recovery import common_prefix, metrics


def arrays(n):
    actual = np.zeros((n+1, 79), dtype=np.float32)
    actual[:, 3] = 1.
    return dict(actual=actual, reference=actual.copy(), canonical_action=np.zeros((n, 29)),
        sole_height=np.zeros((n+1, 2)), load=np.full((n+1, 2), 10.),
        material_point_slip=np.zeros((n+1, 2)), tilt=np.zeros(n+1))


def row(n, failed=False):
    return dict(attempted_steps=n, failure_frame=64+n if failed else None,
        valid_prefix_steps=n-1 if failed else n, anchor=64, clip="fixed",
        supported_steps=80, complete_short_fork=n == 80 and not failed)


class SavedRecoverySupportTests(unittest.TestCase):
    def test_common_prefix_excludes_failure_and_longer_arm_tail(self):
        short, long = arrays(3), arrays(6)
        short["actual"][3, 21] = 1000.
        long["actual"][3:, 21] = 500.
        n, result = common_prefix([row(3, True), row(6, True)], [short, long], [14])
        self.assertEqual(n, 2)
        self.assertEqual([v["upper_joint_rmse_rad"] for v in result], [0., 0.])

    def test_different_initial_state_or_target_is_rejected(self):
        a, b = arrays(3), arrays(6)
        b["actual"][0, 0] = .1
        with self.assertRaisesRegex(ValueError, "initial states"):
            common_prefix([row(3, True), row(6, True)], [a, b], [14])
        b["actual"][0, 0] = 0.
        b["reference"][1, 0] = .1
        with self.assertRaisesRegex(ValueError, "common target"):
            common_prefix([row(3, True), row(6, True)], [a, b], [14])

    def test_missing_one_loaded_foot_has_no_two_foot_average(self):
        a = arrays(4)
        a["load"][:, 0] = 0.
        result = metrics(a, 4, [14])
        self.assertIsNone(result["equal_foot_loaded_slip_mean_m_s"])
        self.assertEqual(result["loaded_samples"], [0, 4])

    def test_terminal_transition_cannot_be_relabelled_valid(self):
        a, b = arrays(3), arrays(6)
        r = row(3, True); r["valid_prefix_steps"] = 3
        with self.assertRaisesRegex(ValueError, "terminating transition"):
            common_prefix([r, row(6, True)], [a, b], [14])


if __name__ == "__main__":
    unittest.main()
