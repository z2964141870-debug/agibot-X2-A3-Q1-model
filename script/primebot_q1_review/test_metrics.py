"""Early stopping cannot use a missing, failed or incomplete observation."""

import unittest
import importlib.util
import math
from metrics import complete_library, scheduled, sustained_stop, tracking


class Monitoring(unittest.TestCase):
    def test_only_three_consecutive_passes_stop(self):
        passed = dict(status="PASSED", gate=dict(passed=True, provisional_only=True))
        self.assertFalse(sustained_stop([passed] * 2))
        self.assertTrue(sustained_stop([passed] * 3))
        self.assertFalse(sustained_stop([passed, dict(status="FAILED"), passed]))
        self.assertFalse(sustained_stop([dict(status="PASSED", gate=None), passed, passed]))
        self.assertFalse(sustained_stop([passed, dict(status="PASSED", gate=dict(passed=False)), passed]))

    def test_schedule_has_baseline_and_first_thousand(self):
        self.assertTrue(all(scheduled(n) for n in (0, 1000, 5000, 10000, 250000)))
        self.assertFalse(scheduled(8))
        self.assertFalse(scheduled(1001))

    def test_library_missing_or_duplicate_never_completes(self):
        self.assertTrue(complete_library([dict(relative="a"), dict(relative="b")], ["a", "b"]))
        self.assertFalse(complete_library([dict(relative="a")], ["a", "b"]))
        self.assertFalse(complete_library([dict(relative="a"), dict(relative="a")], ["a", "b"]))

    @unittest.skipIf(importlib.util.find_spec("numpy") is None, "Requires the locked runtime NumPy")
    def test_prefix_excludes_initial_and_later_invalid_frames(self):
        import numpy as np
        actual = np.zeros((4, 79))
        reference = actual.copy()
        actual[:, 7:29] = np.array([999., 1., 2., 999.])[:, None]
        body = np.zeros((4, 4, 3))
        target = body.copy()
        body[:, 2:, 0] = np.array([999., 1., 2., 999.])[:, None]
        result = tracking(actual, reference, body, target, 2)
        self.assertAlmostEqual(result["joint_rmse_rad"], math.sqrt(2.5))
        self.assertAlmostEqual(result["body_relative_rmse_m"], math.sqrt(2.5 / 3))


if __name__ == "__main__":
    unittest.main()
