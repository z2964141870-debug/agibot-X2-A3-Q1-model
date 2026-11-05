import math
import unittest

from preview_benchmark import DT, preview


class CausalPreviewTests(unittest.TestCase):
    def test_constant_velocity_exact_and_observed_slots_preserved(self):
        rows = [[1 + 2*t*DT] for t in range(25)]
        for buffer_frames in (0, 4, 5, 10):
            pred = preview(rows[:7+buffer_frames], 6, "cv_ls5")
            for i, value in enumerate(pred):
                self.assertAlmostEqual(value[0], rows[7+i][0], places=12)
            self.assertEqual(pred[:buffer_frames], rows[7:7+buffer_frames])

    def test_constant_acceleration_exact(self):
        rows = [[.3 + 2*t*DT + .5*3*(t*DT)**2] for t in range(25)]
        pred = preview(rows[:7], 6, "ca_ls7")
        for i, value in enumerate(pred):
            self.assertAlmostEqual(value[0], rows[7+i][0], places=11)

    def test_hidden_future_cannot_change_prediction(self):
        rows = [[math.sin(t*DT)] for t in range(25)]
        changed = [row[:] for row in rows]
        changed[11:] = [[1e8]] * (len(changed)-11)
        for method in ("hold", "cv_difference", "cv_ls5", "ca_ls7", "damped_cv_ls5"):
            self.assertEqual(preview(rows[:11], 6, method), preview(changed[:11], 6, method))

    def test_invalid_buffer_rejected(self):
        for observed in (6, 18):
            with self.assertRaises(ValueError):
                preview([[0.0]]*observed, 6, "cv_ls5")


if __name__ == "__main__":
    unittest.main()
