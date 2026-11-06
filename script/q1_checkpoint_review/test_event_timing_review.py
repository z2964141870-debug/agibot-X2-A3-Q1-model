"""Clock alignment and denominator checks for saved-call localization."""

import unittest

from event_timing_review import locate, statistics_for


class TimingReviewTests(unittest.TestCase):
    def test_capture_window_and_recovery_keep_frame_identity(self):
        values = [dict(frame=i, simulated_control_ns=(i+4)*20_000_000,
            event=dict(first_frame=50, duration_frames=10), gpu_policy_wall_compute_ns=5_000_000,
            fallback="hold" if 46 <= i < 55 else None) for i in range(70)]
        row = dict(event="dropout", method="damped_cv", horizon_frames=4, clip="fixture")
        points = locate(values, row)
        self.assertEqual(points[45]["region"], "before_event")
        self.assertEqual(points[46]["capture_clock_since_event_ms"], 0)
        self.assertEqual(points[55]["region"], "event_window")
        self.assertEqual(points[56]["region"], "after_event")
        self.assertTrue(points[55]["recovered_this_frame"])
        self.assertTrue(points[53]["within_two_frames_of_recovery"])
        self.assertFalse(points[52]["within_two_frames_of_recovery"])
        values[12]["simulated_control_ns"] += 1
        with self.assertRaises(ValueError):
            locate(values, row)

    def test_rates_keep_denominators_and_strict_deadlines(self):
        measured = statistics_for([dict(compute_ms=v) for v in (1, 20, 21, 100, 101)])
        self.assertEqual((measured["calls"], measured["over20ms"], measured["over100ms"]), (5, 3, 1))
        self.assertAlmostEqual(measured["p95_ms"], 100.8)
        self.assertEqual(statistics_for([])["calls"], 0)


if __name__ == "__main__":
    unittest.main()
