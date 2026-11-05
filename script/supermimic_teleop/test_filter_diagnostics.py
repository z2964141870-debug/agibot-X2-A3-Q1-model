"""Analytic causal-filter properties; never tune on the human TEST_ONLY set."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from filter_diagnostics import ScalarFilter, filter_signal, harmonic_response, load_one_euro


class FilterDiagnosticsTests(unittest.TestCase):
    def test_trailing_mean_has_40ms_fundamental_delay_at50hz(self):
        times = np.arange(1001)/50
        for frequency in (.5, 1., 2., 5.):
            values = .25*np.sin(2*np.pi*frequency*times)
            output = filter_signal("mean5", times, values)
            row = harmonic_response(times, output, frequency)
            self.assertAlmostEqual(row["fundamental_phase_lag_ms"], 40., places=9)
            self.assertLess(row["harmonic_residual_rms_rad"], 1e-12)
            self.assertFalse(row["phase_is_measured_end_to_end_delay"])

    def test_slew_limit_uses_irregular_real_intervals(self):
        times = np.asarray([0., .01, .04, .11, .13])
        output = filter_signal("slew2", times, [0., 1., -1., 1., -1.])
        speed = np.abs(np.diff(output)/np.diff(times))
        self.assertLessEqual(float(speed.max()), 2.+1e-12)

    def test_future_poison_does_not_change_past_outputs(self):
        times = np.arange(50)/50
        values = np.sin(times)
        poisoned = values.copy()
        poisoned[20:] = 1e5
        for method in ("raw", "mean5", "slew2"):
            with self.subTest(method=method):
                a = filter_signal(method, times, values)
                b = filter_signal(method, times, poisoned)
                np.testing.assert_array_equal(a[:20], b[:20])

    def test_spike_averaging_reduces_peak_and_retains_explicit_tail(self):
        times = np.arange(20)/50
        values = np.zeros(20)
        values[10] = .5
        output = filter_signal("mean5", times, values)
        np.testing.assert_allclose(output[10:15], .1)
        self.assertEqual(output[15], 0.)

    def test_nonfinite_clock_rewind_and_long_gaps_rejected(self):
        filter_ = ScalarFilter("slew2")
        filter_(0., 0)
        for value, stamp in ((float("nan"), 20_000_000), (1., 0), (1., 201_000_000)):
            with self.subTest(value=value, stamp=stamp), self.assertRaises(ValueError):
                filter_(value, stamp)
        with self.assertRaises(ValueError):
            filter_signal("raw", [0., .01], [0., np.inf])
        with self.assertRaises(ValueError):
            ScalarFilter("one_euro_beta01")

    def test_untrusted_wheel_never_imported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"not_the_pinned_wheel.whl"
            path.write_bytes(b"not executable dependency content")
            with self.assertRaisesRegex(ValueError, "exact pinned"):
                load_one_euro(path)


if __name__ == "__main__":
    unittest.main()
