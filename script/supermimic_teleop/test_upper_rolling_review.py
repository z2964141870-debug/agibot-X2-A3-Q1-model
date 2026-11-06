"""Check rolling target alignment, causality and explicitly generated dropouts."""

import json
from pathlib import Path
import unittest

import numpy as np

from upper_rolling_review import (dense_windows, fixed_sources, metrics,
                                  observations, specification, stale_gate, timing_review)


class RollingReviewTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.repeat((np.arange(150)*.02)[:, None], 8, axis=1)

    def test_preregistered_rules_match(self):
        path = Path(__file__).with_name("upper_rolling_review_plan_20261106.json")
        self.assertEqual(json.loads(path.read_text()), specification())

    def test_future_mutation_does_not_change_visible_history(self):
        cutoffs, history, labels = dense_windows(self.reference, self.reference)
        altered = self.reference.copy()
        altered[70:] += 1000.
        _, changed, changed_labels = dense_windows(altered, altered)
        np.testing.assert_array_equal(history[cutoffs < 70], changed[cutoffs < 70])
        self.assertFalse(np.array_equal(labels, changed_labels))
        self.assertEqual(cutoffs[0], 24)
        self.assertEqual(cutoffs[-1], len(self.reference)-11)

    def test_noise_is_one_stream_and_dropouts_keep_real_support(self):
        noisy, support = observations(self.reference, "continuous_iid_0.01")
        np.testing.assert_array_equal(support, np.arange(150))
        _, histories, _ = dense_windows(self.reference, noisy)
        np.testing.assert_array_equal(histories[1, :-1], histories[0, 1:])
        dropped, source = observations(self.reference, "iid_0.01_periodic5_drop")
        np.testing.assert_array_equal(source[50:55], np.full(5, 49))
        np.testing.assert_array_equal(dropped[50:55], np.repeat(noisy[49:50], 5, axis=0))
        self.assertEqual(source[55], 55)
        np.testing.assert_array_equal(dropped[55], noisy[55])

    def test_oracle_has_zero_same_absolute_target_revision(self):
        _, history, target = dense_windows(self.reference, self.reference)
        report = metrics(target, target)
        self.assertEqual(report["same_target_revision_rad"]["absolute_max"], 0.)
        self.assertEqual(report["error_rad"]["rms"], 0.)
        self.assertLess(report["generated_signal_proxies"][0]["speed_error_rad_s"]["absolute_max"], 1e-12)
        self.assertEqual(history.shape[1:], (25, 8))

    def test_gate_uses_explicit_age_and_preserves_inputs(self):
        _, history, target = dense_windows(self.reference, self.reference)
        ages = np.zeros(len(history), dtype=int)
        ages[:3] = [1, 2, 3]
        original = target.copy()
        gated, replaced = stale_gate(target, history, ages)
        np.testing.assert_array_equal(replaced[:3], [False, False, True])
        np.testing.assert_array_equal(gated[:2], target[:2])
        np.testing.assert_array_equal(gated[2], np.repeat(history[2, -1, None], 10, axis=0))
        np.testing.assert_array_equal(target, original)
        with self.assertRaises(ValueError):
            stale_gate(target, history, np.full(len(history), -1))

    def test_wrong_fixed_split_is_rejected(self):
        with self.assertRaises(ValueError):
            fixed_sources(dict(clips=[]))

    def test_timing_waits_for_materialization_and_reports_scope(self):
        events = []
        value = timing_review(lambda: events.append("call"), lambda: events.append("sync"))
        self.assertEqual(events[:3], ["sync", "call", "sync"])
        self.assertEqual(events.count("call"), 221)
        self.assertIsNone(value["robot_end_to_end_ms"])
        self.assertEqual(value["measured_calls"], 200)


if __name__ == "__main__":
    unittest.main()
