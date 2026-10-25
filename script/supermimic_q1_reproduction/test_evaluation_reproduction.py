"""Focused regression checks for full-clip accounting and fixed clip selection."""

import unittest

import numpy as np

from evaluate_reproduction import aggregate_rows, select_clips, summarize_trace


class EvaluationAccountingTests(unittest.TestCase):
    def test_early_termination_keeps_full_reference_denominator(self):
        state = np.zeros((2, 57))
        state[:, 0] = [1., 2.]
        reference = np.zeros((5, 3))
        reference[:, 0] = np.arange(5)
        row = summarize_trace(state, [2., 2.], [1., 3.], [4., 6.], [False, True],
                              reference, np.zeros(3), 2.)
        self.assertEqual(row["normalized_return"], .4)
        self.assertEqual(row["full_clip_completion"], .4)
        self.assertTrue(row["early_terminated"])
        self.assertEqual(row["termination_step"], 2)
        self.assertEqual(row["root_xy_path_length_m"], 2.)
        self.assertEqual(row["root_xy_displacement_m"], 2.)
        self.assertEqual(row["reference_root_xy_displacement_full_m"], 4.)
        self.assertEqual(row["reference_root_xy_displacement_executed_m"], 1.)
        self.assertAlmostEqual(row["executed_joint_rmse_rad"], np.sqrt(5.))
        group = aggregate_rows([{"split": "training", **row}])["training"]
        self.assertEqual(group["failures"], 1)
        self.assertEqual(group["early_terminations"], 1)
        self.assertEqual(group["mean_full_clip_completion"], .4)

    def test_fixed_sorted_selection_and_explicit_order(self):
        available = ["z", "a", "c", "b"]
        self.assertEqual(select_clips(available, count=2), ["a", "b"])
        self.assertEqual(select_clips(available, ["z", "a"]), ["z", "a"])
        with self.assertRaises(ValueError):
            select_clips(available, ["a", "a"])
        with self.assertRaises(ValueError):
            select_clips(available, ["unknown"])

    def test_displacement_and_travel_differ_without_a_performance_gate(self):
        state = np.zeros((2, 57))
        state[:, 0] = [11., 10.]
        reference = np.array([[10., 0., 0.], [11., 0., 0.]])
        row = summarize_trace(state, [-1., -1.], [7., 7.], [8., 8.], [False, False],
                              reference, np.array([10., 0., 0.]), 1.)
        self.assertEqual(row["normalized_return"], -1.)
        self.assertEqual(row["root_xy_path_length_m"], 2.)
        self.assertEqual(row["root_xy_displacement_m"], 0.)
        self.assertFalse(row["failed"])
        self.assertEqual(row["full_clip_completion"], 1.)


if __name__ == "__main__":
    unittest.main()
