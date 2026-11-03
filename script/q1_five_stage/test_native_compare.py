"""Saved native traces cannot turn falls or a changed reference into an improvement."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from native_compare import compare_traces
from native_metrics import tracking


class NativeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def case(self, name, value=1., failure=None):
        reference = np.zeros((5, 269))
        actual = reference.copy()[:failure + 1 if failure else len(reference)]
        actual[1:, 0] = value
        if failure:
            actual[-1, 0] = 99.
        mask = np.zeros(29, dtype=bool)
        mask[0] = True
        path = Path(self.temp.name) / (name + ".npz")
        np.savez_compressed(path, actual=actual, reference=reference, joint_presence=mask)
        row = dict(clip="walk", seed=0, robot="q1", interface="motion", complete=failure is None,
            failure_frame=failure, valid_steps=4 if failure is None else failure - 1,
            attempted_steps=4 if failure is None else failure, reference_steps=4,
            reference_sha256="a" * 64, human_sha256="b" * 64, trace=str(path))
        if failure is None:
            row["full_clip_metrics"] = tracking(actual, reference, mask, 4)
        return row

    def test_fall_frame_excluded_and_full_motion_error_independently_checked(self):
        left, right = self.case("initial", 2., failure=2), self.case("final", .5)
        paired = compare_traces(left, right)
        self.assertEqual(paired["common_steps"], 1)
        self.assertEqual(paired["initial_joint_rmse_rad"], 2.)
        self.assertEqual(paired["final_joint_rmse_rad"], .5)
        right["full_clip_metrics"]["joint_rmse_rad"] = 0.
        with self.assertRaises(RuntimeError):
            compare_traces(left, right)

    def test_zero_prefix_remains_missing_and_invalid_failure_counters_rejected(self):
        left, right = self.case("initial", failure=1), self.case("final")
        paired = compare_traces(left, right)
        self.assertEqual(paired["common_steps"], 0)
        self.assertNotIn("initial_joint_rmse_rad", paired)
        left["valid_steps"] = 1
        with self.assertRaises(RuntimeError):
            compare_traces(left, right)

    def test_mixed_plant_inputs_and_reference_or_mask_changes_are_rejected(self):
        left, right = self.case("initial"), self.case("final")
        with self.assertRaises(RuntimeError):
            compare_traces(left, dict(right, interface="human"))
        with np.load(right["trace"]) as data:
            saved = {key: data[key].copy() for key in data.files}
        saved["reference"][-1, 0] = 1.
        np.savez_compressed(right["trace"], **saved)
        with self.assertRaises(RuntimeError):
            compare_traces(left, right)


if __name__ == "__main__":
    unittest.main()
