"""Counterexamples for low-force conditional command and clock statistics."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_q1_saved_low_force_control import conditions, joint_metrics, previous_angles, published_inputs, read_pose
from analyze_q1_saved_sample_runs import identity


class LowForceControlTests(unittest.TestCase):
    def test_published_artifact_list_and_nested_list_both_validate_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "summary.json"
            path.write_text("{}\n")
            record = identity(path)
            record["relative_path"] = "data/saved/summary.json"
            manifest = root / "manifest.json"
            for artifacts in ([record], {"files": [record]}):
                manifest.write_text(json.dumps(dict(artifacts=artifacts)))
                self.assertEqual(published_inputs(manifest, root, ("summary.json",)), [manifest, path])
            path.write_text("{\"changed\":true}\n")
            with self.assertRaises(ValueError):
                published_inputs(manifest, root, ("summary.json",))

    def test_request_toward_reference_can_have_motion_away(self):
        row = joint_metrics([0., 0.], [-.1, -.1], [.2, .2], [.1, .1], [0., 0.], [.2, .2])
        self.assertEqual(row["request_toward_reference_fraction"], 1.)
        self.assertEqual(row["observed_step_toward_reference_fraction"], 0.)
        self.assertLess(row["observed_reference_squared_error_reduction_rad2"], 0)

    def test_residual_size_and_reference_error_improvement_are_distinct(self):
        row = joint_metrics([0.], [.02], [.19], [.1], [-.01], [.2])
        self.assertAlmostEqual(row["residual_rms_rad"], .01)
        self.assertGreater(row["residual_reference_squared_error_reduction_rad2"], 0)
        row = joint_metrics([0.], [.02], [.21], [.1], [.01], [.2])
        self.assertLess(row["residual_reference_squared_error_reduction_rad2"], 0)

    def test_empty_and_near_zero_reference_gaps_keep_null(self):
        row = joint_metrics([], [], [], [], [], [])
        self.assertEqual(row["samples"], 0)
        self.assertIsNone(row["observed_step_rms_rad"])
        row = joint_metrics([.1], [.1], [.1], [.1005], [0.], [.1])
        self.assertEqual(row["active_reference_gap_samples"], 0)
        self.assertIsNone(row["request_toward_reference_fraction"])

    def test_previous_state_uses_initial_without_wrap(self):
        post = np.array([np.full(22, 2.), np.full(22, 3.)])
        pre = previous_angles(np.full(22, 1.), post)
        np.testing.assert_array_equal(pre, np.array([np.full(22, 1.), np.full(22, 2.)]))
        np.testing.assert_array_equal(post[-1], np.full(22, 3.))

    def test_masks_keep_force_height_and_window_boundaries(self):
        def foot(force, height=.005, reference=.01, swing=True):
            return dict(native_force_N=force, sole_z_m=height, reference_sole_z_m=reference, intent_swing=swing)
        feet = dict(right=[foot(5.), foot(5.01), foot(0., .011), foot(0., .01), foot(0., swing=False)],
                    left=[foot(5.01), foot(20.), foot(5.), foot(20.), foot(20.)])
        masks = conditions(feet, "right")
        self.assertEqual(np.flatnonzero(masks["raised_low"]).tolist(), [0, 3])
        self.assertEqual(np.flatnonzero(masks["raised_low_near5"]).tolist(), [0])
        self.assertFalse(masks["raised_low_high10"].any())
        self.assertEqual(np.flatnonzero(masks["raised_both_low"]).tolist(), [2])

    def test_invalid_shapes_or_nonfinite_joint_values_rejected(self):
        for initial, post in ((np.zeros(21), np.zeros((2, 22))), (np.zeros(22), np.zeros((0, 22))),
                              (np.zeros(22), np.full((2, 22), np.nan))):
            with self.assertRaises(ValueError):
                previous_angles(initial, post)
        with self.assertRaises(ValueError):
            joint_metrics([0.], [float("nan")], [0.], [0.], [0.], [0.])

    def test_pose_join_checks_clock_frame_force_and_budget(self):
        key = "base", "initial", 0, "synthetic"
        sample = dict(training_seconds=0., global_reference_frame=12, intent_swing=True,
                      native_force_N=0., reference_sole_z_m=.02, sole_z_m=.003)
        feet = {key: dict(left=[sample.copy()], right=[sample.copy()])}
        rows = [dict(line=key[0], label=key[1], repeat=0, clip=key[3], side=side, sample=0,
            control_start_s=0., reference_sample_end_s=.02, training_seconds=0., global_reference_frame=12,
            intent_swing=True, reference_sole_z_m=.02, native_force_final_solve_N=0., pre_sole_z_m=.003,
            post_sole_z_m=.003, target_sole_z_m=.03, nominal_sole_z_m=.03,
            reference_same_root_sole_z_m=.02, target_leg_any_joint_outside=False) for side in ("left", "right")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.csv"
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(values)
            write(rows)
            self.assertEqual(len(read_pose(path, feet)), 2)
            for name, value in (("sample", 1), ("control_start_s", .01), ("global_reference_frame", 13),
                                ("training_seconds", 1.), ("native_force_final_solve_N", 1.),
                                ("post_sole_z_m", .005), ("intent_swing", False)):
                altered = [dict(row) for row in rows]
                altered[0][name] = value
                write(altered)
                with self.subTest(field=name), self.assertRaises(ValueError):
                    read_pose(path, feet)


if __name__ == "__main__":
    unittest.main()
