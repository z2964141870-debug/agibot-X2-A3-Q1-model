"""Source isolation, stance geometry and installed native collision scaling controls."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
from probe_walking_contact import fixed_selection
from probe_walking_retarget import select_training
from test_select_walking_holdout import alias, motion
from walking_contact_targets import contact_targets, native_collision_limit, runs, stance_mask


class SourceTests(unittest.TestCase):
    def test_reuses_p11_exact_selection_and_rejects_changed_roles(self):
        rows = [motion(f"{i:06d}") for i in range(1, 7)]
        aliases = dict(alias(f"{i:06d}", f"BMLrub/rub{i:03d}/circle_walk") for i in range(1, 7))
        selected, proof = select_training(rows, aliases, ["BMLrub/rub001"])
        plan = {"selected": [{"row": row, "lineage": info} for row, info in selected],
                "selection_proof": proof, "role": "TRAIN_DIAGNOSTIC_ONLY_NO_REFERENCE_ADMISSION"}
        self.assertEqual(fixed_selection(plan, rows, aliases, ["BMLrub/rub001"]), selected)
        changed = copy.deepcopy(plan)
        changed["selected"][0]["row"]["split"] = "test"
        with self.assertRaises(ValueError):
            fixed_selection(changed, rows, aliases, ["BMLrub/rub001"])
        changed = dict(plan, role="train")
        with self.assertRaises(ValueError):
            fixed_selection(changed, rows, aliases, ["BMLrub/rub001"])
        with self.assertRaises(ValueError):
            fixed_selection(plan, rows, aliases, ["BMLrub/rub002"])


class MaskTests(unittest.TestCase):
    def test_relative_height_translation_invariant_and_input_preserved(self):
        heights = np.c_[np.r_[np.zeros(6), np.ones(6) * .05],
                        np.r_[np.ones(6) * .05, np.zeros(6)]]
        before = heights.copy()
        mask = stance_mask(heights)
        np.testing.assert_array_equal(mask, stance_mask(heights + np.arange(12)[:, None] * .01))
        np.testing.assert_array_equal(heights, before)
        self.assertEqual(runs(mask[:, 0]), [(0, 6)])
        self.assertEqual(runs(mask[:, 1]), [(6, 12)])

    def test_bad_shape_and_nonfinite_rejected(self):
        for value in (np.zeros((8, 3)), np.zeros((4, 2)), np.full((8, 2), np.nan)):
            with self.assertRaises(ValueError):
                stance_mask(value)


class ServerGeometryTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(HERE.parent / "q1"))
        sys.path.insert(0, str(HERE.parent / "supermimic_q1_analysis"))

    def test_planted_geometry_and_continuous_swing_without_mutating_source(self):
        from scipy.spatial.transform import Rotation
        from analyze_phuma_support import pose_min_z

        count = 41
        positions = np.zeros((count, 2, 3))
        positions[:, :, 0] = np.arange(count)[:, None] * .003
        positions[:, :, 2] = .04
        positions[:, 1, 1] = -.15
        rotations = np.broadcast_to(Rotation.from_euler("z", .3).as_matrix(), (count, 2, 3, 3)).copy()
        mask = np.zeros((count, 2), bool)
        mask[:9] = True
        mask[30:] = True
        shape = {"kind": "sphere", "position": np.zeros(3), "rotation": np.eye(3), "size": np.array([.01, 0, 0])}
        before, before_r = positions.copy(), rotations.copy()
        corrected, corrected_r, events = contact_targets(positions, rotations, mask,
            [[shape], [shape]], -.02, [np.eye(3), np.eye(3)])
        np.testing.assert_array_equal(positions, before)
        np.testing.assert_array_equal(rotations, before_r)
        for side in range(2):
            for event in events[side]:
                start, end = event["start_frame"], event["end_frame_exclusive"]
                np.testing.assert_allclose(corrected[start:end, side], np.tile(corrected[start, side], (end - start, 1)))
                self.assertAlmostEqual(pose_min_z([shape], corrected[start, side], corrected_r[start, side]), -.02)
        self.assertLess(np.abs(np.diff(corrected[:, :, 0], axis=0)).max(), .01)
        np.testing.assert_allclose(np.linalg.det(corrected_r), 1, rtol=0, atol=1e-12)

    def test_native_collision_bound_matches_displacement_at_both_time_steps(self):
        import importlib.metadata
        import mink
        import mujoco

        self.assertEqual(importlib.metadata.version("mink"), "1.2.0")
        xml = """<mujoco><worldbody>
          <body name="left_wrist_yaw_link"><joint type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".03" mass="1"/></body>
          <body name="pelvis" pos=".1 0 0"><joint type="slide" axis="0 1 0"/>
          <geom type="sphere" size=".03" mass="1"/></body>
          </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        configuration = mink.Configuration(model)
        bounds = []
        for dt in (.005, .02):
            limit = native_collision_limit(mink, model, dt)
            inequality = limit.compute_qp_inequalities(configuration, dt)
            self.assertEqual(len(limit.geom_id_pairs), 1)
            self.assertTrue(np.isfinite(inequality.h).all())
            bounds.append(inequality.h)
        np.testing.assert_allclose(bounds, .85 * (.04 - .002), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
