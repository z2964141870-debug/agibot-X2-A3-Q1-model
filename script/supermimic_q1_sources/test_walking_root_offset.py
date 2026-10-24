"""P14 common-interval rejection and preservation of native-time target geometry."""

from pathlib import Path
import sys
import unittest

import numpy as np

from probe_walking_root_offset import constant_offset, translate_targets


class RootOffsetTests(unittest.TestCase):
    def test_registered_constant_covers_every_interval(self):
        result = constant_offset([-.7, -.6], [-.04, -.08], np.ones(2, dtype=bool))
        self.assertEqual(result["status"], "ACCEPTED_NECESSARY_BOUND_ONLY")
        self.assertAlmostEqual(result["root_delta_z_m"], -.085)
        self.assertEqual(constant_offset([-.2], [.1], np.ones(1, dtype=bool))["root_delta_z_m"], 0)

    def test_cap_empty_common_interval_and_invalid_frame_rejected(self):
        for lower, upper, valid in (([-.8], [-.116], [True]),
                                    ([-.05, -.8], [.1, -.07], [True, True]),
                                    ([-.5], [-.04], [False])):
            self.assertEqual(constant_offset(lower, upper, np.array(valid))["status"], "REJECTED")
        for lower, upper, valid in (([], [], []), ([np.nan], [0], [True]), ([0], [1], [1])):
            with self.assertRaises(ValueError):
                constant_offset(lower, upper, np.array(valid))

    def test_transform_preserves_root_shape_foot_anchors_and_relative_nonfeet(self):
        names = ["pelvis", "left_ankle", "right_ankle", "left_wrist"]
        roots = np.array([[0, 1, .6, 1, 0, 0, 0], [1, 2, .63, 1, 0, 0, 0]], float)
        positions = np.arange(24, dtype=float).reshape(2, 4, 3) / 100
        original_roots, original_positions = roots.copy(), positions.copy()
        moved, targets = translate_targets(roots, positions, names, -.08)
        np.testing.assert_array_equal(roots, original_roots)
        np.testing.assert_array_equal(positions, original_positions)
        np.testing.assert_array_equal(targets[:, [1, 2]], positions[:, [1, 2]])
        np.testing.assert_array_equal(moved[:, [0, 1, 3, 4, 5, 6]], roots[:, [0, 1, 3, 4, 5, 6]])
        np.testing.assert_allclose(np.diff(moved, axis=0), np.diff(roots, axis=0), atol=1e-15)
        np.testing.assert_allclose(targets[:, [0, 3]] - moved[:, None, :3],
                                   positions[:, [0, 3]] - roots[:, None, :3], atol=1e-15)

    def test_real_q1_constant_translation_moves_hips_and_keeps_joint_limits(self):
        import mujoco as mj

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supermimic_q1_phuma"))
        from retarget import ContactCorrectedAssets
        assets = ContactCorrectedAssets(Path("/opt/data/private/YuanQi_Q1"))
        model, data = assets.model, mj.MjData(assets.model)
        bids = [model.body(side + "_hip_pitch_link").id for side in ("left", "right")]
        rng = np.random.default_rng(2026102514)
        limits = model.jnt_range[assets.jids]
        for _ in range(32):
            data.qpos[assets.qadr] = rng.uniform(limits[:, 0], limits[:, 1])
            original = data.qpos.copy()
            mj.mj_kinematics(model, data)
            hips = data.xpos[bids].copy()
            data.qpos[2] -= .08
            mj.mj_kinematics(model, data)
            np.testing.assert_allclose(data.xpos[bids] - hips, np.tile([0, 0, -.08], (2, 1)), atol=1e-12)
            np.testing.assert_array_equal(data.qpos[7:], original[7:])


if __name__ == "__main__":
    unittest.main()
