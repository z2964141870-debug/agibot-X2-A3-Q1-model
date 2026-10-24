"""Analytic necessary reach bounds and their real Q1 FK preconditions."""

from pathlib import Path
import sys
import unittest

import numpy as np

from diagnose_walking_reach import leg_chain, reach_bounds


class ReachTests(unittest.TestCase):
    def test_vertical_lowering_and_unreachable_horizontal_target(self):
        bound = reach_bounds([[0, 0, 0], [2, 0, 0]], [[0, 0, 1], [0, 0, 0]], .5)
        np.testing.assert_allclose(bound["excess_m"], [.5, 1.5])
        np.testing.assert_allclose(bound["root_delta_lower_m"], [-1.5, 0])
        np.testing.assert_allclose(bound["root_delta_upper_m"], [-.5, 0])
        self.assertEqual(bound["horizontal_feasible"].tolist(), [True, False])

    def test_root_origin_translation_does_not_change_bounds(self):
        target, hip = np.array([[.1, .2, .3]]), np.array([[.3, .1, .8]])
        original = reach_bounds(target, hip, .6)
        shifted = reach_bounds(target + [4, -3, 2], hip + [4, -3, 2], .6)
        for key in original:
            np.testing.assert_allclose(original[key], shifted[key], atol=1e-14)

    def test_invalid_geometry_rejected(self):
        for target, hip, radius in (([[0, 0]], [[0, 0]], 1), ([[0, 0, 0]], [[0, 0, 0]], -1),
                                   ([[0, 0, np.nan]], [[0, 0, 0]], 1)):
            with self.assertRaises(ValueError):
                reach_bounds(target, hip, radius)

    def test_real_q1_bound_and_fixed_hip_for_joint_limit_samples(self):
        import mujoco as mj

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supermimic_q1_phuma"))
        from retarget import ContactCorrectedAssets
        assets = ContactCorrectedAssets(Path("/opt/data/private/YuanQi_Q1"))
        model, data = assets.model, mj.MjData(assets.model)
        chains = [leg_chain(model, side) for side in ("left", "right")]
        rng = np.random.default_rng(2026102513)
        limits = model.jnt_range[assets.jids]
        anchors = None
        for _ in range(256):
            data.qpos[assets.qadr] = rng.uniform(limits[:, 0], limits[:, 1])
            mj.mj_kinematics(model, data)
            current = np.array([data.xpos[hip].copy() for hip, _, _, _ in chains])
            if anchors is None:
                anchors = current
            np.testing.assert_allclose(current, anchors, atol=1e-12)
            for hip, ankle, radius, _ in chains:
                self.assertLessEqual(np.linalg.norm(data.xpos[ankle] - data.xpos[hip]), radius + 1e-12)


if __name__ == "__main__":
    unittest.main()
