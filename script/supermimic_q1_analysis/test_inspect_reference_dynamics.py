import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inspect_reference_dynamics import checked, derivatives, identity, inverse_with_check, summarize


class ReferenceDynamicsTests(unittest.TestCase):
    def test_derivatives_match_analytic_constant_and_cubic_velocity(self):
        times = np.arange(15) * .02
        velocity = np.stack([np.ones_like(times), times, times ** 3], axis=1)
        result = derivatives(velocity, .02)
        expected = np.stack([np.zeros(11), np.ones(11), 3 * times[2:-2] ** 2], axis=1)
        np.testing.assert_allclose(result["central5"], expected, atol=1e-13)
        expected[:, 2] += .02 ** 2
        np.testing.assert_allclose(result["central3"], expected, atol=1e-13)

    def test_bad_velocity_input_is_rejected(self):
        for velocity, dt in ((np.zeros((4, 28)), .02), (np.ones((8, 28)), 0),
                             (np.full((8, 28), np.nan), .02)):
            with self.assertRaises(ValueError):
                derivatives(velocity, dt)

    def test_proximity_empty_sets_and_effort_denominators(self):
        arrays = {"foot_gap_m": np.array([[0., .04], [.02, .04]]),
                  "foot_tangent_speed_m_s": np.array([[.1, 9.], [.02, 9.]]),
                  "active_foot_floor_contacts": np.array([1, 0])}
        for label in ("central3", "central5"):
            force = np.zeros((2, 28))
            force[:, 2] = 10
            force[0, 6] = 2
            arrays[label + "_inverse"] = force
            arrays[label + "_qacc"] = np.zeros((2, 28))
            arrays[label + "_decomposition_error"] = np.zeros(2)
        result = summarize(arrays, np.ones(22), 10)
        self.assertEqual(result["foot_floor_contact_fraction"], .5)
        self.assertEqual(result["proximity"]["0.001"]["point_tangent_speed_m_s"]["count"], 1)
        self.assertEqual(result["proximity"]["0.03"]["point_tangent_speed_m_s"]["count"], 2)
        self.assertEqual(result["inverse"]["central3"]["frames_any_active_effort_over_cap_fraction"], .5)
        self.assertEqual(result["inverse"]["central3"]["active_effort_over_cap"]["count"], 44)
        arrays["foot_gap_m"] += 1
        empty = summarize(arrays, np.ones(22), 10)["proximity"]["0.03"]
        self.assertIsNone(empty["point_speed_above_0_05_fraction"])
        self.assertIsNone(empty["point_tangent_speed_m_s"]["p95"])

    def test_input_changes_and_outside_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root / "input"
            path.write_bytes(b"fixture")
            expected = identity(path)
            self.assertEqual(checked(root, expected), path)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                checked(root, expected)
            with self.assertRaises(ValueError):
                checked(root / "subproject", identity(path))

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo available on training server")
    def test_inverse_static_weight_freefall_and_decomposition(self):
        import mujoco as mj
        model = mj.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 -9.81"/><worldbody><body>'
            '<joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1" mass="2"/>'
            '</body></worldbody></mujoco>')
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        force, error = inverse_with_check(model, data, np.array([0.]), mj)
        np.testing.assert_allclose(force, [19.62], atol=1e-12)
        self.assertLess(error, 1e-12)
        force, _ = inverse_with_check(model, data, np.array([-9.81]), mj)
        np.testing.assert_allclose(force, [0.], atol=1e-12)
        self.assertEqual(data.time, 0.)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo available on training server")
    def test_soft_contact_decomposition_and_no_time_advance(self):
        import mujoco as mj
        model = mj.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
            '<body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="2"/>'
            '</body></worldbody></mujoco>')
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        force, error = inverse_with_check(model, data, np.zeros(6), mj)
        self.assertLess(error, 1e-9)
        self.assertTrue(np.isfinite(force).all())
        self.assertEqual(data.time, 0.)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo available on training server")
    def test_armature_inertia_is_included_in_decomposition(self):
        import mujoco as mj
        model = mj.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 -9.81"/><worldbody><body>'
            '<joint type="slide" axis="0 0 1" armature=".5"/>'
            '<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>')
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        force, error = inverse_with_check(model, data, np.array([1.]), mj)
        np.testing.assert_allclose(force, [22.12], atol=1e-12)
        self.assertLess(error, 1e-12)


if __name__ == "__main__":
    unittest.main()
