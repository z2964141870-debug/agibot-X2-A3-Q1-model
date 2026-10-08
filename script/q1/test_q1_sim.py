"""Check indexing, reference input validation and simulation control boundaries."""

from pathlib import Path
import unittest

import mujoco
import numpy as np

from q1_sim import JointReference, Q1Sim


CONFIG = Path(__file__).with_name("sim_config.json")


class ControlContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Q1Sim(CONFIG, supported=True)

    def setUp(self):
        self.sim.reset()

    def test_every_motor_and_sensor_maps_to_named_joint(self):
        sim = self.sim
        for i, name in enumerate(sim.names):
            self.assertEqual(sim.model.actuator_trnid[sim.motor_indices[i], 0], sim.joint_ids[i])
            np.testing.assert_allclose(sim.sensor("jointpos_" + name),
                                       sim.data.qpos[sim.q_indices[i]:sim.q_indices[i] + 1])
        self.assertEqual(len(set(sim.q_indices)), 22)
        self.assertEqual(len(set(sim.dq_indices)), 22)

    def test_invalid_targets_do_not_advance_or_change_controls(self):
        sim = self.sim
        bad = sim.neutral.copy()
        bad[0] = sim.upper[0] + 0.1
        for values in (bad, np.zeros(21), np.full(22, np.nan)):
            with self.assertRaises(ValueError):
                sim.advance(values)
            self.assertEqual(sim.data.time, 0)
            np.testing.assert_array_equal(sim.data.ctrl, np.zeros(22))

    def test_torque_and_action_limits(self):
        sim = self.sim
        target = sim.decode_action(np.full(22, 100.))
        self.assertTrue(np.all(target <= sim.upper))
        self.assertTrue(np.all(target >= sim.lower))
        sim.advance(target)
        self.assertEqual(sim.torque_samples, sim.substeps * 22)
        self.assertTrue(np.all(sim.last_torque >= sim.torque_bounds[:, 0]))
        self.assertTrue(np.all(sim.last_torque <= sim.torque_bounds[:, 1]))

    def test_timing_and_reset(self):
        sim = self.sim
        initial = sim.data.qpos.copy()
        sim.advance(sim.neutral)
        self.assertAlmostEqual(sim.data.time, 0.02)
        sim.reset()
        np.testing.assert_array_equal(sim.data.qpos, initial)
        self.assertEqual(sim.torque_samples, 0)
        self.assertEqual(sim.data.time, 0)

    def test_projected_gravity_and_world_quaternion(self):
        obs = self.sim.observe()
        np.testing.assert_allclose(obs["orientation_wxyz_world"], [1, 0, 0, 0])
        np.testing.assert_allclose(obs["projected_gravity_body"], [0, 0, -1])

    def test_source_has_free_base_and_support_is_explicit(self):
        free = Q1Sim(CONFIG)
        self.assertEqual(free.model.neq, 0)
        self.assertEqual(self.sim.model.neq, 1)
        self.assertEqual(free.model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)


class ReferenceTests(unittest.TestCase):
    def test_shuffled_columns_are_reordered_by_name(self):
        ref = JointReference([0, 1], [[10, 1], [20, 3]], ["b", "a"], ["a", "b"])
        pos, vel = ref.sample(0.5)
        np.testing.assert_allclose(pos, [2, 15])
        np.testing.assert_allclose(vel, [2, 10])
        pos, vel = ref.sample(2)
        np.testing.assert_allclose(pos, [3, 20])
        np.testing.assert_array_equal(vel, [0, 0])

    def test_ambiguous_names_and_times_are_rejected(self):
        for names in (["a", "a"], ["a", "c"], ["a"]):
            with self.assertRaises(ValueError):
                JointReference([0, 1], np.zeros((2, len(names))), names, ["a", "b"])
        for times in ([0, 0], [0, -1], [1, 2], [0, np.nan]):
            with self.assertRaises(ValueError):
                JointReference(times, np.zeros((2, 2)), ["a", "b"], ["a", "b"])

    def test_bad_shapes_nonfinite_and_sampling_times_are_rejected(self):
        for values in (np.zeros((2, 1)), np.full((2, 2), np.inf)):
            with self.assertRaises(ValueError):
                JointReference([0, 1], values, ["a", "b"], ["a", "b"])
        ref = JointReference([0, 1], np.zeros((2, 2)), ["a", "b"], ["a", "b"])
        for time in (-1, np.nan):
            with self.assertRaises(ValueError):
                ref.sample(time)


if __name__ == "__main__":
    unittest.main()
