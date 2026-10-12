"""A finite rotational step must check the new lowest corner of the sole."""

import unittest

import mujoco
import numpy as np

from script.a3.smpl_corpus_qualify import configure_helpers
configure_helpers()
from script.a3.smpl_checked_integration import FloorCheckedConfiguration


class CheckedIntegrationTests(unittest.TestCase):
    def configuration(self, height):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="foot" pos="0 0 .021">
            <joint name="tilt" type="hinge" axis="0 1 0"/>
            <geom name="sole" type="box" size=".1 .05 .02"/>
          </body></worldbody></mujoco>''')
        model.body_pos[1, 2] = height
        vertices = np.array([[x, y, z] for x in (-.1, .1)
            for y in (-.05, .05) for z in (-.02, .02)])
        return FloorCheckedConfiguration(model, np.zeros(1), [[(0, vertices)]])

    def test_rotation_crossing_floor_is_reduced_and_recorded(self):
        configuration = self.configuration(.021)
        configuration.integrate_inplace(np.array([24.]), 1 / 240)
        row = configuration.integration_records[-1]
        self.assertLess(row[1], 0.)
        self.assertGreaterEqual(configuration.minimum_foot_z(), 0.)
        self.assertLess(row[3], 1.)
        self.assertGreater(row[3], 0.)

    def test_clear_step_keeps_full_velocity_budget(self):
        configuration = self.configuration(.2)
        configuration.integrate_inplace(np.array([24.]), 1 / 240)
        np.testing.assert_allclose(configuration.data.qpos, [.1])
        self.assertEqual(configuration.integration_records[-1][3], 1.)

    def test_initial_penetration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.configuration(.01)


if __name__ == '__main__':
    unittest.main()
