"""A physical plane approach distinguishes velocity from displacement bounds."""

import unittest

import mink
import mujoco
import numpy as np

from script.a3.smpl_corpus_qualify import configure_helpers
configure_helpers()
from script.a3.smpl_collision_units import DisplacementCollisionLimit


class CollisionUnitChecks(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 .1"/>
          <body name="carrier">
            <joint name="horizontal" type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".005" contype="0" conaffinity="0"/>
            <body name="foot" pos="0 0 .06">
              <joint name="height" type="slide" axis="0 0 1"/>
              <geom name="sole" type="sphere" size=".01"/>
            </body>
          </body>
        </worldbody></mujoco>''')
        self.configuration = mink.Configuration(self.model)
        self.raw = mink.CollisionAvoidanceLimit(self.model,
            [(['sole'], ['floor'])], gain=.85,
            minimum_distance_from_collisions=.001,
            collision_detection_distance=.15, bound_relaxation=0.)
        self.assertEqual(len(self.raw.geom_id_pairs), 1)
        self.fixed = DisplacementCollisionLimit(self.raw)

    def test_installed_bound_allows_sixty_mm_approach_to_fifty_mm_gap(self):
        dt = 1 / 240
        bad_step = np.array([0., -.06])
        raw = self.raw.compute_qp_inequalities(self.configuration, dt)
        fixed = self.fixed.compute_qp_inequalities(self.configuration, dt)
        self.assertTrue(np.all(raw.G @ bad_step <= raw.h))
        self.assertFalse(np.all(fixed.G @ bad_step <= fixed.h))
        self.configuration.integrate_inplace(bad_step / dt, dt)
        sole = self.model.geom('sole').id
        floor = self.model.geom('floor').id
        self.assertLess(mujoco.mj_geomDistance(self.model,
            self.configuration.data, sole, floor, .15, None), 0.)

    def test_displacement_budget_does_not_change_with_timestep(self):
        for dt in (1 / 30, 1 / 120, 1 / 240):
            value = self.fixed.compute_qp_inequalities(self.configuration, dt)
            np.testing.assert_allclose(value.h, [.85 * (.05 - .001)])

    def test_solved_step_respects_actual_plane_gap(self):
        task = mink.FrameTask('foot', 'body', position_cost=1., orientation_cost=0.)
        task.set_target(mink.SE3.from_translation(np.array([0., 0., -.04])))
        dt = 1 / 240
        velocity = mink.solve_ik(self.configuration, [task], dt, 'quadprog',
            damping=.001, limits=[self.fixed])
        constraint = self.fixed.compute_qp_inequalities(self.configuration, dt)
        self.assertTrue(np.all(constraint.G @ (velocity * dt) <= constraint.h + 1e-12))
        self.configuration.integrate_inplace(velocity, dt)
        distance = mujoco.mj_geomDistance(self.model, self.configuration.data,
            self.model.geom('sole').id, self.model.geom('floor').id, .15, None)
        self.assertGreaterEqual(distance, .001)


if __name__ == '__main__':
    unittest.main()
