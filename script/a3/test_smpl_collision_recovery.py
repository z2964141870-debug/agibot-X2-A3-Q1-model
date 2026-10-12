"""The plane constraint must drive a penetrating foot out of the floor."""

import unittest

import mink
import mujoco
import numpy as np

from script.a3 import test_smpl_collision_units as checks
from script.a3.smpl_collision_recovery import RecoveringFloorLimit


class CollisionRecoveryChecks(checks.CollisionUnitChecks):
    def test_penetrating_foot_must_recover_signed_gap(self):
        dt = 1 / 240
        self.configuration.update(np.array([0., -.06]))
        fixed = RecoveringFloorLimit(self.raw)
        old = self.fixed.compute_qp_inequalities(self.configuration, dt)
        new = fixed.compute_qp_inequalities(self.configuration, dt)
        self.assertTrue(np.all(old.G @ np.zeros(2) <= old.h))
        self.assertFalse(np.all(new.G @ np.zeros(2) <= new.h))
        task = mink.FrameTask('foot', 'body', position_cost=1., orientation_cost=0.)
        task.set_target(mink.SE3.from_translation(np.array([0., 0., -.04])))
        distances = []
        for _ in range(8):
            velocity = mink.solve_ik(self.configuration, [task], dt, 'quadprog',
                damping=.001, limits=[fixed])
            self.configuration.integrate_inplace(velocity, dt)
            distances.append(mujoco.mj_geomDistance(self.model, self.configuration.data,
                self.model.geom('sole').id, self.model.geom('floor').id, .15, None))
        self.assertTrue(np.all(np.diff(distances) > 0))
        self.assertGreaterEqual(distances[-1], .001 - 1e-8)


if __name__ == '__main__':
    unittest.main()
