"""Reject root assignment before IK and verify continuous-base guards."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.smpl_corpus_qualify import configure_helpers
configure_helpers()
from script.a3.smpl_base_retarget import ContinuousBaseA3Retargeter
from q1_retarget import BodyReference


class ContinuousBaseChecks(unittest.TestCase):
    def test_floor_constraint_still_instantiated(self):
        retargeter = ContinuousBaseA3Retargeter()
        self.assertGreater(retargeter.floor_collision_pairs, 0)

    def test_lower_root_goal_cannot_teleport_feet_through_floor(self):
        class LowRootTarget(ContinuousBaseA3Retargeter):
            def targets(self, source, scale):
                positions, rotations = super().targets(source, scale)
                positions[:, 0, 2] -= .1
                return positions, rotations
        r = LowRootTarget()
        r.set_human_source(np.zeros((3, 24, 3)))
        quaternion = Rotation.from_matrix(r.rest_rotation).as_quat(scalar_first=True)
        source = BodyReference(np.arange(3) / 30, r.names,
            np.tile(r.rest_position, (3,1,1)), np.tile(quaternion, (3,1,1)),
            r.rest_position, quaternion, r.names, 'SYNTHETIC_LOWER_ROOT_PLANTED_FEET')
        with tempfile.TemporaryDirectory() as temporary:
            r.solve(source, 1., Path(temporary))
            with np.load(Path(temporary) / 'kinematic_trace.npz') as trace:
                self.assertGreaterEqual(float(trace['foot_geometry_min_z_m'].min()), -.005)
                self.assertGreater(float(trace['position_error_m'][:, 0].max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
