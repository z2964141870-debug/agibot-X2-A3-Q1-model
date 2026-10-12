"""Check compatible segment geometry independently of IK convergence."""

import unittest
import numpy as np

from script.a3.smpl_corpus_qualify import configure_helpers
configure_helpers()
from script.a3.smpl_chain_retarget import ChainA3Retargeter, align_direction
from q1_retarget import BodyReference


class ChainContract(unittest.TestCase):
    def test_direction_alignment(self):
        for before, after in (([1,0,0],[0,0,-1]), ([1,0,0],[-1,0,0]), ([0,1,0],[0,1,0])):
            matrix = align_direction(before, after)
            np.testing.assert_allclose(matrix @ before, after, atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.)

    def test_robot_reference_identity(self):
        from scipy.spatial.transform import Rotation
        r = ChainA3Retargeter()
        quaternions = Rotation.from_matrix(r.rest_rotation).as_quat(scalar_first=True)
        source = BodyReference([0., 1/30], r.names, np.tile(r.rest_position,(2,1,1)),
            np.tile(quaternions,(2,1,1)), r.rest_position, quaternions, r.names, 'SYNTHETIC_A3_IDENTITY')
        positions, orientations = r.targets(source, 1.)
        np.testing.assert_allclose(positions, np.tile(r.rest_position,(2,1,1)), atol=1e-12)
        np.testing.assert_allclose(orientations, np.tile(r.rest_rotation,(2,1,1,1)), atol=1e-12)

    def test_zero_direction_rejected(self):
        with self.assertRaises(ValueError):
            align_direction([0,0,0], [1,0,0])

    def test_all_limb_lengths_are_robot_lengths(self):
        from scipy.spatial.transform import Rotation
        r = ChainA3Retargeter()
        human = r.rest_position.copy()
        for side in ('left','right'):
            s,e,w=[r.names.index(side+'_'+name) for name in ('shoulder','elbow','wrist')]
            human[e]=human[s]+[0,.3 if side=='left' else -.3,0]
            human[w]=human[e]+[0,.25 if side=='left' else -.25,0]
        quaternions=Rotation.from_matrix(r.rest_rotation).as_quat(scalar_first=True)
        source=BodyReference([0.,1/30],r.names,np.tile(human,(2,1,1)),np.tile(quaternions,(2,1,1)),
            human,quaternions,r.names,'SYNTHETIC_T_POSE')
        positions,_=r.targets(source,1.)
        for side in ('left','right'):
            for parent,child in (('hip','knee'),('knee','ankle'),('shoulder','elbow'),('elbow','wrist')):
                p,c=[r.names.index(side+'_'+name) for name in (parent,child)]
                np.testing.assert_allclose(np.linalg.norm(positions[:,c]-positions[:,p],axis=-1),
                    np.linalg.norm(r.rest_position[c]-r.rest_position[p]),atol=1e-12)


if __name__ == '__main__':
    unittest.main()
