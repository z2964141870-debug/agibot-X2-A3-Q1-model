"""Check planted-foot targets and root-correction determinism before IK."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.smpl_corpus_qualify import configure_helpers
configure_helpers()
from q1_retarget import BodyReference
from script.a3.smpl_contact_retarget import ContactA3Retargeter


class ContactTargets(unittest.TestCase):
    def source(self,r):
        count=300
        positions=np.tile(r.rest_position,(count,1,1))
        positions[:,:,0]+=np.arange(count)[:,None]*.001
        quaternions=Rotation.from_matrix(r.rest_rotation).as_quat(scalar_first=True)
        return BodyReference(np.arange(count)/30,r.names,positions,np.tile(quaternions,(count,1,1)),
            r.rest_position,quaternions,r.names,'SYNTHETIC_PLANTED_FEET')

    def test_contact_requires_original_toe_labels(self):
        r=ContactA3Retargeter()
        with self.assertRaises(ValueError):
            r.targets(self.source(r),1.)

    def test_planted_feet_and_geometry_are_grounded(self):
        r=ContactA3Retargeter()
        r.set_human_source(np.zeros((300,24,3)))
        target,rotations=r.targets(self.source(r),1.)
        self.assertTrue(r.contact_mask.all())
        for side,name in enumerate(('left_ankle','right_ankle')):
            ankle=r.names.index(name)
            np.testing.assert_allclose(target[:,ankle,:2],np.tile(target[0,ankle,:2],(300,1)),atol=1e-12)
            for frame in (0,150,299):
                minimum=(r.ankle_vertices[side]@rotations[frame,ankle].T+target[frame,ankle])[:,2].min()
                self.assertAlmostEqual(float(minimum),.001)
        np.testing.assert_allclose(target[:,0,:2],np.tile(target[0,0,:2],(300,1)),atol=1e-12)

    def test_target_reconstruction_is_repeatable(self):
        r=ContactA3Retargeter()
        r.set_human_source(np.zeros((300,24,3)))
        first=r.targets(self.source(r),1.)
        second=r.targets(self.source(r),1.)
        for a,b in zip(first,second,strict=True):
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main()
