import unittest

import numpy as np

from analyze_q1_substep_observation import foot_masks, pd_force, runs, stats, validate_clocks


class SavedSubstepAnalysisTests(unittest.TestCase):
    def native(self):
        q = np.arange(9*29,dtype=float).reshape(9,1,29)
        return dict(control_index=np.repeat(np.arange(2),4), substep_index=np.tile(np.arange(4),2),
                    pre_time=np.arange(8)[:,None]*.005,post_time=np.arange(1,9)[:,None]*.005,
                    pre_qpos=q[:-1],post_qpos=q[1:])

    def pd(self):
        gain=np.zeros((1,22,10));gain[:,:,0]=2
        bias=np.zeros((1,22,10));bias[:,:,1]=-2;bias[:,:,2]=-.5
        limits=np.tile([-22.,22.],(1,22,1))
        ctrl=np.arange(22,dtype=float)[None,None,:]
        length=np.zeros_like(ctrl);velocity=np.ones_like(ctrl)
        force=np.clip(2*ctrl-.5,-22,22)
        contract=dict(native_actuator_ids_in_q1_joint_order=list(range(21,-1,-1)),
            actuator_coefficients=dict(actuator_gainprm=gain.tolist(),actuator_biasprm=bias.tolist(),
                                       actuator_forcerange=limits.tolist()))
        return dict(pre_ctrl=ctrl,solve_actuator_length=length,solve_actuator_velocity=velocity,
                    solve_actuator_force=force),contract

    def test_runs_include_boundary_samples(self):
        self.assertEqual(runs(np.array([True,True,False,True])),[(0,2),(3,4)])
        self.assertEqual(runs(np.zeros(4,bool)),[])
        first,stop=runs(np.ones(12,bool))[0]
        self.assertAlmostEqual((stop-first-1)*.005,.055)
        self.assertAlmostEqual((stop-first)*.005,.060)

    def test_empty_condition_is_missing_not_zero(self):
        self.assertEqual(stats(np.array([3.]),np.array([False])),
                         dict(samples=0,mean=None,rms=None,abs_p95=None,abs_max=None))

    def test_native_clock_keeps_solve_before_post(self):
        np.testing.assert_allclose(validate_clocks(self.native(),2),np.arange(8)*.005)

    def test_missing_substep_rejected(self):
        value=self.native();value['substep_index'][5]=0
        with self.assertRaises(ValueError): validate_clocks(value,2)

    def test_discontinuous_native_state_rejected(self):
        value=self.native();value['pre_qpos']=value['pre_qpos'].copy();value['pre_qpos'][3,0,0]+=1
        with self.assertRaises(ValueError): validate_clocks(value,2)

    def test_actuator_mapping_and_force_limit(self):
        value,contract=self.pd();request,force,near,error=pd_force(value,contract)
        self.assertEqual(request[0,0],41.5);self.assertEqual(force[0,0],22.)
        self.assertTrue(near[0,0]);self.assertFalse(near[0,-1]);self.assertEqual(error,0)

    def test_mismatched_solve_force_rejected(self):
        value,contract=self.pd();value['solve_actuator_force'][0,0,0]+=1
        with self.assertRaises(ValueError): pd_force(value,contract)

    def test_low_force_requires_loaded_other_side_and_matching_height(self):
        force=np.array([[0.,0.],[5.,6.],[4.,8.],[6.,8.]])
        height=np.array([[0.,0.],[.001,0.],[.02,0.],[.001,0.]])
        masks=foot_masks(force,height,0,5.)
        np.testing.assert_array_equal(masks['low_other_loaded'],[False,True,True,False])
        np.testing.assert_array_equal(masks['low_near5'],[False,True,False,False])
        np.testing.assert_array_equal(masks['low_high10'],[False,False,True,False])


if __name__ == '__main__':
    unittest.main()
