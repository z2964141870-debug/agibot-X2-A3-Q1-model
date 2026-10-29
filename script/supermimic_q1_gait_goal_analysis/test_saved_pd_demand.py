"""Guard signs, force clipping, state timing and unsupported actuator contracts."""

from types import SimpleNamespace
import unittest
import numpy as np

from saved_pd_demand import model_contract, pd_demand, preceding_states


def model_fixture():
    m = SimpleNamespace(nu=22, na=0, actuator_trntype=np.zeros(22, int),
        actuator_trnid=np.column_stack((np.arange(22), np.full(22, -1))),
        actuator_gear=np.column_stack((np.ones(22), np.zeros((22, 5)))),
        actuator_gainprm=np.zeros((22, 10)), actuator_biasprm=np.zeros((22, 10)),
        actuator_dyntype=np.zeros(22, int), actuator_gaintype=np.zeros(22, int),
        actuator_biastype=np.ones(22, int), actuator_ctrllimited=np.zeros(22, bool),
        actuator_forcelimited=np.ones(22, bool), actuator_delay=np.zeros(22),
        actuator_history=np.zeros((22, 2), int), jnt_actfrclimited=np.zeros(22, bool),
        jnt_actgravcomp=np.zeros(22, bool), body_gravcomp=np.zeros(23), jnt_type=np.full(22, 3),
        jnt_qposadr=np.arange(22), jnt_dofadr=np.arange(22),
        actuator_forcerange=np.tile([-22., 22.], (22, 1)),
        opt=SimpleNamespace(disableflags=0, enableflags=0, timestep=.005),
        joint=lambda name: SimpleNamespace(id=int(name)))
    m.actuator_gainprm[:, 0] = 15.
    m.actuator_biasprm[:, 1], m.actuator_biasprm[:, 2] = -15., -1.
    return m


class SavedPdDemandTests(unittest.TestCase):
    def test_pd_signs_and_asymmetric_limits(self):
        request, applied, ratio = pd_demand([[2., -2.]], [[.5, -.5]], [[1., -1.]],
            [10., 10.], [2., 2.], [[-8., 12.], [-9., 15.]])
        np.testing.assert_array_equal(request, [[13., -13.]])
        np.testing.assert_array_equal(applied, [[12., -9.]])
        np.testing.assert_allclose(ratio, [[13 / 12, 13 / 9]])

    def test_pre_state_uses_initial_then_preceding_post_state(self):
        initial, post = np.full(57, -1.), np.stack([np.full(57, i) for i in (1, 2, 3)])
        pre = preceding_states(initial, post)
        np.testing.assert_array_equal(pre[:, 13], [-1., 1., 2.])
        request, _, _ = pd_demand(np.zeros((3, 1)), pre[:, 13:14], np.zeros((3, 1)),
            [10.], [0.], [[-100., 100.]])
        np.testing.assert_array_equal(request[:, 0], [10., -10., -20.])

    def test_residual_can_be_blocked_by_force_clamp(self):
        values = [pd_demand([[target]], [[0.]], [[0.]], [10.], [1.], [[-5., 5.]]) for target in (1., 1.1)]
        self.assertAlmostEqual(float(values[1][0][0, 0] - values[0][0][0, 0]), 1.)
        self.assertEqual(float(values[1][1][0, 0] - values[0][1][0, 0]), 0.)

    def test_model_contract_rejects_delay_nonunit_gear_and_wrong_bias(self):
        names = [str(i) for i in range(22)]
        np.testing.assert_array_equal(model_contract(model_fixture(), names)["kp"], 15.)
        for field, index, value in (("actuator_delay", 0, .005), ("actuator_gear", (0, 0), 2.),
                ("actuator_biasprm", (0, 1), -16.), ("actuator_dyntype", 0, 1),
                ("actuator_ctrllimited", 0, True), ("jnt_actfrclimited", 0, True)):
            model = model_fixture()
            getattr(model, field)[index] = value
            with self.assertRaises(ValueError):
                model_contract(model, names)

    def test_nonfinite_state_and_unsigned_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            preceding_states(np.full(57, np.nan), np.zeros((2, 57)))
        with self.assertRaises(ValueError):
            pd_demand([[1.]], [[0.]], [[0.]], [1.], [1.], [[1., 2.]])


if __name__ == "__main__":
    unittest.main()
