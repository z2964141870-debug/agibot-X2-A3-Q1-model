"""Coordinate, memory and no-integration contracts in the real MuJoCo runtime."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from native_state_cpu import KinematicState


class Translator:
    def to_canonical(self, value):
        result = torch.zeros(1, 29)
        result[:, :22] = value
        return result

    to_canonical_delta = to_canonical


def fixture():
    child = '<body name="b1" pos="1 0 0"><joint name="j1" axis="0 0 1" pos="-.2 0 0" ref=".3"/>'
    child += '<inertial pos=".05 .1 0" mass="1" diaginertia=".1 .1 .1"/>'
    for i in range(2, 23):
        child += f'<body name="b{i}" pos=".01 0 0"><joint name="j{i}" axis="0 0 1"/><geom size=".01"/></body>'
    child += '</body>'
    xml = '<mujoco><compiler angle="radian"/><worldbody><body name="root"><freejoint/>'
    xml += '<inertial pos=".1 .2 .3" mass="2" diaginertia="1 1 1"/>'+child+'</body></worldbody><actuator>'
    xml += ''.join(f'<motor joint="j{i}"/>' for i in range(1, 23))+'</actuator></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    robot = SimpleNamespace(model=model, parents=model.body_parentid.tolist(),
        qindexes=(model.jnt_qposadr-7).tolist(), **{k:torch.tensor(v, dtype=torch.float32)
        for k, v in dict(body_pos=model.body_pos, body_quat=model.body_quat,
            joint_axis=model.jnt_axis, joint_anchor=model.jnt_pos, default_q=model.qpos0[7:]).items()})
    spec = SimpleNamespace(tracked_bodies=('root',)+tuple(f'b{i}' for i in range(1, 14)),
                           translator=Translator())
    value = torch.zeros(1, 79)
    value[:, :29] = torch.tensor(model.qpos0, dtype=torch.float32)
    return robot, spec, value


class NativeStateCpuTests(unittest.TestCase):
    def setUp(self):
        self.robot, self.spec, self.value = fixture()
        self.convert = KinematicState(self.robot, self.spec)

    def test_world_root_velocity_at_rotated_link_origin(self):
        self.value[:, :3] = torch.tensor([.3, -.4, 1.2])
        self.value[:, 3:7] = torch.tensor([.5, .5, .5, .5])
        self.value[:, 29:35] = torch.tensor([.1, .2, .3, 1.2, -.7, .9])
        out = self.convert(self.value)
        np.testing.assert_allclose(out[0, 156:159], [.1, .2, .3], atol=2e-6)
        np.testing.assert_allclose(out[0, 198:201], [1.2, -.7, .9], atol=2e-6)
        expected = np.asarray([.1, .2, .3])+np.cross([1.2, -.7, .9], [0., 1., 0.])
        np.testing.assert_allclose(out[0, 159:162], expected, atol=2e-6)

    def test_nonzero_joint_reference_anchor_and_com_are_distinct(self):
        self.value[:, :3] = torch.tensor([.3, -.4, 1.2])
        self.value[:, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
        self.value[:, 7] += np.pi/2
        self.value[:, 29:35] = torch.tensor([.1, .2, .3, 0., 0., 2.])
        self.value[:, 35] = 3.
        out = self.convert(self.value)
        np.testing.assert_allclose(out[0, 61:64], [.1, .4, 1.2], atol=2e-6)
        np.testing.assert_allclose(out[0, 159:162], [-1.5, -.8, .3], atol=2e-6)
        np.testing.assert_allclose(out[0, 201:204], [0., 0., 5.], atol=2e-6)
        com = np.empty(6)
        mujoco.mj_objectVelocity(self.robot.model, self.convert.data, mujoco.mjtObj.mjOBJ_BODY, 2, com, 0)
        self.assertGreater(np.max(np.abs(com[3:]-out[0, 159:162].numpy())), .1)

    def test_sign_scale_missing_slots_and_result_ownership(self):
        self.value[:, 3:7] = torch.tensor([-2., 0., 0., 0.])
        self.value[:, 57:] = torch.linspace(-1., 1., 22)
        out = self.convert(self.value)
        self.assertEqual(out.shape, (1, 269))
        np.testing.assert_array_equal(out[0, 100:104], [-1., 0., 0., 0.])
        for start in (0, 29, 240):
            self.assertTrue(torch.equal(out[:, start+22:start+29], torch.zeros(1, 7)))
        self.assertTrue(torch.equal(out[:, 240:262], self.value[:, 57:79]))
        saved = out.clone()
        self.value[:, 0] = 4.
        self.convert(self.value)
        self.assertTrue(torch.equal(out, saved))

    def test_no_dynamics_no_input_or_model_mutation(self):
        fields = ('qpos0', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass',
                  'body_inertia', 'jnt_axis', 'jnt_pos', 'jnt_qposadr', 'jnt_type')
        before = {k:getattr(self.robot.model, k).copy() for k in fields}
        state = self.value.clone()
        self.convert.data.time = .75
        dynamics = {k:getattr(self.convert.data, k).copy() for k in
                    ('qacc', 'ctrl', 'act', 'qfrc_applied', 'xfrc_applied')}
        with ExitStack() as stack:
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse'):
                stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No dynamics')))
            self.convert(self.value)
        self.assertEqual(self.convert.data.time, .75)
        self.assertTrue(torch.equal(state, self.value))
        for k, v in before.items():
            np.testing.assert_array_equal(v, getattr(self.robot.model, k))
        for k, v in dynamics.items():
            np.testing.assert_array_equal(v, getattr(self.convert.data, k))

    def test_rejects_unadmitted_contracts_and_buffer_changes(self):
        for value in (self.value.double(), self.value.repeat(2, 1), self.value[0],
                      self.value.clone().requires_grad_(), torch.full((1, 79), float('nan'))):
            with self.assertRaises(ValueError):
                self.convert(value)
        with self.assertRaises(ValueError):
            self.convert(torch.zeros(1, 79))
        self.robot.default_q[0] += .1
        with self.assertRaises(ValueError):
            KinematicState(self.robot, self.spec)


if __name__ == '__main__':
    unittest.main()
