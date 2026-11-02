"""Focused CPU tests for sparse mappings and missing-DOF numerical isolation."""

from dataclasses import replace
import unittest

import torch

from canonical import Q1_MISSING, SparseJointMap, morphology_with_presence
from control import delayed_target, motor_control, parameters, torque_bounds


def names():
    legs = tuple(f"{s}_{j}_joint" for s in ("left", "right") for j in
                 ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
    arms = tuple(f"{s}_{j}_joint" for s in ("left", "right") for j in
                 ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw"))
    canonical = legs + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + arms
    native = legs + ("waist_roll_joint", "waist_yaw_joint") + tuple(
        n for n in arms if n not in Q1_MISSING)
    return canonical, native


class SparseTests(unittest.TestCase):
    def setUp(self):
        self.map = SparseJointMap.q1(*names())

    def test_native_roundtrip_and_waist_order(self):
        native = torch.arange(44, dtype=torch.float64).reshape(2, 22)
        canon = self.map.canonical(native)
        torch.testing.assert_close(self.map.native(canon), native)
        self.assertEqual(canon[0, 12].item(), 13)
        self.assertEqual(canon[0, 13].item(), 12)
        self.assertEqual(canon[:, ~self.map.mask(canon)].count_nonzero().item(), 0)

    def test_missing_outputs_and_nan_cannot_drive_real_motor(self):
        canon = torch.arange(29, dtype=torch.float64)
        expected = self.map.native(canon)
        canon[~self.map.mask(canon)] = float("nan")
        torch.testing.assert_close(self.map.native(canon), expected)
        self.assertTrue(torch.isfinite(self.map.project(canon)).all())

    def test_missing_output_gradients_are_zero(self):
        action = torch.ones(3, 29, requires_grad=True)
        self.map.native(action, delta=True).square().sum().backward()
        self.assertEqual(action.grad[:, ~self.map.mask(action)].count_nonzero().item(), 0)
        self.assertTrue((action.grad[:, self.map.mask(action)] == 2).all())

    def test_signed_offset_roundtrip_and_limits(self):
        mapping = SparseJointMap(("a", "b", "missing"), ("b", "a", "extra"),
                                 (1, 0, None), (-1., 1., 1.), (0.5, -0.2, 0.))
        q = torch.tensor([0.7, 1.2, 9.])
        canon = mapping.canonical(q)
        torch.testing.assert_close(mapping.native(canon, q), q)
        torch.testing.assert_close(mapping.native(mapping.canonical(q, delta=True),
                                                  delta=True), torch.tensor([.7, 1.2, 0.]))
        limits = mapping.limits(torch.tensor([[-1., 1.], [-2., 2.], [0., 10.]]))
        torch.testing.assert_close(limits, torch.tensor([[-1.5, 2.5], [-.8, 1.2], [0., 0.]]))

    def test_bad_mapping_and_unexpected_missing_rejected(self):
        for bad in (-1, 22, True):
            with self.assertRaises(ValueError):
                replace(self.map, native_for_slot=(bad,) + self.map.native_for_slot[1:])
        with self.assertRaises(ValueError):
            replace(self.map, native_for_slot=(1,) + self.map.native_for_slot[1:])
        with self.assertRaises(ValueError):
            SparseJointMap.q1(names()[0], names()[1][:-1] + ("fake_joint",))

    def test_all_joint_channels_projected_without_touching_body_data(self):
        state = torch.ones(2, 240)
        out = self.map.state(state)
        torch.testing.assert_close(out[..., 58:], state[..., 58:])
        self.assertEqual(out[..., :58].sum().item(), 2 * 44)
        obs = self.map.proprio(torch.ones(93))
        self.assertEqual(obs.sum().item(), 22 * 3 + 6)
        context = self.map.context(torch.ones(33))
        self.assertEqual(context.sum().item(), 26)
        goal = self.map.goal(torch.ones(670))
        self.assertEqual(goal.sum().item(), 10 * (44 + 9))
        with self.assertRaises(ValueError):
            self.map.state(torch.ones(239))

    def test_real_joint_rmse_not_diluted_by_seven_padding_slots(self):
        predicted = self.map.project(torch.ones(29))
        self.assertEqual(self.map.joint_rmse(predicted, torch.zeros(29)).item(), 1)

    def test_descriptor_ignores_nan_absence_and_declares_presence(self):
        terms = torch.ones(2, 4, 29)
        presence = torch.ones(2, 29, dtype=torch.bool)
        presence[1] = self.map.mask(terms)
        terms[1, :, ~presence[1]] = float("nan")
        vec, bounds = morphology_with_presence(terms, torch.ones(2, 9), presence)
        self.assertEqual(vec.shape, (2, 154))
        self.assertTrue(torch.isfinite(vec).all())
        torch.testing.assert_close(vec[:, -29:], presence.float())
        self.assertEqual(vec[:, :125].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(bounds["min"]).all())

    def test_descriptor_bounds_fit_present_robots_only(self):
        terms = torch.full((3, 4, 29), 2.)
        terms[1] = 8
        presence = torch.ones(3, 29, dtype=torch.bool)
        presence[2] = self.map.mask(terms)
        terms[2, :, ~presence[2]] = 0
        vec, _ = morphology_with_presence(terms, torch.ones(3, 9), presence)
        self.assertEqual(vec[0, 14].item(), -1)
        self.assertEqual(vec[1, 14].item(), 1)
        self.assertEqual(vec[2, 14].item(), 0)
        terms[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            morphology_with_presence(terms, torch.ones(3, 9), presence)

    def test_delay_uses_previous_target_until_quantized_substep(self):
        current, previous = torch.ones(1, 22), torch.zeros(1, 22)
        delay = torch.tensor([[.006]])
        for i in range(10):
            self.assertEqual(delayed_target(current, previous, delay, i).sum().item(),
                             0 if i < 3 else 22)
        for value in (0, .003, float("nan")):
            with self.assertRaises(ValueError):
                delayed_target(current, previous, delay, 0, value)

    def test_torque_curve_retains_braking_at_high_speed(self):
        v = torch.tensor([-20., 0., 20.])
        peak, knee, cutoff = torch.tensor(34.), torch.tensor(7.5), torch.tensor(17.)
        low, high = torque_bounds(v, peak, knee, cutoff)
        torch.testing.assert_close(low, torch.tensor([0., -34., -34.]))
        torch.testing.assert_close(high, torch.tensor([34., 34., 0.]))

    def test_motor_limits_and_gains_use_native_names(self):
        kp, kd, peak, knee, cutoff = parameters(names()[1])
        self.assertEqual(kp[12].item(), 80)
        self.assertEqual(kp[4].item(), 20)
        self.assertEqual(kp[5].item(), 10)
        effort = motor_control(torch.ones(22) * 100, torch.zeros(22), torch.zeros(22),
                               kp, kd, peak, knee, cutoff, torch.ones(22),
                               torch.tensor([[-36., 36.]]).expand(22, 2))
        torch.testing.assert_close(effort, peak)
        with self.assertRaises(ValueError):
            parameters(("wrist_pitch_joint",))


if __name__ == "__main__":
    unittest.main()
