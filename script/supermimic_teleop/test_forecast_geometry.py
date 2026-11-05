"""Analytic, sign, frame-equivariance and history-only encoding checks."""

import math
import unittest

import numpy as np

from causal_tail import multiply, rotation_quaternion, rotation_vector
from forecast_geometry import (FEATURE_DIM, TARGET_DIM, decode_targets, encode_history,
    encode_targets, error_by_horizon, exp_rotation, inverse, log_rotation, product,
    rotation_columns, yaw_matrix)


def motion(frames=40):
    times = np.arange(frames)*.02
    positions = np.zeros((frames, 24, 3))
    positions[:, :, 0] = times[:, None] + np.arange(24)[None]*.03
    positions[:, :, 1] = np.sin(times[:, None]*2+np.arange(24)[None]*.1)*.1
    positions[:, :, 2] = .8+np.arange(24)[None]*.02
    vectors = np.stack((times*.2, times*.3, times*.4), axis=-1)
    quaternions = np.repeat(exp_rotation(vectors)[:, None], 24, axis=1)
    return positions, quaternions


class ForecastGeometryTests(unittest.TestCase):
    def test_vectorized_quaternion_math_matches_existing_independent_scalar_paths(self):
        rng = np.random.default_rng(14)
        vectors = rng.normal(size=(100, 3))*.3
        quaternions = exp_rotation(vectors)
        for vector, quaternion in zip(vectors, quaternions):
            np.testing.assert_allclose(quaternion, rotation_quaternion(tuple(vector)), atol=2e-15)
            np.testing.assert_allclose(log_rotation(quaternion), rotation_vector(tuple(quaternion)), atol=2e-15)
        right = exp_rotation(rng.normal(size=(100, 3))*.2)
        expected = np.array([multiply(a, b) for a, b in zip(quaternions, right)])
        np.testing.assert_allclose(product(quaternions, right), expected, atol=2e-15)
        np.testing.assert_allclose(product(quaternions, inverse(quaternions)),
                                   np.tile([1., 0., 0., 0.], (100, 1)), atol=2e-15)

    def test_roundtrip_root_body_and_rotations_at_original_time_slots(self):
        positions, quaternions = motion()
        history = encode_history(positions[:25], quaternions[:25])
        targets = encode_targets(positions[25:35], quaternions[25:35], history.anchor)
        decoded = decode_targets(targets, history.anchor)
        self.assertEqual(history.features.shape, (25, FEATURE_DIM))
        self.assertEqual(targets.shape, (10, TARGET_DIM))
        np.testing.assert_allclose(decoded.positions, positions[25:35], atol=3e-15)
        np.testing.assert_allclose(log_rotation(product(inverse(decoded.quaternions), quaternions[25:35])), 0., atol=3e-15)
        self.assertEqual(decoded.target_offsets_ns, tuple(range(20_000_000, 200_000_001, 20_000_000)))
        self.assertFalse(decoded.bone_contact_constraints_applied)
        errors = error_by_horizon(decoded, positions[25:35], quaternions[25:35])
        self.assertLess(max(errors["position_rmse_m"]), 3e-15)
        self.assertLess(max(errors["orientation_geodesic_rmse_rad"]), 3e-15)
        self.assertFalse(errors["temporal_realignment_applied"])
        self.assertIsNone(errors["measured_robot_latency_ns"])

    def test_translation_and_common_world_yaw_do_not_change_features_or_labels(self):
        positions, quaternions = motion()
        rotation = yaw_matrix(2.1)
        shift = np.array([30., -20., 2.])
        changed_positions = positions @ rotation.T + shift
        changed_quaternions = product(np.array(rotation_quaternion((0., 0., 2.1))), quaternions)
        a = encode_history(positions[:25], quaternions[:25])
        b = encode_history(changed_positions[:25], changed_quaternions[:25])
        np.testing.assert_allclose(a.features, b.features, atol=2e-14)
        first = encode_targets(positions[25:35], quaternions[25:35], a.anchor)
        second = encode_targets(changed_positions[25:35], changed_quaternions[25:35], b.anchor)
        np.testing.assert_allclose(first, second, atol=2e-14)
        decoded = decode_targets(first, b.anchor)
        np.testing.assert_allclose(decoded.positions, changed_positions[25:35], atol=2e-14)

    def test_quaternion_antipodes_and_near_pi_are_equivalent(self):
        positions, quaternions = motion()
        changed = quaternions.copy()
        changed[::2, ::2] *= -1
        a = encode_history(positions[:25], quaternions[:25])
        b = encode_history(positions[:25], changed[:25])
        np.testing.assert_array_equal(a.features, b.features)
        first = encode_targets(positions[25:35], quaternions[25:35], a.anchor)
        second = encode_targets(positions[25:35], changed[25:35], b.anchor)
        np.testing.assert_array_equal(first, second)
        for angle in (1e-13, math.pi-1e-9, math.pi+1e-9):
            q = exp_rotation(np.array([0., angle, 0.]))
            restored = exp_rotation(log_rotation(q))
            self.assertGreater(abs(float(np.dot(q, restored))), 1-2e-15)
        # Known 90-degree z rotation: its first two matrix columns are +y and -x.
        np.testing.assert_allclose(rotation_columns(exp_rotation(np.array([0., 0., math.pi/2]))),
                                   [0., 1., 0., -1., 0., 0.], atol=3e-16)

    def test_zero_residual_is_exact_hold_without_inventing_future_motion(self):
        positions, quaternions = motion()
        history = encode_history(positions[:25], quaternions[:25])
        decoded = decode_targets(np.zeros((10, TARGET_DIM)), history.anchor)
        np.testing.assert_allclose(decoded.positions, np.repeat(positions[24:25], 10, axis=0), atol=3e-15)
        np.testing.assert_allclose(decoded.linear_velocity_w, 0., atol=3e-13)
        np.testing.assert_allclose(decoded.angular_velocity_w, 0., atol=3e-13)
        self.assertFalse(history.features.flags.writeable)
        self.assertFalse(decoded.positions.flags.writeable)

    def test_input_anchor_cannot_depend_on_modified_future(self):
        positions, quaternions = motion()
        first = encode_history(positions[:25], quaternions[:25])
        positions[25:] = 1e6
        quaternions[25:] = exp_rotation(np.array([2., 0., 0.]))
        second = encode_history(positions[:25], quaternions[:25])
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.anchor.root_position, second.anchor.root_position)
        np.testing.assert_array_equal(first.anchor.last_quaternions, second.anchor.last_quaternions)

    def test_world_linear_and_angular_velocities_use_same_backward_interval(self):
        positions, quaternions = motion()
        times = np.arange(40)*.02
        speed = np.array([1., -2., .4])
        positions[:] = times[:, None, None]*speed+np.arange(24)[None, :, None]*.03
        omega = np.array([.2, -.3, .7])
        quaternions[:] = exp_rotation(times[:, None]*omega)[:, None]
        history = encode_history(positions[:25], quaternions[:25])
        decoded = decode_targets(encode_targets(positions[25:35], quaternions[25:35], history.anchor), history.anchor)
        np.testing.assert_allclose(decoded.linear_velocity_w, np.broadcast_to(speed, (10, 24, 3)), atol=1e-13)
        np.testing.assert_allclose(decoded.angular_velocity_w, np.broadcast_to(omega, (10, 24, 3)), atol=1e-13)
        expected = np.diff(positions[24:35], axis=0)/.02
        np.testing.assert_allclose(decoded.linear_velocity_w, expected, atol=1e-13)

    def test_invalid_quaternions_heading_and_horizon_fail_explicitly(self):
        positions, quaternions = motion()
        with self.assertRaisesRegex(ValueError, "unit"):
            encode_history(positions[:25], np.zeros_like(quaternions[:25]))
        with self.assertRaisesRegex(ValueError, "vertical"):
            encode_history(positions[:25], np.broadcast_to(exp_rotation(np.array([0., math.pi/2, 0.])), (25, 24, 4)))
        history = encode_history(positions[:25], quaternions[:25])
        for shape in ((11, TARGET_DIM), (0, TARGET_DIM), (10, TARGET_DIM-1)):
            with self.assertRaises(ValueError):
                decode_targets(np.zeros(shape), history.anchor)


if __name__ == "__main__":
    unittest.main()
