"""Arrival causality, joint order and upstream ABI contracts."""

from types import SimpleNamespace
import unittest

import numpy as np

from script.a3.joint_forecast_replay import JointForecastReplay, forecast_window


class JointReplayTests(unittest.TestCase):
    def test_constant_velocity_and_horizon_guard(self):
        times = np.arange(6) / 30
        q = times[:, None] * np.arange(1, 30)[None]
        query = times[-1] + np.arange(11) / 50
        for mode in ("cv", "smooth_cv"):
            np.testing.assert_allclose(forecast_window(q, times, query, mode),
                                       query[:, None] * np.arange(1, 30)[None], atol=1e-12)
        with self.assertRaises(ValueError):
            forecast_window(q, times, query + .001, "cv")

    def test_startup_and_known_end_hold(self):
        q = np.arange(29)[None].astype(float)
        np.testing.assert_array_equal(forecast_window(q, [0], [.18], "E12"), q)
        np.testing.assert_array_equal(forecast_window(q, [0], [.21], "cv", end_seen=True), q)
        with self.assertRaises(ValueError):
            forecast_window(q * np.nan, [0], [.1], "hold")

    def make_replay(self):
        raw = np.arange(90)[:, None] / 30 * np.arange(1, 30)[None]
        resampled = np.arange(148)[:, None] / 50 * np.arange(1, 30)[None]
        permutation = np.arange(29)[::-1]
        reference = SimpleNamespace(path="motion.csv", dof_mj29=resampled,
            dof_il=resampled[:, permutation], dof_vel_il=np.ones_like(resampled),
            anchor_quat_wxyz=np.ones((148, 4)), num_frames=148)
        sim = SimpleNamespace(load_loop_motion_reference=lambda *a, **k: reference,
            LoopSimRunner=SimpleNamespace(_record_metrics_step=lambda *a: None),
            load_a3_flat_csv=lambda *a, **k: (None, None, raw, len(raw)),
            root_ori_diff_6d=lambda anchor, quat: np.ones((len(quat), 6)))
        replay = JointForecastReplay(sim, "cv")
        replay.register(reference, permutation, 30, 4)
        return replay, reference, permutation

    def test_future_mutation_cannot_change_joint_commands(self):
        replay, reference, _ = self.make_replay()
        for frame in range(10, 130):
            before = replay.build(reference, frame, None)
            row = replay.rows[-1]
            self.assertLessEqual(row[3], row[0] + 1e-10)
            self.assertLessEqual(row[6], .20 + 1e-10)
            times, observed, _ = replay.sources[str(reference.path)]
            future = times > row[0] + 1e-10
            old = observed.copy()
            observed[future] = 1e6
            np.testing.assert_array_equal(before, replay.build(reference, frame, None))
            observed[:] = old

    def test_packing_and_joint_mapping(self):
        replay, reference, permutation = self.make_replay()
        value = replay.build(reference, 10, None)
        base = replay.last_target_frame / 50
        positions = (base + np.arange(10) / 50)[:, None] * np.arange(1, 30)[permutation]
        velocity = np.broadcast_to(np.arange(1, 30)[permutation], (10, 29))
        expected = np.concatenate((positions.reshape(-1), velocity.reshape(-1)))
        np.testing.assert_allclose(value[:, :58].reshape(-1), expected, atol=1e-6)
        bad = permutation.copy()
        bad[0] = bad[1]
        with self.assertRaises(ValueError):
            replay.register(reference, bad, 30, 4)


if __name__ == "__main__":
    unittest.main()
