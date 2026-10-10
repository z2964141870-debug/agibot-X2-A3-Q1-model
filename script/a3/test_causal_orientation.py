"""SO(3) forecasting and causality of all actual policy reference tokens."""

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.causal_orientation import FullCausalReplay, orientation_window


class OrientationTests(unittest.TestCase):
    def test_constant_rotation_and_sign_equivalence(self):
        times = np.arange(6)/30
        q = Rotation.from_euler("z", times*.7).as_quat()[:, [3, 0, 1, 2]]
        query = times[-1]+np.arange(10)/50
        expected = Rotation.from_euler("z", query*.7)
        actual = Rotation.from_quat(orientation_window(q, times, query)[:, [1, 2, 3, 0]])
        np.testing.assert_allclose((actual*expected.inv()).magnitude(), 0, atol=1e-12)
        q[::2] *= -1
        signed = Rotation.from_quat(orientation_window(q, times, query)[:, [1, 2, 3, 0]])
        np.testing.assert_allclose((actual*signed.inv()).magnitude(), 0, atol=1e-12)
        with self.assertRaises(ValueError):
            orientation_window(q, times, [times[-1]+.201])

    def test_startup_and_arrived_end(self):
        q = np.array([[1., 0, 0, 0], [np.cos(.1), 0, 0, np.sin(.1)]])
        np.testing.assert_array_equal(orientation_window(q[:1], [0], [0, .18]), np.tile(q[0], (2, 1)))
        np.testing.assert_allclose(orientation_window(q, [0, .1], [.2, .3], end_seen=True), np.tile(q[-1], (2, 1)))

    def test_future_joint_and_orientation_mutations_do_not_change_input(self):
        times = np.arange(90)/30
        raw_q = times[:, None] * np.arange(1, 30)[None]
        raw_rot = Rotation.from_euler("z", times*.7).as_quat()[:, [3, 0, 1, 2]]
        sampled_t = np.arange(148)/50
        sampled_q = sampled_t[:, None] * np.arange(1, 30)[None]
        reference = SimpleNamespace(path="test.csv", num_frames=148, dof_mj29=sampled_q, dof_il=sampled_q,
            dof_vel_il=np.ones_like(sampled_q),
            anchor_quat_wxyz=Rotation.from_euler("z", sampled_t*.7).as_quat()[:, [3, 0, 1, 2]])
        sim = SimpleNamespace(load_loop_motion_reference=lambda *a, **k: reference,
            LoopSimRunner=SimpleNamespace(_record_metrics_step=lambda *a: None),
            load_a3_flat_csv=lambda *a, **k: (None, raw_rot, raw_q, len(raw_q)),
            root_ori_diff_6d=lambda anchor, quat: np.column_stack((quat, np.zeros((len(quat), 2)))))
        replay = FullCausalReplay(sim, "cv")
        replay.register(reference, np.arange(29), 30, 4)
        for tick in range(10, 80):
            before = replay.build(reference, tick, [1, 0, 0, 0])
            latest = int(replay.rows[-1][4])
            observed = replay.sources[str(reference.path)][1]
            rotation = replay.rotations[str(reference.path)]
            old_q, old_r = observed.copy(), rotation.copy()
            observed[latest+1:] += 10000
            rotation[latest+1:] = [0, 0, 0, 1]
            np.testing.assert_array_equal(before, replay.build(reference, tick, [1, 0, 0, 0]))
            observed[:], rotation[:] = old_q, old_r


if __name__ == "__main__":
    unittest.main()
