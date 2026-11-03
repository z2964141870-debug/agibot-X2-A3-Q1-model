"""Real G1/Q1 CPU kinematics only: no integration, policy, optimizer, or CUDA."""

import os
from pathlib import Path
import sys
import tempfile
import unittest


class RealNativeKinematicsTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("Q1_NATIVE_TEST_SOURCE"), "Requires the fixed server source and assets")
    def test_reference_native_roundtrip_and_actual_collision_surfaces(self):
        import mujoco
        import numpy as np
        import torch
        sys.path[:0] = [str(Path(os.environ["Q1_NATIVE_TEST_SOURCE"]) / "src"),
                       str(Path(__file__).resolve().parents[1] / "morphmimic_q1")]
        from bootstrap import install
        spec = install(os.environ.get("Q1_NATIVE_TEST_ROBOT", "q1"))["robot"]
        from morphmimic.eval.benchmark import make_eval_env
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        self.assertEqual(make_eval_env.__globals__["MorphMimicEnv"].__name__, "NativeEnv")
        self.assertIs(make_eval_env.__globals__["morphmimic_env_cfg"], morphmimic_env_cfg)
        cfg = morphmimic_env_cfg(spec.name, num_envs=1, eval=True)
        self.assertFalse(cfg.auto_reset)
        self.assertEqual(cfg.actions["joint_pos"].max_delay_s, 0.)
        from native_metrics import foot_shapes, reference_poses, sole_series
        self.assertFalse(torch.cuda.is_initialized())
        model = spec.entity_cfg_fn().spec_fn().compile()
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.mjb"
            mujoco.mj_saveModel(model, str(scene), None)
            recovered = mujoco.MjModel.from_binary_path(str(scene))
            self.assertEqual((recovered.nq, recovered.ngeom), (model.nq, model.ngeom))
        data = mujoco.MjData(model)
        tracked = [model.body(name).id for name in spec.tracked_bodies]
        addresses = [model.jnt_qposadr[model.joint(name).id] for name in spec.joint_names]
        root = int(model.jnt_qposadr[np.flatnonzero(model.jnt_type == 0)[0]])
        states, poses = [], []
        for i in range(4):
            data.qpos[:] = model.qpos0
            data.qpos[root:root + 3] = [.03 * i, -.04 * i, 1. + .02 * i]
            data.qpos[root + 3:root + 7] = [np.cos(.07 * i), 0., 0., np.sin(.07 * i)]
            data.qpos[addresses] += .01 * i
            mujoco.mj_kinematics(model, data)
            state = np.zeros(269)
            state[:29] = spec.translator.to_canonical(torch.tensor(data.qpos[addresses])).numpy()
            state[58:100], state[100:156] = data.xpos[tracked].flatten(), data.xquat[tracked].flatten()
            state[156:198] = np.tile([.3, -.4, 0.], 14)
            states.append(state)
            poses.append(data.qpos.copy())
        states, poses = np.array(states), np.array(poses)
        reconstructed, recovered_root = reference_poses(model, spec, states)
        self.assertEqual(recovered_root, root)
        np.testing.assert_allclose(reconstructed, poses, atol=2e-12, rtol=0.)
        groups = foot_shapes(model, spec)
        self.assertTrue(all(groups))
        height, speed = sole_series(model, spec, poses, states)
        np.testing.assert_allclose(speed, .5, atol=1e-12, rtol=0.)
        shifted = poses.copy()
        shifted[:, root + 2] += .2
        raised, _ = sole_series(model, spec, shifted, states)
        np.testing.assert_allclose(raised - height, .2, atol=1e-12, rtol=0.)
        self.assertTrue(np.isfinite(height).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
