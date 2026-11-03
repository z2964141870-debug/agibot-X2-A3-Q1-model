"""Real Primebot weights and paired Q1 references in the Morph Python runtime."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class CommonPlantRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("Q1_COMMON_TEST_RUNTIME"), "Requires the fixed server runtime and baseline")
    def test_real_baseline_and_fixed_reference_adapters_without_integration(self):
        from common import FIXED, ROOT, identity
        runtime = json.loads(Path(os.environ["Q1_COMMON_TEST_RUNTIME"]).read_text())
        shared = json.loads(Path(os.environ["Q1_COMMON_TEST_SHARED_CONFIG"]).read_text())
        self.assertEqual(Path.cwd().resolve(), ROOT)
        self.assertEqual(Path(sys.executable).absolute(), Path(runtime["morph_python"]))
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        frozen_path = Path(shared["frozen_data"])
        self.assertEqual(identity(frozen_path)["sha256"], shared["frozen_data_sha256"])
        frozen = json.loads(frozen_path.read_text())
        selected = [r for r in frozen["rows"] if Path(r["q1_primebot"]["path"]).stem in FIXED]
        self.assertEqual(len(selected), 3)
        self.assertEqual({Path(r["q1_primebot"]["path"]).stem for r in selected}, set(FIXED))

        import mujoco
        import numpy as np
        import torch
        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())
        sys.path[:0] = [str(Path(runtime["morph_source"]) / "src"),
                       str(Path(runtime["adapted_source"]) / "src"),
                       str(ROOT / "script/morphmimic_q1"), str(ROOT / "script/primebot_q1_review")]
        from bootstrap import install
        spec = install("q1")["robot"]
        from evaluate import load_actor
        from q1_native_review import assert_paired_reference, native_state, reference_window_indices
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.data.motion_library import MotionLibrary
        from morphmimic.model.human import pack_human_goal
        from morphmimic.model.rewards_torch import analytic_obs, goal_features
        from morphmimic.model.state import split_state
        from supermimic.data import Q1Motions
        from supermimic.models import goal_features as primebot_goals
        from supermimic.simulator import Q1Simulator
        from snapshots import rng_sha

        audit = json.loads((Path(shared["primebot_review"]) / "checkpoint_audit.json").read_text())
        checkpoint = audit["final"]["checkpoint"]
        self.assertEqual(identity(checkpoint["path"]), checkpoint)
        robot, actor, config, model_id, iteration, transitions = load_actor(checkpoint["path"])
        self.assertGreater(iteration, 0)
        self.assertGreater(transitions, 0)
        self.assertEqual(Path(config["robot_xml"]), Path(runtime["xml"]))
        self.assertEqual(config["action_scale"], .5)
        default = spec.translator.to_canonical(torch.zeros(22))
        present = torch.tensor([v is not None for v in spec.translator.perm])
        rows = []

        # Exercise real construction/reset and FK while making any integration fail.
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("CPU compatibility cannot integrate")), \
             patch.object(mujoco, "mj_step1", side_effect=AssertionError("CPU compatibility cannot integrate")), \
             patch.object(mujoco, "mj_step2", side_effect=AssertionError("CPU compatibility cannot integrate")), \
             patch.object(Q1Simulator, "step", side_effect=AssertionError("CPU compatibility cannot roll out")):
            simulator = Q1Simulator(robot, 1, "cpu", config["action_scale"],
                use_tn=config.get("use_tn", False), physics_timestep=config.get("physics_timestep"))
            self.assertEqual(simulator.substeps, 10)
            for item in selected:
                for key in ("q1_primebot", "q1_morph", "human"):
                    self.assertEqual(identity(item[key]["path"]), item[key])
                reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
                library = MotionLibrary([item["q1_morph"]["path"]], device="cpu", robot=spec,
                                        human_root=shared["human_root"])
                paired = assert_paired_reference(robot, reference, library, spec)
                with np.load(item["human"]["path"], allow_pickle=False) as human:
                    np.testing.assert_array_equal(library.human_pos_w.numpy(), human["human_pos_w"])
                    np.testing.assert_array_equal(library.human_quat_w.numpy(), human["human_quat_w"])
                simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
                state = simulator.state()
                converted = native_state(robot, state, spec)
                self.assertEqual(tuple(converted.shape), (1, 269))
                offsets = torch.tensor(reference_window_indices(0, len(reference)))
                future = library.state(offsets)[None]
                observation = analytic_obs(converted, torch.zeros(29), default, converted[:, 240:], spec.imu_slot)
                observation = torch.cat((torch.where(present, observation[:, :29], 0.),
                    torch.where(present, observation[:, 29:58], 0.), observation[:, 58:64],
                    torch.where(present, observation[:, 64:], 0.)), -1)
                motion = goal_features(converted, future, anchor="reference")
                human = pack_human_goal(library.human_pos_w[offsets][None],
                    library.human_quat_w[offsets][None], split_state(converted).body_quat[:, ANCHOR_IDX])
                self.assertEqual(tuple(observation.shape), (1, 93))
                self.assertEqual(tuple(motion.shape), (1, 670))
                self.assertEqual(tuple(human.shape), (1, 1684))
                self.assertTrue(all(torch.isfinite(x).all().item() for x in (converted, observation, motion, human)))
                history = robot.observation(state, simulator.context)[:, None].expand(-1, 10, -1).clone()
                before_rng = rng_sha()
                with torch.inference_mode():
                    action = actor(history, primebot_goals(robot, reference[offsets][None]))
                self.assertEqual(tuple(action.shape), (1, 22))
                self.assertTrue(torch.isfinite(action).all().item())
                self.assertEqual(before_rng, rng_sha())
                self.assertTrue(all(identity(item[key]["path"]) == item[key]
                                    for key in ("q1_primebot", "q1_morph", "human")))
                rows.append(dict(clip=Path(item["q1_primebot"]["path"]).stem, **paired,
                    native_state=269, observation=93, motion_goal=670, human_goal=1684,
                    baseline_action_finite=True, policy_rng_unchanged=True))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(identity(checkpoint["path"]), checkpoint)
        print("Q1_COMMON_CPU_EVIDENCE=" + json.dumps(dict(
            python=sys.version, packages={name: importlib.metadata.version(name)
                for name in ("torch", "mujoco", "mujoco-warp", "warp-lang")},
            checkpoint=checkpoint, iteration=iteration, transitions=transitions,
            asset_sha256=model_id["asset_sha256"], cases=rows,
            cuda_initialized=False, physical_transitions=0, optimizer_updates=0,
            note="real Primebot CPU forwards; no trained shared checkpoint or shared policy quality test")), flush=True)


if __name__ == "__main__":
    unittest.main()
