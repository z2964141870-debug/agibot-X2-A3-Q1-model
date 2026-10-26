from dataclasses import asdict
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "leg_control"))
sys.path.insert(0, str(HERE.parents[1] / "supermimic_q1_sources"))

from feedback import (FeedbackQ1Sim, FeedbackSlottedQ1Sim, augment_manifest,
                      configure_imagination, input_contract, reference_feedback)
from control import Leg50Assets
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.q1_assets import TRACKED
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, GOAL_DIM, STATE_DIM, XI_DIM
from supermimic.world_model.heads import analytic_observation, anchor_relative_obs
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


class FeedbackMathTests(unittest.TestCase):
    def test_body_frame_sign_and_rotation_derivative(self):
        position = torch.tensor([[1., 2., 0.]], dtype=torch.float64, requires_grad=True)
        half_yaw = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
        zero = torch.zeros_like(half_yaw)
        rotation = torch.stack((half_yaw.cos(), zero, zero, half_yaw.sin()))[None]
        target = torch.tensor([[2., 2., 0.]], dtype=torch.float64)
        target_rotation = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
        feedback = anchor_relative_obs(position, rotation, target, target_rotation)
        torch.testing.assert_close(feedback[0, :3],
            torch.stack((torch.cos(2. * half_yaw), -torch.sin(2. * half_yaw), zero)))
        position_grad, angle_grad = torch.autograd.grad(feedback[0, 0] + feedback[0, 3],
                                                      (position, half_yaw))
        self.assertGreater(float(position_grad.norm()), .9)
        self.assertTrue(torch.isfinite(angle_grad))
        self.assertGreater(float(angle_grad.abs()), .1)


@unittest.skipUnless(os.environ.get("Q1_FEEDBACK_TEST_ROOT"), "requires actual Q1 assets/reference")
class FeedbackPhysicalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261026)
        cls.root = Path(os.environ["Q1_FEEDBACK_TEST_ROOT"])
        cls.assets = Leg50Assets(cls.root)
        cls.reference = cls.root / "data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references/train_3419f09ba93c91d75fc1.npz"
        cls.library = MotionLibrary([cls.reference], TRACKED, cls.assets.q0, device="cpu",
                                    active_indices=cls.assets.active_indices)
        cls.config = SimCfg(num_envs=2, threads=1, domain_randomization=False,
                            observation_noise=False, action_noise=0.)
        configure_imagination()

    def make_sim(self, slotted=False):
        factory = FeedbackSlottedQ1Sim if slotted else FeedbackQ1Sim
        extra = {"slots": 1} if slotted else {}
        sim = factory(self.assets, self.library, self.config, self.assets.heads(), 0, **extra)
        sim.reset(sim.ids, torch.tensor([0, 60]))
        self.addCleanup(sim.close)
        return sim

    def policy_cfg(self):
        return PolicyCfg(encoder_hidden=(64, 32), decoder_hidden=(64, 32), goal_dim=GOAL_DIM + 9)

    def test_feedback_resolves_translation_alias_without_changing_proprioception(self):
        sim = self.make_sim()
        previous = sim.state.clone()
        base = analytic_observation(previous, sim.previous_action, sim.context,
                                    sim.heads.default_joint_pos, sim.heads.active_indices)
        _, first = sim.policy_inputs()
        sim.state[:, 0] += .2
        sim.state[:, 1] -= .3
        moved = analytic_observation(sim.state, sim.previous_action, sim.context,
                                     sim.heads.default_joint_pos, sim.heads.active_indices)
        _, second = sim.policy_inputs()
        torch.testing.assert_close(base, moved, rtol=0., atol=0.)
        torch.testing.assert_close(first[:, :GOAL_DIM], second[:, :GOAL_DIM], rtol=0., atol=0.)
        self.assertGreater(float((first[:, -9:-6] - second[:, -9:-6]).norm()), .4)

    def test_native_and_slotted_paths_preserve_the_physical_transition(self):
        native, slotted = self.make_sim(), self.make_sim(slotted=True)
        base = Q1Sim(self.assets, self.library, self.config, self.assets.heads(), 0)
        base.reset(base.ids, torch.tensor([0, 60]))
        self.addCleanup(base.close)
        action = torch.linspace(-.2, .2, ACT_DIM)[None].expand(2, -1)
        for _ in range(3):
            expected = base.step(action, auto_reset=False)
            for sim in (native, slotted):
                actual = sim.step(action, auto_reset=False)
                for field in ("state", "reward", "terminated"):
                    torch.testing.assert_close(actual[field], expected[field], rtol=0., atol=0.)
            for actual, expected_input in zip(native.policy_inputs(), slotted.policy_inputs()):
                torch.testing.assert_close(actual, expected_input, rtol=0., atol=0.)

    def test_noise_free_imagination_goals_match_simulator_and_keep_h32_gradients(self):
        sim = self.make_sim(slotted=True)
        replay = ReplayBuffer(2, 40, STATE_DIM, ACT_DIM, XI_DIM, "cpu")
        for _ in range(12):
            sim.step(torch.zeros(2, ACT_DIM), replay, auto_reset=False)
        start = replay.sample_starts(2, 10)
        wm = WorldModel(WorldModelCfg(hidden=(64, 64), term_hidden=(32, 32)), self.assets.heads(), "cpu")
        last = [layer for layer in wm.dynamics.modules() if isinstance(layer, torch.nn.Linear)][-1]
        torch.nn.init.normal_(last.weight, std=1e-4)
        wm.requires_grad_(False)
        policy = Policy(self.policy_cfg())
        ro = imagine(wm, policy, self.library, start, 32, obs_noise=False, anchor_obs=True)
        self.assertEqual(ro.goals.shape, (33, 2, GOAL_DIM + 9))
        for index in (0, 1, 32):
            expected = reference_feedback(wm.heads, ro.states[index], reference_at(self.library, ro.frames[index]))
            torch.testing.assert_close(ro.goals[index, :, -9:], expected, rtol=0., atol=0.)
        sim.state = ro.states[0].detach()
        sim.frame = ro.frames[0]
        torch.testing.assert_close(sim.policy_inputs()[1], ro.goals[0], rtol=0., atol=0.)
        ro.goals[-1, :, -9:].square().sum().backward()
        norm = sum(float(p.grad.square().sum()) for p in policy.parameters() if p.grad is not None) ** .5
        self.assertTrue(torch.isfinite(torch.tensor(norm)))
        self.assertGreater(norm, 1e-9)
        self.assertFalse(any(p.grad is not None for p in wm.parameters()))

    def test_policy_critic_and_metadata_round_trip(self):
        cfg = self.policy_cfg()
        policy, critic = Policy(cfg), Critic(CriticCfg(), pcfg=cfg)
        sim = self.make_sim()
        observation, goal = sim.policy_inputs()
        self.assertEqual(policy(observation, goal).shape, (2, ACT_DIM))
        self.assertEqual(critic(torch.zeros(2, 57), goal, torch.zeros(2, XI_DIM), torch.zeros(2, 4)).shape, (2,))
        manifest = {"assets": self.assets.manifest(), "policy": asdict(cfg),
            "policy_trainer": asdict(PolicyTrainerCfg(anchor_obs=True)), "training_entry": {"path": "/base.py"},
            "paper_deviations": ["CPU MuJoCo backend; Q1 simulation PD50/2 and 25% model torque bounds"]}
        augment_manifest(manifest, HERE / "entry_root_feedback.py")
        self.assertEqual(manifest["reference_feedback"], input_contract())
        stream = io.BytesIO()
        torch.save({"policy": policy.state_dict(), "manifest": manifest}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        restored = Policy(PolicyCfg(**saved["manifest"]["policy"]))
        restored.load_state_dict(saved["policy"], strict=True)
        torch.testing.assert_close(restored(observation, goal), policy(observation, goal), rtol=0., atol=0.)

    def test_actual_training_dispatch_enables_both_feedback_paths(self):
        import entry_root_feedback as entry
        import train_reproduction
        import train_q1_with_dashboard
        import contact_protocol
        from supermimic import q1_sim
        from supermimic.imagination import rollout
        with ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            attributes = [(train_reproduction, name) for name in
                ("Q1Assets", "Q1Sim", "PolicyCfg", "PolicyTrainerCfg", "atomic_json")]
            attributes += [(contact_protocol, "ContactCorrectedAssets"), (q1_sim, "Q1Sim"),
                           (rollout, "ANCHOR_NOISE_HALF")]
            for module, name in attributes:
                stack.enter_context(patch.object(module, name, getattr(module, name)))
            called = []

            def fake_training_loop():
                cfg = train_reproduction.PolicyCfg()
                trainer = train_reproduction.PolicyTrainerCfg(compile=None)
                self.assertEqual(cfg.goal_dim, GOAL_DIM + 9)
                self.assertTrue(trainer.anchor_obs)
                sim = train_reproduction.Q1Sim(self.assets, self.library, self.config,
                                               self.assets.heads(), 0, slots=1)
                try:
                    self.assertEqual(sim.policy_inputs()[1].shape[-1], cfg.goal_dim)
                finally:
                    sim.close()
                manifest = {"assets": train_reproduction.Q1Assets(self.root).manifest(),
                    "policy": asdict(cfg), "policy_trainer": asdict(trainer),
                    "training_entry": {"path": "/base.py"}, "paper_deviations": [
                        "CPU MuJoCo backend; Q1 simulation PD50/2 and 25% model torque bounds"]}
                train_reproduction.atomic_json(Path(directory) / "admission.json", manifest)
                self.assertEqual(manifest["reference_feedback"], input_contract())
                called.append(True)

            stack.enter_context(patch.object(train_q1_with_dashboard, "main", fake_training_loop))
            stack.enter_context(patch.object(sys, "argv", [str(HERE / "entry_root_feedback.py"), "train"]))
            entry.main()
            self.assertEqual(called, [True])


if __name__ == "__main__":
    unittest.main()
