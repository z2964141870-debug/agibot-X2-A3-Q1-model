"""Native inference must reject mismatched, stale and profile checkpoints."""

import copy
import unittest

from inference import CONTRACT, validate_metadata
from contracts import expected_transitions
from controller import updates


def checkpoint(iteration=8):
    cfg = dict(robot="q1", morphology=True, actor_geometry=True, policy_goal="motion",
        kp_interface=True, kp_student="return", hist_len=10, goal_anchor="reference",
        num_envs=1024, warmup_env_steps=400, utd=.4, wm_only_iters=2000, iterations=160000)
    return dict(native_q1_contract=dict(contracts=CONTRACT, code_commit="a" * 40,
        plan=dict(code_commit="a" * 40, purpose="train", robots=["g1", "q1"], cfg=cfg.copy())),
        actual_training=dict(iteration=iteration, code_commit="a" * 40,
            fresh_initialization=True, profile_weights_reused=False,
            optimizer_updates=updates(cfg, iteration),
            collected_environment_transitions=0 if not iteration else expected_transitions(dict(cfg, iterations=iteration))),
        cfg=cfg, iteration=iteration, schedule=dict(step=iteration, world_size=2), goal_version=2, kp_goal_version=10)


class InferenceTests(unittest.TestCase):
    def test_initial_and_updated_actual_metadata(self):
        for iteration in (0, 8, 160000):
            self.assertEqual(validate_metadata(checkpoint(iteration), "q1", "a" * 40)["iteration"], iteration)

    def test_profile_reused_weights_and_wrong_robot_are_rejected(self):
        for key, edit in (
            ("profile", lambda x: x["native_q1_contract"]["plan"].update(purpose="profile")),
            ("reused", lambda x: x["actual_training"].update(profile_weights_reused=True)),
            ("robot", lambda x: x["cfg"].update(robot="g1")),
            ("contract", lambda x: x["native_q1_contract"]["contracts"].update(state=240)),
            ("commit", lambda x: x["actual_training"].update(code_commit="b" * 40)),
            ("exposure", lambda x: x["actual_training"].update(collected_environment_transitions=0)),
            ("human", lambda x: x.update(kp_goal_version=2)),
            ("history", lambda x: x["cfg"].update(hist_len=1)),
            ("updates", lambda x: x["actual_training"]["optimizer_updates"].update(wm_opt=0)),
            ("shared", lambda x: x["schedule"].update(world_size=1)),
            ("plan", lambda x: x["native_q1_contract"]["plan"]["cfg"].update(num_envs=512)),
            ("frame", lambda x: x["cfg"].update(goal_anchor="robot"))):
            value = copy.deepcopy(checkpoint())
            edit(value)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_metadata(value, "q1", "a" * 40)


if __name__ == "__main__":
    unittest.main()
