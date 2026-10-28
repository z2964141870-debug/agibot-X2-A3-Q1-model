"""Interpolate physical arm targets while retaining real lower-body feedback."""

from __future__ import annotations

import math


def blended_upper_actions(actions, reference, base, scale, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if actions.shape != reference.shape or actions.shape[-1] != 22:
        raise ValueError("Expected matching Q1 action/reference tensors")
    result = actions * 1
    if alpha == 0:
        return result
    reference_action = (reference[..., 14:] - base[..., 14:]) / scale[..., 14:]
    result[..., 14:] = (1 - alpha) * actions[..., 14:] + alpha * reference_action
    return result


class BlendedUpper:
    def __init__(self, policy, env, alpha):
        import torch
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import (
            Q1_JOINT_NAMES, action_scale,
        )

        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.policy, self.env, self.alpha = policy, env, alpha
        self.scale = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES],
                                  device=env.device)
        if not bool((self.scale > 0).all()):
            raise ValueError("Q1 action scales must be positive")

    def __call__(self, history, goal):
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

        actions = self.policy(history, goal).clamp(-20, 20)
        if self.alpha == 0:
            return actions
        current = self.env.state.info[SUPERMIMIC_STATE_KEY]
        motion = self.env.supermimic_mdp.motion
        frames = current["f"]
        following = (frames + 1).minimum(motion.clip_end[motion.clip_id[frames]])
        base = self.env._supermimic_command.default_q + current["xi"][:, 1:23]
        return blended_upper_actions(actions, motion.q[following], base,
                                     self.scale, self.alpha)
