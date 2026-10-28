"""Reference-centered arm targets for frozen-policy diagnostic comparisons."""

from __future__ import annotations

import math


def reference_centered_actions(actions, reference, base, scale, residual_radius):
    """Bound eight arm PD-target residuals in radians; retain lower-body actions."""
    if not math.isfinite(residual_radius) or residual_radius < 0:
        raise ValueError("residual_radius must be finite and nonnegative")
    if actions.shape != reference.shape or actions.shape[-1] != 22:
        raise ValueError("Expected matching Q1 action/reference tensors with 22 joints")
    result = actions * 1
    policy_target = base + scale * actions
    residual = (policy_target[..., 14:] - reference[..., 14:]).clip(
        -residual_radius, residual_radius
    )
    upper_target = reference[..., 14:] + residual
    result[..., 14:] = (upper_target - base[..., 14:]) / scale[..., 14:]
    return result


class ReferenceCenteredUpper:
    def __init__(self, policy, env, residual_radius):
        import torch
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import (
            Q1_JOINT_NAMES, action_scale,
        )

        self.policy, self.env, self.radius = policy, env, residual_radius
        self.scale = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES],
                                  device=env.device)
        if not bool((self.scale > 0).all()):
            raise ValueError("Q1 PD action scales must be positive")

    def __call__(self, history, goal):
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

        actions = self.policy(history, goal).clamp(-20, 20)
        current = self.env.state.info[SUPERMIMIC_STATE_KEY]
        motion = self.env.supermimic_mdp.motion
        frames = current["f"]
        next_frames = (frames + 1).minimum(motion.clip_end[motion.clip_id[frames]])
        base = self.env._supermimic_command.default_q + current["xi"][:, 1:23]
        return reference_centered_actions(actions, motion.q[next_frames], base,
                                          self.scale, self.radius)
