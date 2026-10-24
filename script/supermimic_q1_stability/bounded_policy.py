"""Use the real Q1 action domain during differentiable imagination."""

from pathlib import Path
import sys

import torch

CORE = Path(__file__).resolve().parents[1] / "supermimic_for_q1"
sys.path.insert(0, str(CORE))

from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.policy.nets import Policy
from supermimic.world_model.wm_trainer import WorldModelTrainer


ACTION_LIMIT = 20.


def bounded_action(raw):
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Nonfinite raw Q1 policy action before clipping")
    return raw.clamp(-ACTION_LIMIT, ACTION_LIMIT)


class BoundedPolicy(Policy):
    def forward(self, obs_hist, goal, interface="motion"):
        return bounded_action(super().forward(obs_hist, goal, interface))


def reject_nonfinite_gradients(optimizer, label):
    def before_step(current, args, kwargs):
        gradients = [parameter.grad.detach() for group in current.param_groups
                     for parameter in group["params"] if parameter.grad is not None]
        if gradients and not bool(torch.isfinite(torch.stack([gradient.norm() for gradient in gradients])).all()):
            raise FloatingPointError("Nonfinite " + label + " gradients before Adam step")

    return optimizer.register_step_pre_hook(before_step)


class GuardedPolicyTrainer(PolicyTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_step_hooks = [reject_nonfinite_gradients(self.opt, "actor"),
                                  reject_nonfinite_gradients(self.critic_opt, "critic")]


class GuardedWorldModelTrainer(WorldModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_step_hook = reject_nonfinite_gradients(self.opt, "world model")
