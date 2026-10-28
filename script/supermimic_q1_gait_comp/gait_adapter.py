"""RSL-RL lower residual with immutable base and explicit foot targets."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "supermimic_q1_balance"), str(HERE.parent / "supermimic_q1_ppo")]
from balance_adapter import BalanceAdapter, BalanceActor, make_ppo as base_make_ppo, bootstrap_rewards
from control import fixed_blocks, physical_base, compose, RADIUS
from adapter import termination_masks
sys.path.insert(0, str(HERE))
from rewards import FEATURES, REWARD_MAXIMUM, FootReference
from overnight import bounded_learning_rate


def make_ppo(actor, obs, count):
    algorithm = base_make_ppo(actor, obs, count)
    algorithm.schedule, algorithm.desired_kl = "adaptive", .01
    algorithm.learning_rate = 5e-5
    for group in algorithm.optimizer.param_groups:
        group["lr"] = algorithm.learning_rate
    def cap_adaptive_rate(optimizer, args, kwargs):
        algorithm.learning_rate = bounded_learning_rate(algorithm.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = algorithm.learning_rate
    algorithm.optimizer.register_step_pre_hook(cap_adaptive_rate)
    return algorithm


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 14))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value):
        return fixed_blocks(self.net, value.float())


class Actor(BalanceActor):
    def __init__(self):
        super().__init__()
        self.head = Head()


class Adapter(BalanceAdapter):
    def __init__(self, env, teacher, thresholds, upper_alpha=.75):
        self.foot_reference = FootReference(env.supermimic_mdp)
        super().__init__(env, teacher, thresholds, fallback_enabled=False)
        if not 0 <= upper_alpha <= 1:
            raise ValueError("Upper reference blend must be in [0,1]")
        self.cap = float(upper_alpha)
        self.reset_gate(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.history = None
        self.observations = self.observe(self.current)

    def build(self, history, current, alpha, risk, cooldown, cap, indices=None):
        obs, nominal, reference = super().build(history, current, alpha, risk, cooldown, cap, indices)
        features = self.foot_reference.features(current["s"], current["f"])
        obs["features"] = torch.cat((obs["features"], features), -1)
        if obs["features"].shape[-1] != FEATURES:
            raise ValueError("Wrong gait feature width")
        return obs, nominal, reference

    @torch.no_grad()
    def step(self, latent):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
        alpha = self.gate.alpha.clone()
        base = physical_base(self.command.default_q, self.current["xi"]).clone()
        previous_targets = base + self.scale * self.current["a_prev"]
        executed, residual = compose(self.nominal, self.reference_action, latent, alpha, self.scale)
        targets = base + self.scale * executed
        state = self.env.step(executed)
        tr = state.info[SUPERMIMIC_TRANSITION_KEY]
        self.last_transition = tr
        torch.testing.assert_close(tr["a"], executed, atol=0, rtol=0)
        _, timeouts = termination_masks(tr, self.mdp.motion, self.max_episode_length)
        self.gate.advance(self.measured_risk(tr["s1"], tr["f"]+1), self.caps())
        self.gate.alpha = self.caps()
        terminal_obs = None
        if bool(timeouts.any()):
            terminal = {"s": tr["s1"][timeouts], "f": tr["f"][timeouts]+1, "xi": tr["xi"][timeouts]}
            measured = self.mdp.add_noise(self.mdp.obs(unflatten_state(terminal["s"]), tr["a"][timeouts],
                                                       terminal["xi"], terminal["f"]))
            history = torch.cat((self.history[timeouts, 1:], measured[:, None]), 1)
            terminal_obs = self.build(history, terminal, self.gate.alpha[timeouts], self.gate.filtered[timeouts],
                                      self.gate.cooldown[timeouts], self.caps()[timeouts], indices=timeouts)[0]
        rewards, parts = self.foot_reference.reward(tr["r"], tr["s1"], tr["f"]+1, targets,
                                                  previous_targets, residual, *self.contacts.last)
        if not bool(torch.isfinite(rewards).all()):
            raise FloatingPointError("Nonfinite physical gait reward")
        done = tr["done"].bool()
        self.episode_steps += 1
        self.episode_return += rewards
        self.episode_normalized += rewards / REWARD_MAXIMUM
        self.episode_original += tr["r"]
        report = {"completed_episodes": int(done.sum()), "hard_terminations": int(tr["term"].sum()),
                  "normalized_return_sum": float((self.episode_normalized[done] / self.episode_steps[done]).sum()),
                  "length_steps_sum": float(self.episode_steps[done].sum()), "reward_sum": float(rewards.sum()),
                  "original_reward_sum": float(tr["r"].sum()), "transitions": self.num_envs,
                  "normalized_step_reward_sum": float((rewards / REWARD_MAXIMUM).sum()),
                  "alpha_sum": float(alpha.sum()), "residual_squared_sum": float(residual.square().sum()),
                  "artificial_timeouts": int(timeouts.sum()),
                  **{key+"_sum": float(value.sum()) for key,value in parts.items()}}
        self.last_reward_parts = {"original": tr["r"], **parts}
        for value in (self.episode_steps, self.episode_return, self.episode_normalized, self.episode_original):
            value[done] = 0
        self.reset_gate(done)
        self.current = state.info[SUPERMIMIC_STATE_KEY]
        self.observations = self.observe(self.current, reset=done)
        return self.observations, rewards, done, {"terminal_observations": terminal_obs,
                                                 "artificial_timeouts": timeouts, "report": report}
