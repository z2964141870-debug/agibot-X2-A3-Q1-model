"""RSL-RL uses 14 latent actions; physical execution retains all 22 Q1 joints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from evaluation import NativeContacts
from adapter import termination_masks, bootstrap_rewards
from control import Fallback, ResidualHead, inputs, compose, risk_features, balance_rewards, physical_base


class BalanceActor(MLPModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.head = ResidualHead()
        self.distribution = GaussianDistribution(14, init_std=.1, learn_std=False)
        self.obs_normalization = False

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None or hidden_state is not None:
            raise ValueError("Feedforward unpadded latent action model")
        mean = self.head(obs["features"])
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def update_normalization(self, obs):
        return


class BalanceAdapter:
    def __init__(self, env, teacher, thresholds, fallback_enabled=False):
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale
        self.env, self.teacher = env, teacher.requires_grad_(False).eval()
        self.actor = type("TeacherBinding", (), {"policy": self.teacher})()
        self.mdp, self.command = env.supermimic_mdp, env._supermimic_command
        self.num_envs, self.num_actions, self.device = env.num_envs, 14, env.device
        self.scale = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=self.device)
        self.max_episode_length = self.command.max_episode_steps
        self.cap = 0.
        self.fallback_enabled = fallback_enabled
        self.gate = Fallback(self.num_envs, thresholds, self.device)
        self.keep_baseline = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contacts = NativeContacts(env).install()
        self.history = None
        if env.state is None:
            env.init_state()
        else:
            env.reset()
        self.current = env.state.info[SUPERMIMIC_STATE_KEY]
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_normalized = torch.zeros_like(self.episode_return)
        self.episode_original = torch.zeros_like(self.episode_return)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_gate(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.observations = self.observe(self.current)
        self.last_transition = None

    def caps(self):
        return torch.where(self.keep_baseline, 0., self.cap)

    def reset_gate(self, mask):
        self.keep_baseline[mask] = torch.rand(int(mask.sum()), device=self.device) < .25
        self.gate.reset(mask, self.caps())

    def measured_risk(self, state, frames):
        from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
        force, slip = self.contacts.last
        return risk_features(state, self.mdp.motion.root_p[frames],
                             mat_to_quat(self.mdp.motion.root_R[frames]), force, slip)

    def build(self, history, current, alpha, risk, cooldown, cap, indices=None):
        frames = torch.minimum(current["f"] + 1, self.mdp.motion.clip_end[self.mdp.motion.clip_id[current["f"]]])
        base = physical_base(self.command.default_q, current["xi"], indices)
        features, nominal, reference = inputs(self.teacher, history, self.mdp.goal(current["f"], 10),
                                              self.mdp.motion.q[frames], base, self.scale, alpha, risk, cooldown, cap)
        privileged = torch.cat((current["s"][:, :35], current["s"][:, 35:] * .05,
                                self.mdp.normalize_context(current["xi"])), -1)
        return TensorDict({"features": features, "privileged": privileged},
                          batch_size=[len(history)], device=self.device), nominal, reference

    @torch.no_grad()
    def observe(self, current, reset=None):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        measured = self.mdp.add_noise(self.mdp.obs(unflatten_state(current["s"]), current["a_prev"],
                                                   current["xi"], current["f"]))
        if self.history is None:
            self.history = measured[:, None].repeat(1, 10, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], measured[:, None]), 1)
            if reset is not None:
                self.history[reset] = measured[reset, None].expand(-1, 10, -1)
        obs, self.nominal, self.reference_action = self.build(self.history, current, self.gate.alpha,
            self.gate.filtered, self.gate.cooldown, self.caps())
        return obs

    @torch.no_grad()
    def step(self, latent):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
        applied_alpha = self.gate.alpha.clone()
        executed, residual = compose(self.nominal, self.reference_action, latent, applied_alpha, self.scale)
        state = self.env.step(executed)
        tr = state.info[SUPERMIMIC_TRANSITION_KEY]
        self.last_transition = tr
        torch.testing.assert_close(tr["a"], executed, rtol=0, atol=0)
        _, timeouts = termination_masks(tr, self.mdp.motion, self.max_episode_length)
        risk = self.measured_risk(tr["s1"], tr["f"] + 1)
        self.gate.advance(risk, self.caps())
        if not self.fallback_enabled:
            self.gate.alpha = self.caps()
        terminal_obs = None
        if bool(timeouts.any()):
            terminal = {"s": tr["s1"][timeouts], "f": tr["f"][timeouts] + 1, "xi": tr["xi"][timeouts]}
            measured = self.mdp.add_noise(self.mdp.obs(unflatten_state(terminal["s"]), tr["a"][timeouts],
                                                       terminal["xi"], terminal["f"]))
            history = torch.cat((self.history[timeouts, 1:], measured[:, None]), 1)
            terminal_obs = self.build(history, terminal, self.gate.alpha[timeouts], self.gate.filtered[timeouts],
                                      self.gate.cooldown[timeouts], self.caps()[timeouts], indices=timeouts)[0]
        rewards, root, leg, cost = balance_rewards(tr["r"], tr["s1"], self.mdp.motion.root_p[tr["f"] + 1],
                                                  self.mdp.motion.q[tr["f"] + 1], residual)
        done = tr["done"].bool()
        self.episode_steps += 1
        self.episode_return += rewards
        self.episode_normalized += rewards / .13
        self.episode_original += tr["r"]
        report = {"completed_episodes": int(done.sum()), "hard_terminations": int(tr["term"].sum()),
                  "normalized_return_sum": float((self.episode_normalized[done] / self.episode_steps[done]).sum()),
                  "length_steps_sum": float(self.episode_steps[done].sum()), "reward_sum": float(rewards.sum()),
                  "original_reward_sum": float(tr["r"].sum()), "root_bonus_sum": float((.02 * root).sum()),
                  "leg_bonus_sum": float((.01 * leg).sum()), "residual_cost_sum": float((.002 * cost).sum()),
                  "transitions": self.num_envs, "normalized_step_reward_sum": float((rewards / .13).sum()),
                  "alpha_sum": float(applied_alpha.sum()),
                  "fallback_rows": int((applied_alpha + 1e-5 < self.caps()).sum()) if self.fallback_enabled else 0,
                  "residual_squared_sum": float(residual.square().sum()),
                  "lower_action_kl_proxy_sum": float((.5 * ((executed[:, :14] - self.nominal[:, :14]) / .1).square().sum(-1)).sum()),
                  "artificial_timeouts": int(timeouts.sum())}
        for value in (self.episode_return, self.episode_normalized, self.episode_original, self.episode_steps):
            value[done] = 0
        self.reset_gate(done)
        self.current = state.info[SUPERMIMIC_STATE_KEY]
        self.observations = self.observe(self.current, reset=done)
        return self.observations, rewards, done, {"terminal_observations": terminal_obs,
                                                 "artificial_timeouts": timeouts, "report": report}

    def close(self):
        self.contacts.close()
        self.env.close()
        self.env.backend.close()


def make_ppo(actor, obs, count):
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    critic = MLPModel(obs, {"critic": ["features", "privileged"]}, "critic", 1,
                      hidden_dims=[256, 128], activation="elu", obs_normalization=False).to(obs.device)
    storage = RolloutStorage("rl", count, 24, obs, [14], device=str(obs.device))
    algorithm = PPO(actor, critic, storage, num_learning_epochs=5, num_mini_batches=4,
                    clip_param=.2, gamma=.99, lam=.95, learning_rate=1e-4, max_grad_norm=1.,
                    schedule="fixed", desired_kl=None, entropy_coef=0., use_mixed_precision=False,
                    device=str(obs.device))
    def finite_gradients(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError("Nonfinite residual PPO gradient; update prevented")
    algorithm.optimizer.register_step_pre_hook(finite_gradients)
    return algorithm
