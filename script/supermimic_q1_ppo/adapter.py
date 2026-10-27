"""RSL-RL interfaces around the frozen HP policy and physical Q1 transitions."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution


class HPActor(MLPModel):
    """Keep HP's encoder/decoder; use RSL's fixed Gaussian likelihoods."""

    def __init__(self, policy):
        torch.nn.Module.__init__(self)
        self.policy = policy
        self.distribution = GaussianDistribution(22, init_std=.1, learn_std=False)
        self.obs_normalization = False

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None or hidden_state is not None:
            raise ValueError("RF12 uses feedforward, unpadded batches")
        history, goal = obs["history"].reshape(-1, 10, 80), obs["goal"]
        means = []
        # BF16 GEMM may select different arithmetic for different batch shapes.
        # Fixed blocks keep collection and likelihood recomputation identical.
        for start in range(0, len(goal), 4096):
            h, g = history[start:start + 4096], goal[start:start + 4096]
            count = len(g)
            if count < 4096:
                h = torch.cat((h, h.new_zeros(4096 - count, 10, 80)))
                g = torch.cat((g, g.new_zeros(4096 - count, 590)))
            means.append(self.policy(h, g)[:count].float())
        mean = torch.cat(means)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def update_normalization(self, obs):
        # A fixed normalizer keeps old/new PPO likelihoods on identical inputs.
        return


def tensor_observations(policy, history, goal, state, context, mdp):
    critic = torch.cat((policy.obs_norm(history).flatten(1), policy.goal_norm(goal),
                        state[:, :35], state[:, 35:] * .05, mdp.normalize_context(context)), -1)
    return TensorDict({"history": history.flatten(1), "goal": goal, "critic": critic},
                      batch_size=[len(state)], device=state.device)


def termination_masks(transition, motion, max_episode_steps):
    frames = transition["f"]
    natural = frames + 1 >= motion.clip_end[motion.clip_id[frames]]
    artificial = transition["ep_step"] + 1 >= max_episode_steps
    hard, done = transition["term"].bool(), transition["done"].bool()
    if not torch.equal(done, hard | natural | artificial):
        raise AssertionError("Unrecognized physical done cause")
    return natural, done & ~hard & ~natural


def reward_components(mdp, next_state, frames, alpha):
    from unisupermimic.rl.algos.supermimic.geometry import yaw_mat

    q_error = next_state[4][:, 14:] - mdp.motion.q[frames, 14:]
    joint = torch.exp(-(q_error / .8).square()).mean(-1)
    wrists = [mdp.kin.tracked_idx.index(mdp.kin.body_names.index(side + "_wrist_yaw_link"))
              for side in ("left", "right")]
    actual = mdp.kin.forward(*next_state)[0][:, wrists]
    reference = mdp.motion.body_p[frames][:, wrists]
    actual_local = (yaw_mat(next_state[1]).transpose(-1, -2).unsqueeze(1)
                    @ (actual - next_state[0].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    reference_local = (yaw_mat(mdp.motion.root_R[frames]).transpose(-1, -2).unsqueeze(1)
                       @ (reference - mdp.motion.root_p[frames].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    wrist = torch.exp(-((actual_local - reference_local) / .15).square().sum(-1)).mean(-1)
    return joint, wrist, alpha * (.02 * joint + .01 * wrist)


class PhysicalAdapter:
    def __init__(self, env, actor, old_clip_count=300):
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

        self.env, self.actor = env, actor
        self.mdp, self.command = env.supermimic_mdp, env._supermimic_command
        self.num_envs, self.num_actions, self.device = env.num_envs, 22, env.device
        self.max_episode_length = self.command.max_episode_steps
        self.alpha, self.new_fraction = 0., 0.
        self.history = None
        self.old_clip_count = old_clip_count
        self.install_sampler()
        if env.state is None:
            env.init_state()
        else:
            env.reset()
        self.current = env.state.info[SUPERMIMIC_STATE_KEY]
        self.observations = self.observe(self.current, reset=torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_normalized = torch.zeros_like(self.episode_return)
        self.episode_original = torch.zeros_like(self.episode_return)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_transition = None

    def install_sampler(self):
        command, motion = self.command, self.mdp.motion
        module = sys.modules[type(command).__module__]
        original = module.start_bin_probabilities
        groups = ((motion.bin_clip < self.old_clip_count).nonzero().flatten(),
                  (motion.bin_clip >= self.old_clip_count).nonzero().flatten())
        if not len(groups[0]):
            raise ValueError("Original AP02 training bins are missing")

        def sample(motion, bin_g, bin_avg, gamma, num_samples, rng):
            if motion is not self.mdp.motion:
                return self.original_sampler(motion, bin_g, bin_avg, gamma, num_samples, rng)
            probabilities = torch.zeros_like(bin_g)
            for group, mass in zip(groups, (1 - self.new_fraction, self.new_fraction)):
                if mass == 0:
                    continue
                if not len(group):
                    raise ValueError("AP03 additions are required for expansion")
                average = bin_avg[group].clone()
                probabilities[group] = mass * original(bin_g[group], average, gamma)
                bin_avg[group] = average
            bins = rng.choice(motion.num_bins, (num_samples,), p=probabilities)
            starts = motion.bin_start[bins]
            ends = torch.minimum(starts + 50, motion.clip_end[motion.bin_clip[bins]])
            frames = starts + (rng.random((num_samples,)) * (ends - starts).float()).long()
            if not bool((frames < motion.clip_end[motion.clip_id[frames]]).all()):
                raise AssertionError("Sampled terminal frame as a transition start")
            return frames, bins

        self.original_sampler = module.sample_start_frames
        self.sampler_module = module
        module.sample_start_frames = sample

    def observe(self, current, reset=None):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

        obs = self.mdp.add_noise(self.mdp.obs(unflatten_state(current["s"]), current["a_prev"],
                                               current["xi"], current["f"]))
        if self.history is None:
            self.history = obs[:, None].repeat(1, 10, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            if reset is not None:
                self.history[reset] = obs[reset, None].expand(-1, 10, -1)
        return tensor_observations(self.actor.policy, self.history, self.mdp.goal(current["f"], 10),
                                   current["s"], current["xi"], self.mdp)

    @torch.no_grad()
    def step(self, actions):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY

        executed = actions.clamp(-20, 20)
        state = self.env.step(executed)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        self.last_transition = transition
        torch.testing.assert_close(transition["a"], executed, rtol=0, atol=0)
        _, timeouts = termination_masks(transition, self.mdp.motion, self.max_episode_length)
        terminal_obs = None
        if bool(timeouts.any()):
            subset = {key: value[timeouts] for key, value in transition.items()
                      if key in ("s1", "f", "a", "xi")}
            measured = self.mdp.add_noise(self.mdp.obs(unflatten_state(subset["s1"]), subset["a"],
                                                        subset["xi"], subset["f"] + 1))
            terminal_history = torch.cat((self.history[timeouts, 1:], measured[:, None]), 1)
            terminal_obs = tensor_observations(self.actor.policy, terminal_history,
                                               self.mdp.goal(subset["f"] + 1, 10), subset["s1"], subset["xi"], self.mdp)
        joint, wrist, bonus = reward_components(self.mdp, unflatten_state(transition["s1"]), transition["f"] + 1, self.alpha)
        rewards = transition["r"] + bonus
        done = transition["done"].bool()
        self.episode_steps += 1
        self.episode_return += rewards
        self.episode_normalized += rewards / (self.mdp.max_step_reward + self.alpha * .03)
        self.episode_original += transition["r"]
        count = int(done.sum())
        report = {"completed_episodes": count, "hard_terminations": int(transition["term"].sum()),
                  "normalized_return_sum": float((self.episode_normalized[done] / self.episode_steps[done]).sum()),
                  "length_steps_sum": float(self.episode_steps[done].sum()),
                  "old_episode_return_sum": float(self.episode_original[done].sum()),
                  "reward_sum": float(rewards.sum()), "original_reward_sum": float(transition["r"].sum()),
                  "joint_reward_sum": float(joint.sum()), "wrist_reward_sum": float(wrist.sum()),
                  "bonus_reward_sum": float(bonus.sum()), "transitions": self.num_envs,
                  "normalized_step_reward_sum": float((rewards / (self.mdp.max_step_reward + self.alpha * .03)).sum()),
                  "artificial_timeouts": int(timeouts.sum()), "reward_alpha": self.alpha,
                  "new_motion_fraction": self.new_fraction}
        for accumulator in (self.episode_return, self.episode_normalized, self.episode_original, self.episode_steps):
            accumulator[done] = 0
        self.current = state.info[SUPERMIMIC_STATE_KEY]
        self.observations = self.observe(self.current, reset=done)
        return self.observations, rewards, done, {"terminal_observations": terminal_obs,
                                                "artificial_timeouts": timeouts, "report": report}

    def close(self):
        self.sampler_module.sample_start_frames = self.original_sampler
        self.env.close()
        self.env.backend.close()


def bootstrap_rewards(rewards, extras, critic, gamma=.99):
    """RSL's stock time_outs path bootstraps V(s_t); use actual V(s_t+1)."""
    result = rewards.clone()
    if extras["terminal_observations"] is not None:
        result[extras["artificial_timeouts"]] += gamma * critic(extras["terminal_observations"]).squeeze(-1)
    return result


def make_ppo(actor, obs, num_envs):
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage

    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1,
                      hidden_dims=[256, 128], activation="elu", obs_normalization=False).to(obs.device)
    storage = RolloutStorage("rl", num_envs, 24, obs, [22], device=str(obs.device))
    algorithm = PPO(actor, critic, storage, num_learning_epochs=5, num_mini_batches=4,
                    clip_param=.2, gamma=.99, lam=.95, learning_rate=1e-4,
                    max_grad_norm=1., schedule="fixed", desired_kl=None,
                    entropy_coef=0., use_mixed_precision=False, device=str(obs.device))

    def finite_gradients(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Nonfinite PPO gradient; optimizer step prevented")

    algorithm.optimizer.register_step_pre_hook(finite_gradients)
    return algorithm
