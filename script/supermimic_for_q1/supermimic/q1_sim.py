"""Q1 physics transitions through MuJoCo's native threaded rollout engine."""

import copy
from dataclasses import dataclass

import mujoco
from mujoco.rollout import Rollout
import numpy as np
import torch

from supermimic.robot_spec import ACT_DIM, NQ, OBS_DIM, STATE_DIM, XI_DIM
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at


@dataclass
class SimCfg:
    num_envs: int = 16
    threads: int = 4
    domain_randomization: bool = True
    observation_noise: bool = True
    action_noise: float = .1
    episode_steps: int = 500
    history: int = 10


class Q1Sim:
    def __init__(self, assets, library, cfg, heads, seed=0):
        self.assets, self.lib, self.cfg, self.heads = assets, library, cfg, heads
        self.device = library.device
        self.rng = np.random.default_rng(seed)
        self.E = cfg.num_envs
        self.ids = torch.arange(self.E, device=self.device)
        model = copy.copy(assets.model)
        for i, ai in enumerate(assets.aids):
            model.actuator_gainprm[ai, 0] = assets.kp[i]
            model.actuator_biasprm[ai, :3] = [0., -assets.kp[i], -assets.kd[i]]
            model.actuator_biastype[ai] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_forcelimited[ai] = True
            model.actuator_forcerange[ai] = [-assets.effort[i], assets.effort[i]]
        if model.na or mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS) != STATE_DIM + 1:
            raise ValueError("Q1 requires [time, qpos29, qvel28], with no actuator activation")
        self.models = [copy.copy(model) for _ in range(self.E)]
        self.data = [mujoco.MjData(model) for _ in range(min(cfg.threads, self.E))]
        self.pool = Rollout(nthread=len(self.data))
        self.initial = np.zeros((self.E, STATE_DIM + 1), dtype=np.float64)
        self.torso_id = model.body("torso_link").id
        self.base_com = model.body_ipos[self.torso_id].copy()
        self.context = torch.zeros(self.E, XI_DIM, device=self.device)
        self.state = torch.zeros(self.E, STATE_DIM, device=self.device)
        self.frame = torch.zeros(self.E, device=self.device, dtype=torch.long)
        self.previous_action = torch.zeros(self.E, ACT_DIM, device=self.device)
        self.hist = torch.zeros(cfg.history, self.E, OBS_DIM, device=self.device)
        self.steps = torch.zeros(self.E, device=self.device, dtype=torch.long)
        self.ep_start = torch.ones(self.E, device=self.device, dtype=torch.bool)
        self.bin_return = torch.zeros(library.num_bins, device=self.device)
        self.bin_memory = torch.ones(library.num_bins, device=self.device)
        self.trajectories = [[] for _ in range(self.E)]
        self.noise_half = torch.tensor([.01] * ACT_DIM + [.5] * ACT_DIM + [.2] * 3
                                      + [.05] * 3 + [0.] * ACT_DIM, device=self.device)
        self.reset(self.ids)

    def close(self):
        self.pool.close()

    @torch.no_grad()
    def reset(self, ids, frames=None):
        n = len(ids)
        if frames is None:
            soft = torch.softmax(-25. * self.bin_return / 10., dim=0)
            self.bin_memory.add_(soft)
            prob = .5 / self.lib.num_bins + .4 * soft + .1 * self.bin_memory / self.bin_memory.sum()
            bins = torch.multinomial(prob, n, replacement=True)
            lo, hi = self.lib.bin_start[bins], self.lib.bin_end[bins]
            frames = lo + (torch.rand(n, device=self.device) * (hi - lo).float()).long()
        self.frame[ids] = frames
        qpos, qvel = self.lib.ref_qpos_qvel(frames)
        self.state[ids] = torch.cat([qpos, qvel], dim=-1)
        self.previous_action[ids] = 0.
        self.steps[ids] = 0
        self.ep_start[ids] = True
        self.context[ids] = 0.
        self.context[ids, 0] = 1.
        if self.cfg.domain_randomization:
            self.context[ids, 0] = torch.rand(n, device=self.device) * .9 + .3
            self.context[ids, 1:1+ACT_DIM] = (torch.rand(n, ACT_DIM, device=self.device) * 2. - 1.) * .01
            self.context[ids, 1+ACT_DIM:] = ((torch.rand(n, 3, device=self.device) * 2. - 1.)
                                             * torch.tensor([.025, .05, .05], device=self.device))
        for idx in ids.tolist():
            model = self.models[idx]
            model.geom_friction[self.assets.foot_geoms, 0] = float(self.context[idx, 0])
            model.body_ipos[self.torso_id] = self.base_com + self.context[idx, 1+ACT_DIM:].cpu().numpy()
            mujoco.mj_setConst(model, self.data[0])
            self.trajectories[idx] = []
        self.hist[:, ids] = self.observation(ids)[None]

    def observation(self, ids=None):
        ids = self.ids if ids is None else ids
        obs = analytic_observation(self.state[ids], self.previous_action[ids], self.context[ids],
                                   self.heads.default_joint_pos, self.heads.active_indices)
        if self.cfg.observation_noise:
            obs = obs + (torch.rand_like(obs) * 2. - 1.) * self.noise_half
        return obs

    def policy_inputs(self):
        return flatten_history_term_major(self.hist), self.lib.goal_window(self.frame, 10)

    def _update_returns(self, ids):
        values, bins = [], []
        for idx in ids.tolist():
            ret = 0.
            for bin_id, reward in reversed(self.trajectories[idx]):
                ret = reward + .99 * ret
                bins.append(bin_id)
                values.append(ret)
        if bins:
            bins = torch.tensor(bins, device=self.device)
            values = torch.tensor(values, device=self.device)
            sums = torch.zeros_like(self.bin_return).scatter_add_(0, bins, values)
            counts = torch.zeros_like(self.bin_return).scatter_add_(0, bins, torch.ones_like(values))
            selected = counts > 0
            self.bin_return[selected] = .9 * self.bin_return[selected] + .1 * sums[selected] / counts[selected]

    @torch.no_grad()
    def step(self, action, buffer=None, auto_reset=True):
        old_state, old_frame = self.state.clone(), self.frame.clone()
        old_action = self.previous_action.clone()
        clean = action.clamp(-20., 20.)
        executed = (clean + torch.randn_like(clean) * self.cfg.action_noise).clamp(-20., 20.)
        self.initial[:, 0] = 0.
        self.initial[:, 1:] = self.state.cpu().numpy()
        control = np.zeros((self.E, 1, self.assets.model.nu), dtype=np.float64)
        control[:, 0, self.assets.aids] = self.assets.q0 + executed.cpu().numpy() * self.assets.scale
        states, _ = self.pool.rollout(self.models, self.data, self.initial, control, nstep=20)
        if not np.isfinite(states).all() or not np.allclose(states[:, -1, 0], .02, atol=1e-10):
            raise RuntimeError("Q1 physics did not advance a complete finite control interval")
        self.state = torch.as_tensor(states[:, -1, 1:].copy(), device=self.device, dtype=torch.float32)
        ref = reference_at(self.lib, old_frame)
        bodies = self.heads.robot_bodies(self.state)
        terms = self.heads.reward(self.state, clean, old_action, ref, bodies)
        reward = terms["total"]
        terminated = self.heads.terminated(self.state, ref)
        self.steps += 1
        timeout = (self.steps >= self.cfg.episode_steps) | (old_frame >= self.lib.last_frame_of(old_frame))
        done = terminated | timeout
        self.frame = torch.minimum(old_frame + 1, self.lib.last_frame_of(old_frame))
        clip = self.lib.frame_clip[old_frame]
        if buffer is not None:
            buffer.add(self.ids, old_state, clean, executed, self.context, reward, terminated.float(),
                       old_frame, clip, torch.zeros_like(done), self.ep_start)
        self.ep_start[:] = False
        for idx, (bin_id, value) in enumerate(zip(self.lib.bin_of_frame(old_frame).tolist(), reward.tolist())):
            self.trajectories[idx].append((bin_id, value))
        self.previous_action = clean
        self.hist = torch.cat([self.hist[1:], self.observation()[None]], dim=0)
        result = {"reward": reward.clone(), "terminated": terminated.clone(), "timeout": timeout.clone(),
                  "state": self.state.clone(), "terms": terms, "frames": old_frame,
                  "joint_rmse": ((self.state[:, 7:NQ] - self.lib.ref(old_frame).joint_pos) ** 2).mean(-1).sqrt(),
                  "body_error_m": (bodies["pos"] - ref.body_pos_w).norm(dim=-1).mean(-1)}
        done_ids = done.nonzero().squeeze(-1)
        if len(done_ids) and auto_reset:
            if buffer is not None:
                zero_action = torch.zeros(len(done_ids), ACT_DIM, device=self.device)
                zeros = torch.zeros(len(done_ids), device=self.device)
                buffer.add(done_ids, self.state[done_ids], zero_action, zero_action, self.context[done_ids],
                           zeros, terminated[done_ids].float(), self.frame[done_ids], clip[done_ids],
                           torch.ones(len(done_ids), dtype=torch.bool, device=self.device),
                           torch.zeros(len(done_ids), dtype=torch.bool, device=self.device))
                buffer.end_episode(done_ids)
            self._update_returns(done_ids)
            self.reset(done_ids)
        return result
