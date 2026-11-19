"""H002-SONIC v3: 2,000-Clip Scaled Training, 4,096-Env GPU mjwarp Push/Delay Closed-Loop RL
& Massive-Batch (batch=4096) 4-Modality Causal Tokenizer Alignment.

Stage A (RTX 3090 mjwarp GPU Closed-Loop Co-Training):
  - 4,096 parallel GPU physics envs across 1,800 Train clips (~85k steps/s)
  - Randomized per-episode 0/20/40 ms actuator delay FIFO, PD gain mismatch, friction, CoM,
    and interval linear velocity pushes (x,y in [-0.55, 0.55] m/s)
  - Live batched GPU CGFA + Torque-Safe Hardware PD filter inside simulation & WorldModel
  - Phase A1 (120 iters): WM + Critic warmup on 3.1M-slot push/delay replay buffer (pi frozen)
  - Phase A2 (360 iters): Proximal closed-loop policy co-training anchored to v2b + stance no-slip,
    gated every 90 iters against a 200-clip Held-Out mjwarp validation benchmark (Nominal + Push/Delay Stress)

Stage B (Massive-Batch 4-Modality SONIC Tokenizers on 1,800 Train Clips):
  - Freeze TokenHead + Enc_RobotPose (SHA-256 codec_fingerprint)
  - Train wider causal tokenizers (HumanSMPL 430D, VR3Pt 360D, PlannerCmd 100D) at batch_size=4096
    for 3,600 steps/modality with 0/20/40 ms arrival jitter & realistic sensor noise
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgfa_estimator import LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, ContactGatedFootAnchor
from multimodal_features import (
    MODALITY_DIMS,
    build_human_smpl_goal,
    build_planner_cmd_goal,
    build_robot_pose_goal,
    build_vr_3pt_goal,
)
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import so3_exp, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Critic, Policy, RunningNorm, WorldModel, mlp, run_mlp
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, gains_of, motor_of
from unisupermimic.teleop.realtime_teleop import build_causal_history_goal, load_world_model

MJCF_PATH = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"


class ModalityTokenizer(nn.Module):
    """SONIC-style causal modality tokenizer mapping modal features -> shared 32D token."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], z_dim: int = 32):
        super().__init__()
        self.in_norm = RunningNorm(in_dim)
        self.enc = mlp(in_dim, hidden, z_dim)

    def encode_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = run_mlp(self.enc, self.in_norm(x))
        return u, torch.tanh(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(run_mlp(self.enc, self.in_norm(x)))


def compute_codec_fingerprint(token_head: Policy) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(
        list(token_head.dec.state_dict().items()) + list(token_head.obs_norm.state_dict().items())
    ):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _load_default_q(device: torch.device) -> torch.Tensor:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    qpos = model.key_qpos[0] if model.nkey > 0 else model.qpos0
    return torch.tensor(qpos[7 : 7 + len(Q1_JOINT_NAMES)], dtype=torch.float32, device=device)


def apply_torque_safe_pd_batch(
    act_raw: torch.Tensor,
    q_cur: torch.Tensor,
    qd_cur: torch.Tensor,
    xi_joint_off: torch.Tensor,
    default_q: torch.Tensor,
    alpha_vec: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    kp_vec: torch.Tensor,
    kd_vec: torch.Tensor,
    eff_vec: torch.Tensor,
) -> torch.Tensor:
    q_tgt_raw = default_q + xi_joint_off + alpha_vec * act_raw
    dq_tau = 0.5 * eff_vec / kp_vec
    q_tgt = torch.clamp(q_tgt_raw, q_lo - dq_tau, q_hi + dq_tau)
    tau_raw = kp_vec * (q_tgt - q_cur) - kd_vec * qd_cur
    tau_cl = torch.clamp(tau_raw, -eff_vec, eff_vec)
    q_tgt_safe = q_cur + (tau_cl + kd_vec * qd_cur) / kp_vec
    return (q_tgt_safe - default_q - xi_joint_off) / alpha_vec


def _make_mjwarp_env(motion_files: list[str], num_envs: int, *, eval_mode: bool) -> Any:
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config", overrides=["task=q1_super_mimic/mjwarp"])
    OmegaConf.set_struct(cfg, False)
    cfg.env.ctrl_dt = 0.02
    cfg.env.commands.motion.ctrl_dt = 0.02
    cfg.env.commands.motion.motion_files = motion_files
    cfg.env.commands.motion.observation_mode = "root_xy"
    if not eval_mode:
        cfg.env.events.startup_context.params.randomize_pd_gains = True
        cfg.env.events.reset_context.params.randomize_pd_gains = True
        cfg.env.events.startup_context.params.randomize_friction = True
        cfg.env.events.reset_context.params.randomize_friction = True
        cfg.env.events.startup_context.params.randomize_com = True
        cfg.env.events.reset_context.params.randomize_com = True
        cfg.env.events.push_robot = {
            "_target_": "unilab.managers.EventTermCfg",
            "func": "unilab.envs.mdp.push_by_setting_velocity",
            "mode": "interval",
            "interval_range_s": [1.8, 3.5],
            "params": {
                "velocity_range": {"x": [-0.55, 0.55], "y": [-0.55, 0.55]},
                "asset_cfg": {"_target_": "unilab.managers.SceneEntityCfg", "name": "robot"},
            },
        }
    else:
        cfg.env.events.startup_context.params.randomize_pd_gains = False
        cfg.env.events.reset_context.params.randomize_pd_gains = False
        cfg.env.events.startup_context.params.randomize_friction = False
        cfg.env.events.reset_context.params.randomize_friction = False
        cfg.env.events.startup_context.params.randomize_com = False
        cfg.env.events.reset_context.params.randomize_com = False
        cfg.env.events.pop("joint_offsets", None)
        cfg.env.events.push_robot = {
            "_target_": "unilab.managers.EventTermCfg",
            "func": "unilab.envs.mdp.push_by_setting_velocity",
            "mode": "interval",
            "interval_range_s": [1.0e9, 1.0e9],
            "params": {
                "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0]},
                "asset_cfg": {"_target_": "unilab.managers.SceneEntityCfg", "name": "robot"},
            },
        }
    ov = build_env_override(cfg, eval_mode=eval_mode, learner_device="cuda:0")
    return create_env(cfg, num_envs=num_envs, env_cfg_override=ov)


def _sample_delay_jitter_frames(motion: MotionLib, frames: torch.Tensor) -> torch.Tensor:
    """0/20/40 ms causal arrival jitter: 70% 0, 20% -1 (20ms), 10% -2 (40ms)."""
    u = torch.rand((frames.shape[0],), device=frames.device)
    loff = torch.where(u < 0.10, -2, torch.where(u < 0.30, -1, 0))
    cs = motion.clip_start[motion.clip_id[frames]]
    ce = motion.clip_end[motion.clip_id[frames]]
    return torch.clamp(frames + loff, min=cs, max=ce)


class HardenedCausalRunner:
    """4,096-env GPU mjwarp closed-loop runner with CGFA, 0-40ms delay FIFO, and v2b proximal anchoring."""

    def __init__(
        self,
        env: Any,
        wm: WorldModel,
        pi: Policy,
        critic: Critic,
        pi_v2b_ref: Policy,
        *,
        buffer_steps: int,
        wm_batch: int,
        pi_batch: int,
        horizon: int,
        goal_frames: int,
        hist: int,
        device: torch.device,
    ):
        self.env = env
        self.mdp: MDP = env.supermimic_mdp
        self.device = device
        self.num_envs = int(env.num_envs)
        self.goal_frames = goal_frames
        self.hist = hist
        self.horizon = horizon
        self.wm_batch = wm_batch
        self.pi_batch = pi_batch
        self.wm_window = 8
        self.gamma = 0.99
        self.lam = 0.95

        self.wm = wm
        self.pi = pi
        self.critic = critic
        self.pi_v2b_ref = pi_v2b_ref

        self.buffer = ReplayBuffer(
            buffer_steps,
            self.num_envs,
            device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
        )
        self.obs_hist = torch.zeros((self.num_envs, hist, self.mdp.obs_dim), device=device)
        self.act_fifo = torch.zeros((self.num_envs, 3, self.mdp.num_joints), device=device)
        self.delay_steps = torch.randint(0, 3, (self.num_envs,), device=device)
        self.env_arange = torch.arange(self.num_envs, device=device)

        self.est_collect = ContactGatedFootAnchor(self.mdp.kin, self.num_envs, dt=0.02, device=device)
        self.est_imag = ContactGatedFootAnchor(self.mdp.kin, pi_batch, dt=0.02, device=device)
        self.sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32, device=device)

        self.alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
        self.kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
        self.kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
        self.eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
        self.q_lo = self.mdp.kin.q_lo.to(device).view(1, -1)
        self.q_hi = self.mdp.kin.q_hi.to(device).view(1, -1)

        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(device)

        self.opt_wm = torch.optim.AdamW(self.wm.parameters(), lr=1.5e-4, weight_decay=1e-5)
        self.opt_v = torch.optim.AdamW(self.critic.parameters(), lr=1.5e-4, weight_decay=1e-5)
        self.opt_pi = torch.optim.AdamW(
            [
                {"params": list(self.pi.enc.parameters()), "lr": 3.5e-5},
                {"params": list(self.pi.dec.parameters()), "lr": 1.5e-5},
            ],
            weight_decay=0.0,
        )
        self.sim_steps = 0

    @torch.no_grad()
    def collect(self, n_steps: int, *, action_noise: float = 0.04) -> dict[str, float]:
        rew_acc, term_cnt, done_cnt = [], 0.0, 0.0
        for _ in range(n_steps):
            state = self.env.state if self.env.state is not None else self.env.init_state()
            info = state.info[SUPERMIMIC_STATE_KEY]
            st = unflatten_state(info["s"])
            f_cur = info["f"]
            fresh = info["ep_step"] == 0
            if bool(fresh.any()):
                self.act_fifo[fresh] = 0.0
                self.delay_steps = torch.where(
                    fresh, torch.randint(0, 3, (self.num_envs,), device=self.device), self.delay_steps
                )
                self.est_collect.reset_mask(
                    fresh,
                    st[1],
                    st[3],
                    st[4],
                    st[5],
                    self.mdp.motion.root_p[f_cur, :2],
                    init_error_xy=(self.mdp.motion.root_p[f_cur, :2] - st[0][:, :2]),
                    R_ref=self.mdp.motion.root_R[f_cur],
                )

            obs_raw = self.mdp.obs(st, info["a_prev"], info["xi"], f_cur)
            est_xy, _ = self.est_collect.step(
                st[1],
                st[3],
                st[4],
                st[5],
                self.mdp.motion.root_p[f_cur, :2],
                R_ref=self.mdp.motion.root_R[f_cur],
                mode="cgfa",
            )
            obs_cur = torch.cat(
                (obs_raw[:, :78], torch.where(fresh.unsqueeze(-1), obs_raw[:, 78:80], est_xy)), dim=-1
            )
            obs_cur = self.mdp.add_noise(obs_cur)
            shifted = torch.cat((self.obs_hist[:, 1:], obs_cur.unsqueeze(1)), dim=1)
            self.obs_hist = torch.where(
                fresh[:, None, None], obs_cur.unsqueeze(1).expand_as(self.obs_hist), shifted
            )

            f_jit = _sample_delay_jitter_frames(self.mdp.motion, f_cur)
            goal = build_causal_history_goal(self.mdp.motion, f_jit, self.goal_frames, latest_offset=0)
            act_raw = self.pi(self.obs_hist, goal)
            if action_noise > 0.0:
                act_raw = torch.clamp(act_raw + action_noise * torch.randn_like(act_raw), -12.0, 12.0)

            self.act_fifo = torch.cat((act_raw.unsqueeze(1), self.act_fifo[:, :-1, :]), dim=1)
            act_del = self.act_fifo[self.env_arange, self.delay_steps]
            act_sent = apply_torque_safe_pd_batch(
                act_del,
                st[4],
                st[5],
                info["xi"][:, 1:23],
                self.mdp.default_q.view(1, -1),
                self.alpha_vec,
                self.q_lo,
                self.q_hi,
                self.kp_vec,
                self.kd_vec,
                self.eff_vec,
            )

            step_s = self.env.step(act_sent)
            tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
            self.buffer.add(tr)
            s_in = unflatten_state(tr["s"])
            s_out = unflatten_state(tr["s1"])
            self.wm.in_norm.update(self.wm.inputs(s_in, tr["a"], self.mdp.normalize_context(tr["xi"])))
            self.wm.out_norm.update(self.wm.targets(s_in, s_out))

            self.sim_steps += self.num_envs
            rew_acc.append(float((tr["r"].mean() / self.mdp.max_step_reward).item()))
            if "term" in tr:
                term_cnt += float(tr["term"].float().sum().item())
            if "done" in tr:
                done_cnt += float(tr["done"].float().sum().item())

        return {
            "sim_rew": float(np.mean(rew_acc)) if rew_acc else 0.0,
            "term_frac": term_cnt / max(done_cnt, 1.0),
        }

    def wm_update(self) -> tuple[float, float]:
        w = self.buffer.sample_windows(self.wm_batch, self.wm_window)
        state = unflatten_state(w["s"][:, 0])
        preds = []
        for k in range(self.wm_window):
            state = self.wm.step(state, w["a"][:, k], self.mdp.normalize_context(w["xi"][:, k]))
            preds.append(state)
        pred_tuple = tuple(torch.stack([p[idx] for p in preds], dim=1) for idx in range(6))
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        err = ((self.wm.state_error(pred_tuple, unflatten_state(w["s1"])) / std) ** 2).mean(-1)
        loss = (err * w["mask"]).sum() / w["s1"].shape[0]
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        m_last = w["mask"][:, -1]
        err8 = float(((err[:, -1].detach() * m_last).sum() / m_last.sum().clamp(min=1.0)).item())
        return float((loss.detach() / self.wm_window).item()), err8

    def _sample_valid_starts(self, batch_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        starts, histories, count = [], [], 0
        while count < batch_size:
            s, h = self.buffer.sample_starts(2 * batch_size, self.hist)
            valid = self.mdp.valid_imagination_start(s["f"], self.horizon + 2)
            idx = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({k: v[idx] for k, v in s.items()})
            histories.append({k: v[idx] for k, v in h.items()})
            count += len(idx)
        return (
            {k: torch.cat([p[k] for p in starts]) for k in starts[0]},
            {k: torch.cat([p[k] for p in histories]) for k in histories[0]},
        )

    def _compute_stance_slip(self, states_tuple: tuple[torch.Tensor, ...], flat_frames: torch.Tensor) -> torch.Tensor:
        p, R, v, w, q, qd = states_tuple
        _, bR, bv, bw = self.mdp.kin.forward(p, R, v, w, q, qd)
        ref_ap = self.mdp.motion.body_p[flat_frames][:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
        ref_av = self.mdp.motion.body_v[flat_frames][:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2]
        c_stance = (
            torch.sigmoid((0.062 - ref_ap[..., 2]) / 0.008)
            * torch.sigmoid((0.15 - ref_av.norm(dim=-1)) / 0.04)
        ).detach()
        fR = bR[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)]
        fv = bv[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)]
        fw = bw[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)]
        r_sole = (fR @ self.sole_off.view(1, 1, 3, 1)).squeeze(-1)
        v_sole = fv + torch.cross(fw, r_sole, dim=-1)
        return (c_stance * (v_sole[..., :2] ** 2).sum(dim=-1)).sum(dim=-1).mean()

    def policy_and_critic_update(self, *, update_pi: bool) -> dict[str, float]:
        start, history = self._sample_valid_starts(self.pi_batch)
        state = unflatten_state(start["s"])
        xi, frames, a_prev = start["xi"], start["f"], start["a_prev"]
        norm_xi = self.mdp.normalize_context(xi)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]
            )
            obs_hist = self.mdp.add_noise(obs_hist)
            init_err = self.mdp.motion.root_p[frames, :2] - state[0][:, :2]
            self.est_imag.reset(
                state[1], state[3], state[4], state[5],
                self.mdp.motion.root_p[frames, :2], init_error_xy=init_err, R_ref=self.mdp.motion.root_R[frames]
            )

        for p in self.wm.parameters():
            p.requires_grad_(False)
        for p in self.critic.parameters():
            p.requires_grad_(False)

        init_goal_f = self.mdp.goal(frames, self.goal_frames)
        states_l, acts_l, prev_acts_l, frames_l, goals_f_l, ref_acts_l = [], [], [], [], [], []
        init_state = state
        f_cur = frames
        B = self.pi_batch
        batch_arange = torch.arange(B, device=self.device)
        is_hard = batch_arange >= (B // 2)
        delay_k = torch.where(
            is_hard,
            torch.randint(0, 3, (B,), device=self.device),
            torch.zeros((B,), dtype=torch.long, device=self.device),
        )
        imag_fifo = a_prev.unsqueeze(1).repeat(1, 3, 1)
        dv_push = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        dv_push[:, :2] = (
            torch.empty((B, 2), dtype=torch.float32, device=self.device).uniform_(-0.62, 0.62)
            * is_hard.unsqueeze(-1).float()
        )

        grad_ctx = torch.enable_grad() if update_pi else torch.no_grad()
        try:
            with grad_ctx:
                for step_i in range(self.horizon):
                    if step_i == 2:
                        state = (state[0], state[1], state[2] + dv_push, state[3], state[4], state[5])
                    prev_acts_l.append(a_prev)
                    with torch.no_grad():
                        f_jit = torch.where(is_hard, _sample_delay_jitter_frames(self.mdp.motion, f_cur), f_cur)
                        g_c = build_causal_history_goal(self.mdp.motion, f_jit, self.goal_frames, latest_offset=0)
                        a_ref = self.pi_v2b_ref(obs_hist, g_c)
                    action = self.pi(obs_hist, g_c)
                    imag_fifo = torch.cat((action.unsqueeze(1), imag_fifo[:, :-1, :]), dim=1)
                    act_del = imag_fifo[batch_arange, delay_k]
                    act_exec = apply_torque_safe_pd_batch(
                        act_del,
                        state[4],
                        state[5],
                        xi[:, 1:23],
                        self.mdp.default_q.view(1, -1),
                        self.alpha_vec,
                        self.q_lo,
                        self.q_hi,
                        self.kp_vec,
                        self.kd_vec,
                        self.eff_vec,
                    )
                    next_state = self.wm.step(state, act_exec, norm_xi)
                    f_next = f_cur + 1
                    obs_n = self.mdp.obs(next_state, act_exec, xi, f_next)
                    e_xy_n, _ = self.est_imag.step(
                        next_state[1], next_state[3], next_state[4], next_state[5],
                        self.mdp.motion.root_p[f_next, :2], R_ref=self.mdp.motion.root_R[f_next],
                        mode="cgfa", slip_noise_std_m=0.0020, differentiable=update_pi,
                    )
                    obs_n = self.mdp.add_noise(torch.cat((obs_n[..., :78], e_xy_n), dim=-1))
                    obs_hist = torch.cat((obs_hist[:, 1:], obs_n.unsqueeze(1)), dim=1)

                    g_next_f = self.mdp.goal(f_next, self.goal_frames)
                    states_l.append(next_state)
                    acts_l.append(action)
                    ref_acts_l.append(a_ref)
                    frames_l.append(f_next)
                    goals_f_l.append(g_next_f)
                    state = next_state
                    a_prev = act_exec
                    f_cur = f_next

                all_states = tuple(torch.stack([s[i] for s in states_l]).flatten(0, 1) for i in range(6))
                flat_frames = torch.stack(frames_l).flatten()
                flat_acts = torch.stack(acts_l).flatten(0, 1)
                flat_prev = torch.stack(prev_acts_l).flatten(0, 1)
                rewards = self.mdp.reward(all_states, flat_frames, flat_acts, flat_prev)
                dones = self.mdp.soft_termination(all_states, flat_frames)
                rewards = rewards.view(self.horizon, self.pi_batch)
                dones = dones.view(self.horizon, self.pi_batch)

                alive = torch.cumprod(1.0 - dones, dim=0)
                cont = torch.cat((torch.ones((1, self.pi_batch), device=self.device), alive[:-1]), dim=0)
                disc = (self.gamma ** torch.arange(self.horizon, device=self.device, dtype=torch.float32)).unsqueeze(-1)
                objective = (disc * cont * rewards).sum(0)
                survival = alive[-1]
                v_term_in = torch.cat(
                    (self.mdp.critic_features(state, f_cur, norm_xi), goals_f_l[-1]), dim=-1
                )
                objective = objective + (self.gamma ** self.horizon) * survival * self.critic(v_term_in)

                pi_loss_val = 0.0
                if update_pi:
                    mse_per_traj = ((torch.stack(acts_l) - torch.stack(ref_acts_l)) ** 2).mean(dim=-1).mean(dim=0)
                    anchor_w = torch.where(is_hard, 0.03, 0.45)
                    act_anchor = (anchor_w * mse_per_traj).mean()
                    l_prox = sum(
                        (p - p0).pow(2).sum()
                        for p, p0 in zip(
                            list(self.pi.enc.parameters()) + list(self.pi.dec.parameters()),
                            list(self.pi_v2b_ref.enc.parameters()) + list(self.pi_v2b_ref.dec.parameters()),
                        )
                    )
                    l_slip = self._compute_stance_slip(all_states, flat_frames)
                    loss_pi = -objective.mean() + act_anchor + 1.5 * l_prox + 0.04 * l_slip
                    self.opt_pi.zero_grad(set_to_none=True)
                    loss_pi.backward()
                    gn = nn.utils.clip_grad_norm_(
                        list(self.pi.enc.parameters()) + list(self.pi.dec.parameters()), 1.0
                    )
                    if torch.isfinite(gn):
                        self.opt_pi.step()
                    pi_loss_val = float(loss_pi.detach().item())
        finally:
            for p in self.wm.parameters():
                p.requires_grad_(True)
            for p in self.critic.parameters():
                p.requires_grad_(True)

        # Critic TD(lambda) update
        with torch.no_grad():
            c_states = tuple(
                torch.cat((init_state[i].unsqueeze(0), torch.stack([s[i].detach() for s in states_l])), dim=0).flatten(0, 1)
                for i in range(6)
            )
            c_frames = torch.cat((frames.unsqueeze(0), torch.stack(frames_l)), dim=0).flatten()
            c_goals = torch.cat((init_goal_f.unsqueeze(0), torch.stack(goals_f_l)), dim=0).flatten(0, 1)
            c_xi = norm_xi.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
            c_in = torch.cat((self.mdp.critic_features(c_states, c_frames, c_xi), c_goals), dim=-1).view(
                self.horizon + 1, self.pi_batch, -1
            )
        self.critic.norm.update(c_in)
        with torch.no_grad():
            vals = self.critic(c_in)
            tgt = vals[-1]
            tgts = []
            dr = rewards.detach()
            dd = dones.detach()
            for idx in range(self.horizon - 1, -1, -1):
                tgt = dr[idx] + self.gamma * (1.0 - dd[idx]) * ((1.0 - self.lam) * vals[idx + 1] + self.lam * tgt)
                tgts.append(tgt)
            tgts_t = torch.stack(tgts[::-1])
        flat_in = c_in[:-1].reshape(-1, c_in.shape[-1])
        flat_tgt = tgts_t.reshape(-1)
        mb = flat_in.shape[0] // 4
        v_loss_val = 0.0
        for _ in range(3):
            perm = torch.randperm(flat_in.shape[0], device=self.device)
            for k in range(4):
                ind = perm[k * mb : (k + 1) * mb]
                vl = ((self.critic(flat_in[ind]) - flat_tgt[ind]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                vl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
                v_loss_val = float(vl.detach().item())

        imag_rew = float((cont * rewards.detach()).sum(0).mean().item() / self.horizon / self.mdp.max_step_reward)
        return {
            "pi_loss": pi_loss_val,
            "v_loss": v_loss_val,
            "imag_rew": imag_rew,
            "imag_surv": float(survival.detach().mean().item()),
        }


@torch.no_grad()
def _eval_val_200(val_env: Any, pi: Policy, *, stress: bool, device: torch.device) -> dict[str, float]:
    """Deterministic 200-clip mjwarp validation under Nominal or Hardened Push+20ms-Delay stress."""
    mdp: MDP = val_env.supermimic_mdp
    motion = mdp.motion
    n_envs = int(val_env.num_envs)
    n_clips = int(motion.clip_len.numel())
    assert n_envs >= n_clips, f"val_env {n_envs} < {n_clips}"

    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=device).view(1, -1)
    q_lo, q_hi = mdp.kin.q_lo.to(device).view(1, -1), mdp.kin.q_hi.to(device).view(1, -1)
    kp_eff = (
        kp_vec * (1.0 + 0.10 * torch.linspace(-1.0, 1.0, n_envs, device=device).unsqueeze(-1))
        if stress
        else kp_vec
    )
    slip_std = 0.0025 if stress else 0.0

    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(20261119 if not stress else 20261120)
        clip_idx = torch.arange(n_envs, device=device) % n_clips
        _, reset_info = val_env.reset_to_frames(motion.clip_start[clip_idx].clone())
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])
        f0 = cur["f"]

        est = ContactGatedFootAnchor(mdp.kin, n_envs, dt=0.02, device=device)
        est.reset(s0[1], s0[3], s0[4], s0[5], motion.root_p[f0, :2], R_ref=motion.root_R[f0])
        obs_hist = None
        act_prev_delayed = torch.zeros((n_envs, mdp.num_joints), device=device)
        finished = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        failed = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        sum_ret = torch.zeros((n_envs,), dtype=torch.float32, device=device)
        steps_run = torch.zeros((n_envs,), dtype=torch.float32, device=device)

        max_budget = min(int(motion.clip_len[clip_idx].max().item()) - 1, 320)
        robot = val_env.scene["robot"]

        for t in range(max_budget):
            if stress and t in (60, 120):
                dv = torch.zeros((n_envs, 3), dtype=torch.float32, device=device)
                if t == 60:
                    dv[:, 1] = 0.55
                else:
                    dv[:, 0] = 0.40
                    dv[:, 1] = -0.40
                dv *= (~finished).unsqueeze(-1).float()
                robot.apply_root_linear_velocity_delta_to_sim(dv)

            st = unflatten_state(cur["s"])
            f_cur = cur["f"]
            obs_raw = mdp.obs(st, cur["a_prev"], cur["xi"], f_cur)
            exy, _ = est.step(
                st[1], st[3], st[4], st[5], motion.root_p[f_cur, :2],
                R_ref=motion.root_R[f_cur], mode="cgfa", slip_noise_std_m=slip_std,
            )
            obs_78 = obs_raw[:, :78] + (0.012 * torch.randn_like(obs_raw[:, :78]) if stress else 0.0)
            obs_pol = torch.cat((obs_78, obs_raw[:, 78:80] if t == 0 else exy), dim=-1)
            obs_hist = obs_pol.unsqueeze(1).repeat(1, 10, 1) if obs_hist is None else torch.cat((obs_hist[:, 1:], obs_pol.unsqueeze(1)), dim=1)
            goal = build_causal_history_goal(motion, f_cur, 10, latest_offset=0)
            act = pi(obs_hist, goal)

            if stress:
                act_eff = act if t == 0 else act_prev_delayed
                act_prev_delayed = act
            else:
                act_eff = act

            act_sent = apply_torque_safe_pd_batch(
                act_eff, st[4], st[5], cur["xi"][:, 1:23], mdp.default_q.view(1, -1),
                alpha_vec, q_lo, q_hi, kp_eff, kd_vec, eff_vec,
            )
            step_s = val_env.step(act_sent)
            tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
            term = tr["term"].bool()
            done = tr["done"].bool()
            active = ~finished
            sum_ret += active.float() * tr["r"]
            steps_run += active.float()
            failed |= active & term
            finished |= term | done
            if bool(finished[:n_clips].all()):
                break
            cur = step_s.info[SUPERMIMIC_STATE_KEY]

    surv = 1.0 - failed[:n_clips].float()
    norm_r = sum_ret[:n_clips] / (steps_run[:n_clips].clamp(min=1.0) * float(mdp.max_step_reward))
    return {
        "surv": round(float(surv.mean().item()), 4),
        "norm_ret": round(float(norm_r.mean().item()), 4),
    }


def _calibrate_norm(norm: RunningNorm, build_fn: Any, motion: MotionLib, total_frames: int, device: torch.device) -> None:
    with torch.no_grad():
        dim = norm.mean.numel()
        sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
        sum_sq = torch.zeros(dim, dtype=torch.float64, device=device)
        for s in range(0, total_frames, 4096):
            e = min(s + 4096, total_frames)
            fc = torch.arange(s, e, dtype=torch.long, device=device)
            gx = build_fn(motion, fc, 10).double()
            sum_x += gx.sum(dim=0)
            sum_sq += (gx * gx).sum(dim=0)
        mean_x = sum_x / max(total_frames, 1)
        var_x = (sum_sq / max(total_frames, 1) - mean_x * mean_x).clamp(min=1e-3)
        norm.mean.copy_(mean_x.float())
        norm.var.copy_(var_x.float())
        norm.count.fill_(float(total_frames))


def _sample_obs_histories_batch(
    mdp: MDP, teacher: Policy, frames: torch.Tensor, hist: int, noise_scale: float = 0.18
) -> tuple[torch.Tensor, torch.Tensor]:
    motion = mdp.motion
    B = frames.shape[0]
    dev = frames.device
    clip_start = motion.clip_start[motion.clip_id[frames]]
    offsets = torch.arange(-(hist - 1), 1, device=dev, dtype=frames.dtype)
    h_frames = torch.clamp(frames.unsqueeze(-1) + offsets.unsqueeze(0), min=clip_start.unsqueeze(-1))
    flat_f = h_frames.reshape(-1)

    xi = torch.zeros((B, mdp.context_dim), dtype=torch.float32, device=dev)
    xi[:, 0] = 0.45 + 0.80 * torch.rand((B,), device=dev)
    flat_xi = xi.unsqueeze(1).expand(-1, hist, -1).reshape(-1, mdp.context_dim)

    p, R, v, w, q, qd = (t.clone() for t in mdp.ref_state(flat_f))
    n = flat_f.shape[0]
    p = p + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.025, 0.025)
    R = so3_exp(noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.05, 0.05)) @ R
    v = v + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.15, 0.15)
    w = w + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.20, 0.20)
    q = torch.clamp(q + noise_scale * torch.empty((n, mdp.num_joints), device=dev).uniform_(-0.05, 0.05), mdp.kin.q_lo, mdp.kin.q_hi)
    st_hist = (p, R, v, w, q, qd)

    zero_a = torch.zeros((n, mdp.num_joints), device=dev)
    obs0 = mdp.obs(st_hist, zero_a, flat_xi, flat_f).view(B, hist, mdp.obs_dim)
    a_teacher = teacher(obs0, mdp.goal(frames, 10))
    use_zero = torch.rand((B, 1), device=dev) < 0.5
    flat_a = torch.where(use_zero, torch.zeros_like(a_teacher), a_teacher).unsqueeze(1).expand(-1, hist, -1).reshape(-1, mdp.num_joints)

    obs_t = mdp.obs(st_hist, flat_a, flat_xi, flat_f).view(B, hist, mdp.obs_dim)
    obs_s = obs_t.clone()
    drift_bias = torch.randn((B, 1, 2), device=dev) * 0.015
    step_noise = torch.randn((B, hist, 2), device=dev) * 0.004
    obs_s[..., 78:80] = 0.25 * torch.tanh((obs_t[..., 78:80] + drift_bias + step_noise) / 0.25)
    return obs_s, obs_t


def train_pipeline(args: argparse.Namespace) -> None:
    t_total = time.monotonic()
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_files = sorted(str(ROOT / r["csv_rel"]) for r in manifest["clips"] if r["split"] == "train")
    held_out_files = sorted(str(ROOT / r["csv_rel"]) for r in manifest["clips"] if r["split"] == "held_out")
    if args.smoke:
        train_files = train_files[:32]
        held_out_files = held_out_files[:20]
    print(f"Dataset loaded: train_clips={len(train_files)}, held_out_clips={len(held_out_files)}", flush=True)

    base_ck = torch.load(args.base_ckpt, map_location="cpu", weights_only=True)
    v2_bundle = torch.load(args.v2_bundle, map_location="cpu", weights_only=True)
    c_args = base_ck["args"]
    goal_frames = int(c_args["goal_frames"])
    hist = int(c_args["hist"])
    z_dim = int(c_args["z_dim"])
    hidden = tuple(c_args.get("test_hidden_dims") or c_args["policy_hidden"])

    # ---------------- STAGE A: 4,096-Env GPU mjwarp Closed-Loop Hardening ----------------
    networks.AMP_DTYPE = torch.bfloat16
    num_envs = 256 if args.smoke else args.num_envs
    val_envs = len(held_out_files)
    buf_steps = 32 if args.smoke else args.buffer_steps
    wm_batch = 256 if args.smoke else args.wm_batch
    pi_batch = 256 if args.smoke else args.pi_batch
    a1_iters = 6 if args.smoke else args.phase_a1_iters
    a2_iters = 6 if args.smoke else args.phase_a2_iters
    val_every = 6 if args.smoke else args.val_every

    train_env = _make_mjwarp_env(train_files, num_envs, eval_mode=False)
    val_env = _make_mjwarp_env(held_out_files, val_envs, eval_mode=True)
    mdp = train_env.supermimic_mdp

    wm: WorldModel = load_world_model(mdp, base_ck, device)
    wm.train()
    for p in wm.parameters():
        p.requires_grad_(True)
    pi = Policy(mdp.obs_dim, mdp.goal_frame_dim * goal_frames, num_joints=mdp.num_joints, hist=hist, z_dim=z_dim, hidden=hidden).to(device)
    pi.load_state_dict(v2_bundle["token_head_pi"])
    pi.obs_norm.count.fill_(1e9)
    pi.goal_norm.count.fill_(1e9)
    for p in pi.obs_norm.parameters():
        p.requires_grad_(False)
    for p in pi.goal_norm.parameters():
        p.requires_grad_(False)

    pi_v2b_ref = copy.deepcopy(pi).eval()
    for p in pi_v2b_ref.parameters():
        p.requires_grad_(False)

    # Compute critic input dim via 1 test sample
    sample_info = (train_env.state if train_env.state is not None else train_env.init_state()).info[SUPERMIMIC_STATE_KEY]
    with torch.no_grad():
        cf = mdp.critic_features(unflatten_state(sample_info["s"])[:6], sample_info["f"], mdp.normalize_context(sample_info["xi"]))
    critic_dim = int(cf.shape[-1] + mdp.goal_frame_dim * goal_frames)
    critic = Critic(critic_dim, hidden=tuple(c_args.get("critic_hidden", (256, 128)))).to(device)

    runner = HardenedCausalRunner(
        train_env, wm, pi, critic, pi_v2b_ref,
        buffer_steps=buf_steps, wm_batch=wm_batch, pi_batch=pi_batch,
        horizon=20, goal_frames=goal_frames, hist=hist, device=device,
    )

    warmup_n = 12 if args.smoke else args.warmup_steps
    t_stage_a = time.perf_counter()
    print(f"STAGE A: Filling initial ReplayBuffer ({warmup_n} x {num_envs} steps)...", flush=True)
    runner.collect(warmup_n, action_noise=0.04)

    # Validation Gate @ iter 0 (v2b baseline on 200 Held-Out in mjwarp)
    v0_nom = _eval_val_200(val_env, pi, stress=False, device=device)
    v0_str = _eval_val_200(val_env, pi, stress=True, device=device)
    best_score = 0.45 * v0_nom["surv"] + 0.55 * v0_str["surv"]
    best_pi_sd = {k: v.detach().cpu().clone() for k, v in pi.state_dict().items()}
    val_history = [{"iter": 0, "nom": v0_nom, "stress": v0_str, "score": round(best_score, 4), "selected": True}]
    print(f"[VALGATE iter=  0] NOM surv={v0_nom['surv']*100:.1f}% ret={v0_nom['norm_ret']:.3f} | PUSH+DELAY STR surv={v0_str['surv']*100:.1f}% ret={v0_str['norm_ret']:.3f} | score={best_score:.4f}", flush=True)

    # Phase A1: WM + Critic warmup (pi frozen)
    for it in range(1, a1_iters + 1):
        runner.collect(4, action_noise=0.05)
        wm_losses = [runner.wm_update() for _ in range(4)]
        pc = runner.policy_and_critic_update(update_pi=False)
        if it == 1 or it % max(a1_iters // 4, 1) == 0 or it == a1_iters:
            print(
                f"[A1 WM+Crit {it:3d}/{a1_iters}] sim_steps={runner.sim_steps:,} "
                f"wm={np.mean([x[0] for x in wm_losses]):.4f} err8={np.mean([x[1] for x in wm_losses]):.4f} "
                f"v_loss={pc['v_loss']:.4f} imag_surv={pc['imag_surv']:.4f}",
                flush=True,
            )

    # Phase A2: Proximal Closed-Loop Co-Training
    for it in range(1, a2_iters + 1):
        c_stats = runner.collect(6, action_noise=0.05)
        wm_losses = [runner.wm_update() for _ in range(2)]
        pc = runner.policy_and_critic_update(update_pi=True)
        if it == 1 or it % max(a2_iters // 8, 1) == 0 or it == a2_iters:
            print(
                f"[A2 CoTrain {it:3d}/{a2_iters}] sim_steps={runner.sim_steps:,} "
                f"wm={np.mean([x[0] for x in wm_losses]):.4f} imag_rew={pc['imag_rew']:.4f} "
                f"imag_surv={pc['imag_surv']:.4f} sim_rew={c_stats['sim_rew']:.4f} term={c_stats['term_frac']:.4f}",
                flush=True,
            )
        if it % val_every == 0 or it == a2_iters:
            vn = _eval_val_200(val_env, pi, stress=False, device=device)
            vs = _eval_val_200(val_env, pi, stress=True, device=device)
            sc = 0.45 * vn["surv"] + 0.55 * vs["surv"]
            keep = (sc > best_score) and (vn["surv"] >= v0_nom["surv"] - 0.01)
            if keep:
                best_score = sc
                best_pi_sd = {k: v.detach().cpu().clone() for k, v in pi.state_dict().items()}
            val_history.append({"iter": it, "nom": vn, "stress": vs, "score": round(sc, 4), "selected": bool(keep)})
            print(f"[VALGATE iter={it:3d}] NOM surv={vn['surv']*100:.1f}% ret={vn['norm_ret']:.3f} | PUSH+DELAY STR surv={vs['surv']*100:.1f}% ret={vs['norm_ret']:.3f} | score={sc:.4f} KEEP={keep}", flush=True)

    stage_a_wall = round(time.perf_counter() - t_stage_a, 2)
    total_sim_steps = runner.sim_steps
    peak_vram_stage_a = float(torch.cuda.max_memory_allocated() / 1e9)
    print(f"STAGE A DONE in {stage_a_wall}s | sim_steps={total_sim_steps:,} | peak_vram={peak_vram_stage_a:.2f} GB", flush=True)

    # Cleanly free GPU physics envs & ReplayBuffer before Stage B
    del runner, wm, critic, pi_v2b_ref
    train_env.close(); train_env.backend.close()
    val_env.close(); val_env.backend.close()
    del train_env, val_env
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- STAGE B: Massive-Batch (4,096) 4-Modality Tokenizer Training ----------------
    networks.AMP_DTYPE = None
    kin_b = Q1Kinematics(str(MJCF_PATH), device)
    motion_b = MotionLib(train_files, kin_b, device, src_fps=30.0)
    default_q = _load_default_q(device)
    mdp_b = MDP(kin_b, motion_b, default_q, observation_mode="root_xy")
    total_frames = int(motion_b.num_frames)

    teacher = Policy(mdp_b.obs_dim, mdp_b.goal_frame_dim * goal_frames, num_joints=mdp_b.num_joints, hist=hist, z_dim=z_dim, hidden=hidden).to(device)
    teacher.load_state_dict(base_ck["pi"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = Policy(mdp_b.obs_dim, mdp_b.goal_frame_dim * goal_frames, num_joints=mdp_b.num_joints, hist=hist, z_dim=z_dim, hidden=hidden).to(device)
    student.load_state_dict(best_pi_sd)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    codec_fp = compute_codec_fingerprint(student)
    print(f"STAGE B START: train_frames_50hz={total_frames:,}, frozen TokenHead fp={codec_fp[:24]}...", flush=True)

    p2_steps = 20 if args.smoke else args.p2_steps
    p2_batch = 512 if args.smoke else args.p2_batch
    modal_specs = [
        ("human_smpl", MODALITY_DIMS["human_smpl"], (1536, 1024, 512), lambda m, f, k: build_human_smpl_goal(m, f, k, noise_std=0.014), lambda m, f, k: build_human_smpl_goal(m, f, k, noise_std=0.0)),
        ("vr_3pt", MODALITY_DIMS["vr_3pt"], (1536, 1024, 512), lambda m, f, k: build_vr_3pt_goal(m, f, k, pos_noise_m=0.012), lambda m, f, k: build_vr_3pt_goal(m, f, k, pos_noise_m=0.0)),
        ("planner_cmd", MODALITY_DIMS["planner_cmd"], (1024, 512, 256), lambda m, f, k: build_planner_cmd_goal(m, f, k), lambda m, f, k: build_planner_cmd_goal(m, f, k)),
    ]

    tokenizers: dict[str, ModalityTokenizer] = {}
    p2_logs: dict[str, Any] = {}
    for mname, indim, mhid, fn_noisy, fn_clean in modal_specs:
        tm0 = time.perf_counter()
        tok = ModalityTokenizer(indim, mhid, z_dim=z_dim).to(device)
        _calibrate_norm(tok.in_norm, fn_clean, motion_b, total_frames, device)
        for p in tok.in_norm.parameters():
            p.requires_grad_(False)
        for p in tok.enc.parameters():
            p.requires_grad_(True)

        opt_m = torch.optim.AdamW(tok.enc.parameters(), lr=6e-4, weight_decay=1e-4)
        sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=max(p2_steps, 1), eta_min=1e-5)
        mlog: dict[str, Any] = {}
        for step in range(1, p2_steps + 1):
            fb = torch.randint(0, total_frames, (p2_batch,), device=device)
            with torch.no_grad():
                f_jit = _sample_delay_jitter_frames(motion_b, fb)
                g_rob = build_robot_pose_goal(motion_b, fb, goal_frames, latest_offset=0)
                u_rob = run_mlp(student.enc, student.goal_norm(g_rob))
                z_rob = torch.tanh(u_rob)
                g_f = mdp_b.goal(fb, goal_frames)
                z_t = torch.tanh(run_mlp(teacher.enc, teacher.goal_norm(g_f)))
                obs_s, obs_t = _sample_obs_histories_batch(mdp_b, teacher, fb, hist, noise_scale=0.18)
                obs_s_flat = student.obs_norm(obs_s).flatten(-2)
                a_rob = run_mlp(student.dec, torch.cat((z_rob, obs_s_flat), dim=-1))
                a_t = run_mlp(teacher.dec, torch.cat((z_t, teacher.obs_norm(obs_t).flatten(-2)), dim=-1))
                g_m = fn_noisy(motion_b, f_jit, goal_frames)

            u_m, z_m = tok.encode_raw(g_m)
            a_m = run_mlp(student.dec, torch.cat((z_m, obs_s_flat), dim=-1))
            loss_m = (
                0.7 * F.mse_loss(z_m, z_rob)
                + 0.3 * F.mse_loss(z_m, z_t)
                + 0.05 * F.mse_loss(u_m, u_rob)
                + 1.2 * F.mse_loss(a_m, a_rob)
                + 0.4 * F.mse_loss(a_m, a_t)
            )
            opt_m.zero_grad(set_to_none=True)
            loss_m.backward()
            nn.utils.clip_grad_norm_(tok.enc.parameters(), 1.0)
            opt_m.step()
            sched_m.step()

            mlog["s1_cos"] = float(F.cosine_similarity(z_m.detach(), z_rob, dim=-1).mean().item())
            mlog["s1_a_mse"] = float(F.mse_loss(a_m.detach(), a_rob).item())
            mlog["s1_a_mse_t"] = float(F.mse_loss(a_m.detach(), a_t).item())
            if step == 1 or step % max(p2_steps // 4, 1) == 0 or step == p2_steps:
                print(
                    f"[P2 {mname:11s} {step:4d}/{p2_steps}] cos_rob={mlog['s1_cos']:.4f} "
                    f"a_mse_rob={mlog['s1_a_mse']:.4f} a_mse_t={mlog['s1_a_mse_t']:.4f}",
                    flush=True,
                )
        mlog["hidden"] = list(mhid)
        mlog["batch_size"] = p2_batch
        mlog["steps"] = p2_steps
        mlog["samples_seen"] = p2_batch * p2_steps
        mlog["wall_s"] = round(time.perf_counter() - tm0, 2)
        tok.eval()
        tokenizers[mname] = tok.cpu()
        p2_logs[mname] = mlog

    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    summary = {
        "architecture": "H002-SONIC v3 (2,000-Clip Multi-Dataset + 4,096-Env GPU mjwarp Push/Delay RL + 4,096-Batch Wide SONIC Tokenizers)",
        "total_wall_s": round(time.monotonic() - t_total, 2),
        "stage_a_wall_s": stage_a_wall,
        "peak_vram_gb": round(max(peak_vram_gb, peak_vram_stage_a), 3),
        "train_clips": len(train_files),
        "held_out_clips": len(held_out_files),
        "train_frames_50hz": total_frames,
        "mjwarp_num_envs": num_envs,
        "total_sim_steps": total_sim_steps,
        "codec_fingerprint": codec_fp,
        "val_gate_history": val_history,
        "phase2_modalities": p2_logs,
    }

    if args.smoke:
        print("V3_SMOKE_PASS:", json.dumps(summary, indent=2), flush=True)
        return

    out_bundle = {
        "args": c_args,
        "obs_dim": mdp_b.obs_dim,
        "num_joints": mdp_b.num_joints,
        "hist": hist,
        "z_dim": z_dim,
        "hidden": hidden,
        "codec_fingerprint": codec_fp,
        "token_head_pi": student.cpu().state_dict(),
        "robot_pose_hidden": hidden,
        "modal_tokenizers": {
            k: {
                "in_dim": MODALITY_DIMS[k],
                "hidden": tuple(p2_logs[k]["hidden"]),
                "state_dict": v.state_dict(),
            }
            for k, v in tokenizers.items()
        },
        "summary": summary,
    }
    args.out_bundle.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_bundle, args.out_bundle)
    args.out_bundle.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED_V3_BUNDLE -> {args.out_bundle}", flush=True)


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest", type=Path, default=base / "dataset_2000_manifest.json")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/q1_base_soma1k.pt")
    parser.add_argument("--v2-bundle", type=Path, default=ROOT / "runs/h002_sonic_v2/checkpoints/h002_sonic_v2_bundle.pt")
    parser.add_argument("--out-bundle", type=Path, default=base / "checkpoints/h002_sonic_v3_bundle.pt")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--buffer-steps", type=int, default=768)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--wm-batch", type=int, default=4096)
    parser.add_argument("--pi-batch", type=int, default=4096)
    parser.add_argument("--phase-a1-iters", type=int, default=90)
    parser.add_argument("--phase-a2-iters", type=int, default=240)
    parser.add_argument("--val-every", type=int, default=60)
    parser.add_argument("--p2-batch", type=int, default=4096)
    parser.add_argument("--p2-steps", type=int, default=3200)
    args = parser.parse_args()
    train_pipeline(args)


if __name__ == "__main__":
    main()
