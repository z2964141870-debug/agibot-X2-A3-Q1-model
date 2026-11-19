"""Comprehensive H002-SONIC v3 Evaluation Suite:
  1. 200-Clip v3 Held-Out Multi-Dataset Benchmark (GPU mjwarp, Nominal + Hardened Stress)
     across Oracle, Zeroed, V1_H002 (12-clip), V2_SONIC (200-clip), V3_SONIC (1,727-clip)
  2. 50-Clip v2 Held-Out Apples-to-Apples Sim2Sim Benchmark (CPU MuJoCo 3.11 + 1-Thread ORT ONNX)
  3. Push-Impulse (0..0.95 m/s) & Actuator-Delay (0..60 ms) Robustness Sweeps (GPU mjwarp)
  4. 24-Clip Zero-IK Raw Human .bvh Generalization (CPU MuJoCo 3.11 + ORT ONNX, V2 vs V3)
  5. 30.0s (1,500-step) Continuous 4-Modality Mode-Switch + Live Shoves + 20ms Delay + 400ms Watchdog
     + Head-to-Head 0.70 m/s Push + 40ms Delay Rollout Recording (V2 vs V3)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgfa_estimator import (
    LEFT_ANKLE_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_WRIST_IDX,
    ContactGatedFootAnchor,
)
from multimodal_features import (
    Q1_NOMINAL_PELVIS_Z,
    build_human_smpl_goal,
    build_planner_cmd_goal,
    build_robot_pose_goal,
    build_synthetic_planner_stream,
    build_vr_3pt_goal,
    load_bvh_as_mock_motion,
)
from train_h002_sonic_v3_mjwarp import ModalityTokenizer, apply_torque_safe_pd_batch
from unilab.base.config_adapter import create_env
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import ManagerTermBase, SceneEntityCfg
from unilab.training import ensure_registries
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import mat_to_6d, mat_to_quat, so3_exp, unflatten_state, yaw_mat
from unisupermimic.rl.algos.supermimic.networks import Policy, run_mlp
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, gains_of, motor_of
from unisupermimic.teleop.realtime_teleop import build_causal_history_goal


def _sel_defaults(defaults: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    vals = defaults.to(device=ids.device, dtype=torch.float32)
    return vals.expand((ids.numel(), *vals.shape)).clone() if vals.ndim == 2 else vals.index_select(0, ids).clone()


class EvalFrictionEvent(ManagerTermBase):
    """Deterministic reset-time friction scaling across robot and ground for mjwarp & mujoco."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        asset_cfg = cfg.params["asset_cfg"]
        ground_cfg = cfg.params["ground_asset_cfg"]
        self._robot = env.scene[asset_cfg.name]
        self._ground = env.scene[ground_cfg.name]
        self._r_gids, self._r_def = self._robot.bind_geom_friction_write(asset_cfg.geom_ids, term_name="eval_fric")
        self._g_gids, self._g_def = self._ground.bind_geom_friction_write(ground_cfg.geom_ids, term_name="eval_gfric")

    def __call__(self, env: Any, env_ids: torch.Tensor | None, **_: Any) -> None:
        ids = resolve_reset_env_ids(env_ids, num_envs=env.num_envs, device=env.device, owner="EvalFrictionEvent")
        scale = float(getattr(env, "_eval_friction_scale", 1.0))
        rf = _sel_defaults(self._r_def, ids)
        rf[..., 0] *= scale
        self._robot.write_geom_friction_to_sim(rf, geom_ids=self._r_gids, env_ids=ids, term_name="eval_fric")
        gf = _sel_defaults(self._g_def, ids)
        gf[..., 0] *= scale
        self._ground.write_geom_friction_to_sim(gf, geom_ids=self._g_gids, env_ids=ids, term_name="eval_gfric")


def _configure_eval_events(cfg: Any) -> None:
    cfg.env.events.startup_context.params.randomize_pd_gains = False
    cfg.env.events.reset_context.params.randomize_pd_gains = False
    cfg.env.events.startup_context.params.randomize_friction = False
    cfg.env.events.reset_context.params.randomize_friction = False
    cfg.env.events.startup_context.params.randomize_com = False
    cfg.env.events.reset_context.params.randomize_com = False
    cfg.env.events.pop("joint_offsets", None)
    cfg.env.events.eval_friction = {
        "_target_": "unilab.managers.EventTermCfg",
        "func": "eval_sonic_v3_suite.EvalFrictionEvent",
        "mode": "reset",
        "params": {
            "asset_cfg": {"_target_": "unilab.managers.SceneEntityCfg", "name": "robot"},
            "ground_asset_cfg": {"_target_": "unilab.managers.SceneEntityCfg", "name": "ground", "geom_names": ["floor"]},
        },
    }
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


def create_mjwarp_eval_env(motion_files: list[str], num_envs: int) -> Any:
    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"):
        cfg = compose("config", overrides=["task=q1_super_mimic/mjwarp"])
    OmegaConf.set_struct(cfg, False)
    cfg.env.ctrl_dt = 0.02
    cfg.env.commands.motion.ctrl_dt = 0.02
    cfg.env.commands.motion.motion_files = motion_files
    cfg.env.commands.motion.observation_mode = "root_xy"
    _configure_eval_events(cfg)
    ov = build_env_override(cfg, eval_mode=True, learner_device="cuda:0")
    return create_env(cfg, num_envs=num_envs, env_cfg_override=ov)


def create_cpu_mujoco_eval_env(motion_files: list[str]) -> Any:
    torch.set_num_threads(1)
    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"):
        cfg = compose("config", overrides=["task=q1_super_mimic/mujoco"])
    OmegaConf.set_struct(cfg, False)
    cfg.training.sim_backend = "mujoco"
    cfg.training.play_only = True
    cfg.env.ctrl_dt = 0.02
    cfg.env.commands.motion.ctrl_dt = 0.02
    cfg.env.commands.motion.motion_files = motion_files
    cfg.env.commands.motion.observation_mode = "root_xy"
    _configure_eval_events(cfg)
    ov = build_env_override(cfg, eval_mode=True, learner_device="cpu")
    return create_env(cfg, num_envs=1, env_cfg_override=ov)


def _ort_sess(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _load_ckpt_policy(mdp: Any, path: Path, device: str | torch.device) -> Policy:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    args = ck["args"]
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])
    obs_dim = int(ck["pi"]["obs_norm.mean"].numel())
    pi = Policy(
        obs_dim, mdp.goal_frame_dim * int(args["goal_frames"]),
        num_joints=mdp.num_joints, hist=int(args["hist"]), z_dim=int(args["z_dim"]), hidden=hidden,
    ).to(device)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    return pi


def _load_sonic_bundle(mdp: Any, path: Path, device: str | torch.device) -> tuple[Policy, dict[str, ModalityTokenizer]]:
    b = torch.load(path, map_location="cpu", weights_only=True)
    args = b["args"]
    z_dim = int(b["z_dim"])
    pi = Policy(
        int(b["obs_dim"]), mdp.goal_frame_dim * int(args["goal_frames"]),
        num_joints=mdp.num_joints, hist=int(b["hist"]), z_dim=z_dim, hidden=tuple(b["hidden"]),
    ).to(device)
    pi.load_state_dict(b["token_head_pi"])
    pi.eval()
    toks: dict[str, ModalityTokenizer] = {}
    for mname, minfo in b["modal_tokenizers"].items():
        tm = ModalityTokenizer(int(minfo["in_dim"]), tuple(minfo["hidden"]), z_dim=z_dim).to(device)
        tm.load_state_dict(minfo["state_dict"])
        tm.eval()
        toks[mname] = tm
    return pi, toks


def _build_goal(motion: Any, f_ref: torch.Tensor, modality: str, **kwargs: Any) -> torch.Tensor:
    if modality == "robot_pose":
        return build_causal_history_goal(motion, f_ref, 10, latest_offset=0)
    if modality == "human_smpl":
        return build_human_smpl_goal(motion, f_ref, 10, latest_offset=0, h_nom=kwargs.get("h_nom", Q1_NOMINAL_PELVIS_Z))
    if modality == "vr_3pt":
        return build_vr_3pt_goal(motion, f_ref, 10, latest_offset=0)
    if modality == "planner_cmd":
        return build_planner_cmd_goal(motion, f_ref, 10, latest_offset=0, h_nom=kwargs.get("h_nom", Q1_NOMINAL_PELVIS_Z))
    if modality == "oracle_future10":
        return motion.goal(f_ref, 10)
    raise ValueError(modality)


def _apply_torque_safe_pd_cpu(
    q_tgt_raw: torch.Tensor,
    q_cur: torch.Tensor,
    qd_cur: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    kp_vec: torch.Tensor,
    kd_vec: torch.Tensor,
    eff_vec: torch.Tensor,
    *,
    q_init: torch.Tensor | None = None,
    step_i: int = 0,
    engage_steps: int = 0,
) -> torch.Tensor:
    if engage_steps > 0 and step_i < engage_steps and q_init is not None:
        blend = 0.5 * (1.0 - np.cos(np.pi * float(step_i + 1) / float(engage_steps)))
        q_tgt_raw = q_init + float(blend) * (q_tgt_raw - q_init)
    dq_tau = 0.5 * eff_vec / kp_vec
    q_tgt = torch.clamp(q_tgt_raw, q_lo - dq_tau, q_hi + dq_tau)
    tau_raw = kp_vec * (q_tgt - q_cur) - kd_vec * qd_cur
    tau_cl = torch.clamp(tau_raw, -eff_vec, eff_vec)
    return q_cur + (tau_cl + kd_vec * qd_cur) / kp_vec


@torch.no_grad()
def eval_mjwarp_batch(
    env: Any,
    *,
    pi: Policy,
    tok: ModalityTokenizer | None,
    est_mode: str,
    modality: str,
    deploy_safety: bool,
    num_clips: int,
    friction_scale: float = 1.0,
    delay_steps: int = 0,
    pushes: list[tuple[int, float, float]] | tuple[tuple[int, float, float], ...] = (),
    stress_sensor: bool = False,
    clip_meta: list[dict[str, Any]] | None = None,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Deterministic batched GPU mjwarp rollout across up to num_clips distinct motions."""
    mdp = env.supermimic_mdp
    motion = mdp.motion
    n_envs = int(env.num_envs)
    dev = env.device
    assert n_envs >= num_clips

    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32, device=dev).view(1, -1)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=dev).view(1, -1)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=dev).view(1, -1)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32, device=dev).view(1, -1)
    q_lo, q_hi = mdp.kin.q_lo.to(dev).view(1, -1), mdp.kin.q_hi.to(dev).view(1, -1)
    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32, device=dev).view(1, 1, 3, 1)

    kp_eff = (
        kp_vec * (1.0 + 0.10 * torch.linspace(-1.0, 1.0, n_envs, device=dev).unsqueeze(-1))
        if stress_sensor
        else kp_vec
    )
    yaw_bias_R = so3_exp(torch.tensor([[0.0, 0.0, float(np.radians(1.5))]], dtype=torch.float32, device=dev))[0]
    slip_std = 0.0025 if stress_sensor else 0.0
    push_map = {step: (vx, vy) for step, vx, vy in pushes}

    env._eval_friction_scale = float(friction_scale)
    try:
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(seed)
            clip_idx = torch.arange(n_envs, device=dev) % num_clips
            _, reset_info = env.reset_to_frames(motion.clip_start[clip_idx].clone())
            cur = reset_info[SUPERMIMIC_STATE_KEY]
            s0 = unflatten_state(cur["s"])
            f0 = cur["f"]

            est = ContactGatedFootAnchor(mdp.kin, n_envs, dt=0.02, device=dev)
            est.reset(s0[1], s0[3], s0[4], s0[5], motion.root_p[f0, :2], R_ref=motion.root_R[f0])
            obs_hist = None
            act_fifo = torch.zeros((n_envs, delay_steps + 1, mdp.num_joints), dtype=torch.float32, device=dev)
            finished = torch.zeros((n_envs,), dtype=torch.bool, device=dev)
            failed = torch.zeros((n_envs,), dtype=torch.bool, device=dev)

            sum_ret = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            sum_body_sq = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            sum_root_sq = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            sum_wrist_sq = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            sum_slip = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            cnt_slip = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            pd_oob = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            pd_tot = torch.zeros((n_envs,), dtype=torch.float32, device=dev)
            steps_run = torch.zeros((n_envs,), dtype=torch.float32, device=dev)

            max_budget = int(motion.clip_len[:num_clips].max().item()) - 1
            robot = env.scene["robot"]

            for t in range(max_budget):
                if t in push_map:
                    vx, vy = push_map[t]
                    dv = torch.zeros((n_envs, 3), dtype=torch.float32, device=dev)
                    dv[:, 0] = vx
                    dv[:, 1] = vy
                    dv *= (~finished).unsqueeze(-1).float()
                    robot.apply_root_linear_velocity_delta_to_sim(dv)

                st = unflatten_state(cur["s"])
                f_cur = cur["f"]
                R_obs, w_obs = st[1], st[3]
                if stress_sensor:
                    R_obs = yaw_bias_R.unsqueeze(0) @ R_obs
                    w_obs = w_obs + 0.03 * torch.randn_like(w_obs)

                obs_raw = mdp.obs((st[0], R_obs, st[2], w_obs, st[4], st[5]), cur["a_prev"], cur["xi"], f_cur)
                exy, _ = est.step(
                    R_obs, w_obs, st[4], st[5], motion.root_p[f_cur, :2],
                    R_ref=motion.root_R[f_cur], mode=est_mode, true_obs_root_xy=obs_raw[:, 78:80],
                    slip_noise_std_m=slip_std,
                )
                obs_78 = obs_raw[:, :78] + (0.012 * torch.randn_like(obs_raw[:, :78]) if stress_sensor else 0.0)
                obs_pol = torch.cat((obs_78, obs_raw[:, 78:80] if t == 0 and est_mode != "zeroed" else exy), dim=-1)
                obs_hist = obs_pol.unsqueeze(1).repeat(1, 10, 1) if obs_hist is None else torch.cat((obs_hist[:, 1:], obs_pol.unsqueeze(1)), dim=1)

                goal = _build_goal(motion, f_cur, modality)
                if tok is not None:
                    z = tok(goal)
                    act = run_mlp(pi.dec, torch.cat((z, pi.obs_norm(obs_hist).flatten(-2)), dim=-1))
                else:
                    act = pi(obs_hist, goal)

                act_fifo = torch.cat((act.unsqueeze(1), act_fifo[:, :-1, :]), dim=1)
                act_eff = act_fifo[:, min(t, delay_steps), :].contiguous()

                if deploy_safety:
                    act_sent = apply_torque_safe_pd_batch(
                        act_eff, st[4], st[5], cur["xi"][:, 1:23], mdp.default_q.view(1, -1),
                        alpha_vec, q_lo, q_hi, kp_eff, kd_vec, eff_vec,
                    )
                else:
                    act_sent = act_eff

                q_tgt_sent = mdp.default_q.view(1, -1) + cur["xi"][:, 1:23] + alpha_vec * act_sent
                tau = kp_vec * (q_tgt_sent - st[4]) - kd_vec * st[5]
                oob = ((torch.abs(tau) - eff_vec) > 1e-4).sum(dim=-1).float()

                step_s = env.step(act_sent.contiguous())
                tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
                f_next = motion.future_frames(tr["f"], 1).squeeze(-1)
                s1 = unflatten_state(tr["s1"])
                _, terms = mdp.reward(s1, f_next, tr["a"], tr["a_prev"], terms=True)
                bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)

                active = (~finished).float()
                sum_ret += active * tr["r"]
                sum_body_sq += active * terms["e_body_pos"]
                sum_root_sq += active * ((s1[0][:, :2] - motion.root_p[f_next, :2]) ** 2).sum(dim=-1)
                wrist_sq = ((bp1[:, (LEFT_WRIST_IDX, RIGHT_WRIST_IDX)] - motion.body_p[f_next][:, (LEFT_WRIST_IDX, RIGHT_WRIST_IDX)]) ** 2).sum(-1).mean(-1)
                sum_wrist_sq += active * wrist_sq

                ref_az = motion.body_p[f_next][:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]
                ref_av = motion.body_v[f_next][:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].norm(dim=-1)
                smask = ((ref_az <= 0.062) & (ref_av <= 0.15)) & (~finished).unsqueeze(-1)
                v_sole = bv1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] + torch.cross(
                    bw1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)],
                    (bR1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] @ sole_off).squeeze(-1),
                    dim=-1,
                )
                slip_spd = v_sole[..., :2].norm(dim=-1) * 1000.0
                sum_slip += (slip_spd * smask.float()).sum(dim=-1)
                cnt_slip += smask.float().sum(dim=-1)

                pd_oob += active * oob
                pd_tot += active * float(mdp.num_joints)
                steps_run += active

                term = tr["term"].bool()
                done = tr["done"].bool()
                failed |= (~finished) & term
                finished |= term | done
                if bool(finished[:num_clips].all()):
                    break
                cur = step_s.info[SUPERMIMIC_STATE_KEY]
    finally:
        env._eval_friction_scale = 1.0

    k = num_clips
    budgets = (motion.clip_len[:k] - 1).float().clamp(min=1.0)
    completed = ((~failed[:k]) & (steps_run[:k] >= budgets)).cpu().numpy()
    surv_frac = (steps_run[:k] / budgets).cpu().numpy()
    norm_ret = (sum_ret[:k] / (steps_run[:k].clamp(min=1.0) * float(mdp.max_step_reward))).cpu().numpy()
    body_cm = (torch.sqrt(sum_body_sq[:k] / steps_run[:k].clamp(min=1.0)) * 100.0).cpu().numpy()
    root_cm = (torch.sqrt(sum_root_sq[:k] / steps_run[:k].clamp(min=1.0)) * 100.0).cpu().numpy()
    wrist_cm = (torch.sqrt(sum_wrist_sq[:k] / steps_run[:k].clamp(min=1.0)) * 100.0).cpu().numpy()
    slip_mm = (sum_slip[:k] / cnt_slip[:k].clamp(min=1.0)).cpu().numpy()
    oob_pct = (100.0 * pd_oob[:k] / pd_tot[:k].clamp(min=1.0)).cpu().numpy()

    def _agg(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n_clips": int(mask.sum()),
            "survival_rate": round(float(completed[mask].mean()), 4),
            "survival_fraction": round(float(surv_frac[mask].mean()), 4),
            "norm_return": round(float(norm_ret[mask].mean()), 4),
            "body_pos_rmse_cm": round(float(body_cm[mask].mean()), 2),
            "root_xy_rmse_cm": round(float(root_cm[mask].mean()), 2),
            "wrist_world_rmse_cm": round(float(wrist_cm[mask].mean()), 2),
            "stance_slip_mm_s": round(float(slip_mm[mask].mean()), 1),
            "pd_oob_pct": round(float(oob_pct[mask].mean()), 4),
        }

    all_mask = np.ones(k, dtype=bool)
    res: dict[str, Any] = {"all": _agg(all_mask)}
    if clip_meta is not None:
        sources = np.array([c["source"] for c in clip_meta[:k]])
        cats = np.array([c["category"] for c in clip_meta[:k]])
        res["by_source"] = {src: _agg(sources == src) for src in sorted(set(sources.tolist()))}
        res["by_category"] = {cat: _agg(cats == cat) for cat in sorted(set(cats.tolist()))}
    return res


def rollout_cpu_clip(
    env: Any,
    *,
    clip_idx: int,
    est_mode: str,
    modality: str,
    tok_sess: ort.InferenceSession | None = None,
    head_sess: ort.InferenceSession | None = None,
    deploy_safety: bool = True,
    friction_scale: float = 1.0,
    delay_steps: int = 0,
    pushes: tuple[tuple[int, float, float], ...] = (),
    stress_sensor: bool = False,
    save_qpos: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """CPU MuJoCo 3.11 + 1-Thread ORT ONNX rollout for exact Sim2Sim verification."""
    mdp = env.supermimic_mdp
    motion = mdp.motion
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()
    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32).view(1, 1, 3, 1)

    budget = int(motion.clip_len[clip_idx]) - 1
    rng = np.random.default_rng(seed + clip_idx)
    yaw_bias_R = so3_exp(torch.tensor([[0.0, 0.0, float(np.radians(1.5))]], dtype=torch.float32))[0]
    push_map = {s: (vx, vy) for s, vx, vy in pushes}

    env._eval_friction_scale = float(friction_scale)
    try:
        with torch.no_grad():
            _, reset_info = env.reset_to_frames(motion.clip_start[clip_idx : clip_idx + 1].clone())
            cur = reset_info[SUPERMIMIC_STATE_KEY]
            s0 = unflatten_state(cur["s"])
            f0 = cur["f"]
            est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
            est.reset(s0[1], s0[3], s0[4], s0[5], motion.root_p[f0, :2], R_ref=motion.root_R[f0])

            history = None
            act_fifo = torch.zeros((delay_steps + 1, mdp.num_joints), dtype=torch.float32)
            sum_ret = sum_body_sq = 0.0
            root_xy_sq, slip_spds, tau_utils, step_ms = [], [], [], []
            pd_oob = pd_tot = 0
            failed = False
            steps_run = 0
            qpos_pol, pz_trace = [], []
            if save_qpos:
                qpos_pol.append(torch.cat((s0[0][0], mat_to_quat(s0[1])[0], s0[4][0]), -1).numpy().copy())
                pz_trace.append(float(s0[0][0, 2]))

            robot = env.scene["robot"]
            for step_i in range(budget):
                if step_i in push_map:
                    vx, vy = push_map[step_i]
                    robot.apply_root_linear_velocity_delta_to_sim(torch.tensor([[vx, vy, 0.0]], dtype=torch.float32))

                state = unflatten_state(cur["s"])
                f_cur = cur["f"]
                R_obs, w_obs = state[1], state[3]
                if stress_sensor:
                    R_obs = (yaw_bias_R @ R_obs[0]).unsqueeze(0)
                    w_obs = w_obs + torch.from_numpy(rng.normal(0.0, 0.03, size=(1, 3)).astype(np.float32))

                t0 = time.perf_counter()
                obs = mdp.obs((state[0], R_obs, state[2], w_obs, state[4], state[5]), cur["a_prev"], cur["xi"], f_cur)
                exy, _ = est.step(
                    R_obs, w_obs, state[4], state[5], motion.root_p[f_cur, :2],
                    R_ref=motion.root_R[f_cur], mode=est_mode, true_obs_root_xy=obs[:, 78:80],
                    slip_noise_std_m=0.0025 if stress_sensor else 0.0,
                )
                obs_pol = torch.cat((obs[:, :78], exy), dim=-1)
                history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)
                goal = _build_goal(motion, f_cur, modality)

                tok = tok_sess.run(["token"], {"modal_input": goal.numpy().astype(np.float32)})[0]
                act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])

                act_fifo = torch.cat((act, act_fifo[:-1]), dim=0)
                act_eff = act_fifo[min(step_i, delay_steps)].unsqueeze(0).contiguous()

                if deploy_safety:
                    q_tgt_raw = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act_eff[0]
                    q_tgt_safe = _apply_torque_safe_pd_cpu(q_tgt_raw, state[4][0], state[5][0], q_lo, q_hi, kp_vec, kd_vec, eff_vec)
                    act_sent = ((q_tgt_safe - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0)
                else:
                    act_sent = act_eff
                step_ms.append((time.perf_counter() - t0) * 1000.0)

                q_tgt_sent = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act_sent[0]
                tau = kp_vec * (q_tgt_sent - state[4][0]) - kd_vec * state[5][0]
                tau_utils.append(float((torch.abs(tau) / eff_vec).max().item()))
                pd_oob += int(((torch.abs(tau) - eff_vec) > 1e-4).sum().item())
                pd_tot += int(q_tgt_sent.numel())

                step_s = env.step(act_sent.contiguous())
                tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
                f_next = motion.future_frames(tr["f"], 1).squeeze(-1)
                s1 = unflatten_state(tr["s1"])
                _, terms = mdp.reward(s1, f_next, tr["a"], tr["a_prev"], terms=True)

                sum_ret += float(tr["r"][0].item())
                sum_body_sq += float(terms["e_body_pos"][0].item())
                steps_run += 1
                root_xy_sq.append(float(((s1[0][0, :2] - motion.root_p[f_next[0], :2]) ** 2).sum().item()))

                bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
                ref_az = motion.body_p[f_next[0], (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]
                ref_av = motion.body_v[f_next[0], (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].norm(dim=-1)
                smask = (ref_az <= 0.062) & (ref_av <= 0.15)
                if bool(smask.any()):
                    v_sole = bv1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] + torch.cross(
                        bw1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)],
                        (bR1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] @ sole_off).squeeze(-1),
                        dim=-1,
                    )
                    slip_spds.extend(v_sole[0, :, :2].norm(dim=-1)[smask].tolist())

                if save_qpos:
                    qpos_pol.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
                    pz_trace.append(float(s1[0][0, 2]))

                if bool(tr["term"][0].item()):
                    failed = True
                    break
                if bool(tr["done"][0].item()):
                    break
                cur = step_s.info[SUPERMIMIC_STATE_KEY]
    finally:
        env._eval_friction_scale = 1.0

    max_r = float(mdp.max_step_reward)
    out: dict[str, Any] = {
        "completed": (not failed) and (steps_run == budget),
        "steps_run": steps_run,
        "budget": budget,
        "survival_fraction": steps_run / max(budget, 1),
        "norm_return": sum_ret / max(steps_run * max_r, 1e-8),
        "body_pos_rmse_cm": float(np.sqrt(sum_body_sq / max(steps_run, 1)) * 100.0),
        "root_xy_rmse_cm": float(np.sqrt(np.mean(root_xy_sq)) * 100.0) if root_xy_sq else 0.0,
        "stance_slip_mm_s": float(np.mean(slip_spds) * 1000.0) if slip_spds else 0.0,
        "pd_oob_pct": 100.0 * pd_oob / max(pd_tot, 1),
        "tau_util_p99": float(np.percentile(tau_utils, 99)) if tau_utils else 0.0,
        "cpu_step_ms_p50": float(np.percentile(step_ms, 50)) if step_ms else 0.0,
    }
    if save_qpos:
        out["qpos_pol"] = np.stack(qpos_pol, axis=0)
        out["pelvis_z_trace"] = pz_trace
    return out


def _agg_cpu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n_clips": len(rows), "survival_rate": round(float(np.mean([float(r["completed"]) for r in rows])), 4)}
    for k in ("survival_fraction", "norm_return", "body_pos_rmse_cm", "root_xy_rmse_cm", "stance_slip_mm_s", "pd_oob_pct", "tau_util_p99", "cpu_step_ms_p50"):
        out[k] = round(float(np.mean([float(r[k]) for r in rows])), 4 if k in ("survival_fraction", "norm_return", "pd_oob_pct", "tau_util_p99") else 2)
    return out


def _enter_free_roaming_mode(env: Any, stand_frame_idx: int) -> None:
    env.termination_manager._term_names = []
    env.termination_manager._term_cfgs = []
    env._compute_truncated = lambda s: torch.zeros_like(s.truncated)
    env._supermimic_command.f[:] = stand_frame_idx
    env._supermimic_command.ep_step[:] = 0


def eval_zero_ik_bvh_24(env: Any, stand_clip_idx: int, tok_smpl: ort.InferenceSession, head_sess: ort.InferenceSession) -> dict[str, Any]:
    bvh_dir = Path("/home/yu/datasets/heldout_raw_bvh_24")
    mdp = env.supermimic_mdp
    stand_frame_idx = int(mdp.motion.clip_start[stand_clip_idx])
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()

    results: dict[str, Any] = {}
    sample_bvh_name = ""
    sample_bvh_qpos = None
    for bvh_i, p in enumerate(sorted(bvh_dir.glob("*.bvh"))):
        kind = "lafan1" if p.name.startswith("lafan_") else "soma"
        mock_m, h_nom, _, _ = load_bvh_as_mock_motion(p, kind=kind)
        with torch.no_grad():
            _, reset_info = env.reset_to_frames(mdp.motion.clip_start[stand_clip_idx : stand_clip_idx + 1].clone())
            _enter_free_roaming_mode(env, stand_frame_idx)
            cur = reset_info[SUPERMIMIC_STATE_KEY]
            s0 = unflatten_state(cur["s"])
            R_align = yaw_mat((s0[1][0] @ mock_m.root_R[0].transpose(-1, -2)).unsqueeze(0))[0]
            p0_rob, p0_hum = s0[0][0, :2], mock_m.root_p[0, :2]

            est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
            est.reset(s0[1], s0[3], s0[4], s0[5], p0_rob.unsqueeze(0), R_ref=(R_align @ mock_m.root_R[0]).unsqueeze(0))
            history = None
            q_init = s0[4][0].clone()
            budget = min(mock_m.num_frames - 1, 280)
            steps_ok = 0
            failed = False
            root_errs = []
            qpos_tr = []

            for t in range(budget):
                st = unflatten_state(cur["s"])
                f_t = torch.tensor([t], dtype=torch.long)
                R_ref_t = (R_align @ mock_m.root_R[t]).unsqueeze(0)
                p_ref_xy = ((R_align[:2, :2] @ (mock_m.root_p[t, :2] - p0_hum)) + p0_rob).unsqueeze(0)
                obs_tmp = mdp.obs(st, cur["a_prev"], cur["xi"], cur["f"])
                rel_rot = mat_to_6d(st[1].transpose(-1, -2) @ R_ref_t)
                exy, _ = est.step(st[1], st[3], st[4], st[5], p_ref_xy, R_ref=R_ref_t, mode="cgfa")
                obs_pol = torch.cat((obs_tmp[:, :72], rel_rot, exy), dim=-1)
                history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)

                gm = build_human_smpl_goal(mock_m, f_t, 10, h_nom=h_nom)
                tok = tok_smpl.run(["token"], {"modal_input": gm.numpy().astype(np.float32)})[0]
                act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])

                q_tgt = _apply_torque_safe_pd_cpu(
                    mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0],
                    st[4][0], st[5][0], q_lo, q_hi, kp_vec, kd_vec, eff_vec,
                    q_init=q_init, step_i=t, engage_steps=15,
                )
                act_sent = ((q_tgt - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0)
                env._supermimic_command.f[:] = stand_frame_idx
                env._supermimic_command.ep_step[:] = 0
                step_s = env.step(act_sent)
                s1 = unflatten_state(step_s.info[SUPERMIMIC_TRANSITION_KEY]["s1"])
                steps_ok += 1
                root_errs.append(float(((s1[0][0, :2] - p_ref_xy[0]) ** 2).sum().item()))
                if sample_bvh_qpos is None or "walk" in p.stem or "dance" in p.stem:
                    qpos_tr.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
                if float(s1[0][0, 2]) < 0.18 or float(s1[1][0, 2, 2]) < 0.35:
                    failed = True
                    break
                cur = step_s.info[SUPERMIMIC_STATE_KEY]

        results[p.stem] = {
            "kind": kind,
            "completed": not failed,
            "steps_run": steps_ok,
            "budget": budget,
            "root_xy_rmse_cm": round(float(np.sqrt(np.mean(root_errs)) * 100.0), 2),
        }
        if qpos_tr and ((not failed and (sample_bvh_qpos is None or "walk" in p.stem or "dance" in p.stem)) or sample_bvh_qpos is None):
            sample_bvh_name = p.stem
            sample_bvh_qpos = np.stack(qpos_tr, axis=0)
    rows = list(results.values())
    by_k = {
        k: round(float(np.mean([float(v["completed"]) for v in rows if v["kind"] == k])), 4)
        for k in ("lafan1", "soma")
        if any(v["kind"] == k for v in rows)
    }
    return {
        "n_bvhs": len(rows),
        "survival_rate": round(float(np.mean([float(v["completed"]) for v in rows])), 4),
        "mean_root_xy_rmse_cm": round(float(np.mean([v["root_xy_rmse_cm"] for v in rows])), 2),
        "by_kind": by_k,
        "sample_bvh_name": sample_bvh_name,
        "sample_bvh_qpos": sample_bvh_qpos,
        "per_bvh": results,
    }


def run_30s_sonic_v3_modeswitch(
    env: Any,
    stand_clip_idx: int,
    walk_clip_idx: int,
    vr_clip_idx: int,
    onnx_dir: Path,
    smpl_bvh: Path,
    out_npz: Path,
    out_json: Path,
) -> dict[str, Any]:
    """30.0s (1,500-step) unbroken closed-loop 4-modality showcase with 20ms delay, live shoves, and 400ms Watchdog."""
    mdp = env.supermimic_mdp
    stand_frame_idx = int(mdp.motion.clip_start[stand_clip_idx])
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()
    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32).view(1, 1, 3, 1)

    head_sess = _ort_sess(onnx_dir / "q1_h002_v3_token_head.onnx")
    tok_robot = _ort_sess(onnx_dir / "q1_v3_tok_robot_pose.onnx")
    tok_smpl = _ort_sess(onnx_dir / "q1_v3_tok_human_smpl.onnx")
    tok_vr = _ort_sess(onnx_dir / "q1_v3_tok_vr_3pt.onnx")
    tok_plan = _ort_sess(onnx_dir / "q1_v3_tok_planner_cmd.onnx")

    kind_s = "lafan1" if smpl_bvh.name.startswith("lafan_") else "soma"
    mock_smpl, h_nom_s, _, _ = load_bvh_as_mock_motion(smpl_bvh, kind=kind_s)

    cmd_D = build_synthetic_planner_stream([
        (110, 0.36, 0.0, 0.0, 1.0, 0.35),
        (70, 0.25, 0.0, 0.35, 1.0, 0.35),
        (70, 0.0, 0.0, 0.0, 0.85, 0.40),
        (60, 0.0, 0.0, 0.0, 1.0, 0.35),
    ])
    cmd_idle = build_synthetic_planner_stream([(220, 0.0, 0.0, 0.0, 1.0, 0.35)])

    T_TOTAL = 1500
    phases = [
        (0, 300, "A_RobotPose_Walk_Push65"),
        (300, 600, "B_ZeroIK_HumanSMPL"),
        (600, 900, "C_VR3Pt_PICO_Push60"),
        (900, 1200, "D_PlannerCmd_TurnSquat"),
        (1200, 1325, "E_RobotPose_PreDropout"),
        (1325, 1345, "F_400ms_Dropout_Watchdog"),
        (1345, 1500, "G_PlannerRecover_Stand"),
    ]

    walk_s = int(mdp.motion.clip_start[walk_clip_idx])
    walk_L = int(mdp.motion.clip_len[walk_clip_idx])
    vr_s = int(mdp.motion.clip_start[vr_clip_idx])
    vr_L = int(mdp.motion.clip_len[vr_clip_idx])

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(mdp.motion.clip_start[walk_clip_idx : walk_clip_idx + 1].clone())
        _enter_free_roaming_mode(env, stand_frame_idx)
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])

        est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
        est.reset(s0[1], s0[3], s0[4], s0[5], mdp.motion.root_p[walk_s : walk_s + 1, :2], R_ref=mdp.motion.root_R[walk_s : walk_s + 1])

        history = None
        q_init = s0[4][0].clone()
        prev_q_tgt = s0[4][0].clone()
        last_act = torch.zeros((1, 22), dtype=torch.float32)
        delayed_act = torch.zeros((1, 22), dtype=torch.float32)
        crossfade_origin_act = last_act.clone()
        crossfade_t0 = -999
        prev_phase = ""

        engage_rob_p = s0[0][0, :2].clone()
        engage_rob_R = yaw_mat((yaw_mat(s0[1])[0] @ mdp.motion.root_R[walk_s].transpose(-1, -2)).unsqueeze(0))[0]
        engage_smpl_p = engage_smpl_R = engage_vr_p = engage_vr_R = None
        p_plan_xy = s0[0][0, :2].clone()
        yaw_plan_R = yaw_mat(s0[1])[0].clone()
        stale_tok = None

        qpos_hist, mode_hist, step_jumps_deg, slip_hist, tau_hist, height_hist, tilt_hist, speed_hist = [], [], [], [], [], [], [], []
        oob_count = 0
        failed = False
        robot = env.scene["robot"]

        for t in range(T_TOTAL):
            if t == 150:
                robot.apply_root_linear_velocity_delta_to_sim(torch.tensor([[0.0, 0.65, 0.0]], dtype=torch.float32))
            elif t == 750:
                robot.apply_root_linear_velocity_delta_to_sim(torch.tensor([[0.0, -0.60, 0.0]], dtype=torch.float32))

            st = unflatten_state(cur["s"])
            p_cur, R_cur, v_cur, w_cur, q_cur, qd_cur = st

            pname = next(n for a, b, n in phases if a <= t < b)
            if pname != prev_phase and t > 0:
                crossfade_origin_act = last_act.clone()
                crossfade_t0 = t
                if pname == "B_ZeroIK_HumanSMPL":
                    engage_smpl_p = p_cur[0, :2].clone()
                    engage_smpl_R = yaw_mat((yaw_mat(R_cur)[0] @ mock_smpl.root_R[0].transpose(-1, -2)).unsqueeze(0))[0]
                    est.reset(R_cur, w_cur, q_cur, qd_cur, engage_smpl_p.unsqueeze(0), R_ref=(engage_smpl_R @ mock_smpl.root_R[0]).unsqueeze(0))
                elif pname == "C_VR3Pt_PICO_Push60":
                    engage_vr_p = p_cur[0, :2].clone()
                    vr_R0 = mdp.motion.root_R[vr_s]
                    engage_vr_R = yaw_mat((yaw_mat(R_cur)[0] @ vr_R0.transpose(-1, -2)).unsqueeze(0))[0]
                    est.reset(R_cur, w_cur, q_cur, qd_cur, engage_vr_p.unsqueeze(0), R_ref=(engage_vr_R @ vr_R0).unsqueeze(0))
                elif pname == "D_PlannerCmd_TurnSquat":
                    p_plan_xy = p_cur[0, :2].clone()
                    yaw_plan_R = yaw_mat(R_cur)[0].clone()
                    est.reset(R_cur, w_cur, q_cur, qd_cur, p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))
                elif pname == "E_RobotPose_PreDropout":
                    engage_rob_p = p_cur[0, :2].clone()
                    engage_rob_R = yaw_mat((yaw_mat(R_cur)[0] @ mdp.motion.root_R[walk_s].transpose(-1, -2)).unsqueeze(0))[0]
                    est.reset(R_cur, w_cur, q_cur, qd_cur, engage_rob_p.unsqueeze(0), R_ref=(engage_rob_R @ mdp.motion.root_R[walk_s]).unsqueeze(0))
                elif pname in ("F_400ms_Dropout_Watchdog", "G_PlannerRecover_Stand"):
                    p_plan_xy = p_cur[0, :2].clone()
                    yaw_plan_R = yaw_mat(R_cur)[0].clone()
                    est.reset(R_cur, w_cur, q_cur, qd_cur, p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))
            prev_phase = pname

            if pname == "A_RobotPose_Walk_Push65":
                ka = min(t, walk_L - 1)
                fa = torch.tensor([walk_s + ka], dtype=torch.long)
                p_ref_xy = ((engage_rob_R[:2, :2] @ (mdp.motion.root_p[fa[0], :2] - mdp.motion.root_p[walk_s, :2])) + engage_rob_p).unsqueeze(0)
                R_ref_t = (engage_rob_R @ mdp.motion.root_R[fa[0]]).unsqueeze(0)
                gr = build_causal_history_goal(mdp.motion, fa, 10, latest_offset=0)
                tok = tok_robot.run(["token"], {"modal_input": gr.numpy().astype(np.float32)})[0]
            elif pname == "B_ZeroIK_HumanSMPL":
                kb = min(t - 300, mock_smpl.num_frames - 1)
                p_ref_xy = ((engage_smpl_R[:2, :2] @ (mock_smpl.root_p[kb, :2] - mock_smpl.root_p[0, :2])) + engage_smpl_p).unsqueeze(0)
                R_ref_t = (engage_smpl_R @ mock_smpl.root_R[kb]).unsqueeze(0)
                gm = build_human_smpl_goal(mock_smpl, torch.tensor([kb]), 10, h_nom=h_nom_s)
                tok = tok_smpl.run(["token"], {"modal_input": gm.numpy().astype(np.float32)})[0]
            elif pname == "C_VR3Pt_PICO_Push60":
                kc = min(t - 600, vr_L - 1)
                fc = torch.tensor([vr_s + kc], dtype=torch.long)
                p_ref_xy = ((engage_vr_R[:2, :2] @ (mdp.motion.root_p[fc[0], :2] - mdp.motion.root_p[vr_s, :2])) + engage_vr_p).unsqueeze(0)
                R_ref_t = (engage_vr_R @ mdp.motion.root_R[fc[0]]).unsqueeze(0)
                gv = build_vr_3pt_goal(mdp.motion, fc, 10)
                tok = tok_vr.run(["token"], {"modal_input": gv.numpy().astype(np.float32)})[0]
            elif pname == "D_PlannerCmd_TurnSquat":
                kd_idx = t - 900
                g_row = cmd_D[kd_idx]
                vx_d, wz_d = float(g_row[-10]), float(g_row[-8])
                yaw_dR = so3_exp(torch.tensor([[0.0, 0.0, wz_d * 0.02]], dtype=torch.float32))[0]
                yaw_plan_R = yaw_mat((yaw_dR @ yaw_plan_R).unsqueeze(0))[0]
                p_plan_xy = p_plan_xy + (yaw_plan_R[:2, :2] @ torch.tensor([vx_d, 0.0])) * 0.02
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                tok = tok_plan.run(["token"], {"modal_input": g_row.unsqueeze(0).numpy().astype(np.float32)})[0]
            elif pname == "E_RobotPose_PreDropout":
                ke = min(t - 1200, walk_L - 1)
                fe = torch.tensor([walk_s + ke], dtype=torch.long)
                p_ref_xy = ((engage_rob_R[:2, :2] @ (mdp.motion.root_p[fe[0], :2] - mdp.motion.root_p[walk_s, :2])) + engage_rob_p).unsqueeze(0)
                R_ref_t = (engage_rob_R @ mdp.motion.root_R[fe[0]]).unsqueeze(0)
                gr = build_causal_history_goal(mdp.motion, fe, 10, latest_offset=0)
                tok = tok_robot.run(["token"], {"modal_input": gr.numpy().astype(np.float32)})[0]
                stale_tok = tok.copy()
            elif pname == "F_400ms_Dropout_Watchdog":
                kf = t - 1325
                if kf == 12:
                    crossfade_origin_act = last_act.clone()
                    crossfade_t0 = t
                    p_plan_xy = p_cur[0, :2].clone()
                    yaw_plan_R = yaw_mat(R_cur)[0].clone()
                    est.reset(R_cur, w_cur, q_cur, qd_cur, p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                tok = stale_tok if kf < 12 else tok_plan.run(["token"], {"modal_input": cmd_idle[kf].unsqueeze(0).numpy().astype(np.float32)})[0]
            else:
                kg = t - 1345
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                tok = tok_plan.run(["token"], {"modal_input": cmd_idle[min(kg, 200)].unsqueeze(0).numpy().astype(np.float32)})[0]

            obs_raw = mdp.obs(st, cur["a_prev"], cur["xi"], cur["f"])
            rel_rot = mat_to_6d(R_cur.transpose(-1, -2) @ R_ref_t)
            exy, _ = est.step(R_cur, w_cur, q_cur, qd_cur, p_ref_xy, R_ref=R_ref_t, mode="cgfa")
            obs_pol = torch.cat((obs_raw[:, :72], rel_rot, exy), dim=-1)
            history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)

            act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])
            if t - crossfade_t0 < 12:
                w_cf = 0.5 * (1.0 - np.cos(np.pi * float(t - crossfade_t0 + 1) / 12.0))
                act = (1.0 - float(w_cf)) * crossfade_origin_act + float(w_cf) * act

            # 20ms (1-step) causal bus delay active across entire 30s run
            act_eff = act if t == 0 else delayed_act
            delayed_act = act

            q_tgt = _apply_torque_safe_pd_cpu(
                mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act_eff[0],
                q_cur[0], qd_cur[0], q_lo, q_hi, kp_vec, kd_vec, eff_vec,
                q_init=q_init, step_i=t, engage_steps=5,
            )
            act_sent = ((q_tgt - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0)
            last_act = act_sent.clone()

            tau = kp_vec * (q_tgt - q_cur[0]) - kd_vec * qd_cur[0]
            oob_count += int(((torch.abs(tau) - eff_vec) > 1e-4).sum().item())
            step_jumps_deg.append(float(np.degrees(torch.max(torch.abs(q_tgt - prev_q_tgt)).item())))
            prev_q_tgt = q_tgt.clone()
            tau_hist.append(float((torch.abs(tau) / eff_vec).max().item()))

            env._supermimic_command.f[:] = stand_frame_idx
            env._supermimic_command.ep_step[:] = 0
            step_s = env.step(act_sent)
            s1 = unflatten_state(step_s.info[SUPERMIMIC_TRANSITION_KEY]["s1"])
            bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
            min_f = int(torch.argmin(bp1[0, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]))
            a_idx = LEFT_ANKLE_IDX if min_f == 0 else RIGHT_ANKLE_IDX
            v_sole = bv1[0, a_idx] + torch.cross(bw1[0, a_idx], (bR1[0, a_idx] @ sole_off[0, 0]).squeeze(-1), dim=-1)
            slip_hist.append(float(v_sole[:2].norm().item() * 1000.0))

            pz = float(s1[0][0, 2])
            up_z = float(s1[1][0, 2, 2])
            spd = float(s1[2][0, :2].norm().item())
            height_hist.append(pz)
            tilt_hist.append(up_z)
            speed_hist.append(spd)
            qpos_hist.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
            mode_hist.append(pname)

            if pz < 0.16 or up_z < 0.35:
                failed = True
                break
            cur = step_s.info[SUPERMIMIC_STATE_KEY]

    np.savez_compressed(out_npz, qpos=np.stack(qpos_hist, axis=0))
    push1_min_z = round(min(height_hist[150:225]), 3) if len(height_hist) >= 225 else 0.0
    push2_min_z = round(min(height_hist[750:825]), 3) if len(height_hist) >= 825 else 0.0
    summary = {
        "completed_30s": (not failed) and len(qpos_hist) == T_TOTAL,
        "steps_survived": len(qpos_hist),
        "budget": T_TOTAL,
        "delay_ms": 20,
        "pushes_applied": [{"t_s": 3.0, "dv_y_mps": 0.65}, {"t_s": 15.0, "dv_y_mps": -0.60}],
        "push_3s_min_pelvis_z_m": push1_min_z,
        "push_15s_min_pelvis_z_m": push2_min_z,
        "pd_oob_count": oob_count,
        "max_step_jump_deg": round(max(step_jumps_deg), 2),
        "p99_step_jump_deg": round(float(np.percentile(step_jumps_deg, 99)), 2),
        "mean_stance_slip_mm_s": round(float(np.mean(slip_hist)), 1),
        "tau_util_p99": round(float(np.percentile(tau_hist, 99)), 3),
        "height_range_m": [round(min(height_hist), 3), round(max(height_hist), 3)],
        "min_upright_cos": round(min(tilt_hist), 3),
        "trace_sampled": [
            {
                "t_s": round(i * 0.02, 2),
                "phase": mode_hist[i],
                "pelvis_z_m": round(height_hist[i], 3),
                "speed_mps": round(speed_hist[i], 3),
                "slip_mm_s": round(slip_hist[i], 1),
                "tau_util": round(tau_hist[i], 3),
            }
            for i in range(0, len(qpos_hist), 5)
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SONIC_30S_MODESWITCH:", json.dumps({k: v for k, v in summary.items() if k != "trace_sampled"}, indent=2), flush=True)
    return summary


def main() -> None:
    t0_all = time.monotonic()
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-2000", type=Path, default=base / "dataset_2000_manifest.json")
    parser.add_argument("--manifest-250", type=Path, default=ROOT / "runs/h002_sonic_v2/dataset_250_manifest.json")
    parser.add_argument("--out-json", type=Path, default=base / "v3_sonic_metrics.json")
    cli_args = parser.parse_args()

    networks.AMP_DTYPE = None
    man2k = json.loads(cli_args.manifest_2000.read_text(encoding="utf-8"))
    ho200_meta = sorted([c for c in man2k["clips"] if c["split"] == "held_out"], key=lambda c: c["csv_rel"])
    assert len(ho200_meta) == 200, len(ho200_meta)
    ho200_files = [str(ROOT / c["csv_rel"]) for c in ho200_meta]

    # Balanced 50-clip slice from held_out_200 for push & delay sweeps
    by_src: dict[str, list[int]] = {}
    for idx, c in enumerate(ho200_meta):
        by_src.setdefault(c["source"], []).append(idx)
    quotas = {"lafan1": 6, "soma": 13, "phuma": 28, "synthetic": 3}
    slice50_idx: list[int] = []
    for src, q in quotas.items():
        idxs = by_src.get(src, [])
        pick = np.linspace(0, len(idxs) - 1, q, dtype=int)
        slice50_idx.extend([idxs[int(i)] for i in pick])
    slice50_files = [ho200_files[i] for i in slice50_idx]

    # ---------------- SUITE 1 & 3: GPU mjwarp 200-Clip + Push/Delay Sweeps ----------------
    dev = torch.device("cuda:0")
    val_env = create_mjwarp_eval_env(ho200_files, num_envs=200)
    mdp_gpu = val_env.supermimic_mdp

    pi_oracle = _load_ckpt_policy(mdp_gpu, ROOT / "checkpoints/q1_base_soma1k.pt", dev)
    pi_zero = _load_ckpt_policy(mdp_gpu, ROOT / "checkpoints/q1_scheme_a_1k.pt", dev)
    pi_v1 = _load_ckpt_policy(mdp_gpu, ROOT / "runs/h002_cgfa_pilot/checkpoints/m3_h002_full_0ms.pt", dev)
    pi_v2, toks_v2 = _load_sonic_bundle(mdp_gpu, ROOT / "runs/h002_sonic_v2/checkpoints/h002_sonic_v2_bundle.pt", dev)
    pi_v3, toks_v3 = _load_sonic_bundle(mdp_gpu, base / "checkpoints/h002_sonic_v3_bundle.pt", dev)

    ho200_specs = [
        ("Oracle_90ms", dict(pi=pi_oracle, tok=None, est_mode="oracle", modality="oracle_future10", deploy_safety=False)),
        ("Zeroed_0ms", dict(pi=pi_zero, tok=None, est_mode="zeroed", modality="robot_pose", deploy_safety=False)),
        ("V1_H002_12clip", dict(pi=pi_v1, tok=None, est_mode="cgfa", modality="robot_pose", deploy_safety=True)),
        ("V2_RobotPose_200clip", dict(pi=pi_v2, tok=None, est_mode="cgfa", modality="robot_pose", deploy_safety=True)),
        ("V2_HumanSMPL_200clip", dict(pi=pi_v2, tok=toks_v2["human_smpl"], est_mode="cgfa", modality="human_smpl", deploy_safety=True)),
        ("V2_VR3Pt_200clip", dict(pi=pi_v2, tok=toks_v2["vr_3pt"], est_mode="cgfa", modality="vr_3pt", deploy_safety=True)),
        ("V2_PlannerCmd_200clip", dict(pi=pi_v2, tok=toks_v2["planner_cmd"], est_mode="cgfa", modality="planner_cmd", deploy_safety=True)),
        ("V3_RobotPose", dict(pi=pi_v3, tok=None, est_mode="cgfa", modality="robot_pose", deploy_safety=True)),
        ("V3_HumanSMPL", dict(pi=pi_v3, tok=toks_v3["human_smpl"], est_mode="cgfa", modality="human_smpl", deploy_safety=True)),
        ("V3_VR3Pt", dict(pi=pi_v3, tok=toks_v3["vr_3pt"], est_mode="cgfa", modality="vr_3pt", deploy_safety=True)),
        ("V3_PlannerCmd", dict(pi=pi_v3, tok=toks_v3["planner_cmd"], est_mode="cgfa", modality="planner_cmd", deploy_safety=True)),
    ]

    ho200_nom: dict[str, Any] = {}
    ho200_str: dict[str, Any] = {}
    for name, kw in ho200_specs:
        rn = eval_mjwarp_batch(
            val_env, num_clips=200, friction_scale=1.0, delay_steps=0, pushes=(), stress_sensor=False, clip_meta=ho200_meta, seed=20261119, **kw,
        )
        ho200_nom[name] = rn
        a = rn["all"]
        print(f"[HO-200 NOM] {name:22s}: surv={a['survival_rate']*100:5.1f}% ret={a['norm_return']:.3f} body={a['body_pos_rmse_cm']:5.2f}cm slip={a['stance_slip_mm_s']:5.1f}mm/s", flush=True)

    for name, kw in ho200_specs[1:]:
        rs = eval_mjwarp_batch(
            val_env, num_clips=200, friction_scale=0.50, delay_steps=1,
            pushes=[(60, 0.0, 0.55), (120, 0.40, -0.40)], stress_sensor=True, clip_meta=ho200_meta, seed=20261120, **kw,
        )
        ho200_str[name] = rs
        a = rs["all"]
        print(f"[HO-200 STR] {name:22s}: surv={a['survival_rate']*100:5.1f}% ret={a['norm_return']:.3f} body={a['body_pos_rmse_cm']:5.2f}cm slip={a['stance_slip_mm_s']:5.1f}mm/s", flush=True)

    val_env.close(); val_env.backend.close()
    del val_env
    gc.collect()
    torch.cuda.empty_cache()

    sweep_env = create_mjwarp_eval_env(slice50_files, num_envs=50)
    mdp_sw = sweep_env.supermimic_mdp
    # Rebind policies cleanly (architecture & weights unchanged, only mdp device helpers identical on cuda:0)
    sweep_specs = [
        ("Zeroed_0ms", dict(pi=pi_zero, tok=None, est_mode="zeroed", modality="robot_pose", deploy_safety=False)),
        ("V1_H002_12clip", dict(pi=pi_v1, tok=None, est_mode="cgfa", modality="robot_pose", deploy_safety=True)),
        ("V2_RobotPose", dict(pi=pi_v2, tok=None, est_mode="cgfa", modality="robot_pose", deploy_safety=True)),
        ("V3_RobotPose", dict(pi=pi_v3, tok=None, est_mode="cgfa", modality="robot_pose", deploy_safety=True)),
        ("V3_HumanSMPL", dict(pi=pi_v3, tok=toks_v3["human_smpl"], est_mode="cgfa", modality="human_smpl", deploy_safety=True)),
    ]

    push_mags = [0.0, 0.35, 0.55, 0.75, 0.95]
    push_sweep: dict[str, dict[str, Any]] = {}
    for name, kw in sweep_specs:
        push_sweep[name] = {}
        for dv in push_mags:
            ps = [(60, 0.0, dv), (120, 0.0, -dv)] if dv > 0.0 else []
            rp = eval_mjwarp_batch(sweep_env, num_clips=50, friction_scale=1.0, delay_steps=0, pushes=ps, stress_sensor=False, seed=20261121, **kw)
            push_sweep[name][f"{dv:.2f}"] = rp["all"]
        print(f"[PUSH SWEEP] {name:16s}: " + " | ".join(f"{m:.2f}={push_sweep[name][f'{m:.2f}']['survival_rate']*100:4.0f}%" for m in push_mags), flush=True)

    delay_steps_list = [0, 1, 2, 3]
    delay_sweep: dict[str, dict[str, Any]] = {}
    for name, kw in sweep_specs:
        delay_sweep[name] = {}
        for ds in delay_steps_list:
            ms_label = f"{ds*20}ms"
            rd = eval_mjwarp_batch(sweep_env, num_clips=50, friction_scale=1.0, delay_steps=ds, pushes=[(75, 0.0, 0.35)], stress_sensor=False, seed=20261122, **kw)
            delay_sweep[name][ms_label] = rd["all"]
        print(f"[DELAY SWP] {name:16s}: " + " | ".join(f"{ds*20}ms={delay_sweep[name][f'{ds*20}ms']['survival_rate']*100:4.0f}%" for ds in delay_steps_list), flush=True)

    del mdp_sw, mdp_gpu, pi_oracle, pi_zero, pi_v1, pi_v2, toks_v2, pi_v3, toks_v3
    sweep_env.close(); sweep_env.backend.close()
    del sweep_env
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- SUITE 2, 4, 5: CPU MuJoCo 3.11 + 1-Thread ORT ONNX Sim2Sim ----------------
    man250 = json.loads(cli_args.manifest_250.read_text(encoding="utf-8"))
    ho50_meta = [c for c in man250["clips"] if c["split"] == "held_out"]
    assert len(ho50_meta) == 50, len(ho50_meta)
    ho50_files = [str(ROOT / c["csv_rel"]) for c in ho50_meta]

    stand_csv = str(ROOT / "src/unisupermimic/assets/motions/q1/multidataset_2000/train/syn_stand_idle_1.csv")
    if stand_csv not in ho50_files:
        ho50_files.append(stand_csv)
    stand_idx = ho50_files.index(stand_csv)

    walk_idx = next((i for i, c in enumerate(ho50_meta) if "walk2_subject1" in c["alias"]), 0)
    vr_idx = next((i for i, c in enumerate(ho50_meta) if "eating_apple" in c["alias"] or "aiming1" in c["alias"]), 1)

    cpu_env = create_cpu_mujoco_eval_env(ho50_files)
    v2_onnx = ROOT / "runs/h002_sonic_v2/onnx"
    v3_onnx = base / "onnx"

    s_v2_head = _ort_sess(v2_onnx / "q1_h002_token_head.onnx")
    s_v2_rob = _ort_sess(v2_onnx / "q1_tok_robot_pose.onnx")
    s_v2_smpl = _ort_sess(v2_onnx / "q1_tok_human_smpl.onnx")

    s_v3_head = _ort_sess(v3_onnx / "q1_h002_v3_token_head.onnx")
    s_v3_rob = _ort_sess(v3_onnx / "q1_v3_tok_robot_pose.onnx")
    s_v3_smpl = _ort_sess(v3_onnx / "q1_v3_tok_human_smpl.onnx")
    s_v3_vr = _ort_sess(v3_onnx / "q1_v3_tok_vr_3pt.onnx")
    s_v3_plan = _ort_sess(v3_onnx / "q1_v3_tok_planner_cmd.onnx")

    ho50_specs = [
        ("V2_RobotPose_ONNX", dict(est_mode="cgfa", modality="robot_pose", tok_sess=s_v2_rob, head_sess=s_v2_head, deploy_safety=True)),
        ("V3_RobotPose_ONNX", dict(est_mode="cgfa", modality="robot_pose", tok_sess=s_v3_rob, head_sess=s_v3_head, deploy_safety=True)),
        ("V3_HumanSMPL_ONNX", dict(est_mode="cgfa", modality="human_smpl", tok_sess=s_v3_smpl, head_sess=s_v3_head, deploy_safety=True)),
        ("V3_VR3Pt_ONNX", dict(est_mode="cgfa", modality="vr_3pt", tok_sess=s_v3_vr, head_sess=s_v3_head, deploy_safety=True)),
        ("V3_PlannerCmd_ONNX", dict(est_mode="cgfa", modality="planner_cmd", tok_sess=s_v3_plan, head_sess=s_v3_head, deploy_safety=True)),
    ]

    ho50_nom, ho50_str = {}, {}
    for name, kw in ho50_specs:
        rows = [rollout_cpu_clip(cpu_env, clip_idx=i, **kw) for i in range(50)]
        ho50_nom[name] = _agg_cpu(rows)
        a = ho50_nom[name]
        print(f"[CPU HO-50 NOM] {name:20s}: surv={a['survival_rate']*100:5.1f}% ret={a['norm_return']:.3f} body={a['body_pos_rmse_cm']:5.2f}cm ms={a['cpu_step_ms_p50']:.2f}", flush=True)

    stress20_idx = [i for i, c in enumerate(ho50_meta) if c.get("v1_clip") or c["source"] in ("lafan1", "synthetic")]
    for i in range(50):
        if len(stress20_idx) >= 20:
            break
        if i not in stress20_idx:
            stress20_idx.append(i)
    stress20_idx = stress20_idx[:20]

    for name, kw in ho50_specs:
        srows = [
            rollout_cpu_clip(
                cpu_env, clip_idx=ci, friction_scale=0.50, delay_steps=1,
                pushes=((60, 0.0, 0.55), (120, 0.40, -0.40)), stress_sensor=True, **kw,
            )
            for ci in stress20_idx
        ]
        ho50_str[name] = _agg_cpu(srows)
        sa = ho50_str[name]
        print(f"[CPU HO-20 STR] {name:20s}: surv={sa['survival_rate']*100:5.1f}% ret={sa['norm_return']:.3f} body={sa['body_pos_rmse_cm']:5.2f}cm slip={sa['stance_slip_mm_s']:5.1f}mm/s", flush=True)

    # Head-to-Head 0.70 m/s push + 40ms delay qpos recording (V2 vs V3)
    push_clip_idx = next((i for i, c in enumerate(ho50_meta) if "walk2_subject1" in c["alias"]), walk_idx)
    h2h_v2 = rollout_cpu_clip(cpu_env, clip_idx=push_clip_idx, est_mode="cgfa", modality="robot_pose", tok_sess=s_v2_rob, head_sess=s_v2_head, deploy_safety=True, delay_steps=2, pushes=((60, 0.0, 0.70),), save_qpos=True)
    h2h_v3 = rollout_cpu_clip(cpu_env, clip_idx=push_clip_idx, est_mode="cgfa", modality="robot_pose", tok_sess=s_v3_rob, head_sess=s_v3_head, deploy_safety=True, delay_steps=2, pushes=((60, 0.0, 0.70),), save_qpos=True)
    np.savez_compressed(
        base / "v2_vs_v3_push_qpos.npz",
        v2_qpos=h2h_v2.pop("qpos_pol"),
        v3_qpos=h2h_v3.pop("qpos_pol"),
        v2_pz=np.array(h2h_v2.pop("pelvis_z_trace")),
        v3_pz=np.array(h2h_v3.pop("pelvis_z_trace")),
    )
    print(f"[H2H PUSH 0.70m/s + 40ms ({ho50_meta[push_clip_idx]['alias']})] V2 completed={h2h_v2['completed']} ({h2h_v2['steps_run']}/{h2h_v2['budget']}) | V3 completed={h2h_v3['completed']} ({h2h_v3['steps_run']}/{h2h_v3['budget']})", flush=True)

    # Zero-IK Raw Human BVH 24 (V2 vs V3)
    bvh_v2 = eval_zero_ik_bvh_24(cpu_env, stand_idx, s_v2_smpl, s_v2_head)
    bvh_v3 = eval_zero_ik_bvh_24(cpu_env, stand_idx, s_v3_smpl, s_v3_head)
    if bvh_v3["sample_bvh_qpos"] is not None:
        np.savez_compressed(base / "v3_sample_bvh_qpos.npz", qpos=bvh_v3["sample_bvh_qpos"], name=bvh_v3["sample_bvh_name"])
    bvh_v2.pop("sample_bvh_qpos", None)
    bvh_v3.pop("sample_bvh_qpos", None)
    print(f"[RAW BVH 24] V2_HumanSMPL_ONNX={bvh_v2['survival_rate']*100:.1f}% ({bvh_v2['n_bvhs']}) | V3_HumanSMPL_ONNX={bvh_v3['survival_rate']*100:.1f}% ({bvh_v3['n_bvhs']})", flush=True)

    # 30s Continuous 4-Modality Mode-Switch + Push + 20ms Delay + 400ms Watchdog
    smpl_bvh = Path("/home/yu/datasets/heldout_raw_bvh_24/soma_stat_3_eating_apple_R_002__A276.bvh")
    if not smpl_bvh.is_file():
        smpl_bvh = sorted(Path("/home/yu/datasets/heldout_raw_bvh_24").glob("*.bvh"))[0]
    ms30 = run_30s_sonic_v3_modeswitch(
        cpu_env, stand_idx, walk_idx, vr_idx, v3_onnx, smpl_bvh,
        base / "sonic_v3_modeswitch_30s_qpos.npz",
        base / "sonic_v3_modeswitch_30s.json",
    )

    out_metrics = {
        "eval_wall_s": round(time.monotonic() - t0_all, 2),
        "held_out_200_nominal": ho200_nom,
        "held_out_200_hardened_stress": ho200_str,
        "push_impulse_sweep_50": push_sweep,
        "actuator_delay_sweep_50": delay_sweep,
        "cpu_sim2sim_v2_held_out_50_nominal": ho50_nom,
        "cpu_sim2sim_v2_held_out_20_hardened": ho50_str,
        "head_to_head_push70_delay40ms": {
            "clip_alias": ho50_meta[push_clip_idx]["alias"],
            "V2_RobotPose_ONNX": h2h_v2,
            "V3_RobotPose_ONNX": h2h_v3,
        },
        "zero_ik_raw_bvh_24": {
            "V2_HumanSMPL_ONNX": bvh_v2,
            "V3_HumanSMPL_ONNX": bvh_v3,
        },
        "sonic_v3_modeswitch_30s": {k: v for k, v in ms30.items() if k != "trace_sampled"},
    }
    cli_args.out_json.write_text(json.dumps(out_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"EVAL_V3_ALL_DONE in {out_metrics['eval_wall_s']}s -> {cli_args.out_json}", flush=True)


if __name__ == "__main__":
    main()
