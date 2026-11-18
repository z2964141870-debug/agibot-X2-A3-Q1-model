"""Controlled H002 Training Ablation: M1 (Legacy 0ms), M2 (CGFA 0ms), M3 (CGFA + WM-Contact-Reg + Jitter 0ms).

Trains strictly on the clean 12-clip training split across LAFAN1 + PHUMA + BONES-SEED/SOMA,
leaving 4 clips (stat_fit_squat, soma_stat_3_eating_apple_R_002__A276, dyn_hml_3, walk2_subject1)
completely held out. Keeps 904-clip motor decoder primitives anchored cleanly via frozen
obs_norm/goal_norm/hidden-decoder layers in Stage 1 and head-only proximal co-adaptation in Stage 2.
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
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

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgfa_estimator import (
    LEFT_ANKLE_IDX,
    RIGHT_ANKLE_IDX,
    ContactGatedFootAnchor,
)
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import so3_exp
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel, run_mlp
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale
from unisupermimic.teleop.realtime_teleop import build_causal_history_goal, load_world_model

HELD_OUT_STEMS = {
    "stat_fit_squat",
    "soma_stat_3_eating_apple_R_002__A276",
    "dyn_hml_3",
    "walk2_subject1",
}


def discover_motion_splits(repo_root: Path = ROOT) -> tuple[list[str], list[str]]:
    all_files = sorted(
        glob.glob(str(repo_root / "src/unisupermimic/assets/motions/q1/lafan1_showcase/*.csv"))
        + glob.glob(str(repo_root / "src/unisupermimic/assets/motions/q1/phuma_showcase/*.csv"))
        + glob.glob(str(repo_root / "src/unisupermimic/assets/motions/q1/soma_showcase/*.csv"))
    )
    train_files, held_out_files = [], []
    for f in all_files:
        stem = Path(f).stem
        if stem in HELD_OUT_STEMS:
            held_out_files.append(str(Path(f).resolve()))
        else:
            train_files.append(str(Path(f).resolve()))
    return train_files, held_out_files


def _load_default_q(mjcf_path: Path, num_joints: int, device: torch.device) -> torch.Tensor:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    qpos = model.key_qpos[0] if model.nkey > 0 else model.qpos0
    return torch.tensor(qpos[7 : 7 + num_joints], dtype=torch.float32, device=device)


def _sample_perturbed_state(mdp: MDP, frames: torch.Tensor, noise_scale: float = 0.25) -> tuple[torch.Tensor, ...]:
    p, R, v, w, q, qd = (t.clone() for t in mdp.ref_state(frames))
    if noise_scale <= 0.0:
        return (p, R, v, w, q, qd)
    n = frames.shape[0]
    dev = frames.device
    p = p + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.02, 0.02)
    R = so3_exp(noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.05, 0.05)) @ R
    v = v + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.12, 0.12)
    w = w + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.18, 0.18)
    q = torch.clamp(
        q + noise_scale * torch.empty((n, mdp.num_joints), device=dev).uniform_(-0.04, 0.04),
        mdp.kin.q_lo,
        mdp.kin.q_hi,
    )
    return (p, R, v, w, q, qd)


def _build_stage1_obs_history(
    mdp: MDP,
    frames: torch.Tensor,
    hist: int,
    xi: torch.Tensor,
    mode: str,
    noise_scale: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (obs_hist_student, obs_hist_teacher) where teacher gets oracle root_xy."""
    motion = mdp.motion
    B = frames.shape[0]
    clip_start = motion.clip_start[motion.clip_id[frames]]
    offsets = torch.arange(-(hist - 1), 1, device=frames.device, dtype=frames.dtype)
    h_frames = torch.clamp(frames.unsqueeze(-1) + offsets.unsqueeze(0), min=clip_start.unsqueeze(-1))
    flat_f = h_frames.reshape(-1)
    flat_xi = xi.unsqueeze(1).expand(-1, hist, -1).reshape(-1, mdp.context_dim)
    state = _sample_perturbed_state(mdp, flat_f, noise_scale=noise_scale)
    a_prev = torch.zeros((flat_f.shape[0], mdp.num_joints), device=frames.device)
    obs_teacher = mdp.obs(state, a_prev, flat_xi, flat_f).view(B, hist, mdp.obs_dim)
    obs_student = obs_teacher.clone()

    if mode == "zeroed":
        obs_student[..., 78:80] = 0.0
    elif mode == "cgfa":
        drift_bias = torch.randn((B, 1, 2), device=frames.device) * 0.015
        step_noise = torch.randn((B, hist, 2), device=frames.device) * 0.004
        obs_student[..., 78:80] = 0.25 * torch.tanh((obs_teacher[..., 78:80] + drift_bias + step_noise) / 0.25)
    else:
        raise ValueError(f"Unknown Stage-1 student obs mode: {mode}")
    return obs_student, obs_teacher


def _compute_wm_contact_and_pd_loss(
    mdp: MDP,
    next_state: tuple[torch.Tensor, ...],
    f_next: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    alpha_vec: torch.Tensor,
    sole_offset: torch.Tensor,
    margin_rad: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable stance-sole no-slip, hard PD bound, and action smoothness losses."""
    motion = mdp.motion
    p, R, v, w, q, qd = next_state
    _, body_R, body_v, body_w = mdp.kin.forward(p, R, v, w, q, qd)

    ref_ankle_p = motion.body_p[f_next][:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
    ref_ankle_v = motion.body_v[f_next][:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2]
    c_stance = (
        torch.sigmoid((0.062 - ref_ankle_p[..., 2]) / 0.008)
        * torch.sigmoid((0.15 - ref_ankle_v.norm(dim=-1)) / 0.04)
    ).detach()

    foot_R = body_R[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :, :]
    foot_v = body_v[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
    foot_w = body_w[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
    r_sole_local = (foot_R @ sole_offset.view(1, 1, 3, 1)).squeeze(-1)
    v_sole = foot_v + torch.cross(foot_w, r_sole_local, dim=-1)

    l_slip = (
        c_stance * (v_sole[..., :2] ** 2).sum(dim=-1)
        + 0.02 * c_stance * (foot_w[..., :2] ** 2).sum(dim=-1)
    ).sum(dim=-1)

    q_tgt = mdp.default_q + alpha_vec * action
    l_pd = (
        F.relu(q_tgt - (mdp.kin.q_hi - margin_rad)) ** 2
        + F.relu((mdp.kin.q_lo + margin_rad) - q_tgt) ** 2
    ).sum(dim=-1)

    l_smooth = ((action - prev_action) ** 2).sum(dim=-1)
    return l_slip, l_pd, l_smooth


def train_variant(
    variant_name: str,
    *,
    obs_mode: str,
    use_wm_contact_reg: bool,
    use_causal_jitter: bool,
    ckpt_path: Path,
    scheme_a_ckpt_path: Path,
    train_files: list[str],
    out_ckpt_path: Path,
    stage1_steps: int,
    stage1_batch: int,
    stage2_steps: int,
    stage2_batch: int,
    stage2_horizon: int,
    seed: int = 42,
    device_str: str = "cuda",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    networks.AMP_DTYPE = None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sa_checkpoint = torch.load(scheme_a_ckpt_path, map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    goal_frames = int(args["goal_frames"])
    hist = int(args["hist"])
    z_dim = int(args["z_dim"])
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])

    mjcf_path = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
    kin = Q1Kinematics(str(mjcf_path), device)
    motion = MotionLib(train_files, kin, device, src_fps=30.0)
    default_q = _load_default_q(mjcf_path, len(Q1_JOINT_NAMES), device)
    mdp = MDP(kin, motion, default_q, observation_mode="root_xy")

    teacher = Policy(
        mdp.obs_dim, mdp.goal_frame_dim * goal_frames, num_joints=mdp.num_joints, hist=hist, z_dim=z_dim, hidden=hidden
    ).to(device)
    teacher.load_state_dict(checkpoint["pi"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    wm: WorldModel = load_world_model(mdp, checkpoint, device)

    student = copy.deepcopy(teacher)
    # Copy the 904-clip causal goal_norm statistics from Scheme A and freeze obs_norm + goal_norm
    with torch.no_grad():
        student.goal_norm.mean.copy_(sa_checkpoint["pi"]["goal_norm.mean"].to(device))
        student.goal_norm.var.copy_(sa_checkpoint["pi"]["goal_norm.var"].to(device))
        student.goal_norm.count.copy_(sa_checkpoint["pi"]["goal_norm.count"].to(device))

    for p in student.obs_norm.parameters():
        p.requires_grad_(False)
    for p in student.goal_norm.parameters():
        p.requires_grad_(False)
    for p in student.dec.parameters():
        p.requires_grad_(False)
    for p in student.enc.parameters():
        p.requires_grad_(True)

    total_frames = int(motion.num_frames)
    alpha_vec = torch.tensor([action_scale(j) for j in Q1_JOINT_NAMES], dtype=torch.float32, device=device)
    margin_rad = 0.005
    lo_act = (mdp.kin.q_lo + margin_rad - mdp.default_q) / alpha_vec
    hi_act = (mdp.kin.q_hi - margin_rad - mdp.default_q) / alpha_vec
    nominal_xi = torch.zeros((1, mdp.context_dim), dtype=torch.float32, device=device)
    nominal_xi[:, 0] = 1.0

    # ------------------------------------------------------------------
    # Stage 1: Latent Intent + Frozen-Decoder Action Distillation (enc only)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    opt1 = torch.optim.AdamW(student.enc.parameters(), lr=5e-4, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(stage1_steps, 1), eta_min=1e-5)

    last_s1: dict[str, float] = {}
    for step in range(1, stage1_steps + 1):
        f_batch = torch.randint(0, total_frames, (stage1_batch,), device=device)
        with torch.no_grad():
            g_f = mdp.goal(f_batch, goal_frames)
            g_c0 = build_causal_history_goal(motion, f_batch, goal_frames, latest_offset=0)
            if use_causal_jitter:
                g_c1 = build_causal_history_goal(motion, f_batch, goal_frames, latest_offset=-1)
                jit_mask = torch.rand((stage1_batch, 1), device=device) < 0.25
                g_c = torch.where(jit_mask, g_c1, g_c0)
            else:
                g_c = g_c0
            u_t = run_mlp(teacher.enc, teacher.goal_norm(g_f))
            z_t = torch.tanh(u_t)

            xi_batch = nominal_xi.expand(stage1_batch, -1)
            obs_s, obs_t = _build_stage1_obs_history(mdp, f_batch, hist, xi_batch, mode=obs_mode, noise_scale=0.20)
            obs_t_flat = teacher.obs_norm(obs_t).flatten(-2)
            a_t = run_mlp(teacher.dec, torch.cat((z_t, obs_t_flat), dim=-1))

        u_s = run_mlp(student.enc, student.goal_norm(g_c))
        z_s = torch.tanh(u_s)
        obs_s_flat = student.obs_norm(obs_s).flatten(-2)
        a_s = run_mlp(student.dec, torch.cat((z_s, obs_s_flat), dim=-1))

        loss_z = F.mse_loss(z_s, z_t)
        loss_u = F.mse_loss(u_s, u_t)
        loss_a = F.mse_loss(a_s, a_t)
        loss = loss_z + 0.05 * loss_u + 0.5 * loss_a

        opt1.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.enc.parameters(), 1.0)
        opt1.step()
        sched1.step()
        last_s1 = {
            "s1_latent_mse": float(loss_z.item()),
            "s1_action_mse": float(loss_a.item()),
            "s1_cos_sim": float(F.cosine_similarity(z_s.detach(), z_t, dim=-1).mean().item()),
        }
        if step == 1 or step % max(stage1_steps // 4, 1) == 0 or step == stage1_steps:
            zm = last_s1["s1_latent_mse"]
            am = last_s1["s1_action_mse"]
            cs = last_s1["s1_cos_sim"]
            print(f"[{variant_name} S1 {step:4d}/{stage1_steps}] z_mse={zm:.4f} a_mse={am:.4f} cos={cs:.4f}", flush=True)
    s1_wall_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Stage 2: Differentiable WorldModel Closed-Loop Fine-Tuning
    #          (Unfreeze enc + ONLY final linear output head student.dec[-1])
    # ------------------------------------------------------------------
    for p in student.dec[-1].parameters():
        p.requires_grad_(True)

    t1 = time.perf_counter()
    all_frames = torch.arange(total_frames, dtype=torch.long, device=device)
    valid_starts = all_frames[mdp.valid_imagination_start(all_frames, stage2_horizon + 2)]
    opt2 = torch.optim.AdamW(
        [
            {"params": student.enc.parameters(), "lr": 2e-5},
            {"params": student.dec[-1].parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(stage2_steps, 1), eta_min=2e-6)
    max_r = float(getattr(mdp, "max_step_reward", 0.1))
    est = ContactGatedFootAnchor(mdp.kin, stage2_batch, dt=0.02, device=device)

    last_s2: dict[str, float] = {}
    grad_check_enc = 0.0
    grad_check_dec = 0.0
    for step in range(1, stage2_steps + 1):
        sel = torch.randint(0, valid_starts.numel(), (stage2_batch,), device=device)
        f_cur = valid_starts[sel]
        xi = nominal_xi.expand(stage2_batch, -1)
        norm_xi = mdp.normalize_context(xi)

        with torch.no_grad():
            obs_hist_s, obs_hist_t = _build_stage1_obs_history(
                mdp, f_cur, hist, xi, mode=obs_mode, noise_scale=0.16
            )
            state = _sample_perturbed_state(mdp, f_cur, noise_scale=0.16)
            a_prev = torch.zeros((stage2_batch, mdp.num_joints), device=device)
            if obs_mode == "cgfa":
                init_err = motion.root_p[f_cur, :2] - state[0][:, :2]
                est.reset(
                    state[1], state[3], state[4], state[5],
                    motion.root_p[f_cur, :2], init_error_xy=init_err, R_ref=motion.root_R[f_cur],
                )

        tot_rew = torch.zeros(stage2_batch, device=device)
        tot_term = torch.zeros(stage2_batch, device=device)
        tot_act = torch.zeros(stage2_batch, device=device)
        tot_z = torch.zeros(stage2_batch, device=device)
        tot_slip = torch.zeros((), device=device)
        tot_pd = torch.zeros((), device=device)
        tot_smooth = torch.zeros((), device=device)
        alive = torch.ones(stage2_batch, device=device)

        for h in range(stage2_horizon):
            with torch.no_grad():
                if use_causal_jitter:
                    jit_mask = torch.rand((stage2_batch,), device=device) < 0.25
                    g_c0 = build_causal_history_goal(motion, f_cur, goal_frames, latest_offset=0)
                    g_c1 = build_causal_history_goal(motion, f_cur, goal_frames, latest_offset=-1)
                    g_c = torch.where(jit_mask.unsqueeze(-1), g_c1, g_c0)
                else:
                    jit_mask = None
                    g_c = build_causal_history_goal(motion, f_cur, goal_frames, latest_offset=0)
                g_f = mdp.goal(f_cur, goal_frames)
                z_t = torch.tanh(run_mlp(teacher.enc, teacher.goal_norm(g_f)))

            z_s = torch.tanh(run_mlp(student.enc, student.goal_norm(g_c)))
            obs_s_flat = student.obs_norm(obs_hist_s).flatten(-2)
            a_s = run_mlp(student.dec, torch.cat((z_s, obs_s_flat), dim=-1))

            with torch.no_grad():
                obs_t_flat = teacher.obs_norm(obs_hist_t).flatten(-2)
                a_t = run_mlp(teacher.dec, torch.cat((z_t, obs_t_flat), dim=-1))
                if use_wm_contact_reg:
                    a_t = torch.clamp(a_t, lo_act, hi_act)

            next_state = wm.step(state, a_s, norm_xi)
            f_next = f_cur + 1
            rew = mdp.reward(next_state, f_next, a_s, a_prev) / max_r
            term = mdp.soft_termination(next_state, f_next)

            disc = 0.99**h
            tot_rew = tot_rew + disc * alive * rew
            tot_term = tot_term + disc * alive * term
            tot_act = tot_act + ((a_s - a_t) ** 2).mean(dim=-1)
            tot_z = tot_z + ((z_s - z_t) ** 2).mean(dim=-1)

            if use_wm_contact_reg:
                l_slip, l_pd, l_smooth = _compute_wm_contact_and_pd_loss(
                    mdp, next_state, f_next, a_s, a_prev, alpha_vec, est.sole_offset, margin_rad=margin_rad
                )
                tot_slip = tot_slip + disc * l_slip.mean() / stage2_horizon
                tot_pd = tot_pd + disc * l_pd.mean() / stage2_horizon
                if h >= 1:
                    tot_smooth = tot_smooth + disc * l_smooth.mean() / max(stage2_horizon - 1, 1)

            alive = alive * (1.0 - term.detach())
            obs_t_next = mdp.obs(next_state, a_s.detach(), xi, f_next)
            obs_s_next = mdp.obs(next_state, a_s, xi, f_next)
            if obs_mode == "zeroed":
                obs_s_next = torch.cat((obs_s_next[..., :78], torch.zeros_like(obs_s_next[..., 78:80])), dim=-1)
            elif obs_mode == "cgfa":
                f_ref_next = torch.where(jit_mask, f_cur, f_next) if jit_mask is not None else f_next
                e_xy_next, _ = est.step(
                    next_state[1], next_state[3], next_state[4], next_state[5],
                    motion.root_p[f_ref_next, :2], R_ref=motion.root_R[f_ref_next],
                    mode="cgfa", slip_noise_std_m=0.0020, differentiable=True,
                )
                obs_s_next = torch.cat((obs_s_next[..., :78], e_xy_next), dim=-1)

            obs_hist_s = torch.cat((obs_hist_s[:, 1:], obs_s_next.unsqueeze(1)), dim=1)
            obs_hist_t = torch.cat((obs_hist_t[:, 1:], obs_t_next.unsqueeze(1)), dim=1)
            state = next_state
            a_prev = a_s
            f_cur = f_next

        loss_wm = (
            1.0 * (tot_z.mean() / stage2_horizon)
            + 1.5 * (tot_act.mean() / stage2_horizon)
            + 0.12 * (-tot_rew.mean() / stage2_horizon + 1.5 * (tot_term.mean() / stage2_horizon))
        )
        if use_wm_contact_reg:
            loss_wm = loss_wm + 0.06 * tot_slip + 0.8 * tot_pd + 0.015 * tot_smooth

        opt2.zero_grad(set_to_none=True)
        loss_wm.backward()
        enc_gn = float(torch.sqrt(sum((p.grad.pow(2).sum() for p in student.enc.parameters() if p.grad is not None), torch.tensor(0.0, device=device))).item())
        dec_gn = float(torch.sqrt(sum((p.grad.pow(2).sum() for p in student.dec[-1].parameters() if p.grad is not None), torch.tensor(0.0, device=device))).item())
        grad_check_enc = enc_gn
        grad_check_dec = dec_gn
        all_params = list(student.enc.parameters()) + list(student.dec[-1].parameters())
        gn = nn.utils.clip_grad_norm_(all_params, 0.5)
        if torch.isfinite(gn):
            opt2.step()
        sched2.step()

        last_s2 = {
            "s2_imag_rew": float((tot_rew.mean() / stage2_horizon).item()),
            "s2_imag_surv": float(alive.mean().item()),
            "s2_slip_loss": float(tot_slip.item()),
            "s2_pd_loss": float(tot_pd.item()),
            "s2_smooth_loss": float(tot_smooth.item()),
            "s2_grad_enc": enc_gn,
            "s2_grad_dec_head": dec_gn,
        }
        if step == 1 or step % max(stage2_steps // 4, 1) == 0 or step == stage2_steps:
            rw = last_s2["s2_imag_rew"]
            sv = last_s2["s2_imag_surv"]
            sl = last_s2["s2_slip_loss"]
            pdv = last_s2["s2_pd_loss"]
            print(
                f"[{variant_name} S2 {step:4d}/{stage2_steps}] rew={rw:.4f} surv={sv:.4f} "
                f"slip={sl:.5f} pd={pdv:.5f} |grad_enc|={enc_gn:.3f} |grad_dec_head|={dec_gn:.3f}",
                flush=True,
            )

    s2_wall_s = time.perf_counter() - t1
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

    summary = {
        "variant": variant_name,
        "obs_mode": obs_mode,
        "use_wm_contact_reg": use_wm_contact_reg,
        "use_causal_jitter": use_causal_jitter,
        "train_clips": [Path(p).stem for p in train_files],
        "held_out_clips": sorted(HELD_OUT_STEMS),
        "stage1_steps": stage1_steps,
        "stage2_steps": stage2_steps,
        "stage1_wall_s": round(s1_wall_s, 2),
        "stage2_wall_s": round(s2_wall_s, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "grad_check_finite": bool(torch.isfinite(torch.tensor([grad_check_enc, grad_check_dec])).all()),
        **last_s1,
        **last_s2,
    }

    out_ckpt = dict(checkpoint)
    out_ckpt["pi"] = student.state_dict()
    out_ckpt["h002_ablation"] = summary
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_ckpt_path)
    out_ckpt_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del teacher, student, wm, mdp, kin, motion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 1-min smoke test on M3 only")
    parser.add_argument("--variants", nargs="*", default=["m1", "m2", "m3"])
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints/q1_base_soma1k.pt")
    parser.add_argument("--scheme-a-ckpt", type=Path, default=ROOT / "checkpoints/q1_scheme_a_1k.pt")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--stage1-steps", type=int, default=1200)
    parser.add_argument("--stage1-batch", type=int, default=512)
    parser.add_argument("--stage2-steps", type=int, default=160)
    parser.add_argument("--stage2-batch", type=int, default=192)
    parser.add_argument("--stage2-horizon", type=int, default=8)
    args = parser.parse_args()

    train_files, held_out_files = discover_motion_splits()
    print(f"Split verified: {len(train_files)} TRAIN clips, {len(held_out_files)} HELD_OUT clips", flush=True)
    print("Held-out:", [Path(p).stem for p in held_out_files], flush=True)
    assert len(train_files) == 12 and len(held_out_files) == 4, (len(train_files), len(held_out_files))

    specs = {
        "m1": ("M1_pure_legacy_0ms", "zeroed", False, False, "m1_pure_legacy_0ms.pt"),
        "m2": ("M2_cgfa_only_0ms", "cgfa", False, False, "m2_cgfa_only_0ms.pt"),
        "m3": ("M3_h002_full_0ms", "cgfa", True, True, "m3_h002_full_0ms.pt"),
    }

    if args.smoke:
        name, omode, wreg, jit, fn = specs["m3"]
        res = train_variant(
            name + "_SMOKE",
            obs_mode=omode,
            use_wm_contact_reg=wreg,
            use_causal_jitter=jit,
            ckpt_path=args.ckpt,
            scheme_a_ckpt_path=args.scheme_a_ckpt,
            train_files=train_files,
            out_ckpt_path=args.out_dir / ("smoke_" + fn),
            stage1_steps=30,
            stage1_batch=256,
            stage2_steps=8,
            stage2_batch=96,
            stage2_horizon=6,
        )
        print("SMOKE_OK:", json.dumps(res, indent=2), flush=True)
        (args.out_dir / ("smoke_" + fn)).unlink(missing_ok=True)
        (args.out_dir / ("smoke_" + fn)).with_suffix(".json").unlink(missing_ok=True)
        return

    all_summaries = {}
    for key in args.variants:
        name, omode, wreg, jit, fn = specs[key]
        res = train_variant(
            name,
            obs_mode=omode,
            use_wm_contact_reg=wreg,
            use_causal_jitter=jit,
            ckpt_path=args.ckpt,
            scheme_a_ckpt_path=args.scheme_a_ckpt,
            train_files=train_files,
            out_ckpt_path=args.out_dir / fn,
            stage1_steps=args.stage1_steps,
            stage1_batch=args.stage1_batch,
            stage2_steps=args.stage2_steps,
            stage2_batch=args.stage2_batch,
            stage2_horizon=args.stage2_horizon,
        )
        all_summaries[key] = res

    (args.out_dir.parent / "train_summary.json").write_text(json.dumps(all_summaries, indent=2) + "\n", encoding="utf-8")
    print("ALL_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
