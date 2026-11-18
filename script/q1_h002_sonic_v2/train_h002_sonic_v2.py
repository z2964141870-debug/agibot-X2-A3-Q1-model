"""H002-SONIC v2b Multi-Dataset (200 Train Clips) & 4-Modality Universal Token Trainer.

Key v2b design (verified via 5-way ablation):
  - Freeze the 904-clip closed-loop RL Universal TokenHead (obs_norm + dec) and goal_norm from
    `q1_scheme_a_1k.pt` so contact/balance primitives are 100% preserved (no open-loop WM drift).
  - Phase 1: Proximal adaptation of Enc_RobotPose (590D) across 200 multi-dataset clips with CGFA
    residuals, 50% cold/warm a_prev starts, 20% 1-frame causal arrival jitter, and L2 proximal pull.
  - Phase 2: Freeze TokenHead + compute SHA-256 codec_fingerprint, then train three 0ms causal
    tokenizers into the shared 32D token space via unclamped latent + TokenHead action distillation:
      * Enc_HumanSMPL (430D, scale-invariant bone-chain, zero-IK capable)
      * Enc_VR3Pt     (360D, PICO torso + dual-wrist 6DoF)
      * Enc_PlannerCmd(100D, gamepad/keyboard velocity + height + stand/crouch)
"""

from __future__ import annotations

import argparse
import copy
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

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multimodal_features import (
    MODALITY_DIMS,
    build_human_smpl_goal,
    build_planner_cmd_goal,
    build_robot_pose_goal,
    build_vr_3pt_goal,
)
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import so3_exp
from unisupermimic.rl.algos.supermimic.networks import Policy, RunningNorm, mlp, run_mlp
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES

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


def _sample_perturbed_state(mdp: MDP, frames: torch.Tensor, noise_scale: float = 0.18) -> tuple[torch.Tensor, ...]:
    p, R, v, w, q, qd = (t.clone() for t in mdp.ref_state(frames))
    if noise_scale <= 0.0:
        return (p, R, v, w, q, qd)
    n = frames.shape[0]
    dev = frames.device
    p = p + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.025, 0.025)
    R = so3_exp(noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.05, 0.05)) @ R
    v = v + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.14, 0.14)
    w = w + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.18, 0.18)
    q = torch.clamp(
        q + noise_scale * torch.empty((n, mdp.num_joints), device=dev).uniform_(-0.05, 0.05),
        mdp.kin.q_lo,
        mdp.kin.q_hi,
    )
    return (p, R, v, w, q, qd)


def _sample_obs_histories(
    mdp: MDP,
    teacher: Policy,
    frames: torch.Tensor,
    hist: int,
    obs_mode: str = "cgfa",
    noise_scale: float = 0.18,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 10-frame obs histories with 50% cold zero a_prev / 50% warm teacher a_prev + CGFA residual."""
    motion = mdp.motion
    B = frames.shape[0]
    dev = frames.device
    clip_start = motion.clip_start[motion.clip_id[frames]]
    offsets = torch.arange(-(hist - 1), 1, device=dev, dtype=frames.dtype)
    h_frames = torch.clamp(frames.unsqueeze(-1) + offsets.unsqueeze(0), min=clip_start.unsqueeze(-1))
    flat_f = h_frames.reshape(-1)

    xi = torch.zeros((B, mdp.context_dim), dtype=torch.float32, device=dev)
    xi[:, 0] = 0.55 + 0.70 * torch.rand((B,), device=dev)
    flat_xi = xi.unsqueeze(1).expand(-1, hist, -1).reshape(-1, mdp.context_dim)

    state_hist = _sample_perturbed_state(mdp, flat_f, noise_scale=noise_scale)
    zero_a = torch.zeros((flat_f.shape[0], mdp.num_joints), device=dev)
    obs0 = mdp.obs(state_hist, zero_a, flat_xi, flat_f).view(B, hist, mdp.obs_dim)
    g_cur = mdp.goal(frames, 10)
    a_teacher = teacher(obs0, g_cur)
    use_zero = torch.rand((B, 1), device=dev) < 0.5
    a_prev_b = torch.where(use_zero, torch.zeros_like(a_teacher), a_teacher)
    flat_a_prev = a_prev_b.unsqueeze(1).expand(-1, hist, -1).reshape(-1, mdp.num_joints)

    obs_teacher = mdp.obs(state_hist, flat_a_prev, flat_xi, flat_f).view(B, hist, mdp.obs_dim)
    obs_student = obs_teacher.clone()
    if obs_mode == "cgfa":
        drift_bias = torch.randn((B, 1, 2), device=dev) * 0.015
        step_noise = torch.randn((B, hist, 2), device=dev) * 0.004
        obs_student[..., 78:80] = 0.25 * torch.tanh((obs_teacher[..., 78:80] + drift_bias + step_noise) / 0.25)
    elif obs_mode == "zeroed":
        obs_student[..., 78:80] = 0.0
    return obs_student, obs_teacher


def _sample_1frame_jitter(frames: torch.Tensor) -> torch.Tensor:
    """Clean 1-frame causal arrival jitter only: 20% latest_offset=-1, 80% 0."""
    u = torch.rand((frames.shape[0],), device=frames.device)
    return torch.where(u < 0.20, -1, 0)


def _calibrate_norm(norm: RunningNorm, build_fn: Any, motion: MotionLib, total_frames: int, device: torch.device) -> None:
    with torch.no_grad():
        dim = norm.mean.numel()
        sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
        sum_sq = torch.zeros(dim, dtype=torch.float64, device=device)
        for s in range(0, total_frames, 2048):
            e = min(s + 2048, total_frames)
            fc = torch.arange(s, e, dtype=torch.long, device=device)
            gx = build_fn(motion, fc, 10).double()
            sum_x += gx.sum(dim=0)
            sum_sq += (gx * gx).sum(dim=0)
        mean_x = sum_x / max(total_frames, 1)
        var_x = (sum_sq / max(total_frames, 1) - mean_x * mean_x).clamp(min=1e-3)
        norm.mean.copy_(mean_x.float())
        norm.var.copy_(var_x.float())
        norm.count.fill_(float(total_frames))


def train_pipeline(args: argparse.Namespace) -> None:
    t_total = time.monotonic()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    networks.AMP_DTYPE = None

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_files = sorted(str(ROOT / r["csv_rel"]) for r in manifest["clips"] if r["split"] == "train")
    print(f"Loaded {len(train_files)} TRAIN clips from {args.manifest}", flush=True)

    base_ck = torch.load(args.base_ckpt, map_location="cpu", weights_only=True)
    sa_ck = torch.load(args.scheme_a_ckpt, map_location="cpu", weights_only=True)
    c_args = base_ck["args"]
    goal_frames = int(c_args["goal_frames"])
    hist = int(c_args["hist"])
    z_dim = int(c_args["z_dim"])
    hidden = tuple(c_args.get("test_hidden_dims") or c_args["policy_hidden"])

    kin = Q1Kinematics(str(MJCF_PATH), device)
    motion = MotionLib(train_files, kin, device, src_fps=30.0)
    default_q = _load_default_q(device)
    mdp = MDP(kin, motion, default_q, observation_mode="root_xy")
    total_frames = int(motion.num_frames)
    print(f"Train MotionLib @50Hz: {total_frames} frames ({total_frames/50.0:.1f}s)", flush=True)

    teacher = Policy(
        mdp.obs_dim, mdp.goal_frame_dim * goal_frames, num_joints=mdp.num_joints, hist=hist, z_dim=z_dim, hidden=hidden
    ).to(device)
    teacher.load_state_dict(base_ck["pi"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Freeze Universal TokenHead (obs_norm + dec) and goal_norm 100% at q1_scheme_a_1k.pt!
    student = Policy(
        mdp.obs_dim, mdp.goal_frame_dim * goal_frames, num_joints=mdp.num_joints, hist=hist, z_dim=z_dim, hidden=hidden
    ).to(device)
    student.load_state_dict(sa_ck["pi"])
    for p in student.obs_norm.parameters():
        p.requires_grad_(False)
    for p in student.goal_norm.parameters():
        p.requires_grad_(False)
    for p in student.dec.parameters():
        p.requires_grad_(False)
    for p in student.enc.parameters():
        p.requires_grad_(True)

    sa_enc_ref = copy.deepcopy(student.enc).eval()
    for p in sa_enc_ref.parameters():
        p.requires_grad_(False)

    # Phase 1: Proximal Enc_RobotPose (590D) adaptation across 200 multi-dataset clips
    p1_steps = 20 if args.smoke else args.p1_s1_steps
    batch_size = 256 if args.smoke else 512

    t_p1 = time.perf_counter()
    opt1 = torch.optim.AdamW(student.enc.parameters(), lr=5e-5, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(p1_steps, 1), eta_min=5e-6)

    p1_log: dict[str, Any] = {}
    for step in range(1, p1_steps + 1):
        fb = torch.randint(0, total_frames, (batch_size,), device=device)
        with torch.no_grad():
            loff = _sample_1frame_jitter(fb)
            clip_s = motion.clip_start[motion.clip_id[fb]]
            clip_e = motion.clip_end[motion.clip_id[fb]]
            f_jit = torch.clamp(fb + loff, min=clip_s, max=clip_e)

            g_c0 = build_robot_pose_goal(motion, fb, goal_frames, latest_offset=0)
            g_c_jit = build_robot_pose_goal(motion, f_jit, goal_frames, latest_offset=0)
            g_f = mdp.goal(fb, goal_frames)

            obs_s, obs_t = _sample_obs_histories(mdp, teacher, fb, hist, obs_mode="cgfa", noise_scale=0.18)
            obs_s_flat = student.obs_norm(obs_s).flatten(-2)

            u_t = run_mlp(teacher.enc, teacher.goal_norm(g_f))
            z_t = torch.tanh(u_t)
            a_t = run_mlp(teacher.dec, torch.cat((z_t, teacher.obs_norm(obs_t).flatten(-2)), dim=-1))

            u_sa = run_mlp(sa_enc_ref, student.goal_norm(g_c0))
            z_sa = torch.tanh(u_sa)
            a_sa = run_mlp(student.dec, torch.cat((z_sa, obs_s_flat), dim=-1))

        u_s = run_mlp(student.enc, student.goal_norm(g_c_jit))
        z_s = torch.tanh(u_s)
        a_s = run_mlp(student.dec, torch.cat((z_s, obs_s_flat), dim=-1))
        l_prox = sum((p - p0).pow(2).sum() for p, p0 in zip(student.enc.parameters(), sa_enc_ref.parameters()))

        loss = (
            0.5 * F.mse_loss(z_s, z_sa)
            + 0.5 * F.mse_loss(z_s, z_t)
            + 0.05 * F.mse_loss(u_s, u_t)
            + 0.6 * F.mse_loss(a_s, a_t)
            + 0.4 * F.mse_loss(a_s, a_sa)
            + 5.0 * l_prox
        )
        opt1.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.enc.parameters(), 1.0)
        opt1.step()
        sched1.step()

        p1_log["s1_cos"] = float(F.cosine_similarity(z_s.detach(), z_t, dim=-1).mean().item())
        p1_log["s1_cos_sa"] = float(F.cosine_similarity(z_s.detach(), z_sa, dim=-1).mean().item())
        p1_log["s1_a_mse"] = float(F.mse_loss(a_s.detach(), a_t).item())
        p1_log["s1_l_prox"] = float(l_prox.detach().item())
        if step == 1 or step % max(p1_steps // 4, 1) == 0 or step == p1_steps:
            print(
                f"[P1 RobotPose {step:4d}/{p1_steps}] cos_t={p1_log['s1_cos']:.4f} "
                f"cos_sa={p1_log['s1_cos_sa']:.4f} a_mse={p1_log['s1_a_mse']:.4f} prox={p1_log['s1_l_prox']:.5f}",
                flush=True,
            )

    p1_log["wall_s"] = round(time.perf_counter() - t_p1, 2)

    # Freeze TokenHead + Enc_RobotPose and compute SONIC codec_fingerprint
    for p in student.parameters():
        p.requires_grad_(False)
    student.eval()
    codec_fp = compute_codec_fingerprint(student)
    print(f"PHASE1 DONE (Frozen 904-Clip TokenHead). codec_fingerprint={codec_fp[:24]}...", flush=True)

    # Phase 2: Align 3 Multi-Modal Tokenizers (HumanSMPL, VR3Pt, PlannerCmd) to shared TokenHead
    p2_steps = 20 if args.smoke else args.p2_s1_steps

    modal_specs = [
        ("human_smpl", MODALITY_DIMS["human_smpl"], (1024, 512, 256), lambda m, f, k: build_human_smpl_goal(m, f, k, noise_std=0.012), lambda m, f, k: build_human_smpl_goal(m, f, k, noise_std=0.0)),
        ("vr_3pt", MODALITY_DIMS["vr_3pt"], (1024, 512, 256), lambda m, f, k: build_vr_3pt_goal(m, f, k, pos_noise_m=0.010), lambda m, f, k: build_vr_3pt_goal(m, f, k, pos_noise_m=0.0)),
        ("planner_cmd", MODALITY_DIMS["planner_cmd"], (512, 256, 256), lambda m, f, k: build_planner_cmd_goal(m, f, k), lambda m, f, k: build_planner_cmd_goal(m, f, k)),
    ]

    tokenizers: dict[str, ModalityTokenizer] = {}
    p2_logs: dict[str, Any] = {}

    for mname, indim, mhid, fn_noisy, fn_clean in modal_specs:
        tm0 = time.perf_counter()
        tok = ModalityTokenizer(indim, mhid, z_dim=z_dim).to(device)
        _calibrate_norm(tok.in_norm, fn_clean, motion, total_frames, device)
        for p in tok.in_norm.parameters():
            p.requires_grad_(False)
        for p in tok.enc.parameters():
            p.requires_grad_(True)

        opt_m = torch.optim.AdamW(tok.enc.parameters(), lr=6e-4, weight_decay=1e-4)
        sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=max(p2_steps, 1), eta_min=1e-5)
        mlog: dict[str, Any] = {}
        for step in range(1, p2_steps + 1):
            fb = torch.randint(0, total_frames, (batch_size,), device=device)
            with torch.no_grad():
                loff = _sample_1frame_jitter(fb)
                cs = motion.clip_start[motion.clip_id[fb]]
                ce = motion.clip_end[motion.clip_id[fb]]
                f_jit = torch.clamp(fb + loff, min=cs, max=ce)

                g_rob = build_robot_pose_goal(motion, fb, goal_frames, latest_offset=0)
                u_rob = run_mlp(student.enc, student.goal_norm(g_rob))
                z_rob = torch.tanh(u_rob)

                g_f = mdp.goal(fb, goal_frames)
                z_t = torch.tanh(run_mlp(teacher.enc, teacher.goal_norm(g_f)))

                obs_s, obs_t = _sample_obs_histories(mdp, teacher, fb, hist, obs_mode="cgfa", noise_scale=0.18)
                obs_s_flat = student.obs_norm(obs_s).flatten(-2)
                a_rob = run_mlp(student.dec, torch.cat((z_rob, obs_s_flat), dim=-1))
                a_t = run_mlp(teacher.dec, torch.cat((z_t, teacher.obs_norm(obs_t).flatten(-2)), dim=-1))

                g_m = fn_noisy(motion, f_jit, goal_frames)

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

        mlog["wall_s"] = round(time.perf_counter() - tm0, 2)
        tok.eval()
        tokenizers[mname] = tok.cpu()
        p2_logs[mname] = mlog

    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    summary = {
        "architecture": "H002-SONIC v2b (Frozen 904-clip Universal TokenHead + Unclamped Proximal Multi-Dataset Causal Tokenizers)",
        "total_wall_s": round(time.monotonic() - t_total, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "train_clips": len(train_files),
        "train_frames_50hz": total_frames,
        "codec_fingerprint": codec_fp,
        "phase1_token_head": p1_log,
        "phase2_modalities": p2_logs,
    }

    if args.smoke:
        print("V2B_SMOKE_PASS:", json.dumps(summary, indent=2), flush=True)
        return

    out_bundle = {
        "args": c_args,
        "obs_dim": mdp.obs_dim,
        "num_joints": mdp.num_joints,
        "hist": hist,
        "z_dim": z_dim,
        "hidden": hidden,
        "codec_fingerprint": codec_fp,
        "token_head_pi": student.cpu().state_dict(),
        "robot_pose_hidden": hidden,
        "modal_tokenizers": {
            k: {
                "in_dim": MODALITY_DIMS[k],
                "hidden": dict(human_smpl=(1024, 512, 256), vr_3pt=(1024, 512, 256), planner_cmd=(512, 256, 256))[k],
                "state_dict": v.state_dict(),
            }
            for k, v in tokenizers.items()
        },
        "summary": summary,
    }
    args.out_bundle.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_bundle, args.out_bundle)
    args.out_bundle.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED_V2B_BUNDLE -> {args.out_bundle}", flush=True)


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest", type=Path, default=base / "dataset_250_manifest.json")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/q1_base_soma1k.pt")
    parser.add_argument("--scheme-a-ckpt", type=Path, default=ROOT / "checkpoints/q1_scheme_a_1k.pt")
    parser.add_argument("--out-bundle", type=Path, default=base / "checkpoints/h002_sonic_v2_bundle.pt")
    parser.add_argument("--p1-s1-steps", type=int, default=1000)
    parser.add_argument("--p2-s1-steps", type=int, default=2400)
    args = parser.parse_args()
    train_pipeline(args)


if __name__ == "__main__":
    main()
