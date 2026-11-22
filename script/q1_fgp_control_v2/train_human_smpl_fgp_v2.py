"""Train FGP-hardened drop-in `q1_tok_human_smpl.onnx` on the frozen H002-SONIC v2b TokenHead.

Trains on all 1,727 clips (486,571 50Hz frames) of `dataset_2000_manifest.json` with:
  1. Teleop-weighted sampling emphasizing leg lifts/kicks, squats, steps, turns, and standing.
  2. Clean 100% frozen V2b TokenHead (`codec_fingerprint = 33f64eaefe80c12b...`) so all deployed
     ONNX siblings (`planner_cmd` v2, `robot_pose`, `vr_3pt`, `recovery_v2`) remain 100% compatible.
  3. Vectorized FGP-realistic domain randomization on the 430D input (residual tilt bias,
     human height variation, root velocity slip/jitter, human resting thigh/arm offsets, 30->50Hz jitter)
     distilled against clean Q1 teacher/student latents and weighted joint actions.
  4. EMA model weights, opset-17 ONNX export, parity check (< 1e-5), and 1-thread CPU ORT latency benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

SMHP_ROOT = Path("/home/yu/projects/SuperMimicHP")
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (
    str(SMHP_ROOT),
    str(SMHP_ROOT / "src"),
    str(REPO_ROOT / "script/q1_h002_sonic_v2"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from multimodal_features import (
    BONE_PAIRS,
    MODALITY_DIMS,
    Q1_NOMINAL_PELVIS_Z,
    _causal_window_indices,
    build_human_smpl_goal,
    build_robot_pose_goal,
)
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import mat_to_6d, so3_exp, so3_log, yaw_mat
from unisupermimic.rl.algos.supermimic.networks import Policy, RunningNorm, mlp, run_mlp
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import GOAL_FRAME_DIM, MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, TRACKED_BODIES

MJCF_PATH = SMHP_ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
EXPECTED_CODEC_FP = "33f64eaefe80c12b35d7bd0b8bfbf641a3a8d463e9979ea448b7ba79788f13f8"


class ModalityTokenizer(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], z_dim: int = 32):
        super().__init__()
        self.in_norm = RunningNorm(in_dim)
        self.enc = mlp(in_dim, hidden, z_dim)

    def encode_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = run_mlp(self.enc, self.in_norm(x))
        return u, torch.tanh(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(run_mlp(self.enc, self.in_norm(x)))


class SharedTokenHeadONNX(nn.Module):
    def __init__(self, pi: Policy):
        super().__init__()
        self.obs_norm = pi.obs_norm
        self.dec = pi.dec

    def forward(self, obs_hist: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        obs = self.obs_norm(obs_hist).flatten(-2)
        return self.dec(torch.cat((token, obs), dim=-1))


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


def _sample_obs_histories(
    mdp: MDP,
    teacher: Policy,
    frames: torch.Tensor,
    hist: int,
    noise_scale: float = 0.18,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion = mdp.motion
    B = frames.shape[0]
    dev = frames.device
    clip_start = motion.clip_start[motion.clip_id[frames]]
    offsets = torch.arange(-(hist - 1), 1, device=dev, dtype=frames.dtype)
    h_frames = torch.clamp(frames.unsqueeze(-1) + offsets.unsqueeze(0), min=clip_start.unsqueeze(-1))
    flat_f = h_frames.reshape(-1)

    xi = torch.zeros((B, mdp.context_dim), dtype=torch.float32, device=dev)
    xi[:, 0] = 0.50 + 0.75 * torch.rand((B,), device=dev)
    flat_xi = xi.unsqueeze(1).expand(-1, hist, -1).reshape(-1, mdp.context_dim)

    p, R, v, w, q, qd = (t.clone() for t in mdp.ref_state(flat_f))
    n = flat_f.shape[0]
    p = p + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.025, 0.025)
    R = so3_exp(noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.05, 0.05)) @ R
    v = v + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.14, 0.14)
    w = w + noise_scale * torch.empty((n, 3), device=dev).uniform_(-0.18, 0.18)
    q = torch.clamp(
        q + noise_scale * torch.empty((n, mdp.num_joints), device=dev).uniform_(-0.05, 0.05),
        mdp.kin.q_lo,
        mdp.kin.q_hi,
    )
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


def _build_fgp_dr_430(
    motion: MotionLib,
    frames: torch.Tensor,
    num_frames: int = 10,
    *,
    p_dr: float = 0.65,
    tilt_pitch_deg: float = 9.0,
    tilt_roll_deg: float = 5.5,
    bone_noise_std: float = 0.020,
    vel_bias_std: float = 0.075,
) -> torch.Tensor:
    """Vectorized FGP-realistic domain randomization for 430D human_smpl features."""
    dt = float(getattr(motion, "ctrl_dt", 0.02))
    B = frames.shape[0]
    dev = frames.device
    clip_id = motion.clip_id[frames]
    clip_start = motion.clip_start[clip_id]
    clip_end = motion.clip_end[clip_id]

    is_dr = torch.rand((B,), device=dev) < p_dr
    dr_f = is_dr.float()

    # Causal 1-frame jitter on half of DR samples + base causal window
    jit = torch.where(is_dr & (torch.rand((B,), device=dev) < 0.35), -1, 0)
    f_now = torch.clamp(frames + jit, min=clip_start, max=clip_end)

    offsets = torch.arange(-(num_frames - 1), 1, device=dev, dtype=frames.dtype).unsqueeze(0).expand(B, -1).clone()
    # Simulate 30Hz->50Hz intermittent hold ticks in random slots for 20% of DR samples
    stutter_mask = is_dr & (torch.rand((B,), device=dev) < 0.20)
    if bool(stutter_mask.any()):
        # Shift random suffix slots by -1 (duplicate sample hold)
        hold_slot = torch.randint(1, num_frames, (B, 1), device=dev)
        slot_idx = torch.arange(num_frames, device=dev).unsqueeze(0)
        adj = torch.where(stutter_mask.unsqueeze(1) & (slot_idx >= hold_slot), -1, 0)
        offsets = torch.clamp(offsets + adj, max=0)

    hist_f = torch.clamp(f_now.unsqueeze(-1) + offsets, min=clip_start.unsqueeze(-1), max=clip_end.unsqueeze(-1))
    prev_f = torch.clamp(hist_f - 1, min=clip_start.unsqueeze(-1), max=clip_end.unsqueeze(-1))

    heading = yaw_mat(motion.root_R[f_now])
    heading_t = heading.transpose(-1, -2).unsqueeze(1)  # (B, 1, 3, 3)

    p_hist = motion.root_p[hist_f]
    p_prev = motion.root_p[prev_f]
    R_hist = motion.root_R[hist_f]
    R_prev = motion.root_R[prev_f]
    bp_hist = motion.body_p[hist_f]  # (B, K, 14, 3)

    v_world = (p_hist - p_prev) / dt
    w_world = so3_log(R_hist @ R_prev.transpose(-1, -2)) / dt
    R_hist_t = R_hist.transpose(-1, -2)

    h_scale = 1.0 + dr_f.view(B, 1, 1) * torch.empty((B, 1, 1), device=dev).uniform_(-0.04, 0.04)
    h_eff = Q1_NOMINAL_PELVIS_Z * h_scale

    # Heading-aligned root rotation with realistic residual calibration tilt DR
    d_roll = dr_f * torch.empty((B,), device=dev).uniform_(-math.radians(tilt_roll_deg), math.radians(tilt_roll_deg))
    d_pitch = dr_f * torch.empty((B,), device=dev).uniform_(-math.radians(tilt_pitch_deg), math.radians(tilt_pitch_deg))
    R_tilt_bias = so3_exp(torch.stack((d_roll, d_pitch, torch.zeros_like(d_roll)), dim=-1)).unsqueeze(1)  # (B, 1, 3, 3)
    R_rel_h = R_tilt_bias @ (heading_t @ R_hist)

    # Rotate upper-body bones (index 0=torso, 5..8=arms) on half of tilted samples (whole upper-body lean bias)
    upper_rot_mask = (is_dr & (torch.rand((B,), device=dev) < 0.5)).float().view(B, 1, 1, 1)
    R_upper_corr = torch.where(
        upper_rot_mask > 0.5,
        (heading.unsqueeze(1) @ R_tilt_bias @ heading_t),
        torch.eye(3, device=dev, dtype=torch.float32).view(1, 1, 3, 3).expand(B, 1, 3, 3),
    )

    p_delta = (heading_t @ (p_hist - motion.root_p[f_now].unsqueeze(1)).unsqueeze(-1)).squeeze(-1) / h_eff
    z_norm = p_hist[..., 2:3] / h_eff
    rot_6d = mat_to_6d(R_rel_h)
    v_local = (R_hist_t @ v_world.unsqueeze(-1)).squeeze(-1) / h_eff
    w_local = (R_hist_t @ w_world.unsqueeze(-1)).squeeze(-1) * 0.2

    # Inject mild residual velocity jitter & bias after Adapter V2 conditioning
    v_bias = torch.randn((B, 1, 3), device=dev) * torch.tensor([vel_bias_std, vel_bias_std, 0.015], device=dev)
    v_jit = torch.randn_like(v_local) * torch.tensor([0.025, 0.025, 0.015], device=dev)
    w_jit = torch.randn_like(w_local) * 0.02
    v_local = v_local + dr_f.view(B, 1, 1) * (v_bias + v_jit)
    w_local = w_local + dr_f.view(B, 1, 1) * w_jit

    bones = []
    # Constant per-sample thigh Y splay + human shoulder/torso variation + per-step unit noise
    thigh_splay = dr_f.view(B, 1) * torch.empty((B, 1), device=dev).uniform_(-0.03, 0.03)
    for bi, (a, b_idx) in enumerate(BONE_PAIRS):
        vec = bp_hist[:, :, b_idx, :] - bp_hist[:, :, a, :]
        if bi == 0:
            vec = (R_upper_corr @ vec.unsqueeze(-1)).squeeze(-1)
        vec_h = (heading_t @ vec.unsqueeze(-1)).squeeze(-1)
        unit_v = vec_h / vec_h.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        if bi == 1:  # L_thigh (+Y splay)
            unit_v = unit_v + torch.stack((torch.zeros_like(thigh_splay), thigh_splay, torch.zeros_like(thigh_splay)), dim=-1)
        elif bi == 3:  # R_thigh (-Y splay)
            unit_v = unit_v + torch.stack((torch.zeros_like(thigh_splay), -thigh_splay, torch.zeros_like(thigh_splay)), dim=-1)
        is_arm = bi in (5, 6, 7, 8)
        bias_scale = 0.006 if is_arm else (0.012 if bi in (0, 1, 3) else 0.008)
        step_scale = 0.008 if is_arm else bone_noise_std
        bias_b = dr_f.view(B, 1, 1) * torch.randn((B, 1, 3), device=dev) * bias_scale
        step_b = dr_f.view(B, 1, 1) * torch.randn_like(unit_v) * step_scale
        unit_v = F.normalize(unit_v + bias_b + step_b, dim=-1)
        bones.append(unit_v)
    bone_feat = torch.cat(bones, dim=-1)

    feat = torch.cat((p_delta, z_norm, rot_6d, v_local, w_local, bone_feat), dim=-1)
    return feat.reshape(B, -1)


def _calibrate_norm(norm: RunningNorm, motion: MotionLib, total_frames: int, device: torch.device) -> None:
    with torch.no_grad():
        dim = norm.mean.numel()
        sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
        sum_sq = torch.zeros(dim, dtype=torch.float64, device=device)
        for s in range(0, total_frames, 4096):
            e = min(s + 4096, total_frames)
            fc = torch.arange(s, e, dtype=torch.long, device=device)
            gx = build_human_smpl_goal(motion, fc, 10, noise_std=0.0).double()
            sum_x += gx.sum(dim=0)
            sum_sq += (gx * gx).sum(dim=0)
        mean_x = sum_x / max(total_frames, 1)
        var_x = (sum_sq / max(total_frames, 1) - mean_x * mean_x).clamp(min=1e-3)
        norm.mean.copy_(mean_x.float())
        norm.var.copy_(var_x.float())
        norm.count.fill_(float(total_frames))


def _compute_frame_weights(motion: MotionLib, train_files: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-frame sampling weights emphasizing leg-lifts/kicks, squats, steps, turns, and standing."""
    N = int(motion.num_frames)
    dev = motion.root_p.device
    la_idx = TRACKED_BODIES.index("left_ankle_roll_link")
    ra_idx = TRACKED_BODIES.index("right_ankle_roll_link")

    foot_dz = torch.abs(motion.body_p[:, la_idx, 2] - motion.body_p[:, ra_idx, 2])
    pelvis_z = motion.root_p[:, 2]
    is_lift = foot_dz > 0.045
    is_squat = pelvis_z < 0.395

    kw_pat = re.compile(r"kick|squat|step|turn|stand|dance|leg|hop", re.IGNORECASE)
    clip_bonus = torch.zeros((len(train_files),), dtype=torch.float32, device=dev)
    for ci, fp in enumerate(train_files):
        if kw_pat.search(Path(fp).stem):
            clip_bonus[ci] = 1.0
    per_frame_clip = clip_bonus[motion.clip_id]

    weights = torch.ones((N,), dtype=torch.float32, device=dev) + 0.5 * per_frame_clip + 1.5 * is_lift.float() + 1.0 * is_squat.float()
    return weights, (is_lift | is_squat)


def _export_onnx(mod: nn.Module, in_dim: int, path: Path, meta: dict[str, str]) -> None:
    mod.eval()
    dyn = {"modal_input": {0: "batch"}, "token": {0: "batch"}}
    torch.onnx.export(
        mod.cpu(),
        (torch.zeros((1, in_dim)),),
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["modal_input"],
        output_names=["token"],
        dynamic_axes=dyn,
        dynamo=False,
    )
    m = onnx.load(str(path))
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key = k
        p.value = str(v)
    onnx.save(m, str(path))
    onnx.checker.check_model(onnx.load(str(path)))


def train(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    networks.AMP_DTYPE = None

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    v2_manifest = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
    v2_train = sorted(str(SMHP_ROOT / r["csv_rel"]) for r in v2_manifest["clips"] if r["split"] == "train")
    v2_held_out = {str(SMHP_ROOT / r["csv_rel"]) for r in v2_manifest["clips"] if r["split"] == "held_out"}
    v3_train = sorted(str(SMHP_ROOT / r["csv_rel"]) for r in manifest["clips"] if r["split"] == "train")
    assert not (set(v2_train) & v2_held_out), "Held-out leakage detected in v2_train"
    assert not (set(v3_train) & v2_held_out), "Held-out leakage detected in v3_train"

    v2_set = set(v2_train)
    v3_unique = [f for f in v3_train if f not in v2_set]
    if args.smoke:
        v2_use = v2_train[:16]
        v3_use = v3_unique[:16]
    else:
        v2_use = v2_train
        v3_use = v3_unique
    train_files = v2_use + v3_use
    n_v2_clips = len(v2_use)
    n_v3_clips = len(v3_use)
    print(
        f"Loaded {len(train_files)} TRAIN clips (v2={n_v2_clips}, v3={n_v3_clips}, held_out_overlap=0, device={device})",
        flush=True,
    )

    base_ck = torch.load(args.base_ckpt, map_location="cpu", weights_only=True)
    v2_bundle = torch.load(args.v2_bundle, map_location="cpu", weights_only=True)
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
    n_v2_frames = int(motion.clip_end[n_v2_clips - 1].item()) + 1
    assert 0 < n_v2_frames < total_frames, f"Invalid n_v2_frames={n_v2_frames} vs total={total_frames}"
    print(
        f"MotionLib ready: {total_frames:,} 50Hz frames (v2={n_v2_frames:,}, v3={total_frames - n_v2_frames:,})",
        flush=True,
    )

    teacher = Policy(
        mdp.obs_dim, mdp.goal_frame_dim * goal_frames, num_joints=mdp.num_joints, hist=hist, z_dim=z_dim, hidden=hidden
    ).to(device)
    teacher.load_state_dict(base_ck["pi"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = Policy(
        mdp.obs_dim, mdp.goal_frame_dim * goal_frames, num_joints=mdp.num_joints, hist=hist, z_dim=z_dim, hidden=hidden
    ).to(device)
    student.load_state_dict(v2_bundle["token_head_pi"])
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    codec_fp = compute_codec_fingerprint(student)
    assert codec_fp == EXPECTED_CODEC_FP, f"TokenHead fingerprint mismatch: {codec_fp} != {EXPECTED_CODEC_FP}"
    print(f"Verified frozen TokenHead codec_fingerprint={codec_fp}", flush=True)

    frame_weights, is_lift_or_squat = _compute_frame_weights(motion, train_files)
    v3_weights = frame_weights[n_v2_frames:]
    cdf_v3 = torch.cumsum(v3_weights.double(), dim=0)
    cdf_v3 = cdf_v3 / cdf_v3[-1]

    # Warm-start from V2 human_smpl tokenizer and preserve V2's exact in_norm
    tok = ModalityTokenizer(MODALITY_DIMS["human_smpl"], (1024, 512, 256), z_dim=z_dim).to(device)
    tok.load_state_dict(v2_bundle["modal_tokenizers"]["human_smpl"]["state_dict"])
    for p in tok.in_norm.parameters():
        p.requires_grad_(False)
    for p in tok.enc.parameters():
        p.requires_grad_(True)
    v2_enc_ref = copy.deepcopy(tok.enc).eval()
    for p in v2_enc_ref.parameters():
        p.requires_grad_(False)

    ema_tok = copy.deepcopy(tok).eval()
    for p in ema_tok.parameters():
        p.requires_grad_(False)

    steps = 30 if args.smoke else args.steps
    batch_size = 256 if args.smoke else args.batch_size
    opt = torch.optim.AdamW(tok.enc.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=1e-5)

    t_train = time.perf_counter()
    l_prox = torch.zeros((), device=device)
    for step in range(1, steps + 1):
        b_v2 = int(round(batch_size * args.v2_mix))
        b_v3 = batch_size - b_v2
        fb_v2 = torch.randint(0, n_v2_frames, (b_v2,), device=device)
        u = torch.rand((b_v3,), device=device, dtype=torch.float64)
        fb_v3 = n_v2_frames + torch.clamp(torch.searchsorted(cdf_v3, u), max=total_frames - n_v2_frames - 1)
        fb = torch.cat((fb_v2, fb_v3), dim=0)
        with torch.no_grad():
            g_rob = build_robot_pose_goal(motion, fb, goal_frames, latest_offset=0)
            u_rob = run_mlp(student.enc, student.goal_norm(g_rob))
            z_rob = torch.tanh(u_rob)

            g_f = mdp.goal(fb, goal_frames)
            z_t = torch.tanh(run_mlp(teacher.enc, teacher.goal_norm(g_f)))

            obs_s, obs_t = _sample_obs_histories(mdp, teacher, fb, hist, noise_scale=0.18)
            obs_s_flat = student.obs_norm(obs_s).flatten(-2)
            a_rob = run_mlp(student.dec, torch.cat((z_rob, obs_s_flat), dim=-1))
            a_t = run_mlp(teacher.dec, torch.cat((z_t, teacher.obs_norm(obs_t).flatten(-2)), dim=-1))

            g_m = _build_fgp_dr_430(
                motion,
                fb,
                goal_frames,
                p_dr=args.p_dr,
                tilt_pitch_deg=args.dr_pitch_deg,
                tilt_roll_deg=args.dr_roll_deg,
                bone_noise_std=args.dr_bone_std,
                vel_bias_std=args.dr_vel_std,
            )

        u_m, z_m = tok.encode_raw(g_m)
        a_m = run_mlp(student.dec, torch.cat((z_m, obs_s_flat), dim=-1))
        l_prox = sum((p - p0).pow(2).sum() for p, p0 in zip(tok.enc.parameters(), v2_enc_ref.parameters()))
        loss = (
            0.65 * F.mse_loss(z_m, z_rob)
            + 0.30 * F.mse_loss(z_m, z_t)
            + 0.05 * F.mse_loss(u_m, u_rob)
            + 1.25 * F.mse_loss(a_m, a_rob)
            + 0.40 * F.mse_loss(a_m, a_t)
            + args.prox_coeff * l_prox
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(tok.enc.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            for pe, p in zip(ema_tok.enc.parameters(), tok.enc.parameters()):
                pe.mul_(0.995).add_(p, alpha=0.005)

        if step == 1 or step % max(steps // 8, 1) == 0 or step == steps:
            cos_r = float(F.cosine_similarity(z_m.detach(), z_rob, dim=-1).mean().item())
            mse_r = float(F.mse_loss(a_m.detach(), a_rob).item())
            mse_t = float(F.mse_loss(a_m.detach(), a_t).item())
            print(
                f"[{step:4d}/{steps}] loss={float(loss.item()):.4f} cos_rob={cos_r:.4f} a_mse_rob={mse_r:.4f} a_mse_t={mse_t:.4f} prox={float(l_prox.detach().item()):.4f}",
                flush=True,
            )

    train_wall = round(time.perf_counter() - t_train, 2)
    out_dir = args.out_dir.resolve()
    onnx_dir = out_dir / "onnx"
    ckpt_dir = out_dir / "checkpoints"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        "codec_fingerprint": codec_fp,
        "ori_mode": "heading",
        "ctrl_hz": "50",
        "lookahead_ms": "0",
        "modality": "human_smpl",
        "fgp_control_v2": "1",
    }
    tok_onnx_path = onnx_dir / "q1_tok_human_smpl.onnx"
    ema_tok_cpu = ema_tok.cpu().eval()
    _export_onnx(ema_tok_cpu, MODALITY_DIMS["human_smpl"], tok_onnx_path, meta_common)

    # Save checkpoint bundle for auditability
    ckpt_path = ckpt_dir / "human_smpl_fgp_v2_tok.pt"
    torch.save(
        {
            "codec_fingerprint": codec_fp,
            "in_dim": MODALITY_DIMS["human_smpl"],
            "hidden": (1024, 512, 256),
            "z_dim": z_dim,
            "state_dict": ema_tok_cpu.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        ckpt_path,
    )

    # Verify parity & 1-thread CPU ORT latency paired with deployed V2b TokenHead
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    head_sess = ort.InferenceSession(str(args.deployed_head_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    tok_sess = ort.InferenceSession(str(tok_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    head_mod = SharedTokenHeadONNX(student.cpu().eval()).eval()

    rng = np.random.default_rng(20261122)
    test_obs = rng.normal(0.0, 0.4, size=(64, hist, mdp.obs_dim)).astype(np.float32)
    test_in = rng.normal(0.0, 0.3, size=(64, MODALITY_DIMS["human_smpl"])).astype(np.float32)
    with torch.no_grad():
        pt_z = ema_tok_cpu(torch.from_numpy(test_in))
        pt_a = head_mod(torch.from_numpy(test_obs), pt_z).numpy()
    ort_z = tok_sess.run(["token"], {"modal_input": test_in})[0]
    ort_a = head_sess.run(["action"], {"obs_hist": test_obs, "token": ort_z})[0]
    max_diff = float(np.max(np.abs(pt_a - ort_a)))
    assert max_diff < 1e-5, f"ONNX parity diff too high: {max_diff}"

    obs_1 = test_obs[:1]
    in_1 = test_in[:1]
    ms_list = []
    for i in range(1000):
        t_s = time.perf_counter()
        z1 = tok_sess.run(["token"], {"modal_input": in_1})[0]
        _ = head_sess.run(["action"], {"obs_hist": obs_1, "token": z1})[0]
        if i >= 200:
            ms_list.append((time.perf_counter() - t_s) * 1000.0)

    onnx_sha = hashlib.sha256(tok_onnx_path.read_bytes()).hexdigest()
    summary = {
        "warmstart_from_v2": True,
        "v2_train_clips": n_v2_clips,
        "v3_train_clips": n_v3_clips,
        "prox_coeff": args.prox_coeff,
        "final_l_prox": round(float(l_prox.detach().item()), 5),
        "train_clips": len(train_files),
        "train_frames_50hz": total_frames,
        "steps": steps,
        "batch_size": batch_size,
        "train_wall_s": train_wall,
        "total_wall_s": round(time.monotonic() - t0, 2),
        "codec_fingerprint": codec_fp,
        "onnx_path": str(tok_onnx_path),
        "onnx_bytes": tok_onnx_path.stat().st_size,
        "onnx_sha256": onnx_sha,
        "parity_action_max_abs_diff": max_diff,
        "cpu_1thread_tok_plus_head_p50_ms": round(float(np.percentile(ms_list, 50)), 4),
        "cpu_1thread_tok_plus_head_p99_ms": round(float(np.percentile(ms_list, 99)), 4),
        "lift_or_squat_frac": round(float(is_lift_or_squat.float().mean().item()), 4),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TRAIN_DONE:", json.dumps(summary, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=SMHP_ROOT / "runs/h002_sonic_v3/dataset_2000_manifest.json")
    ap.add_argument("--v2-manifest", type=Path, default=SMHP_ROOT / "runs/h002_sonic_v2/dataset_250_manifest.json")
    ap.add_argument("--base-ckpt", type=Path, default=SMHP_ROOT / "checkpoints/q1_base_soma1k.pt")
    ap.add_argument("--v2-bundle", type=Path, default=SMHP_ROOT / "runs/h002_sonic_v2/checkpoints/h002_sonic_v2_bundle.pt")
    ap.add_argument(
        "--deployed-head-onnx",
        type=Path,
        default=Path("/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/onnx/q1_h002_token_head.onnx"),
    )
    ap.add_argument("--out-dir", type=Path, default=SMHP_ROOT / "runs/q1_fgp_human_smpl_v2_20261122")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--prox-coeff", type=float, default=0.02)
    ap.add_argument("--v2-mix", type=float, default=0.50)
    ap.add_argument("--p-dr", type=float, default=0.35)
    ap.add_argument("--dr-pitch-deg", type=float, default=3.0)
    ap.add_argument("--dr-roll-deg", type=float, default=2.0)
    ap.add_argument("--dr-bone-std", type=float, default=0.010)
    ap.add_argument("--dr-vel-std", type=float, default=0.030)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
