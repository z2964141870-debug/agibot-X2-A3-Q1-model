"""16-Clip Multi-Dataset Benchmark (LAFAN1 + PHUMA + BONES-SEED/SOMA), Stress Test & Plotter for H002."""

from __future__ import annotations

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
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
from train_h002_ablation import HELD_OUT_STEMS
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import so3_exp, unflatten_state, yaw_mat
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale
from unisupermimic.teleop.realtime_teleop import compute_teleop_goal


def discover_all_16_clips() -> list[dict[str, Any]]:
    specs = []
    for ds, folder in (
        ("lafan1", "lafan1_showcase"),
        ("phuma", "phuma_showcase"),
        ("soma", "soma_showcase"),
    ):
        for p in sorted(glob.glob(str(ROOT / f"src/unisupermimic/assets/motions/q1/{folder}/*.csv"))):
            stem = Path(p).stem
            cat = "stationary" if (stem.startswith("stat_") or stem.startswith("soma_stat_")) else "dynamic"
            specs.append(
                {
                    "path": str(Path(p).resolve()),
                    "name": stem,
                    "dataset": ds,
                    "category": cat,
                    "held_out": stem in HELD_OUT_STEMS,
                }
            )
    return specs


def create_eval_env(motion_files: list[str]) -> Any:
    torch.set_num_threads(1)
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config", overrides=["task=q1_super_mimic/mujoco"])
    OmegaConf.set_struct(cfg, False)
    cfg.training.sim_backend = "mujoco"
    cfg.training.play_only = True
    cfg.env.ctrl_dt = 0.02
    cfg.env.commands.motion.ctrl_dt = 0.02
    cfg.env.commands.motion.motion_files = motion_files
    cfg.env.commands.motion.observation_mode = "root_xy"
    override = build_env_override(cfg, eval_mode=True, learner_device="cpu")
    return create_env(cfg, num_envs=1, env_cfg_override=override)


def _find_mj_model(env: Any) -> Any:
    for owner in (env.backend, getattr(env.backend, "_sim", None), getattr(env.backend, "sim", None)):
        if owner is None:
            continue
        for attr in ("model", "_model", "mj_model", "_mj_model"):
            val = getattr(owner, attr, None)
            if val is not None and hasattr(val, "geom_friction"):
                return val
    raise RuntimeError(f"Could not locate MuJoCo MjModel in backend {type(env.backend)}")


def load_pt_policy(mdp: Any, ckpt_path: Path) -> tuple[Any, int, int]:
    networks.AMP_DTYPE = None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ck["args"]
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])
    obs_dim = int(ck["pi"]["obs_norm.mean"].numel())
    pi = networks.Policy(
        obs_dim,
        mdp.goal_frame_dim * int(args["goal_frames"]),
        num_joints=mdp.num_joints,
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        hidden=hidden,
    )
    pi.load_state_dict(ck["pi"])
    pi.eval()
    return pi, int(args["goal_frames"]), int(args["hist"])


def load_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def rollout_clip(
    env: Any,
    actor: Any,
    *,
    is_onnx: bool,
    clip_idx: int,
    est_mode: str,
    goal_frames: int = 10,
    hist: int = 10,
    deploy_safety: bool = False,
    stress: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    mdp = env.supermimic_mdp
    motion = mdp.motion
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo = mdp.kin.q_lo.cpu()
    q_hi = mdp.kin.q_hi.cpu()
    ctrl_dt = 0.02
    budget = int(motion.clip_len[clip_idx]) - 1
    rng = np.random.default_rng(seed + clip_idx)
    yaw_bias_R = so3_exp(torch.tensor([[0.0, 0.0, float(np.radians(1.5))]], dtype=torch.float32))[0]

    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32).view(1, 1, 3, 1)

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(motion.clip_start[clip_idx : clip_idx + 1].clone())
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])
        f0 = cur["f"]
        est = ContactGatedFootAnchor(mdp.kin, 1, dt=ctrl_dt, device="cpu")
        est.reset(s0[1], s0[3], s0[4], s0[5], motion.root_p[f0, :2], R_ref=motion.root_R[f0])
        bp0, _, _, _ = mdp.kin.forward(*s0)
        foot0_xy = bp0[0, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].numpy().copy()

        history = None
        q_init = s0[4][0].clone()
        prev_q_tgt = None

        sum_ret = 0.0
        sum_body_sq = 0.0
        root_xy_sq = []
        wrist_world_sq = []
        foot_drift_max_m = 0.0
        stance_slip_speeds = []
        pd_oob_total = 0
        pd_count_total = 0
        pd_max_oob_rad = 0.0
        step0_jump_default_deg = 0.0
        step0_jump_q0_deg = 0.0
        action_jumps_deg = []
        step_ms = []
        failed = False
        steps_run = 0

        for step_i in range(budget):
            state = unflatten_state(cur["s"])
            f_cur = cur["f"]
            clip_s = motion.clip_start[motion.clip_id[f_cur]]

            # Packet hold jitter under stress (25% 1-frame delay)
            loff = -1 if (stress and step_i > 0 and rng.random() < 0.25) else 0
            f_ref = torch.maximum(f_cur + loff, clip_s)

            R_obs = state[1]
            w_obs = state[3]
            if stress:
                R_obs = (yaw_bias_R @ R_obs[0]).unsqueeze(0)
                w_noise = torch.from_numpy(rng.normal(0.0, 0.03, size=(1, 3)).astype(np.float32))
                w_obs = w_obs + w_noise

            t0 = time.perf_counter()
            state_imu = (state[0], R_obs, state[2], w_obs, state[4], state[5])
            obs = mdp.obs(state_imu, cur["a_prev"], cur["xi"], f_ref)
            est_xy, _ = est.step(
                R_obs,
                w_obs,
                state[4],
                state[5],
                motion.root_p[f_ref, :2],
                R_ref=motion.root_R[f_ref],
                mode=est_mode,  # type: ignore[arg-type]
                true_obs_root_xy=obs[:, 78:80],
            )
            obs_pol = torch.cat((obs[:, :78], est_xy), dim=-1)
            history = (
                obs_pol.unsqueeze(1).repeat(1, hist, 1)
                if history is None
                else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), dim=1)
            )
            goal = compute_teleop_goal(mdp, f_cur, goal_frames, "causal_history", latest_offset=loff)
            if is_onnx:
                out_np = actor.run(["action"], {"obs_hist": history.numpy(), "goal": goal.numpy()})[0]
                act = torch.from_numpy(out_np)
            else:
                act = actor(history, goal)

            if deploy_safety:
                q_tgt_raw = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0]
                if step_i < 5:
                    blend = 0.5 * (1.0 - np.cos(np.pi * float(step_i + 1) / 5.0))
                    q_tgt_raw = q_init + float(blend) * (q_tgt_raw - q_init)
                q_tgt_safe = torch.clamp(q_tgt_raw, q_lo, q_hi)
                act = ((q_tgt_safe - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0)

            step_ms.append((time.perf_counter() - t0) * 1000.0)

            q_tgt_sent = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0]
            oob = torch.maximum(q_tgt_sent - q_hi, q_lo - q_tgt_sent)
            oob_pos = oob[oob > 1e-5]
            pd_oob_total += int(oob_pos.numel())
            pd_count_total += int(q_tgt_sent.numel())
            if oob_pos.numel() > 0:
                pd_max_oob_rad = max(pd_max_oob_rad, float(oob_pos.max().item()))

            if step_i == 0:
                step0_jump_default_deg = float(np.degrees(torch.max(torch.abs(q_tgt_sent - mdp.default_q)).item()))
                step0_jump_q0_deg = float(np.degrees(torch.max(torch.abs(q_tgt_sent - q_init)).item()))
            elif prev_q_tgt is not None:
                action_jumps_deg.append(float(np.degrees(torch.max(torch.abs(q_tgt_sent - prev_q_tgt)).item())))
            prev_q_tgt = q_tgt_sent.clone()

            step_state = env.step(act)
            tr = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            f_next = motion.future_frames(tr["f"], 1).squeeze(-1)
            s1 = unflatten_state(tr["s1"])
            _, terms = mdp.reward(s1, f_next, tr["a"], tr["a_prev"], terms=True)

            sum_ret += float(tr["r"][0].item())
            sum_body_sq += float(terms["e_body_pos"][0].item())
            steps_run += 1

            # World-frame XY root error, world-frame wrist error, and sole stance slip
            root_xy_err_sq = float(((s1[0][0, :2] - motion.root_p[f_next[0], :2]) ** 2).sum().item())
            root_xy_sq.append(root_xy_err_sq)

            bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
            w_rob = bp1[0, (LEFT_WRIST_IDX, RIGHT_WRIST_IDX), :]
            w_ref = motion.body_p[f_next[0], (LEFT_WRIST_IDX, RIGHT_WRIST_IDX), :]
            wrist_world_sq.append(float(((w_rob - w_ref) ** 2).sum(dim=-1).mean().item()))

            f_xy_now = bp1[0, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].numpy()
            # Track stance foot drift relative to t=0 initial anchor
            foot_drift_max_m = max(foot_drift_max_m, float(np.linalg.norm(f_xy_now - foot0_xy, axis=-1).min()))

            # Sole linear speed in XY for feet anchored in reference clip
            ref_az = motion.body_p[f_next[0], (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]
            ref_av = motion.body_v[f_next[0], (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].norm(dim=-1)
            stance_mask = (ref_az <= 0.062) & (ref_av <= 0.15)
            if bool(stance_mask.any()):
                fR = bR1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :, :]
                fv = bv1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
                fw = bw1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :]
                v_sole = fv + torch.cross(fw, (fR @ sole_off).squeeze(-1), dim=-1)
                speeds = v_sole[0, :, :2].norm(dim=-1)[stance_mask].tolist()
                stance_slip_speeds.extend(speeds)

            if bool(tr["term"][0].item()):
                failed = True
                break
            if bool(tr["done"][0].item()):
                break
            cur = step_state.info[SUPERMIMIC_STATE_KEY]

    max_r = float(mdp.max_step_reward)
    return {
        "completed": (not failed) and (steps_run == budget),
        "failed": failed,
        "steps_run": steps_run,
        "budget": budget,
        "survival_fraction": steps_run / max(budget, 1),
        "norm_return": sum_ret / max(budget * max_r, 1e-8),
        "body_pos_rmse_cm": float(np.sqrt(sum_body_sq / max(steps_run, 1)) * 100.0),
        "root_xy_rmse_cm": float(np.sqrt(np.mean(root_xy_sq)) * 100.0),
        "wrist_world_rmse_cm": float(np.sqrt(np.mean(wrist_world_sq)) * 100.0),
        "stance_foot_drift_cm": float(foot_drift_max_m * 100.0),
        "stance_slip_mm_s": float(np.mean(stance_slip_speeds) * 1000.0) if stance_slip_speeds else 0.0,
        "pd_oob_pct": 100.0 * pd_oob_total / max(pd_count_total, 1),
        "pd_max_oob_deg": float(np.degrees(pd_max_oob_rad)),
        "step0_jump_default_deg": step0_jump_default_deg,
        "step0_jump_q0_deg": step0_jump_q0_deg,
        "action_jitter_p99_deg": float(np.percentile(action_jumps_deg, 99)) if action_jumps_deg else 0.0,
        "cpu_step_ms_p50": float(np.percentile(step_ms, 50)) if step_ms else 0.0,
        "cpu_step_ms_p99": float(np.percentile(step_ms, 99)) if step_ms else 0.0,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "completed",
        "survival_fraction",
        "norm_return",
        "body_pos_rmse_cm",
        "root_xy_rmse_cm",
        "wrist_world_rmse_cm",
        "stance_foot_drift_cm",
        "stance_slip_mm_s",
        "pd_oob_pct",
        "step0_jump_q0_deg",
        "action_jitter_p99_deg",
        "cpu_step_ms_p50",
        "cpu_step_ms_p99",
    ]
    out: dict[str, Any] = {"n_clips": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k if k != "completed" else "survival_rate"] = float(np.mean(vals))
    out["pd_max_oob_deg"] = float(max(r["pd_max_oob_deg"] for r in rows))
    out["step0_jump_default_deg"] = float(max(r["step0_jump_default_deg"] for r in rows))
    return out


def make_summary_plot(results: dict[str, Any], out_png: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=160)
    conds = [
        ("C0_unadapted_zeroed_0ms", "C0 Zeroed\n(Unadapted)", "#8d99ae"),
        ("C1_m1_pure_legacy_0ms", "C1 Pure\nLegacy", "#e76f51"),
        ("C2_m2_unfiltered_odom_0ms", "C2 Unfilt.\nOdom", "#f4a261"),
        ("C3_m2_cgfa_only_0ms", "C3 CGFA\nOnly", "#2a9d8f"),
        ("C4_m3_h002_full_0ms", "C4 H002\nFull", "#264653"),
        ("C5_m3_h002_onnx_deploy", "C5 ONNX\nDeploy", "#1d3557"),
        ("Oracle_scheme_a_0ms", "Oracle\nCeiling", "#6c757d"),
    ]
    keys = [c[0] for c in conds]
    labels = [c[1] for c in conds]
    colors = [c[2] for c in conds]
    x = np.arange(len(conds))

    # Panel A: Survival Rate (%) across All 16, Held-Out 4, Stress 8
    ax = axes[0, 0]
    w = 0.25
    all_surv = [results["conditions"][k]["all_16"]["survival_rate"] * 100.0 for k in keys]
    ho_surv = [results["conditions"][k]["held_out_4"]["survival_rate"] * 100.0 for k in keys]
    st_surv = [
        results["stress_conditions"][k]["all_8"]["survival_rate"] * 100.0
        if k in results["stress_conditions"] else np.nan
        for k in keys
    ]
    ax.bar(x - w, all_surv, w, label="All 16 Nominal", color="#264653")
    ax.bar(x, ho_surv, w, label="Held-Out 4 Nominal", color="#2a9d8f")
    ax.bar(x + w, np.nan_to_num(st_surv, nan=0.0), w, label="Stress 8 (0.55x Fric+Jitter+IMU)", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 126); ax.set_ylabel("Survival Rate (%)")
    ax.set_title("A. Multi-Dataset Survival: Nominal (16 & Held-Out 4) vs. Stress (8)", fontweight="bold")
    ax.legend(fontsize=7.2, loc="upper left", ncol=2, framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: Spatial Drift Control (Stationary Dual-Wrist World RMSE vs Dynamic Root XY RMSE)
    ax = axes[0, 1]
    stat_wrist = [results["conditions"][k]["stationary_6"]["wrist_world_rmse_cm"] for k in keys]
    dyn_root = [results["conditions"][k]["dynamic_10"]["root_xy_rmse_cm"] for k in keys]
    w2 = 0.34
    ylim_b = max(max(stat_wrist), max(dyn_root)) * 1.18 + 2.0
    b1 = ax.bar(x - w2/2, stat_wrist, w2, label="Stationary (6): Wrist World RMSE (cm)", color="#2a9d8f")
    b2 = ax.bar(x + w2/2, dyn_root, w2, label="Dynamic (10): Root XY RMSE (cm)", color="#f4a261")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(h + 0.6, ylim_b * 0.93), f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, ylim_b)
    ax.set_ylabel("World Position Error (cm)")
    ax.set_title("B. Anchor Fidelity: Stationary Wrist vs. Dynamic Root XY RMSE", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: Contact Stability & PD Safety (Stance Slip mm/s + PD Limit OOB %)
    ax = axes[1, 0]
    slip = [results["conditions"][k]["all_16"]["stance_slip_mm_s"] for k in keys]
    pd_oob = [results["conditions"][k]["all_16"]["pd_oob_pct"] for k in keys]
    ax2 = ax.twinx()
    b_s = ax.bar(x - 0.18, slip, 0.36, label="Stance Sole Slip (mm/s)", color="#1d3557", alpha=0.88)
    b_p = ax2.bar(x + 0.18, pd_oob, 0.36, label="PD Target Limit OOB (%)", color="#e63946", alpha=0.85)
    for bar in b_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.0f}", ha="center", va="bottom", fontsize=6.5)
    for bar in b_p:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}%", ha="center", va="bottom", fontsize=6.5, color="#b00020")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Mean Stance Slip Speed (mm/s)", color="#1d3557")
    ax2.set_ylabel("PD Target Exceedance (%)", color="#e63946")
    ax.set_ylim(0, max(slip) * 1.28 + 10)
    ax2.set_ylim(0, max(max(pd_oob) * 1.38, 0.25))
    ax.set_title("C. Sim2Real Readiness: Stance Slip & MJCF PD Limit Exceedance", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: Real-Time CPU Latency & Per-Dataset Body RMSE (M3/C5 vs Baselines)
    ax = axes[1, 1]
    ds_names = ["lafan1", "phuma", "soma"]
    ds_x = np.arange(len(ds_names))
    w3 = 0.22
    for idx_k, (ckey, clab, ccol) in enumerate([
        ("C0_unadapted_zeroed_0ms", "C0 Zeroed", "#8d99ae"),
        ("C1_m1_pure_legacy_0ms", "C1 Legacy", "#e76f51"),
        ("C4_m3_h002_full_0ms", "C4 H002 PyTorch", "#264653"),
        ("C5_m3_h002_onnx_deploy", "C5 ONNX Deploy", "#2a9d8f"),
    ]):
        vals = [results["conditions"][ckey][f"ds_{ds}"]["body_pos_rmse_cm"] for ds in ds_names]
        ax.bar(ds_x + (idx_k - 1.5) * w3, vals, w3, label=clab, color=ccol)
    ax.set_xticks(ds_x)
    ax.set_xticklabels(["LAFAN1 (4 Walk)", "PHUMA (3 Stat+3 Dyn)", "BONES-SEED/SOMA (3 Stat+3 Dyn)"], fontsize=8)
    ax.set_ylabel("Body Position RMSE (cm)")
    onnx_p50 = results["conditions"]["C5_m3_h002_onnx_deploy"]["all_16"]["cpu_step_ms_p50"]
    onnx_p99 = results["conditions"]["C5_m3_h002_onnx_deploy"]["all_16"]["cpu_step_ms_p99"]
    ax.set_title(f"D. Per-Dataset RMSE | C5 CPU Loop: {onnx_p50:.2f} / {onnx_p99:.2f} ms (p50/p99)", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("H002: Contact-Anchored Causal Whole-Body Teleoperation (0 ms Lookahead, Q1 50 Hz, 16 Clips)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--out-json", type=Path, default=base / "h002_metrics.json")
    parser.add_argument("--out-png", type=Path, default=base / "h002_ablation_summary.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
        make_summary_plot(payload, args.out_png)
        print(f"PLOT_ONLY_DONE -> {args.out_png} ({args.out_png.stat().st_size} B)", flush=True)
        return

    clips = discover_all_16_clips()
    assert len(clips) == 16, f"Expected 16 clips, found {len(clips)}"
    motion_files = [c["path"] for c in clips]
    env = create_eval_env(motion_files)
    mdp = env.supermimic_mdp

    base_dir = ROOT / "runs/h002_cgfa_pilot"
    pi_sa, gf_sa, h_sa = load_pt_policy(mdp, ROOT / "checkpoints/q1_scheme_a_1k.pt")
    pi_m1, gf_m1, h_m1 = load_pt_policy(mdp, base_dir / "checkpoints/m1_pure_legacy_0ms.pt")
    pi_m2, gf_m2, h_m2 = load_pt_policy(mdp, base_dir / "checkpoints/m2_cgfa_only_0ms.pt")
    pi_m3, gf_m3, h_m3 = load_pt_policy(mdp, base_dir / "checkpoints/m3_h002_full_0ms.pt")
    onnx_m3 = load_onnx_session(base_dir / "q1_h002_0ms.onnx")

    eval_specs = [
        ("C0_unadapted_zeroed_0ms", pi_sa, False, "zeroed", gf_sa, h_sa, False),
        ("C1_m1_pure_legacy_0ms", pi_m1, False, "zeroed", gf_m1, h_m1, False),
        ("C2_m2_unfiltered_odom_0ms", pi_m2, False, "unfiltered_odom", gf_m2, h_m2, False),
        ("C3_m2_cgfa_only_0ms", pi_m2, False, "cgfa", gf_m2, h_m2, False),
        ("C4_m3_h002_full_0ms", pi_m3, False, "cgfa", gf_m3, h_m3, False),
        ("C5_m3_h002_onnx_deploy", onnx_m3, True, "cgfa", gf_m3, h_m3, True),
        ("Oracle_scheme_a_0ms", pi_sa, False, "oracle", gf_sa, h_sa, False),
    ]

    nom_results: dict[str, Any] = {}
    for cname, actor, is_onnx, emode, gf, h, dep_safe in eval_specs:
        per_clip = {}
        rows_all, rows_ho, rows_stat, rows_dyn = [], [], [], []
        by_ds: dict[str, list[dict[str, Any]]] = {"lafan1": [], "phuma": [], "soma": []}
        for ci, cmeta in enumerate(clips):
            r = rollout_clip(
                env, actor, is_onnx=is_onnx, clip_idx=ci, est_mode=emode,
                goal_frames=gf, hist=h, deploy_safety=dep_safe, stress=False,
            )
            r_meta = {**cmeta, **r}
            per_clip[cmeta["name"]] = r_meta
            rows_all.append(r)
            if cmeta["held_out"]:
                rows_ho.append(r)
            if cmeta["category"] == "stationary":
                rows_stat.append(r)
            else:
                rows_dyn.append(r)
            by_ds[cmeta["dataset"]].append(r)
        nom_results[cname] = {
            "all_16": aggregate_rows(rows_all),
            "held_out_4": aggregate_rows(rows_ho),
            "stationary_6": aggregate_rows(rows_stat),
            "dynamic_10": aggregate_rows(rows_dyn),
            "ds_lafan1": aggregate_rows(by_ds["lafan1"]),
            "ds_phuma": aggregate_rows(by_ds["phuma"]),
            "ds_soma": aggregate_rows(by_ds["soma"]),
            "per_clip": per_clip,
        }
        a16 = nom_results[cname]["all_16"]
        ho4 = nom_results[cname]["held_out_4"]
        print(
            f"[NOMINAL] {cname:26s} | All16 surv={a16['survival_rate']*100:5.1f}% ret={a16['norm_return']:.3f} "
            f"body={a16['body_pos_rmse_cm']:.2f}cm rootXY={a16['root_xy_rmse_cm']:.1f}cm slip={a16['stance_slip_mm_s']:.1f}mm/s "
            f"pdOOB={a16['pd_oob_pct']:.3f}% | HO4 surv={ho4['survival_rate']*100:5.1f}%",
            flush=True,
        )

    # Stress Test (0.55x floor friction + 25% 20ms packet hold + 0.03rad/s IMU noise + 1.5deg yaw bias) on 8 clips (4 Held-Out + LAFAN1 + dyn_hml_1)
    stress_indices = [
        ci for ci, c in enumerate(clips)
        if c["held_out"] or c["dataset"] == "lafan1" or c["name"] == "dyn_hml_1"
    ]
    assert len(stress_indices) == 8, f"Expected 8 stress clips, got {len(stress_indices)}"
    mj_model = _find_mj_model(env)
    orig_friction = mj_model.geom_friction.copy()
    mj_model.geom_friction[:, 0] *= 0.55
    print(f"Applied Stress MuJoCo friction 0.55x (num geoms={len(orig_friction)}) across {len(stress_indices)} clips", flush=True)

    stress_results: dict[str, Any] = {}
    try:
        for cname, actor, is_onnx, emode, gf, h, dep_safe in eval_specs[:-1]:
            rows_st, rows_st_ho = [], []
            per_clip_st = {}
            for ci in stress_indices:
                cmeta = clips[ci]
                r = rollout_clip(
                    env, actor, is_onnx=is_onnx, clip_idx=ci, est_mode=emode,
                    goal_frames=gf, hist=h, deploy_safety=dep_safe, stress=True,
                )
                per_clip_st[cmeta["name"]] = {**cmeta, **r}
                rows_st.append(r)
                if cmeta["held_out"]:
                    rows_st_ho.append(r)
            stress_results[cname] = {
                "all_8": aggregate_rows(rows_st),
                "held_out_4": aggregate_rows(rows_st_ho),
                "per_clip": per_clip_st,
            }
            s8 = stress_results[cname]["all_8"]
            print(
                f"[STRESS ] {cname:26s} | Stress8 surv={s8['survival_rate']*100:5.1f}% ret={s8['norm_return']:.3f} "
                f"body={s8['body_pos_rmse_cm']:.2f}cm rootXY={s8['root_xy_rmse_cm']:.1f}cm slip={s8['stance_slip_mm_s']:.1f}mm/s",
                flush=True,
            )
    finally:
        mj_model.geom_friction[:] = orig_friction

    payload = {
        "clips": clips,
        "stress_clip_names": [clips[i]["name"] for i in stress_indices],
        "conditions": nom_results,
        "stress_conditions": stress_results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    make_summary_plot(payload, args.out_png)
    print(f"EVAL_DONE -> {args.out_json} & {args.out_png} ({args.out_png.stat().st_size} B)", flush=True)


if __name__ == "__main__":
    main()
