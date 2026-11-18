"""Quantitative Sim2Real Deployment Readiness Audit for SuperMimicHP Q1 Teleop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CLIPS = [
    "walk1_subject1",
    "walk2_subject1",
    "walk3_subject1",
    "walk4_subject1",
]
MOTION_DIR = ROOT / "src/unisupermimic/assets/motions/q1/lafan1_showcase"
OUT_DIR = ROOT / "runs/deploy_readiness_audit"


def _create_multi_clip_env(motion_files: list[str], *, ctrl_dt: float, obs_mode: str) -> Any:
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from unilab.base.config_adapter import create_env
    from unilab.training import ensure_registries
    from unisupermimic.scripts.train_supermimic import build_env_override

    torch.set_num_threads(2)
    ensure_registries()
    task_override = (
        "task=q1_super_mimic/mjwarp_66hz_causal"
        if abs(ctrl_dt - 0.015) < 1e-6
        else "task=q1_super_mimic/mujoco"
    )
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config", overrides=[task_override])
    OmegaConf.set_struct(cfg, False)
    cfg.training.sim_backend = "mujoco"
    cfg.training.play_only = True
    cfg.env.ctrl_dt = ctrl_dt
    cfg.env.commands.motion.ctrl_dt = ctrl_dt
    cfg.env.commands.motion.motion_files = motion_files
    cfg.env.commands.motion.observation_mode = obs_mode
    override = build_env_override(cfg, eval_mode=True, learner_device="cpu")
    return create_env(cfg, num_envs=1, env_cfg_override=override)


def _load_policy(mdp: Any, ckpt_path: Path):
    import torch
    from unisupermimic.rl.algos.supermimic import networks

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])
    networks.AMP_DTYPE = None
    policy = networks.Policy(
        mdp.obs_dim,
        mdp.goal_frame_dim * int(args["goal_frames"]),
        num_joints=mdp.num_joints,
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        hidden=hidden,
    )
    policy.load_state_dict(ckpt["pi"])
    policy.eval()
    return policy, int(args["goal_frames"]), int(args["hist"])


def _apply_root_xy_mode(
    obs: Any,
    state: Any,
    mdp: Any,
    frames: Any,
    root_xy_mode: str,
    step_idx: int,
    ctrl_dt: float,
) -> Any:
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import yaw_mat

    if root_xy_mode == "oracle" or mdp.observation_mode != "root_xy":
        return obs
    out = obs.clone()
    if root_xy_mode == "zeroed":
        out[:, -2:] = 0.0
        return out
    if root_xy_mode == "drift":
        t_sec = step_idx * ctrl_dt
        drift_w = torch.tensor(
            [[0.15 * t_sec, 0.08 * t_sec, 0.0]], dtype=torch.float32, device=obs.device
        )
        pos_drifted = state[0] + drift_w
        heading_t = yaw_mat(state[1]).transpose(-1, -2)
        root_err = (heading_t @ (mdp.motion.root_p[frames] - pos_drifted).unsqueeze(-1)).squeeze(-1)
        out[:, -2:] = root_err[..., :2]
        return out
    raise ValueError(f"Unknown root_xy_mode: {root_xy_mode}")


def _rollout_clip(
    env: Any,
    policy: Any,
    *,
    clip_idx: int,
    goal_frames: int,
    hist: int,
    goal_mode: str,
    latest_offset: int,
    root_xy_mode: str = "oracle",
    max_seconds: float = 8.0,
) -> dict[str, Any]:
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat, unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import (
        Q1_JOINT_NAMES,
        action_scale,
        gains_of,
        motor_of,
    )
    from unisupermimic.teleop.realtime_teleop import compute_teleop_goal

    alpha = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    effort = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)

    mdp = env.supermimic_mdp
    motion = mdp.motion
    ctrl_dt = float(getattr(motion, "ctrl_dt", 0.02))
    budget = min(int(motion.clip_len[clip_idx]) - 1, max(1, round(max_seconds / ctrl_dt)))

    def _flat_qpos(state_flat: torch.Tensor) -> np.ndarray:
        p, R, _, _, q, _ = unflatten_state(state_flat)
        return torch.cat((p, mat_to_quat(R), q), dim=-1)[0].cpu().numpy()

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(motion.clip_start[clip_idx : clip_idx + 1].clone())
        current = reset_info[SUPERMIMIC_STATE_KEY]
        f0 = int(current["f"][0].item())
        qpos_pol = [_flat_qpos(current["s"])]
        ref0 = torch.cat(
            (motion.root_p[f0], mat_to_quat(motion.root_R[f0 : f0 + 1])[0], motion.q[f0]), dim=-1
        ).cpu().numpy()
        qpos_ref = [ref0]
        q_targets = []
        tau_utils = []
        joint_sq = []
        lat_ms = []
        sum_body_sq = 0.0
        sum_root_sq = 0.0
        sum_ret = 0.0
        failed = False
        history = None
        step_idx = 0

        for step_idx in range(1, budget + 1):
            t0 = time.perf_counter()
            state = unflatten_state(current["s"])
            obs = mdp.obs(state, current["a_prev"], current["xi"], current["f"])
            obs = _apply_root_xy_mode(
                obs, state, mdp, current["f"], root_xy_mode, step_idx - 1, ctrl_dt
            )
            if history is None:
                history = obs.unsqueeze(1).repeat(1, hist, 1)
            else:
                history = torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)

            goal = compute_teleop_goal(
                mdp, current["f"], goal_frames, goal_mode, latest_offset=latest_offset  # type: ignore[arg-type]
            )
            action = policy(history, goal)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

            q_tgt = (mdp.default_q + current["xi"][0, 1:23] + alpha * action[0]).cpu().numpy()
            q_cur = state[4][0].cpu().numpy()
            qd_cur = state[5][0].cpu().numpy()
            tau = kp.numpy() * (q_tgt - q_cur) - kd.numpy() * qd_cur
            q_targets.append(q_tgt)
            tau_utils.append(np.abs(tau) / effort.numpy())

            step_state = env.step(action)
            tr = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            f_next = motion.future_frames(tr["f"], 1).squeeze(-1)
            _, terms = mdp.reward(
                unflatten_state(tr["s1"]), f_next, tr["a"], tr["a_prev"], terms=True
            )
            sum_body_sq += float(terms["e_body_pos"][0].item())
            sum_root_sq += float(terms["e_root_pos"][0].item())
            sum_ret += float(tr["r"][0].item())

            s1 = unflatten_state(tr["s1"])
            fn = int(f_next[0].item())
            joint_sq.append(float(torch.mean((s1[4][0] - motion.q[fn]) ** 2).item()))
            qpos_pol.append(_flat_qpos(tr["s1"]))
            qpos_ref.append(
                torch.cat(
                    (motion.root_p[fn], mat_to_quat(motion.root_R[fn : fn + 1])[0], motion.q[fn]),
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
            if bool(tr["term"][0].item()):
                failed = True
                break
            if bool(tr["done"][0].item()):
                break
            current = step_state.info[SUPERMIMIC_STATE_KEY]

    steps_run = max(step_idx, 1)
    max_r = float(mdp.max_step_reward)
    return {
        "ctrl_dt": np.float32(ctrl_dt),
        "steps": np.int32(steps_run),
        "budget": np.int32(budget),
        "failed": np.bool_(failed),
        "norm_return": np.float32(sum_ret / (budget * max_r)),
        "body_rmse_cm": np.float32(np.sqrt(sum_body_sq / steps_run) * 100.0),
        "root_rmse_cm": np.float32(np.sqrt(sum_root_sq / steps_run) * 100.0),
        "joint_rmse_deg": np.float32(np.degrees(np.sqrt(np.mean(joint_sq)))),
        "mean_latency_ms": np.float32(np.mean(lat_ms[1:]) if len(lat_ms) > 1 else np.mean(lat_ms)),
        "q_default": mdp.default_q.cpu().numpy().astype(np.float32),
        "q_lo": mdp.kin.q_lo.cpu().numpy().astype(np.float32),
        "q_hi": mdp.kin.q_hi.cpu().numpy().astype(np.float32),
        "qpos_pol": np.stack(qpos_pol, axis=0).astype(np.float32),
        "qpos_ref": np.stack(qpos_ref, axis=0).astype(np.float32),
        "q_targets": np.stack(q_targets, axis=0).astype(np.float32),
        "tau_utils": np.stack(tau_utils, axis=0).astype(np.float32),
    }


def _run_50hz(out_path: Path) -> None:
    motion_files = [str((MOTION_DIR / f"{c}.csv").resolve()) for c in CLIPS]
    env = _create_multi_clip_env(motion_files, ctrl_dt=0.02, obs_mode="root_xy")
    mdp = env.supermimic_mdp
    pi_base, gf_b, h_b = _load_policy(mdp, ROOT / "checkpoints/q1_base_soma1k.pt")
    pi_a, gf_a, h_a = _load_policy(mdp, ROOT / "checkpoints/q1_scheme_a_1k.pt")

    configs = [
        ("base_oracle", pi_base, gf_b, h_b, "oracle_future10", 1, "oracle"),
        ("base_zoh", pi_base, gf_b, h_b, "causal_zoh", 0, "oracle"),
        ("scheme_a", pi_a, gf_a, h_a, "causal_history", 0, "oracle"),
        ("base_zeroed", pi_base, gf_b, h_b, "oracle_future10", 1, "zeroed"),
        ("base_drift", pi_base, gf_b, h_b, "oracle_future10", 1, "drift"),
        ("scheme_a_zeroed", pi_a, gf_a, h_a, "causal_history", 0, "zeroed"),
        ("scheme_a_drift", pi_a, gf_a, h_a, "causal_history", 0, "drift"),
    ]
    payload: dict[str, Any] = {}
    for cfg_name, pi, gf, h, gmode, loff, rmode in configs:
        for ci, cname in enumerate(CLIPS):
            print(f"[50Hz] {cfg_name} | {cname}...", flush=True)
            res = _rollout_clip(
                env, pi, clip_idx=ci, goal_frames=gf, hist=h,
                goal_mode=gmode, latest_offset=loff, root_xy_mode=rmode,
            )
            for k, v in res.items():
                payload[f"{cfg_name}__{cname}__{k}"] = v
    np.savez_compressed(out_path, **payload)
    sys.stdout.flush()
    os._exit(0)


def _run_66hz(out_path: Path) -> None:
    motion_files = [str((MOTION_DIR / f"{c}.csv").resolve()) for c in CLIPS]
    env = _create_multi_clip_env(motion_files, ctrl_dt=0.015, obs_mode="root_xy")
    mdp = env.supermimic_mdp
    pi_b, gf_b, h_b = _load_policy(mdp, ROOT / "checkpoints/q1_scheme_b_66hz_1k.pt")
    payload: dict[str, Any] = {}
    for ci, cname in enumerate(CLIPS):
        print(f"[66.7Hz] scheme_b | {cname}...", flush=True)
        res = _rollout_clip(
            env, pi_b, clip_idx=ci, goal_frames=gf_b, hist=h_b,
            goal_mode="causal_history", latest_offset=0, root_xy_mode="oracle",
        )
        for k, v in res.items():
            payload[f"scheme_b__{cname}__{k}"] = v
    np.savez_compressed(out_path, **payload)
    sys.stdout.flush()
    os._exit(0)


def _foot_metrics(
    qpos_seq: np.ndarray, dt: float, model: Any, left_geo: Any, right_geo: Any, l_id: int, r_id: int
) -> dict[str, float]:
    import mujoco

    data = mujoco.MjData(model)
    z_l, z_r = np.empty(len(qpos_seq)), np.empty(len(qpos_seq))
    p_l, p_r = np.empty((len(qpos_seq), 2)), np.empty((len(qpos_seq), 2))
    for i, qp in enumerate(qpos_seq):
        data.qpos[:] = qp
        mujoco.mj_kinematics(model, data)
        z_l[i] = left_geo.minimum(data)
        z_r[i] = right_geo.minimum(data)
        p_l[i] = data.xpos[l_id, :2]
        p_r[i] = data.xpos[r_id, :2]

    all_z = np.concatenate([z_l, z_r])
    swing = all_z[all_z > 0.010]
    spd_l = np.linalg.norm(np.diff(p_l, axis=0), axis=-1) / dt
    spd_r = np.linalg.norm(np.diff(p_r, axis=0), axis=-1) / dt
    st_l = spd_l[np.minimum(z_l[:-1], z_l[1:]) < 0.008]
    st_r = spd_r[np.minimum(z_r[:-1], z_r[1:]) < 0.008]
    stance_spd = np.concatenate([st_l, st_r])

    return {
        "swing_frac_pct": float((all_z > 0.010).mean() * 100.0),
        "swing_p50_mm": float(np.percentile(swing, 50) * 1000.0) if len(swing) else float(np.percentile(all_z, 50) * 1000.0),
        "swing_p95_mm": float(np.percentile(swing, 95) * 1000.0) if len(swing) else float(np.percentile(all_z, 95) * 1000.0),
        "max_clearance_mm": float(np.max(all_z) * 1000.0),
        "stance_slip_mean_mms": float(np.mean(stance_spd) * 1000.0) if len(stance_spd) else 0.0,
        "stance_slip_p95_mms": float(np.percentile(stance_spd, 95) * 1000.0) if len(stance_spd) else 0.0,
    }


def _pd_metrics(q_targets: np.ndarray, qpos_pol: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, q_def: np.ndarray, tau_utils: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
    over = np.maximum(q_targets - q_hi[None, :], 0.0) + np.maximum(q_lo[None, :] - q_targets, 0.0)
    viol_mask = over > 1e-5
    step_jumps_deg = np.degrees(np.abs(np.diff(q_targets, axis=0)))
    step0_def_deg = float(np.degrees(np.max(np.abs(q_targets[0] - q_def))))
    step0_sim_deg = float(np.degrees(np.max(np.abs(q_targets[0] - qpos_pol[0, 7:]))))
    per_joint_max_viol = np.degrees(over.max(axis=0))
    top_idx = np.argsort(per_joint_max_viol)[::-1][:3]
    return {
        "oob_pair_pct": float(viol_mask.mean() * 100.0),
        "oob_step_pct": float(viol_mask.any(axis=1).mean() * 100.0),
        "max_oob_deg": float(np.degrees(over.max())),
        "top_oob_joints": [
            {"joint": joint_names[int(j)], "max_oob_deg": round(float(per_joint_max_viol[j]), 2)}
            for j in top_idx if per_joint_max_viol[j] > 0.0
        ],
        "per_joint_max_oob_deg": [round(float(x), 3) for x in per_joint_max_viol],
        "step0_jump_from_default_deg": step0_def_deg,
        "step0_jump_from_init_q_deg": step0_sim_deg,
        "step_jump_p99_deg": float(np.percentile(step_jumps_deg, 99)) if len(step_jumps_deg) else 0.0,
        "step_jump_max_deg": float(step_jumps_deg.max()) if len(step_jumps_deg) else 0.0,
        "per_joint_max_jump_deg": [round(float(x), 3) for x in step_jumps_deg.max(axis=0)] if len(step_jumps_deg) else [0.0] * len(joint_names),
        "tau_util_p99_pct": float(np.percentile(tau_utils, 99) * 100.0),
        "tau_util_max_pct": float(tau_utils.max() * 100.0),
    }


def _analyze_and_plot(npz50_path: Path, npz66_path: Path) -> None:
    import matplotlib.pyplot as plt
    import mujoco
    from tools.x2_geometry import CollisionGeometry
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES

    d50 = np.load(npz50_path)
    d66 = np.load(npz66_path)

    mj_model = mujoco.MjModel.from_xml_path(str(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"))
    l_geo = CollisionGeometry(mj_model, ("left_ankle_roll_link",))
    r_geo = CollisionGeometry(mj_model, ("right_ankle_roll_link",))
    l_id = mj_model.body("left_ankle_roll_link").id
    r_id = mj_model.body("right_ankle_roll_link").id

    all_cfgs = [
        ("base_oracle", d50),
        ("base_zoh", d50),
        ("scheme_a", d50),
        ("scheme_b", d66),
        ("base_zeroed", d50),
        ("base_drift", d50),
        ("scheme_a_zeroed", d50),
        ("scheme_a_drift", d50),
    ]

    summary: dict[str, Any] = {"clips": CLIPS, "schemes": {}}
    for cfg_name, src in all_cfgs:
        per_clip = {}
        for c in CLIPS:
            p = f"{cfg_name}__{c}__"
            dt = float(src[p + "ctrl_dt"].item())
            qp_pol = src[p + "qpos_pol"]
            qp_ref = src[p + "qpos_ref"]
            foot_pol = _foot_metrics(qp_pol, dt, mj_model, l_geo, r_geo, l_id, r_id)
            pd_stat = _pd_metrics(
                src[p + "q_targets"], qp_pol, src[p + "q_lo"], src[p + "q_hi"],
                src[p + "q_default"], src[p + "tau_utils"], Q1_JOINT_NAMES,
            )
            rec = {
                "steps": int(src[p + "steps"].item()),
                "budget": int(src[p + "budget"].item()),
                "survived": not bool(src[p + "failed"].item()),
                "norm_return": round(float(src[p + "norm_return"].item()), 4),
                "body_rmse_cm": round(float(src[p + "body_rmse_cm"].item()), 2),
                "root_rmse_cm": round(float(src[p + "root_rmse_cm"].item()), 2),
                "joint_rmse_deg": round(float(src[p + "joint_rmse_deg"].item()), 2),
                "mean_latency_ms": round(float(src[p + "mean_latency_ms"].item()), 2),
                "foot": foot_pol,
                "pd": pd_stat,
            }
            if cfg_name == "base_oracle":
                rec["foot_ref"] = _foot_metrics(qp_ref, dt, mj_model, l_geo, r_geo, l_id, r_id)
            per_clip[c] = rec

        agg = {
            "survival_rate_pct": round(float(np.mean([v["survived"] for v in per_clip.values()]) * 100.0), 1),
            "norm_return_mean": round(float(np.mean([v["norm_return"] for v in per_clip.values()])), 4),
            "body_rmse_cm_mean": round(float(np.mean([v["body_rmse_cm"] for v in per_clip.values()])), 2),
            "root_rmse_cm_mean": round(float(np.mean([v["root_rmse_cm"] for v in per_clip.values()])), 2),
            "joint_rmse_deg_mean": round(float(np.mean([v["joint_rmse_deg"] for v in per_clip.values()])), 2),
            "mean_latency_ms": round(float(np.mean([v["mean_latency_ms"] for v in per_clip.values()])), 2),
            "foot_swing_p95_mm_mean": round(float(np.mean([v["foot"]["swing_p95_mm"] for v in per_clip.values()])), 1),
            "foot_stance_slip_mean_mms": round(float(np.mean([v["foot"]["stance_slip_mean_mms"] for v in per_clip.values()])), 1),
            "foot_stance_slip_p95_mms": round(float(np.mean([v["foot"]["stance_slip_p95_mms"] for v in per_clip.values()])), 1),
            "pd_oob_pair_pct_mean": round(float(np.mean([v["pd"]["oob_pair_pct"] for v in per_clip.values()])), 2),
            "pd_max_oob_deg": round(float(np.max([v["pd"]["max_oob_deg"] for v in per_clip.values()])), 2),
            "pd_step0_jump_from_default_deg_max": round(float(np.max([v["pd"]["step0_jump_from_default_deg"] for v in per_clip.values()])), 2),
            "pd_step_jump_p99_deg_max": round(float(np.max([v["pd"]["step_jump_p99_deg"] for v in per_clip.values()])), 2),
            "pd_step_jump_max_deg": round(float(np.max([v["pd"]["step_jump_max_deg"] for v in per_clip.values()])), 2),
            "pd_tau_util_p99_pct_max": round(float(np.max([v["pd"]["tau_util_p99_pct"] for v in per_clip.values()])), 1),
        }
        if cfg_name == "base_oracle":
            summary["reference_foot_mean"] = {
                "swing_p95_mm": round(float(np.mean([v["foot_ref"]["swing_p95_mm"] for v in per_clip.values()])), 1),
                "stance_slip_mean_mms": round(float(np.mean([v["foot_ref"]["stance_slip_mean_mms"] for v in per_clip.values()])), 1),
                "stance_slip_p95_mms": round(float(np.mean([v["foot_ref"]["stance_slip_p95_mms"] for v in per_clip.values()])), 1),
            }
        summary["schemes"][cfg_name] = {"aggregate": agg, "per_clip": per_clip}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "audit_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # 2x2 Summary Plot
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=150)

    # Panel 1: 4-Clip Teleop Tracking
    ax = axes[0, 0]
    p1_keys = ["base_oracle", "base_zoh", "scheme_a", "scheme_b"]
    p1_labels = ["Base (+200ms)", "ZOH (0ms)", "Scheme A (0ms,50Hz)", "Scheme B (0ms,66.7Hz)"]
    x = np.arange(len(p1_keys))
    w = 0.35
    b_vals = [summary["schemes"][k]["aggregate"]["body_rmse_cm_mean"] for k in p1_keys]
    r_vals = [summary["schemes"][k]["aggregate"]["root_rmse_cm_mean"] for k in p1_keys]
    ax.bar(x - w / 2, b_vals, w, label="Body RMSE (cm)", color="#3b82f6")
    ax.bar(x + w / 2, r_vals, w, label="Root RMSE (cm)", color="#f97316")
    for xi, (bv, rv) in enumerate(zip(b_vals, r_vals)):
        ax.text(xi - w / 2, bv + 0.4, f"{bv:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, rv + 0.4, f"{rv:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, p1_labels, rotation=12)
    ax.set_ylabel("Error (cm)")
    ax.set_title("1. Teleop Tracking Accuracy (4 LAFAN1 Walk Clips)")
    ax.legend(fontsize=8)

    # Panel 2: root_xy Ablation
    ax = axes[0, 1]
    p2_keys = ["base_oracle", "base_zeroed", "base_drift", "scheme_a", "scheme_a_zeroed", "scheme_a_drift"]
    p2_labels = ["Base\nOracle", "Base\nZeroed", "Base\nDrift", "Sch-A\nOracle", "Sch-A\nZeroed", "Sch-A\nDrift"]
    x2 = np.arange(len(p2_keys))
    b2 = [summary["schemes"][k]["aggregate"]["body_rmse_cm_mean"] for k in p2_keys]
    r2 = [summary["schemes"][k]["aggregate"]["root_rmse_cm_mean"] for k in p2_keys]
    surv2 = [summary["schemes"][k]["aggregate"]["survival_rate_pct"] for k in p2_keys]
    ax.bar(x2 - w / 2, b2, w, label="Local Body RMSE (cm)", color="#10b981")
    ax.bar(x2 + w / 2, r2, w, label="Global Root RMSE (cm)", color="#ef4444")
    for xi, (rv, sv) in enumerate(zip(r2, surv2)):
        ax.text(xi, max(rv, b2[xi]) + 1.0, f"{sv:.0f}% surv", ha="center", va="bottom", fontsize=7.5, color="#1f2937")
    ax.set_xticks(x2, p2_labels)
    ax.set_ylabel("Error (cm)")
    ax.set_title("2. Privilege Ablation: Removing Simulator root_xy")
    ax.legend(fontsize=8)

    # Panel 3: Per-Joint PD OOB & Max Jump (Scheme A on walk1..4 max)
    ax = axes[1, 0]
    per_j_oob = np.max([summary["schemes"]["scheme_a"]["per_clip"][c]["pd"]["per_joint_max_oob_deg"] for c in CLIPS], axis=0)
    per_j_jmp = np.max([summary["schemes"]["scheme_a"]["per_clip"][c]["pd"]["per_joint_max_jump_deg"] for c in CLIPS], axis=0)
    xj = np.arange(len(Q1_JOINT_NAMES))
    short_j = [n.replace("_joint", "").replace("left_", "L_").replace("right_", "R_") for n in Q1_JOINT_NAMES]
    ax.bar(xj - 0.2, per_j_oob, 0.4, label="Max Limit Exceed (deg)", color="#dc2626")
    ax.bar(xj + 0.2, per_j_jmp, 0.4, label="Max Step Jump (deg)", color="#6366f1")
    ax.set_xticks(xj, short_j, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Degrees")
    ax.set_title("3. Q1 Scheme A PD Target Safety (22 DoF)")
    ax.legend(fontsize=8)

    # Panel 4: Foot Clearance & Stance Slip
    ax = axes[1, 1]
    p4_labels = ["Ref Motion", "Base Oracle", "Scheme A (0ms)", "Sch-A Zeroed"]
    sw_vals = [
        summary["reference_foot_mean"]["swing_p95_mm"],
        summary["schemes"]["base_oracle"]["aggregate"]["foot_swing_p95_mm_mean"],
        summary["schemes"]["scheme_a"]["aggregate"]["foot_swing_p95_mm_mean"],
        summary["schemes"]["scheme_a_zeroed"]["aggregate"]["foot_swing_p95_mm_mean"],
    ]
    sl_vals = [
        summary["reference_foot_mean"]["stance_slip_mean_mms"],
        summary["schemes"]["base_oracle"]["aggregate"]["foot_stance_slip_mean_mms"],
        summary["schemes"]["scheme_a"]["aggregate"]["foot_stance_slip_mean_mms"],
        summary["schemes"]["scheme_a_zeroed"]["aggregate"]["foot_stance_slip_mean_mms"],
    ]
    x4 = np.arange(len(p4_labels))
    ax.bar(x4 - w / 2, sw_vals, w, label="Swing Clearance p95 (mm)", color="#0ea5e9")
    ax.bar(x4 + w / 2, sl_vals, w, label="Stance Slip Mean (mm/s)", color="#eab308")
    for xi, (sv, slv) in enumerate(zip(sw_vals, sl_vals)):
        ax.text(xi - w / 2, sv + 2.0, f"{sv:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, slv + 2.0, f"{slv:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x4, p4_labels)
    ax.set_ylabel("mm  /  mm/s")
    ax.set_title("4. Foot Ground Clearance & Stance Slip")
    ax.legend(fontsize=8)

    fig.suptitle("SuperMimicHP Q1 Sim2Real Deployment Readiness Audit (LAFAN1 Walk1-4)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = OUT_DIR / "deploy_audit_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"SAVED {metrics_path} and {fig_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "50hz", "66hz", "analyze"), default="all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--npz50", type=Path)
    parser.add_argument("--npz66", type=Path)
    args = parser.parse_args()

    if args.stage == "50hz":
        _run_50hz(args.out)
        return
    if args.stage == "66hz":
        _run_66hz(args.out)
        return
    if args.stage == "analyze":
        _analyze_and_plot(args.npz50, args.npz66)
        return

    tmp = Path(tempfile.mkdtemp(prefix="smhp_audit_"))
    try:
        p50 = tmp / "rollout_50.npz"
        p66 = tmp / "rollout_66.npz"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", "50hz", "--out", str(p50)], check=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", "66hz", "--out", str(p66)], check=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", "analyze", "--npz50", str(p50), "--npz66", str(p66)], check=True)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
