"""Comprehensive Evaluation Suite for H002-SONIC v2:
  1. Apples-to-Apples v1 16-clip re-evaluation
  2. Unseen 50-clip Held-Out Multi-Dataset Benchmark (Nominal + 20-clip 0.55x/Jitter/IMU Stress)
     across Zeroed, v1 H002, Oracle, and all 4 SONIC Modalities (RobotPose, HumanSMPL, VR3Pt, PlannerCmd)
  3. Zero-IK Direct Raw Human BVH Evaluation (8 unseen SOMA/LAFAN1 BVHs, zero Mink IK)
  4. 24-second (1,200-step) Continuous SONIC Mode-Switch & 400ms Watchdog Dropout Test (saving 3D qpos trajectories)
"""

from __future__ import annotations

import argparse
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
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import mat_to_6d, mat_to_quat, so3_exp, unflatten_state, yaw_mat
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, gains_of, motor_of


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
    raise RuntimeError("Could not find MjModel")


def _ort_sess(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _load_pt_policy(mdp: Any, ckpt_path: Path) -> networks.Policy:
    networks.AMP_DTYPE = None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ck["args"]
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])
    obs_dim = int(ck["pi"]["obs_norm.mean"].numel())
    pi = networks.Policy(obs_dim, mdp.goal_frame_dim * int(args["goal_frames"]), num_joints=mdp.num_joints, hist=int(args["hist"]), z_dim=int(args["z_dim"]), hidden=hidden)
    pi.load_state_dict(ck["pi"]); pi.eval()
    return pi


def _build_goal(mdp_motion: Any, f_ref: torch.Tensor, modality: str, **kwargs: Any) -> torch.Tensor:
    if modality == "robot_pose":
        return build_robot_pose_goal(mdp_motion, f_ref, 10, latest_offset=0)
    if modality == "human_smpl":
        return build_human_smpl_goal(mdp_motion, f_ref, 10, latest_offset=0, h_nom=kwargs.get("h_nom", Q1_NOMINAL_PELVIS_Z))
    if modality == "vr_3pt":
        return build_vr_3pt_goal(mdp_motion, f_ref, 10, latest_offset=0, **kwargs)
    if modality == "planner_cmd":
        return build_planner_cmd_goal(mdp_motion, f_ref, 10, latest_offset=0, h_nom=kwargs.get("h_nom", Q1_NOMINAL_PELVIS_Z))
    if modality == "oracle_future10":
        return mdp_motion.goal(f_ref, 10)
    raise ValueError(modality)


def _apply_torque_safe_pd_filter(
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
    """Hardware-faithful PD filter: optional engage ramp + spring-margin clamp + motor torque saturation."""
    if engage_steps > 0 and step_i < engage_steps and q_init is not None:
        blend = 0.5 * (1.0 - np.cos(np.pi * float(step_i + 1) / float(engage_steps)))
        q_tgt_raw = q_init + float(blend) * (q_tgt_raw - q_init)
    dq_tau = 0.5 * eff_vec / kp_vec
    q_tgt = torch.clamp(q_tgt_raw, q_lo - dq_tau, q_hi + dq_tau)
    tau_raw = kp_vec * (q_tgt - q_cur) - kd_vec * qd_cur
    tau_clamped = torch.clamp(tau_raw, -eff_vec, eff_vec)
    return q_cur + (tau_clamped + kd_vec * qd_cur) / kp_vec


def rollout_clip(
    env: Any,
    *,
    clip_idx: int,
    est_mode: str,
    modality: str,
    tok_sess: ort.InferenceSession | None = None,
    head_sess: ort.InferenceSession | None = None,
    legacy_onnx: ort.InferenceSession | None = None,
    pt_policy: networks.Policy | None = None,
    deploy_safety: bool = True,
    stress: bool = False,
    save_qpos: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
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

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(motion.clip_start[clip_idx : clip_idx + 1].clone())
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])
        f0 = cur["f"]
        est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
        est.reset(s0[1], s0[3], s0[4], s0[5], motion.root_p[f0, :2], R_ref=motion.root_R[f0])

        history = None
        q_init = s0[4][0].clone()
        prev_q_tgt = None
        sum_ret = 0.0
        sum_body_sq = 0.0
        root_xy_sq, wrist_world_sq, stance_slip_speeds, tau_utils = [], [], [], []
        pd_oob_total = pd_count_total = 0
        pd_max_oob_rad = 0.0
        step_ms = []
        failed = False
        steps_run = 0
        qpos_pol, qpos_ref = [], []

        if save_qpos:
            qpos_pol.append(torch.cat((s0[0][0], mat_to_quat(s0[1])[0], s0[4][0]), -1).numpy().copy())
            qpos_ref.append(torch.cat((motion.root_p[f0[0]], mat_to_quat(motion.root_R[f0[0] : f0[0] + 1])[0], motion.q[f0[0]]), -1).numpy().copy())

        for step_i in range(budget):
            state = unflatten_state(cur["s"])
            f_cur = cur["f"]
            clip_s = motion.clip_start[motion.clip_id[f_cur]]
            loff = -1 if (stress and step_i > 0 and rng.random() < 0.25) else 0
            f_ref = torch.maximum(f_cur + loff, clip_s)

            R_obs, w_obs = state[1], state[3]
            if stress:
                R_obs = (yaw_bias_R @ R_obs[0]).unsqueeze(0)
                w_obs = w_obs + torch.from_numpy(rng.normal(0.0, 0.03, size=(1, 3)).astype(np.float32))

            t0 = time.perf_counter()
            obs = mdp.obs((state[0], R_obs, state[2], w_obs, state[4], state[5]), cur["a_prev"], cur["xi"], f_ref)
            est_xy, _ = est.step(
                R_obs, w_obs, state[4], state[5], motion.root_p[f_ref, :2],
                R_ref=motion.root_R[f_ref], mode=est_mode, true_obs_root_xy=obs[:, 78:80],
            )
            obs_pol = torch.cat((obs[:, :78], est_xy), dim=-1)
            history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)
            goal = _build_goal(motion, f_ref, modality)

            if tok_sess is not None and head_sess is not None:
                tok = tok_sess.run(["token"], {"modal_input": goal.numpy().astype(np.float32)})[0]
                act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])
            elif legacy_onnx is not None:
                act = torch.from_numpy(legacy_onnx.run(["action"], {"obs_hist": history.numpy(), "goal": goal.numpy()})[0])
            elif pt_policy is not None:
                act = pt_policy(history, goal)
            else:
                raise RuntimeError("No policy provided")

            if deploy_safety:
                q_tgt_raw = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0]
                q_tgt_safe = _apply_torque_safe_pd_filter(
                    q_tgt_raw, state[4][0], state[5][0], q_lo, q_hi, kp_vec, kd_vec, eff_vec, engage_steps=0
                )
                act = ((q_tgt_safe - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0)

            step_ms.append((time.perf_counter() - t0) * 1000.0)
            q_tgt_sent = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0]
            dq_tau = 0.5 * eff_vec / kp_vec
            tau = kp_vec * (q_tgt_sent - state[4][0]) - kd_vec * state[5][0]
            tau_utils.append(float((torch.abs(tau) / eff_vec).max().item()))
            oob_q = torch.maximum(q_tgt_sent - (q_hi + dq_tau), (q_lo - dq_tau) - q_tgt_sent)
            oob_tau_mask = (torch.abs(tau) - eff_vec) > 1e-4
            oob_mask = (oob_q > 1e-4) & oob_tau_mask
            if deploy_safety:
                oob_mask = oob_tau_mask
            pd_oob_total += int(oob_mask.sum().item())
            pd_count_total += int(q_tgt_sent.numel())
            if bool(oob_mask.any()):
                oob_mag = torch.maximum(oob_q.clamp(min=0.0), (torch.abs(tau) - eff_vec).clamp(min=0.0) / kp_vec)
                pd_max_oob_rad = max(pd_max_oob_rad, float(oob_mag[oob_mask].max().item()))

            step_state = env.step(act)
            tr = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            f_next = motion.future_frames(tr["f"], 1).squeeze(-1)
            s1 = unflatten_state(tr["s1"])
            _, terms = mdp.reward(s1, f_next, tr["a"], tr["a_prev"], terms=True)

            sum_ret += float(tr["r"][0].item())
            sum_body_sq += float(terms["e_body_pos"][0].item())
            steps_run += 1
            root_xy_sq.append(float(((s1[0][0, :2] - motion.root_p[f_next[0], :2]) ** 2).sum().item()))

            bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
            wrist_world_sq.append(float(((bp1[0, (LEFT_WRIST_IDX, RIGHT_WRIST_IDX)] - motion.body_p[f_next[0], (LEFT_WRIST_IDX, RIGHT_WRIST_IDX)]) ** 2).sum(-1).mean().item()))

            ref_az = motion.body_p[f_next[0], (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]
            ref_av = motion.body_v[f_next[0], (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].norm(dim=-1)
            smask = (ref_az <= 0.062) & (ref_av <= 0.15)
            if bool(smask.any()):
                v_sole = bv1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] + torch.cross(bw1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)], (bR1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] @ sole_off).squeeze(-1), dim=-1)
                stance_slip_speeds.extend(v_sole[0, :, :2].norm(dim=-1)[smask].tolist())

            if save_qpos:
                qpos_pol.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
                fn = int(f_next[0])
                qpos_ref.append(torch.cat((motion.root_p[fn], mat_to_quat(motion.root_R[fn : fn + 1])[0], motion.q[fn]), -1).numpy().copy())

            if bool(tr["term"][0].item()):
                failed = True
                break
            if bool(tr["done"][0].item()):
                break
            cur = step_state.info[SUPERMIMIC_STATE_KEY]

    max_r = float(mdp.max_step_reward)
    out = {
        "completed": (not failed) and (steps_run == budget),
        "failed": failed,
        "steps_run": steps_run,
        "budget": budget,
        "survival_fraction": steps_run / max(budget, 1),
        "norm_return": sum_ret / max(budget * max_r, 1e-8),
        "body_pos_rmse_cm": float(np.sqrt(sum_body_sq / max(steps_run, 1)) * 100.0),
        "root_xy_rmse_cm": float(np.sqrt(np.mean(root_xy_sq)) * 100.0) if root_xy_sq else 0.0,
        "wrist_world_rmse_cm": float(np.sqrt(np.mean(wrist_world_sq)) * 100.0) if wrist_world_sq else 0.0,
        "stance_slip_mm_s": float(np.mean(stance_slip_speeds) * 1000.0) if stance_slip_speeds else 0.0,
        "pd_oob_pct": 100.0 * pd_oob_total / max(pd_count_total, 1),
        "pd_max_oob_deg": float(np.degrees(pd_max_oob_rad)),
        "tau_util_p99": float(np.percentile(tau_utils, 99)) if tau_utils else 0.0,
        "cpu_step_ms_p50": float(np.percentile(step_ms, 50)) if step_ms else 0.0,
    }
    if save_qpos:
        out["qpos_pol"] = np.stack(qpos_pol, axis=0)
        out["qpos_ref"] = np.stack(qpos_ref, axis=0)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {"n_clips": len(rows), "survival_rate": float(np.mean([float(r["completed"]) for r in rows]))}
    for k in ("survival_fraction", "norm_return", "body_pos_rmse_cm", "root_xy_rmse_cm", "wrist_world_rmse_cm", "stance_slip_mm_s", "pd_oob_pct", "tau_util_p99", "cpu_step_ms_p50"):
        out[k] = float(np.mean([float(r[k]) for r in rows]))
    out["pd_max_oob_deg"] = float(max(r["pd_max_oob_deg"] for r in rows))
    return out


def _enter_free_roaming_mode(env: Any, stand_frame_idx: int) -> None:
    """Disable episode auto-reset/termination/truncation during free-roaming teleop & pin command frame."""
    env.termination_manager._term_names = []
    env.termination_manager._term_cfgs = []
    env._compute_truncated = lambda s: torch.zeros_like(s.truncated)
    env._supermimic_command.f[:] = stand_frame_idx
    env._supermimic_command.ep_step[:] = 0


def eval_zero_ik_bvh(
    env: Any,
    stand_clip_idx: int,
    tok_smpl_sess: ort.InferenceSession,
    head_sess: ort.InferenceSession,
) -> dict[str, Any]:
    """Run closed-loop MuJoCo teleop directly from 8 raw human SOMA/LAFAN1 BVH files (ZERO Mink IK!)."""
    bvh_dir = Path("/home/yu/datasets/heldout_raw_bvh")
    mdp = env.supermimic_mdp
    stand_frame_idx = int(mdp.motion.clip_start[stand_clip_idx])
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()

    results = {}
    for p in sorted(bvh_dir.glob("*.bvh")):
        kind = "lafan1" if p.name.startswith("lafan_") else "soma"
        mock_m, h_nom, _, _ = load_bvh_as_mock_motion(p, kind=kind)
        with torch.no_grad():
            _, reset_info = env.reset_to_frames(mdp.motion.clip_start[stand_clip_idx : stand_clip_idx + 1].clone())
            _enter_free_roaming_mode(env, stand_frame_idx)
            cur = reset_info[SUPERMIMIC_STATE_KEY]
            s0 = unflatten_state(cur["s"])
            R0_rob = s0[1][0]
            R0_hum = mock_m.root_R[0]
            R_align = yaw_mat((R0_rob @ R0_hum.transpose(-1, -2)).unsqueeze(0))[0]
            p0_rob = s0[0][0, :2]
            p0_hum = mock_m.root_p[0, :2]

            est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
            p_init_ref = ((R_align[:2, :2] @ (mock_m.root_p[0, :2] - p0_hum)) + p0_rob).unsqueeze(0)
            R_init_ref = (R_align @ mock_m.root_R[0]).unsqueeze(0)
            est.reset(s0[1], s0[3], s0[4], s0[5], p_init_ref, R_ref=R_init_ref)
            history = None
            q_init = s0[4][0].clone()
            budget = min(mock_m.num_frames - 1, 280)
            steps_ok = 0
            failed = False
            root_xy_errs = []

            for t in range(budget):
                st = unflatten_state(cur["s"])
                f_t = torch.tensor([t], dtype=torch.long)
                R_ref_t = (R_align @ mock_m.root_R[t]).unsqueeze(0)
                p_ref_xy = ((R_align[:2, :2] @ (mock_m.root_p[t, :2] - p0_hum)) + p0_rob).unsqueeze(0)

                obs_tmp = mdp.obs(st, cur["a_prev"], cur["xi"], cur["f"])
                rel_rot = mat_to_6d(st[1].transpose(-1, -2) @ R_ref_t)
                est_xy, _ = est.step(st[1], st[3], st[4], st[5], p_ref_xy, R_ref=R_ref_t, mode="cgfa")
                obs_pol = torch.cat((obs_tmp[:, :72], rel_rot, est_xy), dim=-1)
                history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)

                gm = build_human_smpl_goal(mock_m, f_t, 10, h_nom=h_nom)
                tok = tok_smpl_sess.run(["token"], {"modal_input": gm.numpy().astype(np.float32)})[0]
                act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])

                q_tgt_raw = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0]
                q_tgt = _apply_torque_safe_pd_filter(
                    q_tgt_raw, st[4][0], st[5][0], q_lo, q_hi, kp_vec, kd_vec, eff_vec,
                    q_init=q_init, step_i=t, engage_steps=15,
                )
                act_sent = ((q_tgt - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0)

                env._supermimic_command.f[:] = stand_frame_idx
                env._supermimic_command.ep_step[:] = 0
                step_s = env.step(act_sent)
                tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
                s1 = unflatten_state(tr["s1"])
                steps_ok += 1
                root_xy_errs.append(float(((s1[0][0, :2] - p_ref_xy[0]) ** 2).sum().item()))

                if float(s1[0][0, 2]) < 0.18 or float(s1[1][0, 2, 2]) < 0.35:
                    failed = True
                    break
                cur = step_s.info[SUPERMIMIC_STATE_KEY]

        results[p.stem] = {
            "kind": kind,
            "completed": not failed,
            "steps_run": steps_ok,
            "budget": budget,
            "h_nom_human_m": round(h_nom, 3),
            "root_xy_rmse_cm": round(float(np.sqrt(np.mean(root_xy_errs)) * 100.0), 2),
        }
        print(f"ZERO_IK_BVH {p.stem:40s} ({kind}): ok={not failed} steps={steps_ok}/{budget} h_nom={h_nom:.2f}m rootXY={results[p.stem]['root_xy_rmse_cm']:.1f}cm", flush=True)

    return {
        "n_bvhs": len(results),
        "survival_rate": float(np.mean([float(v["completed"]) for v in results.values()])),
        "mean_root_xy_rmse_cm": float(np.mean([v["root_xy_rmse_cm"] for v in results.values()])),
        "per_bvh": results,
    }


def run_24s_sonic_modeswitch(
    env: Any,
    stand_clip_idx: int,
    vr_clip_idx: int,
    onnx_dir: Path,
    smpl_bvh: Path,
    out_npz: Path,
    out_json: Path,
) -> dict[str, Any]:
    """Single unbroken 24.0s (1,200-step) closed-loop simulation switching across ALL 4 SONIC modalities + Watchdog."""
    mdp = env.supermimic_mdp
    stand_frame_idx = int(mdp.motion.clip_start[stand_clip_idx])
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()
    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32).view(1, 1, 3, 1)

    head_sess = _ort_sess(onnx_dir / "q1_h002_token_head.onnx")
    tok_plan = _ort_sess(onnx_dir / "q1_tok_planner_cmd.onnx")
    tok_smpl = _ort_sess(onnx_dir / "q1_tok_human_smpl.onnx")
    tok_vr = _ort_sess(onnx_dir / "q1_tok_vr_3pt.onnx")

    cmd_A = build_synthetic_planner_stream([(205, 0.0, 0.0, 0.0, 1.0, 0.35)])
    cmd_B = build_synthetic_planner_stream([
        (100, 0.38, 0.0, 0.0, 1.0, 0.35),
        (20, 0.0, 0.0, 0.0, 1.0, 0.35),
        (80, 0.0, 0.0, 0.0, 0.84, 0.42),
        (55, 0.0, 0.0, 0.0, 1.0, 0.35),
    ])
    cmd_idle = build_synthetic_planner_stream([(200, 0.0, 0.0, 0.0, 1.0, 0.35)])

    mock_smpl, h_nom_s, _, _ = load_bvh_as_mock_motion(smpl_bvh, kind="soma")

    T_TOTAL = 1200
    phases = [
        (0, 200, "A_StandIdle_Planner"),
        (200, 450, "B_WalkSquat_Planner"),
        (450, 700, "C_ZeroIK_HumanSMPL"),
        (700, 950, "D_VR3Pt_PICO"),
        (950, 1050, "E_400ms_Dropout_Watchdog"),
        (1050, 1200, "F_StandRecover_Planner"),
    ]

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(mdp.motion.clip_start[stand_clip_idx : stand_clip_idx + 1].clone())
        _enter_free_roaming_mode(env, stand_frame_idx)
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])
        est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
        est.reset(s0[1], s0[3], s0[4], s0[5], s0[0][:, :2], R_ref=s0[1])

        history = None
        q_init = s0[4][0].clone()
        prev_q_tgt = s0[4][0].clone()
        last_act = torch.zeros((1, 22), dtype=torch.float32)

        p_plan_xy = s0[0][0, :2].clone()
        yaw_plan_R = yaw_mat(s0[1])[0].clone()

        crossfade_origin_act = last_act.clone()
        crossfade_t0 = -999
        prev_phase = ""

        engage_smpl_p = None
        engage_smpl_R = None
        engage_vr_p = None
        engage_vr_R = None
        vr_clip_s = int(mdp.motion.clip_start[vr_clip_idx])

        qpos_hist = []
        mode_hist = []
        step_jumps_deg = []
        slip_hist = []
        tau_hist = []
        oob_count = 0
        height_hist = []
        tilt_hist = []
        failed = False
        stale_tok = None

        for t in range(T_TOTAL):
            st = unflatten_state(cur["s"])
            p_cur, R_cur, v_cur, w_cur, q_cur, qd_cur = st

            pname = next(n for a, b, n in phases if a <= t < b)
            if pname != prev_phase and t > 0:
                crossfade_origin_act = last_act.clone()
                crossfade_t0 = t
                if pname == "B_WalkSquat_Planner":
                    p_plan_xy = p_cur[0, :2].clone()
                    yaw_plan_R = yaw_mat(R_cur)[0].clone()
                    est.reset(R_cur, w_cur, q_cur, qd_cur, p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))
                elif pname == "C_ZeroIK_HumanSMPL":
                    engage_smpl_p = p_cur[0, :2].clone()
                    engage_smpl_R = yaw_mat((yaw_mat(R_cur)[0] @ mock_smpl.root_R[0].transpose(-1, -2)).unsqueeze(0))[0]
                    est.reset(R_cur, w_cur, q_cur, qd_cur, engage_smpl_p.unsqueeze(0), R_ref=(engage_smpl_R @ mock_smpl.root_R[0]).unsqueeze(0))
                elif pname == "D_VR3Pt_PICO":
                    engage_vr_p = p_cur[0, :2].clone()
                    vr_R0 = mdp.motion.root_R[vr_clip_s]
                    engage_vr_R = yaw_mat((yaw_mat(R_cur)[0] @ vr_R0.transpose(-1, -2)).unsqueeze(0))[0]
                    est.reset(R_cur, w_cur, q_cur, qd_cur, engage_vr_p.unsqueeze(0), R_ref=(engage_vr_R @ vr_R0).unsqueeze(0))
                elif pname in ("E_400ms_Dropout_Watchdog", "F_StandRecover_Planner"):
                    p_plan_xy = p_cur[0, :2].clone()
                    yaw_plan_R = yaw_mat(R_cur)[0].clone()
                    est.reset(R_cur, w_cur, q_cur, qd_cur, p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))
            prev_phase = pname

            if pname == "A_StandIdle_Planner":
                g_row = cmd_A[t]
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                tok = tok_plan.run(["token"], {"modal_input": g_row.unsqueeze(0).numpy()})[0]
            elif pname == "B_WalkSquat_Planner":
                k_b = t - 200
                g_row = cmd_B[k_b]
                vx_b = float(g_row[-10])
                v_w = yaw_plan_R[:2, :2] @ torch.tensor([vx_b, 0.0])
                p_plan_xy = p_plan_xy + v_w * 0.02
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                tok = tok_plan.run(["token"], {"modal_input": g_row.unsqueeze(0).numpy()})[0]
            elif pname == "C_ZeroIK_HumanSMPL":
                k_c = min(t - 450, mock_smpl.num_frames - 1)
                p_ref_xy = ((engage_smpl_R[:2, :2] @ (mock_smpl.root_p[k_c, :2] - mock_smpl.root_p[0, :2])) + engage_smpl_p).unsqueeze(0)
                R_ref_t = (engage_smpl_R @ mock_smpl.root_R[k_c]).unsqueeze(0)
                gm = build_human_smpl_goal(mock_smpl, torch.tensor([k_c]), 10, h_nom=h_nom_s)
                tok = tok_smpl.run(["token"], {"modal_input": gm.numpy()})[0]
                stale_tok = tok.copy()
            elif pname == "D_VR3Pt_PICO":
                k_d = min(t - 700, int(mdp.motion.clip_len[vr_clip_idx]) - 1)
                f_d = torch.tensor([vr_clip_s + k_d], dtype=torch.long)
                p_ref_xy = ((engage_vr_R[:2, :2] @ (mdp.motion.root_p[f_d[0], :2] - mdp.motion.root_p[vr_clip_s, :2])) + engage_vr_p).unsqueeze(0)
                R_ref_t = (engage_vr_R @ mdp.motion.root_R[f_d[0]]).unsqueeze(0)
                gv = build_vr_3pt_goal(mdp.motion, f_d, 10)
                tok = tok_vr.run(["token"], {"modal_input": gv.numpy()})[0]
                stale_tok = tok.copy()
            elif pname == "E_400ms_Dropout_Watchdog":
                k_e = t - 950
                if k_e == 13:
                    crossfade_origin_act = last_act.clone()
                    crossfade_t0 = t
                    p_plan_xy = p_cur[0, :2].clone()
                    yaw_plan_R = yaw_mat(R_cur)[0].clone()
                    est.reset(R_cur, w_cur, q_cur, qd_cur, p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                if k_e < 13:
                    tok = stale_tok
                else:
                    tok = tok_plan.run(["token"], {"modal_input": cmd_idle[k_e].unsqueeze(0).numpy()})[0]
            else:
                k_f = t - 1050
                p_ref_xy = p_plan_xy.unsqueeze(0)
                R_ref_t = yaw_plan_R.unsqueeze(0)
                tok = tok_plan.run(["token"], {"modal_input": cmd_idle[k_f].unsqueeze(0).numpy()})[0]

            obs_raw = mdp.obs(st, cur["a_prev"], cur["xi"], cur["f"])
            rel_rot = mat_to_6d(R_cur.transpose(-1, -2) @ R_ref_t)
            est_xy, _ = est.step(R_cur, w_cur, q_cur, qd_cur, p_ref_xy, R_ref=R_ref_t, mode="cgfa")
            obs_pol = torch.cat((obs_raw[:, :72], rel_rot, est_xy), dim=-1)
            history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)

            act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])

            # 0.24s (12-step) cosine crossfade on mode switches & watchdog takeover
            if t - crossfade_t0 < 12:
                w_cf = 0.5 * (1.0 - np.cos(np.pi * float(t - crossfade_t0 + 1) / 12.0))
                act = (1.0 - float(w_cf)) * crossfade_origin_act + float(w_cf) * act

            q_tgt_raw = mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act[0]
            q_tgt = _apply_torque_safe_pd_filter(
                q_tgt_raw, q_cur[0], qd_cur[0], q_lo, q_hi, kp_vec, kd_vec, eff_vec,
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
            tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
            s1 = unflatten_state(tr["s1"])
            bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
            min_f_idx = int(torch.argmin(bp1[0, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]))
            a_idx = LEFT_ANKLE_IDX if min_f_idx == 0 else RIGHT_ANKLE_IDX
            v_sole = bv1[0, a_idx] + torch.cross(bw1[0, a_idx], (bR1[0, a_idx] @ sole_off[0, 0]).squeeze(-1), dim=-1)
            slip_hist.append(float(v_sole[:2].norm().item() * 1000.0))

            pz = float(s1[0][0, 2])
            up_z = float(s1[1][0, 2, 2])
            height_hist.append(pz)
            tilt_hist.append(up_z)
            qpos_hist.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
            mode_hist.append(pname)

            if pz < 0.16 or up_z < 0.35:
                failed = True
                break
            cur = step_s.info[SUPERMIMIC_STATE_KEY]

    np.savez_compressed(out_npz, qpos=np.stack(qpos_hist, axis=0))
    summary = {
        "completed_24s": (not failed) and len(qpos_hist) == T_TOTAL,
        "steps_survived": len(qpos_hist),
        "budget": T_TOTAL,
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
                "slip_mm_s": round(slip_hist[i], 1),
                "jump_deg": round(step_jumps_deg[i], 2),
                "tau_util": round(tau_hist[i], 3),
            }
            for i in range(0, len(qpos_hist), 10)
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SONIC_24S_MODESWITCH:", json.dumps({k: v for k, v in summary.items() if k != "trace_sampled"}, indent=2), flush=True)
    return summary


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-baselines", type=Path, default=base / "v2a_wm_unfrozen_metrics.json")
    parser.add_argument("--reuse-trajs", type=Path, default=base / "v2a_sample_trajs.npz")
    cli_args = parser.parse_args()

    reuse_m = json.loads(cli_args.reuse_baselines.read_text(encoding="utf-8")) if cli_args.reuse_baselines and cli_args.reuse_baselines.is_file() else None
    reuse_t = np.load(cli_args.reuse_trajs) if cli_args.reuse_trajs and cli_args.reuse_trajs.is_file() else None
    if reuse_m is not None:
        print(f"Reusing invariant baselines + Ablation_WM_UnfrozenDec from {cli_args.reuse_baselines}", flush=True)

    manifest = json.loads((base / "dataset_250_manifest.json").read_text(encoding="utf-8"))
    v1_train_clips = [c for c in manifest["clips"] if c.get("v1_clip") and c["split"] == "train"]
    held_out_clips = [c for c in manifest["clips"] if c["split"] == "held_out"]
    assert len(held_out_clips) == 50, len(held_out_clips)

    all_eval_clips = v1_train_clips + held_out_clips
    motion_files = [str(ROOT / c["csv_rel"]) for c in all_eval_clips]
    stand_ref_csv = str(ROOT / "src/unisupermimic/assets/motions/q1/multidataset_250/train/syn_stand_idle_1.csv")
    if stand_ref_csv not in motion_files:
        motion_files.append(stand_ref_csv)
    stand_clip_idx = motion_files.index(stand_ref_csv)

    vr_ref_csv = str(ROOT / "src/unisupermimic/assets/motions/q1/multidataset_250/held_out/lafan_aiming1_subject4_w0.csv")
    if vr_ref_csv not in motion_files:
        vr_ref_csv = str(ROOT / "src/unisupermimic/assets/motions/q1/multidataset_250/held_out/soma_stat_3_eating_apple_R_002__A276.csv")
    vr_clip_idx = motion_files.index(vr_ref_csv)

    n_v1tr = len(v1_train_clips)
    ho_offset = n_v1tr  # held_out indices start at n_v1tr

    env = create_eval_env(motion_files)
    mdp = env.supermimic_mdp

    onnx_dir = base / "onnx"
    head_sess = _ort_sess(onnx_dir / "q1_h002_token_head.onnx")
    tok_robot = _ort_sess(onnx_dir / "q1_tok_robot_pose.onnx")
    tok_smpl = _ort_sess(onnx_dir / "q1_tok_human_smpl.onnx")
    tok_vr = _ort_sess(onnx_dir / "q1_tok_vr_3pt.onnx")
    tok_plan = _ort_sess(onnx_dir / "q1_tok_planner_cmd.onnx")
    v1_onnx = _ort_sess(ROOT / "runs/h002_cgfa_pilot/q1_h002_0ms.onnx")
    pi_sa = _load_pt_policy(mdp, ROOT / "checkpoints/q1_scheme_a_1k.pt")

    # 1. Evaluate v1 12-train clips with V2_RobotPose_ONNX + Oracle for exact 16-clip v1 match
    v1_16_indices = [i for i, c in enumerate(all_eval_clips) if c.get("v1_clip")]
    v1_16_res = {"V2_RobotPose_ONNX": [], "Oracle_scheme_a_0ms": [], "stat_fit_squat": {}, "walk4_subject1": {}}
    saved_trajs: dict[str, np.ndarray] = {}

    baseline_names = ("Oracle_scheme_a_0ms", "Zeroed_unadapted_0ms", "V1_H002_ONNX")
    ho_specs = [
        ("Oracle_scheme_a_0ms", dict(est_mode="oracle", modality="robot_pose", pt_policy=pi_sa, deploy_safety=False)),
        ("Zeroed_unadapted_0ms", dict(est_mode="zeroed", modality="robot_pose", pt_policy=pi_sa, deploy_safety=False)),
        ("V1_H002_ONNX", dict(est_mode="cgfa", modality="robot_pose", legacy_onnx=v1_onnx, deploy_safety=True)),
        ("V2_RobotPose_ONNX", dict(est_mode="cgfa", modality="robot_pose", tok_sess=tok_robot, head_sess=head_sess, deploy_safety=True)),
        ("V2_HumanSMPL_ONNX", dict(est_mode="cgfa", modality="human_smpl", tok_sess=tok_smpl, head_sess=head_sess, deploy_safety=True)),
        ("V2_VR3Pt_ONNX", dict(est_mode="cgfa", modality="vr_3pt", tok_sess=tok_vr, head_sess=head_sess, deploy_safety=True)),
        ("V2_PlannerCmd_ONNX", dict(est_mode="cgfa", modality="planner_cmd", tok_sess=tok_plan, head_sess=head_sess, deploy_safety=True)),
    ]

    ho_results: dict[str, Any] = {}
    for cname, kw in ho_specs:
        if reuse_m is not None and cname in baseline_names and cname in reuse_m.get("held_out_50_nominal", {}):
            ho_results[cname] = reuse_m["held_out_50_nominal"][cname]
            if reuse_t is not None:
                for k in reuse_t.files:
                    if k.startswith(f"{cname}__"):
                        saved_trajs[k] = reuse_t[k]
            a = ho_results[cname]["all_50"]
            print(f"[HO-50 NOM] {cname:22s}: surv={a['survival_rate']*100:5.1f}% ret={a['norm_return']:.3f} body={a['body_pos_rmse_cm']:5.2f}cm rootXY={a['root_xy_rmse_cm']:5.1f}cm slip={a['stance_slip_mm_s']:5.1f}mm/s pdOOB={a['pd_oob_pct']:.3f}% (reused)", flush=True)
            continue

        rows_all, rows_stat, rows_dyn, rows_loco_sq = [], [], [], []
        per_clip = {}
        for i in range(ho_offset, len(all_eval_clips)):
            cmeta = all_eval_clips[i]
            want_save = (cname in ("V2_RobotPose_ONNX", "V1_H002_ONNX", "Oracle_scheme_a_0ms") and cmeta["alias"] in ("stat_fit_squat", "walk2_subject1", "soma_stat_3_eating_apple_R_002__A276", "dyn_hml_3"))
            r = rollout_clip(env, clip_idx=i, save_qpos=want_save, **kw)
            if want_save:
                saved_trajs[f"{cname}__{cmeta['alias']}__pol"] = r.pop("qpos_pol")
                saved_trajs[f"{cname}__{cmeta['alias']}__ref"] = r.pop("qpos_ref")
            r_clean = {k: v for k, v in r.items()}
            per_clip[cmeta["alias"]] = r_clean
            rows_all.append(r_clean)
            if cmeta["category"] == "stationary":
                rows_stat.append(r_clean)
            else:
                rows_dyn.append(r_clean)
            if any(tok in cmeta["alias"] for tok in ("walk", "hml", "fit", "squat", "syn_", "soma_dyn")):
                rows_loco_sq.append(r_clean)
        ho_results[cname] = {
            "all_50": aggregate(rows_all),
            "stationary": aggregate(rows_stat),
            "dynamic": aggregate(rows_dyn),
            "loco_squat_subset": aggregate(rows_loco_sq),
            "per_clip": per_clip,
        }
        a = ho_results[cname]["all_50"]
        print(f"[HO-50 NOM] {cname:22s}: surv={a['survival_rate']*100:5.1f}% ret={a['norm_return']:.3f} body={a['body_pos_rmse_cm']:5.2f}cm rootXY={a['root_xy_rmse_cm']:5.1f}cm slip={a['stance_slip_mm_s']:5.1f}mm/s pdOOB={a['pd_oob_pct']:.3f}%", flush=True)

    if reuse_m is not None and "V2_RobotPose_ONNX" in reuse_m.get("held_out_50_nominal", {}):
        ho_results["Ablation_WM_UnfrozenDec"] = reuse_m["held_out_50_nominal"]["V2_RobotPose_ONNX"]
        aa = ho_results["Ablation_WM_UnfrozenDec"]["all_50"]
        print(f"[HO-50 NOM] {'Ablation_WM_UnfrozenDec':22s}: surv={aa['survival_rate']*100:5.1f}% ret={aa['norm_return']:.3f} (v2a ablation)", flush=True)

    # Also evaluate V2_RobotPose_ONNX & Oracle on v1's 12 train clips to compute exact 16-clip v1 match
    reuse_oracle_v1 = (reuse_m is not None and "Oracle_scheme_a_0ms" in reuse_m.get("v1_16_match", {}))
    v1_16_rows_v2, v1_16_rows_oracle = [], []
    for i in v1_16_indices:
        cmeta = all_eval_clips[i]
        if i >= ho_offset:
            v1_16_rows_v2.append(ho_results["V2_RobotPose_ONNX"]["per_clip"][cmeta["alias"]])
            if not reuse_oracle_v1:
                v1_16_rows_oracle.append(ho_results["Oracle_scheme_a_0ms"]["per_clip"][cmeta["alias"]])
        else:
            want_save = (cmeta["alias"] == "walk4_subject1")
            rv2 = rollout_clip(env, clip_idx=i, est_mode="cgfa", modality="robot_pose", tok_sess=tok_robot, head_sess=head_sess, deploy_safety=True, save_qpos=want_save)
            if want_save:
                saved_trajs[f"V2_RobotPose_ONNX__{cmeta['alias']}__pol"] = rv2.pop("qpos_pol")
                saved_trajs[f"V2_RobotPose_ONNX__{cmeta['alias']}__ref"] = rv2.pop("qpos_ref")
                v1_16_res["walk4_subject1"] = rv2
            v1_16_rows_v2.append(rv2)
            if not reuse_oracle_v1:
                v1_16_rows_oracle.append(rollout_clip(env, clip_idx=i, est_mode="oracle", modality="robot_pose", pt_policy=pi_sa, deploy_safety=False))
    v1_16_res["stat_fit_squat"] = ho_results["V2_RobotPose_ONNX"]["per_clip"]["stat_fit_squat"]
    v1_16_res["V2_RobotPose_ONNX"] = aggregate(v1_16_rows_v2)
    v1_16_res["Oracle_scheme_a_0ms"] = reuse_m["v1_16_match"]["Oracle_scheme_a_0ms"] if reuse_oracle_v1 else aggregate(v1_16_rows_oracle)
    print(f"[V1-16 MATCH] V2_RobotPose_ONNX surv={v1_16_res['V2_RobotPose_ONNX']['survival_rate']*100:.1f}% ret={v1_16_res['V2_RobotPose_ONNX']['norm_return']:.3f} | stat_fit_squat={v1_16_res['stat_fit_squat']['completed']} ({v1_16_res['stat_fit_squat']['steps_run']}/{v1_16_res['stat_fit_squat']['budget']}) | walk4={v1_16_res['walk4_subject1']['completed']}", flush=True)

    # 3. 20-Clip Held-Out Perturbation Stress Test (0.55x friction + 25% jitter + 1.5deg yaw + 0.03rad/s noise)
    stress_indices = [
        i for i in range(ho_offset, len(all_eval_clips))
        if all_eval_clips[i].get("v1_clip") or all_eval_clips[i]["source"] in ("lafan1", "synthetic")
    ]
    for i in range(ho_offset, len(all_eval_clips)):
        if len(stress_indices) >= 20:
            break
        if i not in stress_indices:
            stress_indices.append(i)
    mj_model = _find_mj_model(env)
    orig_fric = mj_model.geom_friction.copy()
    mj_model.geom_friction[:, 0] *= 0.55
    stress_results = {}
    try:
        for cname, kw in ho_specs[1:]:  # Zeroed, V1_H002, V2_RobotPose, V2_HumanSMPL, V2_VR3Pt, V2_PlannerCmd
            if reuse_m is not None and cname in baseline_names and cname in reuse_m.get("held_out_20_stress", {}):
                stress_results[cname] = reuse_m["held_out_20_stress"][cname]
                sa = stress_results[cname]
                print(f"[HO-20 STR] {cname:22s}: surv={sa['survival_rate']*100:5.1f}% ret={sa['norm_return']:.3f} body={sa['body_pos_rmse_cm']:5.2f}cm rootXY={sa['root_xy_rmse_cm']:5.1f}cm slip={sa['stance_slip_mm_s']:5.1f}mm/s pdOOB={sa['pd_oob_pct']:.3f}% (reused)", flush=True)
                continue
            srows = [rollout_clip(env, clip_idx=ci, stress=True, **kw) for ci in stress_indices]
            stress_results[cname] = aggregate(srows)
            sa = stress_results[cname]
            print(f"[HO-20 STR] {cname:22s}: surv={sa['survival_rate']*100:5.1f}% ret={sa['norm_return']:.3f} body={sa['body_pos_rmse_cm']:5.2f}cm rootXY={sa['root_xy_rmse_cm']:5.1f}cm slip={sa['stance_slip_mm_s']:5.1f}mm/s pdOOB={sa['pd_oob_pct']:.3f}%", flush=True)
    finally:
        mj_model.geom_friction[:] = orig_fric

    if reuse_m is not None and "V2_RobotPose_ONNX" in reuse_m.get("held_out_20_stress", {}):
        stress_results["Ablation_WM_UnfrozenDec"] = reuse_m["held_out_20_stress"]["V2_RobotPose_ONNX"]
        sa = stress_results["Ablation_WM_UnfrozenDec"]
        print(f"[HO-20 STR] {'Ablation_WM_UnfrozenDec':22s}: surv={sa['survival_rate']*100:5.1f}% ret={sa['norm_return']:.3f} (v2a ablation)", flush=True)

    # Save trajectories for visual rendering
    np.savez_compressed(base / "v2_sample_trajs.npz", **saved_trajs)

    # 4. Zero-IK Raw Human BVH Test (8 clips)
    zero_ik = eval_zero_ik_bvh(env, stand_clip_idx, tok_smpl, head_sess)

    # 5. 24-Second Continuous SONIC Mode-Switch & Watchdog Test
    smpl_bvh = Path("/home/yu/datasets/heldout_raw_bvh/soma_stat_3_eating_apple_R_002__A276.bvh")
    modeswitch = run_24s_sonic_modeswitch(
        env, stand_clip_idx, vr_clip_idx, onnx_dir, smpl_bvh,
        base / "sonic_modeswitch_24s.npz",
        base / "sonic_modeswitch_24s.json",
    )

    full_metrics = {
        "v1_16_match": v1_16_res,
        "held_out_50_nominal": ho_results,
        "held_out_20_stress": stress_results,
        "zero_ik_raw_bvh_8": zero_ik,
        "sonic_modeswitch_24s": {k: v for k, v in modeswitch.items() if k != "trace_sampled"},
    }
    (base / "v2_sonic_metrics.json").write_text(json.dumps(full_metrics, indent=2) + "\n", encoding="utf-8")
    print("EVAL_V2_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
