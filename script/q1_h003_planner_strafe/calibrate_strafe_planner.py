"""Calibrate Q1 PlannerCmd ONNX tokenizer for smooth symmetric lateral strafe (±vy) while freezing V2b TokenHead."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_strafe_suite import DEPLOY_ROOT, HEAD_ONNX_PATH, ROBOT_XML_PATH, ROOT, SIM_XML, evaluate_all

for p in (
    ROOT,
    ROOT / "src",
    ROOT / "runs/h002_sonic_v2",
    ROOT / "runs/h003_deploy_hybrid",
    DEPLOY_ROOT / "code",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from multimodal_features import build_planner_cmd_goal, build_robot_pose_goal
from q1_cgfa_obs_np import ContactGatedFootAnchorNP, LivePlannerControllerNP, build_obs_80_np, build_planner_cmd_single_10d_np
from q1_kinematics_np import Q1KinematicsNP, yaw_angle_from_mat
from q1_v2b_runtime import apply_torque_safe_pd_np
from train_h002_sonic_v2 import ModalityTokenizer, _sample_obs_histories
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import yaw_mat
from unisupermimic.rl.algos.supermimic.networks import Policy, run_mlp
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import GOAL_FRAME_DIM, MDP, NUM_J
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

BASE_ANCHORS_NPZ = ROOT / "runs/h003_v2b_planner_cal/planner_cem_anchors_true_cgfa.npz"
V2B_BUNDLE_PT = ROOT / "runs/h002_sonic_v2/checkpoints/h002_sonic_v2_bundle.pt"
BASE_TEACHER_PT = ROOT / "checkpoints/q1_base_soma1k.pt"
DATASET_250_MANIFEST = ROOT / "runs/h002_sonic_v2/dataset_250_manifest.json"
DATASET_2000_DIR = ROOT / "src/unisupermimic/assets/motions/q1/multidataset_2000"
DEPLOY_CFG_PATH = DEPLOY_ROOT / "config/q1_deploy_config.json"

PRESERVED_SPECS: list[tuple[str, float, float, float, float]] = [
    ("stand_0", 0.00, 0.00, 0.00, 1.00),
    ("walk_f018", 0.18, 0.00, 0.00, 1.00),
    ("walk_f035", 0.35, 0.00, 0.00, 1.00),
    ("walk_f048", 0.48, 0.00, 0.00, 1.00),
    ("walk_b020", -0.20, 0.00, 0.00, 1.00),
    ("turn_l050", 0.00, 0.00, 0.50, 1.00),
    ("turn_r050", 0.00, 0.00, -0.50, 1.00),
    ("walk_turn_l", 0.30, 0.00, 0.40, 1.00),
    ("walk_turn_r", 0.30, 0.00, -0.40, 1.00),
    ("crouch_088", 0.00, 0.00, 0.00, 0.88),
    ("crouch_080", 0.00, 0.00, 0.00, 0.80),
]

LATERAL_SPECS: list[tuple[str, float]] = [
    ("strafe_l010", 0.10),
    ("strafe_r010", -0.10),
    ("strafe_l005", 0.05),
    ("strafe_r005", -0.05),
    ("strafe_l015", 0.15),
    ("strafe_r015", -0.15),
    ("strafe_l020", 0.20),
    ("strafe_r020", -0.20),
]

_W_M = None
_W_HEAD = None
_W_KIN = None
_W_CFG = None
_W_LA_BID = -1
_W_RA_BID = -1
_W_PERT_Q0 = None
_W_U_STAND = None


def _worker_init() -> None:
    global _W_M, _W_HEAD, _W_KIN, _W_CFG, _W_LA_BID, _W_RA_BID, _W_PERT_Q0, _W_U_STAND
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _W_M = mujoco.MjModel.from_xml_path(str(SIM_XML))
    _W_M.opt.timestep = 0.002
    _W_LA_BID = mujoco.mj_name2id(_W_M, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    _W_RA_BID = mujoco.mj_name2id(_W_M, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    _W_HEAD = ort.InferenceSession(str(HEAD_ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    _W_KIN = Q1KinematicsNP(ROBOT_XML_PATH)

    cfg = json.loads(DEPLOY_CFG_PATH.read_text(encoding="utf-8"))
    dq = np.asarray(cfg["default_q"], dtype=np.float32)
    q_lo = np.asarray(cfg["q_lo"], dtype=np.float32)
    q_hi = np.asarray(cfg["q_hi"], dtype=np.float32)
    _W_CFG = {
        "default_q": dq,
        "ascale": np.asarray(cfg["action_scale"], dtype=np.float32),
        "kp": np.asarray(cfg["kp"], dtype=np.float32),
        "kd": np.asarray(cfg["kd"], dtype=np.float32),
        "eff": np.asarray(cfg["effort_limit_nm"], dtype=np.float32),
        "q_lo": q_lo,
        "q_hi": q_hi,
    }
    rng = np.random.default_rng(77)
    _W_PERT_Q0 = np.clip(dq + rng.normal(0.0, 0.05, size=22).astype(np.float32), q_lo + 0.02, q_hi - 0.02)
    base_anch = dict(np.load(BASE_ANCHORS_NPZ))
    _W_U_STAND = np.arctanh(np.clip(base_anch["stand_0"].astype(np.float32), -0.995, 0.995))


def _rollout_2ms_worker(
    z_vec: np.ndarray,
    vy: float,
    *,
    t_cmd_start: int,
    t_cmd_end: int,
    total_steps: int,
    q0_custom: np.ndarray | None,
    delay_10ms: bool,
) -> tuple[float, dict[str, float]]:
    m = _W_M
    c = _W_CFG
    dq, asc, kp, kd, eff, q_lo, q_hi = (
        c["default_q"],
        c["ascale"],
        c["kp"],
        c["kd"],
        c["eff"],
        c["q_lo"],
        c["q_hi"],
    )

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.4213]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    q_start = dq if q0_custom is None else q0_custom
    d.qpos[7:29] = q_start
    d.ctrl[:] = q_start
    mujoco.mj_forward(m, d)
    for _ in range(100):  # 0.2s settle
        mujoco.mj_step(m, d)

    q_init = d.qpos[7:29].astype(np.float32).copy()
    qd_init = d.qvel[6:28].astype(np.float32).copy()
    R_init = d.xmat[1].reshape(3, 3).astype(np.float32).copy()
    w_init = d.qvel[3:6].astype(np.float32).copy()

    cgfa = ContactGatedFootAnchorNP(_W_KIN, dt=0.02)
    planner = LivePlannerControllerNP(dt=0.02, num_frames=10, smooth_slew=True)
    cgfa.reset(R_init, w_init, q_init, qd_init, np.zeros(2, dtype=np.float32), R_ref=R_init)
    planner.reset(yaw0_rad=yaw_angle_from_mat(R_init))

    u_cand = np.arctanh(np.clip(z_vec.astype(np.float32), -0.995, 0.995))
    vy_abs = max(abs(vy), 1e-4)
    hist: np.ndarray | None = None
    a_prev = np.zeros(22, dtype=np.float32)
    prev_a = np.zeros(22, dtype=np.float32)
    last_tgt = q_init.copy()
    delayed_state = (q_init, qd_init, R_init, w_init)

    pos_xy, fws, rolls, pitches, slips, act_rates, arm_pens, lz, rz = [], [], [], [], [], [], [], [], []
    for t in range(total_steps):
        q_obs, qd_obs, R_obs, w_obs = delayed_state if delay_10ms else (
            d.qpos[7:29].astype(np.float32),
            d.qvel[6:28].astype(np.float32),
            d.xmat[1].reshape(3, 3).astype(np.float32),
            d.qvel[3:6].astype(np.float32),
        )
        vy_cmd = vy if (t_cmd_start <= t < t_cmd_end) else 0.0
        _, p_ref_xy, R_ref = planner.push_and_build(0.0, vy_cmd, 0.0, 1.0, 0.04, cgfa=cgfa, R_imu_aligned=R_obs)
        exy, _ = cgfa.step(R_obs, w_obs, q_obs, qd_obs, p_ref_xy, R_ref=R_ref)
        obs80 = build_obs_80_np(
            q=q_obs, qd=qd_obs, default_q=dq, R_imu=R_obs, w_body=w_obs, a_prev=a_prev, R_ref=R_ref, cgfa_exy=exy
        ).reshape(1, 1, 80)
        hist = np.repeat(obs80, 10, axis=1) if hist is None else np.concatenate((hist[:, 1:], obs80), axis=1)

        alpha = min(1.0, abs(float(planner.cur_cmd[1])) / vy_abs)
        z_in = np.tanh((1.0 - alpha) * _W_U_STAND + alpha * u_cand).astype(np.float32).reshape(1, 32)
        act = _W_HEAD.run(["action"], {"obs_hist": hist, "token": z_in})[0][0]
        q_safe = apply_torque_safe_pd_np(
            dq + asc * act, q_obs, qd_obs, q_lo, q_hi, kp, kd, eff, q_init=q_init, step_i=t, engage_steps=5
        )
        q_safe = np.clip(q_safe, last_tgt - 0.2, last_tgt + 0.2)
        q_safe = np.clip(q_safe, dq - 2.0, dq + 2.0)
        last_tgt = q_safe.copy()
        a_sent = ((q_safe - dq) / asc).astype(np.float32)
        if t > 0:
            act_rates.append(float(np.mean(np.abs(a_sent - prev_a))))
        prev_a = a_sent.copy()
        a_prev = a_sent.copy()

        d.ctrl[:] = q_safe
        for sub in range(10):
            mujoco.mj_step(m, d)
            if delay_10ms and sub == 4:
                delayed_state = (
                    d.qpos[7:29].astype(np.float32).copy(),
                    d.qvel[6:28].astype(np.float32).copy(),
                    d.xmat[1].reshape(3, 3).astype(np.float32).copy(),
                    d.qvel[3:6].astype(np.float32).copy(),
                )

        Rp = d.xmat[1].reshape(3, 3)
        if d.qpos[2] < 0.24 or Rp[2, 2] < 0.65:
            return (-80.0 + 0.1 * t, {"fell_step": t})

        pos_xy.append(d.qpos[0:2].copy())
        q_post = d.qpos[7:29]
        yaw_c = math.atan2(float(Rp[1, 0]), float(Rp[0, 0]))
        ch, sh = math.cos(yaw_c), math.sin(yaw_c)
        dp_lr = d.xpos[_W_LA_BID] - d.xpos[_W_RA_BID]
        fws.append(abs(-sh * float(dp_lr[0]) + ch * float(dp_lr[1])) * 100.0)
        rolls.append(math.degrees(math.asin(max(-1.0, min(1.0, float(Rp[2, 1]))))))
        pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, -float(Rp[2, 0]))))))
        lz.append(float(d.xpos[_W_LA_BID, 2]))
        rz.append(float(d.xpos[_W_RA_BID, 2]))

        b_st = _W_LA_BID if d.xpos[_W_LA_BID, 2] < d.xpos[_W_RA_BID, 2] else _W_RA_BID
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b_st, vel6, 0)
        Rf = d.xmat[b_st].reshape(3, 3)
        vs = vel6[3:6] + np.cross(vel6[0:3], Rf @ np.array([0.020, 0.0, -0.022]))
        slips.append(float(np.linalg.norm(vs[:2]) * 1000.0))

        j_breach = float(np.max(np.maximum((q_lo + 0.02) - q_post, q_post - (q_hi - 0.02))))
        arm_pens.append(
            float(
                abs(q_post[17] - q_post[21])
                + abs(q_post[14] - q_post[18])
                + 0.3 * (abs(q_post[14]) + abs(q_post[18]))
                + max(0.0, j_breach) * 40.0
            )
        )

    pos_arr = np.array(pos_xy)
    dx = float(pos_arr[t_cmd_end - 1, 0] - pos_arr[t_cmd_start - 1, 0])
    dy = float(pos_arr[t_cmd_end - 1, 1] - pos_arr[t_cmd_start - 1, 1])
    stop_drift_m = float(np.linalg.norm(pos_arr[total_steps - 1] - pos_arr[total_steps - 41]))

    Rp = d.xmat[1].reshape(3, 3)
    yaw_err_rad = abs(math.atan2(float(Rp[1, 0]), float(Rp[0, 0])))
    sl_active = slice(t_cmd_start, total_steps)
    fw_mean_cm = float(np.mean(fws[sl_active]))
    fw_min_cm = float(np.min(fws[sl_active]))
    roll_std_deg = float(np.std(rolls[sl_active]))
    roll_max_deg = float(np.max(np.abs(rolls[sl_active])))
    pitch_std_deg = float(np.std(pitches[sl_active]))
    mean_ar = float(np.mean(act_rates[max(0, t_cmd_start - 1) :]))
    mean_arm_pen = float(np.mean(arm_pens[sl_active]))
    mean_slip_mm_s = float(np.mean(slips[sl_active]))
    step_switches = int(np.sum(np.diff((np.array(lz[t_cmd_start:t_cmd_end]) > np.array(rz[t_cmd_start:t_cmd_end])).astype(np.int32)) != 0))

    sgn = 1.0 if vy > 0 else -1.0
    target_dy = vy * ((t_cmd_end - t_cmd_start) * 0.02)
    dy_prog = sgn * dy
    score = -(
        abs(dy - target_dy) * 7.0
        + max(0.0, 0.50 * abs(target_dy) - dy_prog) * 12.0
        + abs(dx) * 6.0
        + abs(yaw_err_rad) * 4.5
        + stop_drift_m * 8.0
        + max(0.0, 11.5 - fw_min_cm) * 0.95
        + max(0.0, fw_mean_cm - 21.5) * 0.35
        + max(0.0, 14.0 - fw_mean_cm) * 0.25
        + roll_std_deg * 0.25
        + max(0.0, roll_max_deg - 8.0) * 0.60
        + max(0.0, pitch_std_deg - 2.3) * 0.80
        + max(0.0, mean_ar - 0.15) * 4.0
        + mean_ar * 0.30
        + mean_arm_pen * 0.25
        + max(0.0, mean_slip_mm_s - 120.0) * 0.015
        + max(0, 4 - step_switches) * 0.35
    )
    return (
        score,
        {
            "dx": dx,
            "dy": dy,
            "stop_drift_m": stop_drift_m,
            "fw_mean": fw_mean_cm,
            "fw_min": fw_min_cm,
            "roll_std": roll_std_deg,
            "roll_max": roll_max_deg,
            "yaw_deg": math.degrees(math.atan2(float(Rp[1, 0]), float(Rp[0, 0]))),
            "step_switches": float(step_switches),
        },
    )


def _eval_strafe_candidate(args: tuple[np.ndarray, float]) -> tuple[float, dict[str, float]]:
    z_vec, vy = args
    score_nom, info_nom = _rollout_2ms_worker(
        z_vec, vy, t_cmd_start=30, t_cmd_end=230, total_steps=300, q0_custom=None, delay_10ms=False
    )
    if "fell_step" in info_nom:
        return (score_nom - 30.0, {"score_nom": score_nom, "score_pert": -80.0, **info_nom})
    score_pert, info_pert = _rollout_2ms_worker(
        z_vec, vy, t_cmd_start=25, t_cmd_end=175, total_steps=225, q0_custom=_W_PERT_Q0, delay_10ms=True
    )
    comb = 0.65 * score_nom + 0.35 * score_pert
    return (comb, {"score_nom": score_nom, "score_pert": score_pert, **info_nom})


def _build_seed_bank(
    base_anch: dict[str, np.ndarray],
    pi: Policy,
    kin_cpu: Q1Kinematics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode lateral motion clips and existing tokens into 32D seed pools."""
    m250 = json.loads(DATASET_250_MANIFEST.read_text(encoding="utf-8"))
    train_files = sorted(str(ROOT / r["csv_rel"]) for r in m250["clips"] if r["split"] == "train")
    mot250 = MotionLib(train_files, kin_cpu, "cpu", src_fps=30.0, ctrl_dt=0.02)

    with torch.no_grad():
        N = int(mot250.num_frames)
        prev = torch.clamp(torch.arange(N) - 1, min=mot250.clip_start[mot250.clip_id])
        v_w = (mot250.root_p - mot250.root_p[prev]) / 0.02
        h_t = yaw_mat(mot250.root_R).transpose(-1, -2)
        v_h = (h_t @ v_w.unsqueeze(-1)).squeeze(-1)

        l_mask = (v_h[:, 1] > 0.08) & (v_h[:, 0].abs() < 0.15) & (mot250.root_p[:, 2] > 0.36)
        r_mask = (v_h[:, 1] < -0.08) & (v_h[:, 0].abs() < 0.15) & (mot250.root_p[:, 2] > 0.36)
        l_idx = torch.nonzero(l_mask, as_tuple=False).squeeze(-1)
        r_idx = torch.nonzero(r_mask, as_tuple=False).squeeze(-1)
        if len(l_idx) > 120:
            l_idx = l_idx[torch.linspace(0, len(l_idx) - 1, steps=120).long()]
        if len(r_idx) > 120:
            r_idx = r_idx[torch.linspace(0, len(r_idx) - 1, steps=120).long()]

        z_l250 = torch.tanh(run_mlp(pi.enc, pi.goal_norm(build_robot_pose_goal(mot250, l_idx, 10)))).numpy()
        z_r250 = torch.tanh(run_mlp(pi.enc, pi.goal_norm(build_robot_pose_goal(mot250, r_idx, 10)))).numpy()

        ss_files = sorted(str(p) for p in DATASET_2000_DIR.rglob("*sidestep*.csv"))
        mot_ss = MotionLib(ss_files, kin_cpu, "cpu", src_fps=30.0, ctrl_dt=0.02)
        n_ss = int(mot_ss.num_frames)
        sel_ss = torch.linspace(0, n_ss - 1, steps=min(120, n_ss)).long()
        z_ss = torch.tanh(run_mlp(pi.enc, pi.goal_norm(build_robot_pose_goal(mot_ss, sel_ss, 10)))).numpy()

    tok_cal = ort.InferenceSession(str(DEPLOY_ROOT / "onnx/q1_tok_planner_cmd.onnx"), providers=["CPUExecutionProvider"])
    tok_bak = ort.InferenceSession(str(DEPLOY_ROOT / "onnx/q1_tok_planner_cmd_uncalibrated_bak.onnx"), providers=["CPUExecutionProvider"])
    tok_seeds = []
    for vy in (-0.20, -0.15, -0.12, -0.10, -0.08, -0.05, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
        m_in = np.tile(build_planner_cmd_single_10d_np(0.0, vy, 0.0, 1.0, 0.04), (1, 10))
        tok_seeds.append(tok_cal.run(["token"], {"modal_input": m_in})[0][0])
        tok_seeds.append(tok_bak.run(["token"], {"modal_input": m_in})[0][0])
    z_tok = np.stack(tok_seeds, axis=0)
    z_base = np.stack([base_anch["stand_0"], base_anch["strafe_l020"], base_anch["strafe_r020"]], axis=0)

    z_left_pool = np.vstack([z_base, z_tok, z_l250, z_ss])
    z_right_pool = np.vstack([z_base, z_tok, z_r250, z_ss])
    return z_left_pool.astype(np.float32), z_right_pool.astype(np.float32), z_tok.astype(np.float32)


def run_stage1_cem(
    out_dir: Path,
    pi: Policy,
    kin_cpu: Q1Kinematics,
    *,
    init_anchors_npz: Path | None = None,
    refine_keys: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_anch = {k: v.astype(np.float32) for k, v in dict(np.load(BASE_ANCHORS_NPZ)).items()}
    init_anch = (
        {k: v.astype(np.float32) for k, v in dict(np.load(init_anchors_npz)).items()}
        if init_anchors_npz is not None
        else {}
    )
    need_bank = (refine_keys is None) or any(k in refine_keys for k, _ in LATERAL_SPECS)
    z_left_pool, z_right_pool, _ = _build_seed_bank(base_anch, pi, kin_cpu) if need_bank else (None, None, None)
    if need_bank:
        print(f"[Stage 1] Seed bank ready: left={len(z_left_pool)}, right={len(z_right_pool)}", flush=True)

    anchors = {k: base_anch[k].copy() for k, _, _, _, _ in PRESERVED_SPECS}
    lateral_meta: dict[str, Any] = {}
    solved_left: list[np.ndarray] = []
    solved_right: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=18, initializer=_worker_init) as pool:
        for key, vy in LATERAL_SPECS:
            ts = time.perf_counter()
            if refine_keys is not None and key not in refine_keys:
                assert key in init_anch, f"--refine-keys omitted {key}, which is missing from --init-anchors-npz"
                best_z = init_anch[key].copy()
                best_r = _eval_strafe_candidate_pool(pool, best_z, vy)
                anchors[key] = best_z
                if vy > 0:
                    solved_left.append(best_z)
                else:
                    solved_right.append(best_z)
                info = best_r[1]
                lateral_meta[key] = {
                    "vy": vy,
                    "kept_from_init": True,
                    "combined_score": round(float(best_r[0]), 3),
                    "score_nom": round(float(info.get("score_nom", -99.0)), 3),
                    "score_pert": round(float(info.get("score_pert", -99.0)), 3),
                    "dx_m": round(float(info.get("dx", 0.0)), 4),
                    "dy_m": round(float(info.get("dy", 0.0)), 4),
                    "stop_drift_m": round(float(info.get("stop_drift_m", 0.0)), 4),
                    "fw_mean_cm": round(float(info.get("fw_mean", 0.0)), 2),
                    "fw_min_cm": round(float(info.get("fw_min", 0.0)), 2),
                    "roll_std_deg": round(float(info.get("roll_std", 0.0)), 2),
                    "roll_max_deg": round(float(info.get("roll_max", 0.0)), 2),
                    "yaw_deg": round(float(info.get("yaw_deg", 0.0)), 2),
                    "step_switches": int(round(float(info.get("step_switches", 0)))),
                }
                print(
                    f"[KEPT {key:11s}] comb={best_r[0]:+6.2f} nom={info.get('score_nom',-99):+6.2f} "
                    f"pert={info.get('score_pert',-99):+6.2f} dx={info.get('dx',0):+.3f} dy={info.get('dy',0):+.3f} "
                    f"stop_d={info.get('stop_drift_m',0):.3f} fw={info.get('fw_mean',0):.1f}(min {info.get('fw_min',0):.1f}) "
                    f"roll_max={info.get('roll_max',0):.1f} sw={int(info.get('step_switches',0))} ({time.perf_counter()-ts:.1f}s)",
                    flush=True,
                )
                continue

            side_pool = z_left_pool if vy > 0 else z_right_pool
            extra: list[np.ndarray] = list(solved_left if vy > 0 else solved_right)
            if init_anch:
                prefix = "strafe_l" if vy > 0 else "strafe_r"
                for ik, iv in init_anch.items():
                    if ik.startswith(prefix):
                        extra.append(iv)
            seeds = np.vstack([side_pool] + [e[None, :] for e in extra]) if extra else side_pool

            seed_res = list(pool.map(_eval_strafe_candidate, [(seeds[i], vy) for i in range(len(seeds))]))
            seed_scores = np.array([r[0] for r in seed_res])
            order = np.argsort(seed_scores)[::-1]
            elites = seeds[order[:10]]
            u_el0 = np.arctanh(np.clip(elites, -0.995, 0.995))
            u_mean = u_el0.mean(axis=0)
            u_std = np.clip(u_el0.std(axis=0), 0.14, 0.36)

            best_r = seed_res[int(order[0])]
            best_z = elites[0].copy()
            rng = np.random.default_rng(202611 + abs(int(round(vy * 100))) + (1000 if vy > 0 else 2000))

            for _ in range(5):
                u_pop = u_mean[None, :] + rng.normal(0.0, 1.0, size=(64, 32)).astype(np.float32) * u_std[None, :]
                z_pop = np.tanh(u_pop).astype(np.float32)
                z_pop[0] = best_z
                pop_res = list(pool.map(_eval_strafe_candidate, [(z_pop[i], vy) for i in range(64)]))
                pop_s = np.array([r[0] for r in pop_res])
                idx = np.argsort(pop_s)[::-1][:10]
                if pop_res[int(idx[0])][0] > best_r[0]:
                    best_r = pop_res[int(idx[0])]
                    best_z = z_pop[int(idx[0])].copy()
                u_el = np.arctanh(np.clip(z_pop[idx], -0.995, 0.995))
                u_mean = 0.65 * u_el.mean(axis=0) + 0.35 * u_mean
                u_std = np.clip(0.65 * u_el.std(axis=0) + 0.35 * u_std, 0.03, 0.38)

            anchors[key] = best_z
            if vy > 0:
                solved_left.append(best_z)
            else:
                solved_right.append(best_z)

            info = best_r[1]
            lateral_meta[key] = {
                "vy": vy,
                "kept_from_init": False,
                "combined_score": round(float(best_r[0]), 3),
                "score_nom": round(float(info.get("score_nom", -99.0)), 3),
                "score_pert": round(float(info.get("score_pert", -99.0)), 3),
                "dx_m": round(float(info.get("dx", 0.0)), 4),
                "dy_m": round(float(info.get("dy", 0.0)), 4),
                "stop_drift_m": round(float(info.get("stop_drift_m", 0.0)), 4),
                "fw_mean_cm": round(float(info.get("fw_mean", 0.0)), 2),
                "fw_min_cm": round(float(info.get("fw_min", 0.0)), 2),
                "roll_std_deg": round(float(info.get("roll_std", 0.0)), 2),
                "roll_max_deg": round(float(info.get("roll_max", 0.0)), 2),
                "yaw_deg": round(float(info.get("yaw_deg", 0.0)), 2),
                "step_switches": int(round(float(info.get("step_switches", 0)))),
            }
            print(
                f"[CEM {key:12s}] comb={best_r[0]:+6.2f} nom={info.get('score_nom',-99):+6.2f} "
                f"pert={info.get('score_pert',-99):+6.2f} dx={info.get('dx',0):+.3f} dy={info.get('dy',0):+.3f} "
                f"stop_d={info.get('stop_drift_m',0):.3f} fw={info.get('fw_mean',0):.1f}(min {info.get('fw_min',0):.1f}) "
                f"roll_max={info.get('roll_max',0):.1f} sw={int(info.get('step_switches',0))} ({time.perf_counter()-ts:.1f}s)",
                flush=True,
            )

    np.savez(out_dir / "anchors_strafe_2ms.npz", **anchors)
    (out_dir / "anchors_strafe_2ms.json").write_text(json.dumps(lateral_meta, indent=2), encoding="utf-8")
    return anchors, lateral_meta


def _eval_strafe_candidate_pool(pool: ProcessPoolExecutor, z_vec: np.ndarray, vy: float) -> tuple[float, dict[str, float]]:
    return list(pool.map(_eval_strafe_candidate, [(z_vec, vy)]))[0]


def _plot_training_curves(csv_path: Path, png_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    steps = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=140)
    specs = [
        ("loss", "Total Loss"),
        ("joy_z_mse", "Joystick Token MSE"),
        ("strafe_z_mse", "Pure Strafe Token MSE"),
        ("anc_z_mse", "Exact Anchor Token MSE"),
        ("joy_act_mse", "TokenHead Action Distill MSE"),
        ("moc_cos", "Mocap Cosine Similarity"),
    ]
    for ax, (col, title) in zip(axes.flat, specs):
        vals = [float(r[col]) for r in rows]
        ax.plot(steps, vals, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
    fig.suptitle("Q1 PlannerCmd Lateral Strafe Tokenizer Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def run_stage2_train_and_export(
    out_dir: Path,
    anchors: dict[str, np.ndarray],
    pi: Policy,
    bundle: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi = pi.to(device).eval()
    for p in pi.parameters():
        p.requires_grad_(False)

    base_ck = torch.load(BASE_TEACHER_PT, map_location="cpu", weights_only=True)
    teacher = Policy(80, GOAL_FRAME_DIM * 10, num_joints=NUM_J, hist=10, z_dim=32, hidden=tuple(bundle["hidden"])).to(device)
    teacher.load_state_dict(base_ck["pi"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    m250 = json.loads(DATASET_250_MANIFEST.read_text(encoding="utf-8"))
    train_files = sorted(str(ROOT / r["csv_rel"]) for r in m250["clips"] if r["split"] == "train")
    kin_gpu = Q1Kinematics(str(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"), device)
    motion = MotionLib(train_files, kin_gpu, device, src_fps=30.0, ctrl_dt=0.02)
    dq_t = torch.tensor(json.loads(DEPLOY_CFG_PATH.read_text(encoding="utf-8"))["default_q"], device=device, dtype=torch.float32)
    mdp = MDP(kin_gpu, motion, dq_t, observation_mode="root_xy")
    n_tot = int(motion.num_frames)

    pts_cfg: list[tuple[str, float, float, float, float, np.ndarray]] = []
    for k, vx, vy, wz, zr in PRESERVED_SPECS:
        pts_cfg.append((k, vx, vy, wz, zr, anchors[k]))
    for k, vy in LATERAL_SPECS:
        pts_cfg.append((k, 0.0, vy, 0.0, 1.00, anchors[k]))

    A_x = torch.tensor([[p[1] / 0.085, p[2] / 0.035, p[3] / 0.22, (p[4] - 1.0) / 0.07] for p in pts_cfg], dtype=torch.float32, device=device)
    A_u = torch.tensor(np.arctanh(np.clip(np.stack([p[5] for p in pts_cfg], axis=0), -0.995, 0.995)), dtype=torch.float32, device=device)

    # Verify RBF sharpness on stand_0 and all 8 lateral anchors
    with torch.no_grad():
        for i, (name, vx, vy, wz, zr, _) in enumerate(pts_cfg):
            if name == "stand_0" or name.startswith("strafe_"):
                q4 = torch.tensor([[vx / 0.085, vy / 0.035, wz / 0.22, (zr - 1.0) / 0.07]], dtype=torch.float32, device=device)
                d2 = ((q4.unsqueeze(1) - A_x.unsqueeze(0)) ** 2).sum(dim=-1)
                w = torch.softmax(-2.5 * d2, dim=-1)[0]
                assert float(w[i]) > 0.985, f"RBF peak too wide at {name}: w={float(w[i]):.5f}"
    print("[Stage 2] Verified RBF anchor weights > 0.985 for stand_0 and all 8 lateral anchors", flush=True)

    def interp_joy_u(vx_t: torch.Tensor, vy_t: torch.Tensor, wz_t: torch.Tensor, zr_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q4 = torch.stack((vx_t / 0.085, vy_t / 0.035, wz_t / 0.22, (zr_t - 1.0) / 0.07), dim=-1)
        d2 = ((q4.unsqueeze(1) - A_x.unsqueeze(0)) ** 2).sum(dim=-1)
        w = torch.softmax(-2.5 * d2, dim=-1)
        u = w @ A_u
        return u, torch.tanh(u)

    def _build_100d_seq(vx_h: torch.Tensor, vy_h: torch.Tensor, wz_h: torch.Tensor, zr_h: torch.Tensor, rx_noise: torch.Tensor) -> torch.Tensor:
        sp = torch.sqrt(vx_h * vx_h + vy_h * vy_h + (0.25 * wz_h) ** 2)
        sf = torch.sigmoid((0.12 - sp) / 0.03)
        cf = torch.sigmoid((0.88 - zr_h) / 0.04)
        rx = torch.full_like(cf, 0.04) + rx_noise
        return torch.stack((vx_h, vy_h, wz_h, zr_h, 0.10 * cf, torch.zeros_like(cf), sf, rx, rx, cf), dim=-1).reshape(vx_h.shape[0], 100)

    mode_probs = torch.tensor([0.10, 0.18, 0.14, 0.10, 0.22, 0.10, 0.16], device=device)
    strafe_targets = torch.tensor([-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20], device=device)

    def sample_joystick_batch(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = torch.multinomial(mode_probs, B, replacement=True)
        vx = torch.zeros(B, device=device)
        vy = torch.zeros(B, device=device)
        wz = torch.zeros(B, device=device)
        zr = torch.ones(B, device=device)

        m1 = mode == 1
        vx = torch.where(m1, torch.empty(B, device=device).uniform_(-0.24, 0.52), vx)

        m2 = mode == 2
        wz = torch.where(m2, torch.empty(B, device=device).uniform_(-0.65, 0.65), wz)

        m3 = mode == 3
        zr = torch.where(m3, torch.empty(B, device=device).uniform_(0.78, 1.01), zr)

        m4 = mode == 4  # Pure lateral strafe
        vy_unif = torch.empty(B, device=device).uniform_(-0.22, 0.22)
        pick = torch.randint(0, len(strafe_targets), (B,), device=device)
        vy_clust = strafe_targets[pick] + torch.randn(B, device=device) * 0.015
        vy_m4 = torch.where(torch.rand(B, device=device) < 0.5, vy_unif, vy_clust)
        vy = torch.where(m4, torch.clamp(vy_m4, -0.23, 0.23), vy)

        m5 = mode == 5  # Combined
        vx = torch.where(m5, torch.empty(B, device=device).uniform_(-0.15, 0.40), vx)
        vy = torch.where(m5, torch.empty(B, device=device).uniform_(-0.18, 0.18), vy)
        wz = torch.where(m5, torch.empty(B, device=device).uniform_(-0.50, 0.50), wz)

        m6 = mode == 6  # Smooth 10-frame slew ramps
        is_strafe_ramp = m6 & (torch.rand(B, device=device) < 0.40)
        is_gen_ramp = m6 & (~is_strafe_ramp)
        vx = torch.where(is_gen_ramp, torch.empty(B, device=device).uniform_(-0.20, 0.45), vx)
        wz = torch.where(is_gen_ramp, torch.empty(B, device=device).uniform_(-0.55, 0.55), wz)
        zr = torch.where(is_gen_ramp, torch.empty(B, device=device).uniform_(0.78, 1.00), zr)

        sr_sub = torch.randint(0, 3, (B,), device=device)
        vy_sr_sample = strafe_targets[torch.randint(0, len(strafe_targets), (B,), device=device)]
        is_sr_accel = is_strafe_ramp & (sr_sub == 0)
        is_sr_decel = is_strafe_ramp & (sr_sub == 1)
        is_sr_rev = is_strafe_ramp & (sr_sub == 2)

        vy = torch.where(is_sr_accel | is_sr_rev, vy_sr_sample, vy)
        vy = torch.where(is_sr_decel, torch.zeros_like(vy), vy)

        vx_prev = vx.clone()
        vy_prev = vy.clone()
        wz_prev = wz.clone()
        zr_prev = zr.clone()

        vx_prev = torch.where(is_gen_ramp, torch.empty(B, device=device).uniform_(-0.20, 0.45), vx_prev)
        wz_prev = torch.where(is_gen_ramp, torch.empty(B, device=device).uniform_(-0.55, 0.55), wz_prev)
        zr_prev = torch.where(is_gen_ramp, torch.empty(B, device=device).uniform_(0.78, 1.00), zr_prev)

        vy_prev = torch.where(is_sr_accel, torch.clamp(torch.zeros_like(vy), vy - 0.18, vy + 0.18), vy_prev)
        vy_prev = torch.where(is_sr_decel, torch.clamp(vy_sr_sample, -0.18, 0.18), vy_prev)
        vy_prev = torch.where(is_sr_rev, torch.clamp(-vy_sr_sample, vy_sr_sample - 0.18, vy_sr_sample + 0.18), vy_prev)

        alphas = torch.linspace(0.0, 1.0, steps=10, device=device).unsqueeze(0)
        vx_h = vx_prev.unsqueeze(1) * (1.0 - alphas) + vx.unsqueeze(1) * alphas
        vy_h = vy_prev.unsqueeze(1) * (1.0 - alphas) + vy.unsqueeze(1) * alphas
        wz_h = wz_prev.unsqueeze(1) * (1.0 - alphas) + wz.unsqueeze(1) * alphas
        zr_h = zr_prev.unsqueeze(1) * (1.0 - alphas) + zr.unsqueeze(1) * alphas

        rx_noise = 0.008 * torch.randn_like(vx_h) * m5.float().unsqueeze(1)
        feat = _build_100d_seq(vx_h, vy_h, wz_h, zr_h, rx_noise)
        u_tgt, z_tgt = interp_joy_u(vx, vy, wz, zr)
        is_pure_strafe = m4 | is_strafe_ramp
        return feat, u_tgt, z_tgt, is_pure_strafe

    # Exact-anchor supervision tensor (19 x 100)
    anc_feats = []
    for _, vx, vy, wz, zr, _ in pts_cfg:
        row10 = build_planner_cmd_single_10d_np(vx, vy, wz, zr, 0.04)
        anc_feats.append(np.tile(row10, 10))
    g_anc = torch.tensor(np.stack(anc_feats, axis=0), dtype=torch.float32, device=device)
    u_anc = A_u.clone()
    z_anc = torch.tanh(u_anc)

    tok = ModalityTokenizer(100, (512, 256, 256), z_dim=32).to(device)
    with torch.no_grad():
        x_joy, _, _, _ = sample_joystick_batch(16384)
        f_moc = torch.linspace(0, n_tot - 1, steps=8192, device=device).long()
        x_moc = build_planner_cmd_goal(motion, f_moc, 10)
        x_cal = torch.cat((x_joy, x_moc), dim=0).double()
        mx = x_cal.mean(dim=0)
        var_x = (x_cal.pow(2).mean(dim=0) - mx.pow(2)).clamp(min=1e-3)
        tok.in_norm.mean.copy_(mx.float())
        tok.in_norm.var.copy_(var_x.float())
        tok.in_norm.count.fill_(float(len(x_cal)))
    for p in tok.in_norm.parameters():
        p.requires_grad_(False)

    total_steps = 4500
    opt = torch.optim.AdamW(tok.enc.parameters(), lr=9e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5)

    csv_path = out_dir / "train_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["step", "loss", "joy_z_mse", "strafe_z_mse", "anc_z_mse", "joy_act_mse", "moc_cos", "lr"])

        for step in range(1, total_steps + 1):
            with torch.no_grad():
                g_joy, u_joy, z_joy, is_strafe = sample_joystick_batch(384)
                fb_joy = torch.randint(0, n_tot, (384,), device=device)
                obs_joy, _ = _sample_obs_histories(mdp, teacher, fb_joy, 10, obs_mode="cgfa", noise_scale=0.22)
                obs_joy_flat = pi.obs_norm(obs_joy).flatten(-2)
                a_joy_tgt = run_mlp(pi.dec, torch.cat((z_joy, obs_joy_flat), dim=-1))

                fb_moc = torch.randint(0, n_tot, (128,), device=device)
                g_moc = build_planner_cmd_goal(motion, fb_moc, 10)
                g_rob = build_robot_pose_goal(motion, fb_moc, 10, latest_offset=0)
                u_rob = run_mlp(pi.enc, pi.goal_norm(g_rob))
                z_rob = torch.tanh(u_rob)
                obs_moc, _ = _sample_obs_histories(mdp, teacher, fb_moc, 10, obs_mode="cgfa", noise_scale=0.18)
                obs_moc_flat = pi.obs_norm(obs_moc).flatten(-2)
                a_rob_tgt = run_mlp(pi.dec, torch.cat((z_rob, obs_moc_flat), dim=-1))

                fb_anc = torch.randint(0, n_tot, (len(pts_cfg),), device=device)
                obs_anc, _ = _sample_obs_histories(mdp, teacher, fb_anc, 10, obs_mode="cgfa", noise_scale=0.22)
                obs_anc_flat = pi.obs_norm(obs_anc).flatten(-2)
                a_anc_tgt = run_mlp(pi.dec, torch.cat((z_anc, obs_anc_flat), dim=-1))

            u_jm, z_jm = tok.encode_raw(g_joy)
            a_jm = run_mlp(pi.dec, torch.cat((z_jm, obs_joy_flat), dim=-1))
            joy_z_mse = F.mse_loss(z_jm, z_joy)
            joy_act_mse = F.mse_loss(a_jm, a_joy_tgt)
            loss_joy = joy_z_mse + 0.08 * F.mse_loss(u_jm, u_joy) + 1.5 * joy_act_mse

            u_mm, z_mm = tok.encode_raw(g_moc)
            a_mm = run_mlp(pi.dec, torch.cat((z_mm, obs_moc_flat), dim=-1))
            loss_moc = F.mse_loss(z_mm, z_rob) + 0.05 * F.mse_loss(u_mm, u_rob) + 1.2 * F.mse_loss(a_mm, a_rob_tgt)

            u_am, z_am = tok.encode_raw(g_anc)
            a_am = run_mlp(pi.dec, torch.cat((z_am, obs_anc_flat), dim=-1))
            anc_z_mse = F.mse_loss(z_am, z_anc)
            loss_anc = 3.0 * anc_z_mse + 0.30 * F.mse_loss(u_am, u_anc) + 1.5 * F.mse_loss(a_am, a_anc_tgt)

            loss = 0.60 * loss_joy + 0.12 * loss_moc + 0.28 * loss_anc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tok.enc.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % 50 == 0 or step == 1:
                strafe_mse = float(F.mse_loss(z_jm.detach()[is_strafe], z_joy[is_strafe])) if bool(is_strafe.any()) else 0.0
                moc_cos = float(F.cosine_similarity(z_mm.detach(), z_rob, dim=-1).mean())
                lr_cur = float(opt.param_groups[0]["lr"])
                writer.writerow([
                    step,
                    round(float(loss.item()), 6),
                    round(float(joy_z_mse.item()), 6),
                    round(strafe_mse, 6),
                    round(float(anc_z_mse.item()), 6),
                    round(float(joy_act_mse.item()), 6),
                    round(moc_cos, 5),
                    round(lr_cur, 7),
                ])
                if step in (1, 900, 1800, 2700, 3600, total_steps):
                    print(
                        f"[Train {step:4d}/{total_steps}] loss={loss.item():.5f} joy_z={joy_z_mse.item():.5f} "
                        f"strafe_z={strafe_mse:.5f} anc_z={anc_z_mse.item():.5f} moc_cos={moc_cos:.4f}",
                        flush=True,
                    )

    _plot_training_curves(csv_path, out_dir / "train_curve.png")

    tok_cpu = tok.cpu().eval()
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    new_onnx = onnx_dir / "q1_tok_planner_cmd.onnx"
    fp = str(bundle["codec_fingerprint"])
    torch.onnx.export(
        tok_cpu,
        (torch.zeros((1, 100), dtype=torch.float32),),
        str(new_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["modal_input"],
        output_names=["token"],
        dynamic_axes={"modal_input": {0: "batch"}, "token": {0: "batch"}},
        dynamo=False,
    )
    om = onnx.load(str(new_onnx))
    for k, v in {
        "codec_fingerprint": fp,
        "ori_mode": "heading",
        "ctrl_hz": "50",
        "lookahead_ms": "0",
        "modality": "planner_cmd",
    }.items():
        pr = om.metadata_props.add()
        pr.key = k
        pr.value = v
    onnx.save(om, str(new_onnx))
    onnx.checker.check_model(onnx.load(str(new_onnx)))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess_new = ort.InferenceSession(str(new_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    x_test = np.random.default_rng(0).normal(0.0, 0.5, size=(64, 100)).astype(np.float32)
    with torch.no_grad():
        y_pt = tok_cpu(torch.from_numpy(x_test)).numpy()
    y_ort = sess_new.run(["token"], {"modal_input": x_test})[0]
    max_diff = float(np.max(np.abs(y_pt - y_ort)))
    assert max_diff < 1e-5, f"ONNX parity error too large: {max_diff}"

    anchor_parity: dict[str, dict[str, float]] = {}
    check_keys = {"stand_0", "walk_f035"} | {k for k, _ in LATERAL_SPECS}
    for name, vx, vy, wz, zr, z_ref in pts_cfg:
        if name not in check_keys:
            continue
        m_in = np.tile(build_planner_cmd_single_10d_np(vx, vy, wz, zr, 0.04), (1, 10)).reshape(1, 100)
        z_onnx = sess_new.run(["token"], {"modal_input": m_in})[0][0]
        l2 = float(np.linalg.norm(z_onnx - z_ref))
        cos = float(np.dot(z_onnx, z_ref) / (np.linalg.norm(z_onnx) * np.linalg.norm(z_ref) + 1e-12))
        anchor_parity[name] = {"l2": round(l2, 5), "cos": round(cos, 6)}
        print(f"[AnchorParity {name:12s}] l2={l2:.5f} cos={cos:.6f}", flush=True)
        assert l2 < 0.05, f"Anchor parity L2 >= 0.05 on {name}: {l2:.5f}"

    sha256_new = hashlib.sha256(new_onnx.read_bytes()).hexdigest()
    train_meta = {
        "codec_fingerprint": fp,
        "new_onnx_path": str(new_onnx),
        "new_onnx_bytes": new_onnx.stat().st_size,
        "new_onnx_sha256": sha256_new,
        "pt_vs_onnx_max_abs_diff": max_diff,
        "anchor_parity": anchor_parity,
        "train_curve_csv": str(csv_path),
        "train_curve_png": str(out_dir / "train_curve.png"),
    }
    return new_onnx, train_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs/h003_v2b_planner_cal_strafe_v2_20261121",
    )
    ap.add_argument("--init-anchors-npz", type=Path, default=None)
    ap.add_argument("--refine-keys", type=str, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    networks.AMP_DTYPE = None
    bundle = torch.load(V2B_BUNDLE_PT, map_location="cpu", weights_only=True)
    pi_cpu = Policy(80, GOAL_FRAME_DIM * 10, num_joints=NUM_J, hist=10, z_dim=32, hidden=tuple(bundle["hidden"])).eval()
    pi_cpu.load_state_dict(bundle["token_head_pi"])
    kin_cpu = Q1Kinematics(str(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"), "cpu")

    refine_set = {s.strip() for s in args.refine_keys.split(",") if s.strip()} if args.refine_keys else None
    anchors, lateral_meta = run_stage1_cem(
        out_dir,
        pi_cpu,
        kin_cpu,
        init_anchors_npz=args.init_anchors_npz,
        refine_keys=refine_set,
    )
    new_onnx, train_meta = run_stage2_train_and_export(out_dir, anchors, pi_cpu, bundle)

    summary: dict[str, Any] = {
        "stage1_lateral_anchors": lateral_meta,
        "stage2_export": train_meta,
        "wall_seconds_calibration": round(time.perf_counter() - t0, 2),
    }
    if not args.skip_eval:
        eval_report = evaluate_all(new_onnx, DEPLOY_ROOT / "onnx/q1_tok_planner_cmd.onnx", out_dir)
        summary["evaluation_path"] = str(out_dir / "eval_strafe_report.json")
        summary["smoke_latency"] = eval_report["smoke_latency"]
        summary["card04_18s"] = eval_report["card04_18s"]
        summary["perturbed_delayed_2ms_summary"] = {
            "old_passed": f"{eval_report['perturbed_delayed_2ms']['old']['passed']}/{eval_report['perturbed_delayed_2ms']['old']['total']}",
            "new_passed": f"{eval_report['perturbed_delayed_2ms']['new']['passed']}/{eval_report['perturbed_delayed_2ms']['new']['total']}",
        }
    summary["total_wall_seconds"] = round(time.perf_counter() - t0, 2)
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Calibration + evaluation finished in {summary['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
