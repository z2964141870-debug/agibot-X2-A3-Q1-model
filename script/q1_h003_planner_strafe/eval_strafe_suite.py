"""Closed-loop 2 ms lateral strafe + Card 04 18s regression verification suite for Q1 PlannerCmd ONNX."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw

ROOT = Path("/home/yu/projects/SuperMimicHP")
DEPLOY_ROOT = Path("/home/yu/projects/Q1_deploy")

for p in (
    ROOT,
    ROOT / "src",
    ROOT / "runs/h002_sonic_v2",
    ROOT / "runs/h003_deploy_hybrid",
    DEPLOY_ROOT / "code",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cgfa_estimator_v2b_deploy import ContactGatedFootAnchor
from q1_cgfa_obs_np import ContactGatedFootAnchorNP, LivePlannerControllerNP, build_obs_80_np, build_planner_cmd_single_10d_np
from q1_kinematics_np import Q1KinematicsNP, yaw_angle_from_mat
from q1_v2b_runtime import apply_torque_safe_pd_np
from render_deploy_hybrid_1x import Q_STAND, _build_dual_vis_xml, _load_fonts
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics

SIM_XML = ROOT / "runs/h003_v2b_planner_cal/xml/q1_hard_floor_sim.xml"
DEPLOY_CFG_PATH = DEPLOY_ROOT / "config/q1_deploy_config.json"
HEAD_ONNX_PATH = DEPLOY_ROOT / "onnx/q1_h002_token_head.onnx"
ROBOT_XML_PATH = DEPLOY_ROOT / "robot_model/supermimic.xml"
TORCH_KIN_XML = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"

LATERAL_VELS = (-0.20, -0.15, -0.12, -0.10, -0.08, -0.05, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
PERT_VELS = (-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)
PERT_SEEDS = (101, 102, 103)


def _load_deploy_arrays() -> dict[str, np.ndarray]:
    cfg = json.loads(DEPLOY_CFG_PATH.read_text(encoding="utf-8"))
    return {
        "default_q": np.asarray(cfg["default_q"], dtype=np.float32),
        "ascale": np.asarray(cfg["action_scale"], dtype=np.float32),
        "kp": np.asarray(cfg["kp"], dtype=np.float32),
        "kd": np.asarray(cfg["kd"], dtype=np.float32),
        "eff": np.asarray(cfg["effort_limit_nm"], dtype=np.float32),
        "q_lo": np.asarray(cfg["q_lo"], dtype=np.float32),
        "q_hi": np.asarray(cfg["q_hi"], dtype=np.float32),
    }


def _make_ort_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class _RunnerContext:
    def __init__(self, tok_onnx_path: Path) -> None:
        self.cfg = _load_deploy_arrays()
        self.tok_sess = _make_ort_session(tok_onnx_path)
        self.head_sess = _make_ort_session(HEAD_ONNX_PATH)
        self.kin_np = Q1KinematicsNP(ROBOT_XML_PATH)
        self.m_2ms = mujoco.MjModel.from_xml_path(str(SIM_XML))
        self.m_2ms.opt.timestep = 0.002
        self.la_bid = mujoco.mj_name2id(self.m_2ms, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.ra_bid = mujoco.mj_name2id(self.m_2ms, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")


def _run_2ms_trial(
    ctx: _RunnerContext,
    schedule_fn: Any,
    total_steps: int,
    *,
    q0_noise_std: float = 0.0,
    seed: int = 0,
    delay_10ms: bool = False,
    record_vis: bool = False,
) -> dict[str, Any]:
    m = ctx.m_2ms
    c = ctx.cfg
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
    q0 = dq.copy()
    if q0_noise_std > 0.0:
        rng = np.random.default_rng(seed)
        q0 = np.clip(
            q0 + rng.normal(0.0, q0_noise_std, size=22).astype(np.float32),
            q_lo + 0.02,
            q_hi - 0.02,
        )
    d.qpos[0:3] = [0.0, 0.0, 0.4213]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:29] = q0
    d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    for _ in range(100):  # 0.2 s settle at 2 ms
        mujoco.mj_step(m, d)

    q_init = d.qpos[7:29].astype(np.float32).copy()
    qd_init = d.qvel[6:28].astype(np.float32).copy()
    R_init = d.xmat[1].reshape(3, 3).astype(np.float32).copy()
    w_init = d.qvel[3:6].astype(np.float32).copy()

    cgfa = ContactGatedFootAnchorNP(ctx.kin_np, dt=0.02)
    planner = LivePlannerControllerNP(dt=0.02, num_frames=10, smooth_slew=True)
    cgfa.reset(R_init, w_init, q_init, qd_init, np.zeros(2, dtype=np.float32), R_ref=R_init)
    planner.reset(yaw0_rad=yaw_angle_from_mat(R_init))

    hist: np.ndarray | None = None
    a_prev = np.zeros(22, dtype=np.float32)
    last_tgt = q_init.copy()
    delayed_state = (q_init, qd_init, R_init, w_init)

    pd_oob = 0
    max_qd = 0.0
    fell_at: float | None = None

    pos_xy: list[np.ndarray] = []
    yaw_list: list[float] = []
    roll_list: list[float] = []
    pitch_list: list[float] = []
    fw_list: list[float] = []
    slip_list: list[float] = []
    lz_list: list[float] = []
    rz_list: list[float] = []
    q_list: list[np.ndarray] = []
    qpos_tr: list[np.ndarray] = []
    ref_qpos_tr: list[np.ndarray] = []

    for t in range(total_steps):
        vx, vy, wz, zr, rx = schedule_fn(t)
        q_obs, qd_obs, R_obs, w_obs = delayed_state if delay_10ms else (
            d.qpos[7:29].astype(np.float32),
            d.qvel[6:28].astype(np.float32),
            d.xmat[1].reshape(3, 3).astype(np.float32),
            d.qvel[3:6].astype(np.float32),
        )

        p_in, p_ref_xy, R_ref = planner.push_and_build(
            vx, vy, wz, zr, rx, cgfa=cgfa, R_imu_aligned=R_obs
        )
        exy, _ = cgfa.step(R_obs, w_obs, q_obs, qd_obs, p_ref_xy, R_ref=R_ref)
        obs80 = build_obs_80_np(
            q=q_obs,
            qd=qd_obs,
            default_q=dq,
            R_imu=R_obs,
            w_body=w_obs,
            a_prev=a_prev,
            R_ref=R_ref,
            cgfa_exy=exy,
        ).reshape(1, 1, 80)
        hist = np.repeat(obs80, 10, axis=1) if hist is None else np.concatenate((hist[:, 1:], obs80), axis=1)

        tk = ctx.tok_sess.run(["token"], {"modal_input": p_in.reshape(1, 100)})[0]
        act = ctx.head_sess.run(["action"], {"obs_hist": hist, "token": tk})[0][0]

        q_raw = dq + asc * act
        q_safe = apply_torque_safe_pd_np(
            q_raw, q_obs, qd_obs, q_lo, q_hi, kp, kd, eff, q_init=q_init, step_i=t, engage_steps=5
        )
        q_safe = np.clip(q_safe, last_tgt - 0.2, last_tgt + 0.2)
        q_safe = np.clip(q_safe, dq - 2.0, dq + 2.0)

        tau = np.abs(kp * (q_safe - q_obs) - kd * qd_obs)
        pd_oob += int(np.sum((tau - eff) > 1e-3))

        last_tgt = q_safe.copy()
        a_prev = ((q_safe - dq) / asc).astype(np.float32)

        d.ctrl[:] = q_safe
        for sub in range(10):
            mujoco.mj_step(m, d)
            max_qd = max(max_qd, float(np.max(np.abs(d.qvel[6:28]))))
            if delay_10ms and sub == 4:
                delayed_state = (
                    d.qpos[7:29].astype(np.float32).copy(),
                    d.qvel[6:28].astype(np.float32).copy(),
                    d.xmat[1].reshape(3, 3).astype(np.float32).copy(),
                    d.qvel[3:6].astype(np.float32).copy(),
                )

        if record_vis:
            qpos_tr.append(d.qpos.copy())
            rq = np.zeros(29, dtype=np.float64)
            rq[0:2] = p_ref_xy
            rq[2] = 0.4213 * zr
            yaw_ref = yaw_angle_from_mat(R_ref)
            rq[3:7] = [math.cos(0.5 * yaw_ref), 0.0, 0.0, math.sin(0.5 * yaw_ref)]
            rq[7:29] = Q_STAND
            ref_qpos_tr.append(rq)

        Rp = d.xmat[1].reshape(3, 3)
        pos_xy.append(d.qpos[0:2].copy())
        yaw_rad = math.atan2(float(Rp[1, 0]), float(Rp[0, 0]))
        yaw_list.append(math.degrees(yaw_rad))
        roll_list.append(math.degrees(math.asin(max(-1.0, min(1.0, float(Rp[2, 1]))))))
        pitch_list.append(math.degrees(math.asin(max(-1.0, min(1.0, -float(Rp[2, 0]))))))
        ch, sh = math.cos(yaw_rad), math.sin(yaw_rad)
        dp_lr = d.xpos[ctx.la_bid] - d.xpos[ctx.ra_bid]
        fw_list.append(abs(-sh * float(dp_lr[0]) + ch * float(dp_lr[1])) * 100.0)
        lz_list.append(float(d.xpos[ctx.la_bid, 2]))
        rz_list.append(float(d.xpos[ctx.ra_bid, 2]))
        q_list.append(d.qpos[7:29].astype(np.float32).copy())

        b_st = ctx.la_bid if d.xpos[ctx.la_bid, 2] < d.xpos[ctx.ra_bid, 2] else ctx.ra_bid
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b_st, vel6, 0)
        Rf = d.xmat[b_st].reshape(3, 3)
        vs = vel6[3:6] + np.cross(vel6[0:3], Rf @ np.array([0.020, 0.0, -0.022]))
        slip_list.append(float(np.linalg.norm(vs[:2]) * 1000.0))

        if d.qpos[2] < 0.24 or Rp[2, 2] < 0.65:
            fell_at = round((t + 1) * 0.02, 2)
            break

    return {
        "fell_at": fell_at,
        "pd_oob": pd_oob,
        "max_qd_rad_s": round(max_qd, 3),
        "pos_xy": np.array(pos_xy) if pos_xy else np.zeros((0, 2)),
        "yaw_deg": np.array(yaw_list),
        "roll_deg": np.array(roll_list),
        "pitch_deg": np.array(pitch_list),
        "fw_cm": np.array(fw_list),
        "slip_mm_s": np.array(slip_list),
        "lz_m": np.array(lz_list),
        "rz_m": np.array(rz_list),
        "q": np.array(q_list) if q_list else np.zeros((0, 22)),
        "qpos": np.array(qpos_tr) if record_vis else None,
        "ref_qpos": np.array(ref_qpos_tr) if record_vis else None,
    }


def _summarize_cmd_trial(tr: dict[str, Any], vx: float, vy: float, wz: float) -> dict[str, Any]:
    # Schedule: 0-50 (1.0s) stand, 50-250 (4.0s) cmd, 250-325 (1.5s) smooth stop
    n = len(tr["pos_xy"])
    s0 = min(49, max(0, n - 1))
    s1 = min(249, max(0, n - 1))
    dx_m = float(tr["pos_xy"][s1, 0] - tr["pos_xy"][s0, 0]) if n > 0 else 0.0
    dy_m = float(tr["pos_xy"][s1, 1] - tr["pos_xy"][s0, 1]) if n > 0 else 0.0
    target_dy_m = round(vy * 4.0, 4)

    cmd_sl = slice(50, min(250, n))
    if n > 50:
        fw_c = tr["fw_cm"][cmd_sl]
        roll_c = tr["roll_deg"][cmd_sl]
        pitch_c = tr["pitch_deg"][cmd_sl]
        slip_c = tr["slip_mm_s"][cmd_sl]
        lz_c = tr["lz_m"][cmd_sl]
        rz_c = tr["rz_m"][cmd_sl]
        q_c = tr["q"][cmd_sl]
        switches = int(np.sum(np.diff((lz_c > rz_c).astype(np.int32)) != 0)) if len(lz_c) > 1 else 0
        l_lift = float(np.max(lz_c) - np.min(lz_c)) * 100.0
        r_lift = float(np.max(rz_c) - np.min(rz_c)) * 100.0
        l_hr = float(np.ptp(q_c[:, 1])) * 180.0 / math.pi
        r_hr = float(np.ptp(q_c[:, 7])) * 180.0 / math.pi
        l_ar = float(np.ptp(q_c[:, 5])) * 180.0 / math.pi
        r_ar = float(np.ptp(q_c[:, 11])) * 180.0 / math.pi
        yaw_deg = float(tr["yaw_deg"][s1])
    else:
        fw_c = tr["fw_cm"]
        roll_c = tr["roll_deg"]
        pitch_c = tr["pitch_deg"]
        slip_c = tr["slip_mm_s"]
        switches = 0
        l_lift = r_lift = l_hr = r_hr = l_ar = r_ar = 0.0
        yaw_deg = float(tr["yaw_deg"][-1]) if len(tr["yaw_deg"]) else 0.0

    stop_drift_m = float(np.linalg.norm(tr["pos_xy"][324] - tr["pos_xy"][274])) if n >= 325 else None

    return {
        "vx": vx,
        "vy": vy,
        "wz": wz,
        "fell_at": tr["fell_at"],
        "dx_m": round(dx_m, 4),
        "dy_m": round(dy_m, 4),
        "target_dy_m": target_dy_m,
        "dy_ratio": round(dy_m / target_dy_m, 3) if abs(target_dy_m) > 1e-6 else None,
        "yaw_deg": round(yaw_deg, 2),
        "fw_mean_cm": round(float(np.mean(fw_c)) if len(fw_c) else 0.0, 2),
        "fw_min_cm": round(float(np.min(fw_c)) if len(fw_c) else 0.0, 2),
        "roll_std_deg": round(float(np.std(roll_c)) if len(roll_c) else 0.0, 2),
        "roll_max_deg": round(float(np.max(np.abs(roll_c))) if len(roll_c) else 0.0, 2),
        "pitch_std_deg": round(float(np.std(pitch_c)) if len(pitch_c) else 0.0, 2),
        "slip_mm_s": round(float(np.mean(slip_c)) if len(slip_c) else 0.0, 1),
        "step_switches": switches,
        "l_foot_lift_cm": round(l_lift, 2),
        "r_foot_lift_cm": round(r_lift, 2),
        "l_hip_roll_deg_range": round(l_hr, 2),
        "r_hip_roll_deg_range": round(r_hr, 2),
        "l_ankle_roll_deg_range": round(l_ar, 2),
        "r_ankle_roll_deg_range": round(r_ar, 2),
        "stop_drift_m": round(stop_drift_m, 4) if stop_drift_m is not None else None,
        "pd_oob": tr["pd_oob"],
        "max_qd_rad_s": tr["max_qd_rad_s"],
    }


def eval_nominal_2ms_sweep(ctx: _RunnerContext) -> dict[str, Any]:
    lateral = {}
    for vy in LATERAL_VELS:
        key = f"vy_{vy:+.2f}"
        tr = _run_2ms_trial(
            ctx,
            lambda t, _vy=vy: (0.0, 0.0, 0.0, 1.0, 0.04) if (t < 50 or t >= 250) else (0.0, _vy, 0.0, 1.0, 0.04),
            325,
        )
        lateral[key] = _summarize_cmd_trial(tr, 0.0, vy, 0.0)

    sanity_specs = [
        ("fwd_+0.25", 0.25, 0.0, 0.0),
        ("bwd_-0.20", -0.20, 0.0, 0.0),
        ("turn_+0.50", 0.0, 0.0, 0.50),
        ("turn_-0.50", 0.0, 0.0, -0.50),
    ]
    sanity = {}
    for name, vx, vy, wz in sanity_specs:
        tr = _run_2ms_trial(
            ctx,
            lambda t, _vx=vx, _vy=vy, _wz=wz: (0.0, 0.0, 0.0, 1.0, 0.04) if (t < 50 or t >= 250) else (_vx, _vy, _wz, 1.0, 0.04),
            325,
        )
        sanity[name] = _summarize_cmd_trial(tr, vx, vy, wz)

    mirror = {}
    for spd in (0.05, 0.10, 0.15, 0.20):
        pos = lateral[f"vy_{spd:+.2f}"]
        neg = lateral[f"vy_{-spd:+.2f}"]
        mag_pos = max(abs(pos["dy_m"]), 1e-6)
        mag_neg = max(abs(neg["dy_m"]), 1e-6)
        sym_ratio = min(mag_pos, mag_neg) / max(mag_pos, mag_neg)
        mirror[f"{spd:.2f}"] = {
            "dy_pos_m": pos["dy_m"],
            "dy_neg_m": neg["dy_m"],
            "mirror_disp_ratio": round(sym_ratio, 3),
            "roll_max_pos_deg": pos["roll_max_deg"],
            "roll_max_neg_deg": neg["roll_max_deg"],
            "fw_min_pos_cm": pos["fw_min_cm"],
            "fw_min_neg_cm": neg["fw_min_cm"],
            "switches_pos": pos["step_switches"],
            "switches_neg": neg["step_switches"],
        }

    return {"lateral": lateral, "sanity": sanity, "mirror_symmetry": mirror}


def eval_perturbed_delayed_2ms(ctx: _RunnerContext) -> dict[str, Any]:
    trials = []
    passed = 0
    for vy in PERT_VELS:
        for seed in PERT_SEEDS:
            tr = _run_2ms_trial(
                ctx,
                lambda t, _vy=vy: (0.0, 0.0, 0.0, 1.0, 0.04) if t < 50 else (0.0, _vy, 0.0, 1.0, 0.04),
                250,
                q0_noise_std=0.05,
                seed=seed,
                delay_10ms=True,
            )
            n = len(tr["pos_xy"])
            s0 = min(49, max(0, n - 1))
            s1 = max(0, n - 1)
            dy_m = float(tr["pos_xy"][s1, 1] - tr["pos_xy"][s0, 1]) if n > 0 else 0.0
            dx_m = float(tr["pos_xy"][s1, 0] - tr["pos_xy"][s0, 0]) if n > 0 else 0.0
            target_dy = vy * 4.0
            fw_c = tr["fw_cm"][50:] if n > 50 else tr["fw_cm"]
            roll_c = tr["roll_deg"][50:] if n > 50 else tr["roll_deg"]
            fw_min = float(np.min(fw_c)) if len(fw_c) else 0.0
            roll_max = float(np.max(np.abs(roll_c))) if len(roll_c) else 99.0
            ok = (
                tr["fell_at"] is None
                and (dy_m * vy > 0.0)
                and (abs(dy_m) >= 0.35 * abs(target_dy))
                and (fw_min >= 6.0)
            )
            if ok:
                passed += 1
            trials.append({
                "vy": vy,
                "seed": seed,
                "passed": ok,
                "fell_at": tr["fell_at"],
                "dx_m": round(dx_m, 4),
                "dy_m": round(dy_m, 4),
                "target_dy_m": round(target_dy, 3),
                "fw_min_cm": round(fw_min, 2),
                "roll_max_deg": round(roll_max, 2),
                "pd_oob": tr["pd_oob"],
            })

    return {
        "passed": passed,
        "total": len(trials),
        "mean_roll_max_deg": round(float(np.mean([t["roll_max_deg"] for t in trials])), 2),
        "worst_roll_max_deg": round(float(np.max([t["roll_max_deg"] for t in trials])), 2),
        "mean_fw_min_cm": round(float(np.mean([t["fw_min_cm"] for t in trials])), 2),
        "worst_fw_min_cm": round(float(np.min([t["fw_min_cm"] for t in trials])), 2),
        "total_pd_oob": int(sum(t["pd_oob"] for t in trials)),
        "trials": trials,
    }


def reversal_cmd_at(t: int) -> tuple[float, float, float, float, float]:
    s = t * 0.02
    if s < 1.0:
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if s < 4.0:
        return (0.0, 0.10, 0.0, 1.0, 0.04)
    if s < 5.0:
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if s < 8.0:
        return (0.0, -0.10, 0.0, 1.0, 0.04)
    if s < 9.0:
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if s < 12.0:
        return (0.0, 0.15, 0.0, 1.0, 0.04)
    if s < 13.0:
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if s < 16.0:
        return (0.0, -0.15, 0.0, 1.0, 0.04)
    return (0.0, 0.0, 0.0, 1.0, 0.04)


def eval_reversal_sequence(ctx: _RunnerContext) -> dict[str, Any]:
    tr = _run_2ms_trial(ctx, reversal_cmd_at, 875, record_vis=True)
    n = len(tr["pos_xy"])
    segments_spec = [
        ("stand_0_1s", 0, 49),
        ("left_+0.10_1_4s", 50, 199),
        ("stop_4_5s", 200, 249),
        ("right_-0.10_5_8s", 250, 399),
        ("stop_8_9s", 400, 449),
        ("left_+0.15_9_12s", 450, 599),
        ("stop_12_13s", 600, 649),
        ("right_-0.15_13_16s", 650, 799),
        ("stop_16_17.5s", 800, 874),
    ]
    segments = {}
    for seg_name, i0, i1 in segments_spec:
        if n <= i0:
            segments[seg_name] = {"reached": False}
            continue
        j1 = min(i1, n - 1)
        sl = slice(i0, j1 + 1)
        segments[seg_name] = {
            "reached": True,
            "dx_m": round(float(tr["pos_xy"][j1, 0] - tr["pos_xy"][i0, 0]), 4),
            "dy_m": round(float(tr["pos_xy"][j1, 1] - tr["pos_xy"][i0, 1]), 4),
            "yaw_deg": round(float(tr["yaw_deg"][j1]), 2),
            "roll_max_deg": round(float(np.max(np.abs(tr["roll_deg"][sl]))), 2),
            "fw_min_cm": round(float(np.min(tr["fw_cm"][sl])), 2),
        }

    return {
        "completed": (tr["fell_at"] is None and n == 875),
        "fell_at": tr["fell_at"],
        "pd_oob": tr["pd_oob"],
        "max_qd_rad_s": tr["max_qd_rad_s"],
        "segments": segments,
        "qpos": tr["qpos"],
        "ref_qpos": tr["ref_qpos"],
    }


def render_reversal_video(
    res_old: dict[str, Any],
    res_new: dict[str, Any],
    out_dir: Path,
    mp4_path: Path,
    qc_path: Path,
) -> None:
    q_new, rq_new = res_new["qpos"], res_new["ref_qpos"]
    q_old, rq_old = res_old["qpos"], res_old["ref_qpos"]
    xml_dir = out_dir / "xml"
    vis_xml_new = _build_dual_vis_xml(xml_dir, q_new[:, :2], rq_new[:, :2], "strafe_new")
    vis_xml_old = _build_dual_vis_xml(xml_dir, q_old[:, :2], rq_old[:, :2], "strafe_old")

    mn = mujoco.MjModel.from_xml_path(str(vis_xml_new))
    dn = mujoco.MjData(mn)
    mo = mujoco.MjModel.from_xml_path(str(vis_xml_old))
    do = mujoco.MjData(mo)
    rn = mujoco.Renderer(mn, height=540, width=640)
    ro = mujoco.Renderer(mo, height=540, width=640)

    cam_n = mujoco.MjvCamera()
    cam_o = mujoco.MjvCamera()
    for cam in (cam_n, cam_o):
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 2.40
        cam.elevation = -20.0
        cam.azimuth = 178.0  # rear view so left/right lateral motion aligns left/right on screen

    _, f_sub, _ = _load_fonts()
    n_tot = max(len(q_new), len(q_old))
    frames_rgb = []
    for k in range(0, n_tot, 2):  # 50 Hz -> 25 fps (1.0x real-time)
        kn = min(k, len(q_new) - 1)
        dn.qpos[:29] = q_new[kn]
        dn.qpos[29:58] = rq_new[kn]
        mujoco.mj_forward(mn, dn)
        cam_n.lookat[:] = [q_new[kn, 0], q_new[kn, 1], 0.44]
        rn.update_scene(dn, camera=cam_n)
        img_n = rn.render()

        ko = min(k, len(q_old) - 1)
        do.qpos[:29] = q_old[ko]
        do.qpos[29:58] = rq_old[ko]
        mujoco.mj_forward(mo, do)
        cam_o.lookat[:] = [q_old[ko, 0], q_old[ko, 1], 0.44]
        ro.update_scene(do, camera=cam_o)
        img_o = ro.render()

        side = np.concatenate((img_o, img_n), axis=1)
        pil = Image.fromarray(side)
        drw = ImageDraw.Draw(pil)
        _, vy_c, _, _, _ = reversal_cmd_at(k)
        ts = k * 0.02
        drw.rectangle([0, 0, 1280, 38], fill=(15, 20, 28))
        drw.text(
            (14, 9),
            f"OLD 6a989b2b (dy={q_old[ko,1]:+.2f}m, ref={rq_old[ko,1]:+.2f}m, z={q_old[ko,2]:.2f}m)",
            fill=(255, 150, 120),
            font=f_sub,
        )
        drw.text(
            (654, 9),
            f"NEW Strafe Calibrated (dy={q_new[kn,1]:+.2f}m, ref={rq_new[kn,1]:+.2f}m, z={q_new[kn,2]:.2f}m)",
            fill=(110, 255, 165),
            font=f_sub,
        )
        drw.rectangle([0, 508, 1280, 540], fill=(15, 20, 28))
        drw.text(
            (14, 514),
            f"1.0x Real-Time (25fps, 2ms physics) | t={ts:5.2f}s/17.50s | Cmd vy={vy_c:+.2f} m/s (L=+vy, R=-vy)",
            fill=(235, 240, 248),
            font=f_sub,
        )
        frames_rgb.append(np.asarray(pil))

    rn.close()
    ro.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "1280x540",
        "-pix_fmt",
        "rgb24",
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames_rgb:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed ({rc}): {err}")

    sample_idx = [int(s * 25) for s in (2.5, 4.5, 6.5, 10.5, 14.5, 17.0)]
    thumbs = [Image.fromarray(frames_rgb[min(i, len(frames_rgb) - 1)]).resize((640, 270)) for i in sample_idx]
    qc = Image.new("RGB", (1280, 810), (12, 16, 22))
    for i, th in enumerate(thumbs):
        qc.paste(th, ((i % 2) * 640, (i // 2) * 270))
    qc.save(qc_path)


def _card04_cmd_at(t: int) -> tuple[float, float, float, float, float]:
    s = t * 0.02
    if s < 3.0:
        return (0.0, 0.0, 0.0, 1.0, 0.04)
    if s < 7.0:
        return (0.35, 0.0, 0.0, 1.0, 0.04)
    if s < 9.0:
        return (0.0, 0.0, 0.50, 1.0, 0.04)
    if s < 11.0:
        return (0.0, 0.0, -0.50, 1.0, 0.04)
    if s < 13.0:
        return (0.0, 0.0, 0.0, 1.0 - 0.20 * ((s - 11.0) / 2.0), 0.04)
    if s < 15.0:
        return (0.0, 0.0, 0.0, 0.80 + 0.20 * ((s - 13.0) / 2.0), 0.04)
    return (0.0, 0.0, 0.0, 1.0, 0.04)


def run_card04_18s_eval(tok_onnx_path: Path) -> dict[str, Any]:
    """Exact Card 04 18.0s regression benchmark (5 ms physics, 4 substeps, identical to fit_and_verify_planner_onnx.py)."""
    m = mujoco.MjModel.from_xml_path(str(SIM_XML))
    m.opt.timestep = 0.005
    la_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ra_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.4213]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:29] = Q_STAND
    d.ctrl[:] = Q_STAND
    mujoco.mj_forward(m, d)

    head_sess = _make_ort_session(HEAD_ONNX_PATH)
    tok_sess = _make_ort_session(tok_onnx_path)
    kin = Q1Kinematics(str(TORCH_KIN_XML), "cpu")
    est = ContactGatedFootAnchor(kin, 1, dt=0.02, device="cpu")
    est.gamma_loco = 0.96
    est.bound_m = 0.25
    R0 = torch.eye(3).unsqueeze(0)
    est.reset(R0, torch.zeros((1, 3)), torch.from_numpy(Q_STAND).unsqueeze(0), torch.zeros((1, 22)), torch.zeros((1, 2)), R_ref=R0)

    c = _load_deploy_arrays()
    dq, asc, kp, kd, eff, q_lo, q_hi = (
        c["default_q"],
        c["ascale"],
        c["kp"],
        c["kd"],
        c["eff"],
        c["q_lo"],
        c["q_hi"],
    )

    buf10 = None
    hist = None
    a_prev = np.zeros(22, dtype=np.float32)
    prev_q_tgt = Q_STAND.copy()
    p_plan = np.zeros(2, dtype=np.float32)
    yaw_plan = 0.0
    was_moving = False

    pos_xy, z_pelvis, yaws_deg, pitches_deg, fws_cm, slips_mm, step_jumps_deg, lat_ms = [], [], [], [], [], [], [], []
    pd_oob = 0
    fell = False

    for t in range(900):
        vx, vy, wz, zr, rx = _card04_cmd_at(t)
        moving = (abs(vx) + abs(vy)) > 0.015
        if was_moving and not moving:
            p_plan = p_plan - est.e_world_xy[0].numpy()
            est.prev_p_ref_xy[0] = torch.from_numpy(p_plan)
            est.e_world_xy.zero_()
        was_moving = moving

        cy, sy = math.cos(yaw_plan), math.sin(yaw_plan)
        p_plan[0] += (cy * vx - sy * vy) * 0.02
        p_plan[1] += (sy * vx + cy * vy) * 0.02
        yaw_plan += wz * 0.02
        R_ref = np.array(
            [
                [math.cos(yaw_plan), -math.sin(yaw_plan), 0.0],
                [math.sin(yaw_plan), math.cos(yaw_plan), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        row = build_planner_cmd_single_10d_np(vx, vy, wz, zr, rx)
        buf10 = np.tile(row, (10, 1)) if buf10 is None else np.vstack((buf10[1:], row[None, :]))

        q_cur = d.qpos[7:29].astype(np.float32)
        qd_cur = d.qvel[6:28].astype(np.float32)
        R_np = d.xmat[1].reshape(3, 3).astype(np.float32)
        w_b = d.qvel[3:6].astype(np.float32)
        w_w = (R_np @ w_b).astype(np.float32)

        exy_t, _ = est.step(
            torch.from_numpy(R_np).unsqueeze(0),
            torch.from_numpy(w_w).unsqueeze(0),
            torch.from_numpy(q_cur).unsqueeze(0),
            torch.from_numpy(qd_cur).unsqueeze(0),
            torch.from_numpy(p_plan.copy()).unsqueeze(0),
            R_ref=torch.from_numpy(R_ref).unsqueeze(0),
            mode="cgfa_clipped_norm",
        )
        obs80 = np.concatenate(
            (q_cur - dq, qd_cur, w_b, -R_np[2, :], a_prev, (R_np.T @ R_ref)[:, :2].reshape(6), exy_t[0].numpy())
        ).astype(np.float32).reshape(1, 1, 80)
        hist = np.repeat(obs80, 10, axis=1) if hist is None else np.concatenate((hist[:, 1:], obs80), axis=1)

        t_inf0 = time.perf_counter()
        tk = tok_sess.run(["token"], {"modal_input": buf10.reshape(1, 100)})[0]
        act = head_sess.run(["action"], {"obs_hist": hist, "token": tk})[0][0]
        lat_ms.append((time.perf_counter() - t_inf0) * 1000.0)

        q_safe = apply_torque_safe_pd_np(dq + asc * act, q_cur, qd_cur, q_lo, q_hi, kp, kd, eff, q_init=Q_STAND, step_i=t, engage_steps=5)
        tau = np.abs(kp * (q_safe - q_cur) - kd * qd_cur)
        pd_oob += int(np.sum((tau - eff) > 1e-3))
        if t > 0:
            step_jumps_deg.append(float(np.mean(np.abs(q_safe - prev_q_tgt)) * 180.0 / math.pi))
        prev_q_tgt = q_safe.copy()
        a_prev = ((q_safe - dq) / asc).astype(np.float32)

        d.ctrl[:] = q_safe
        for _ in range(4):
            mujoco.mj_step(m, d)

        R_post = d.xmat[1].reshape(3, 3)
        if d.qpos[2] < 0.22 or R_post[2, 2] < 0.55:
            fell = True
            break

        pos_xy.append(d.qpos[0:2].copy())
        z_pelvis.append(float(d.qpos[2]))
        yaw_deg = math.degrees(math.atan2(float(R_post[1, 0]), float(R_post[0, 0])))
        yaws_deg.append(yaw_deg)
        pitches_deg.append(math.degrees(math.asin(max(-1.0, min(1.0, -float(R_post[2, 0]))))))
        ch, sh = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        dp_lr = d.xpos[la_bid] - d.xpos[ra_bid]
        fws_cm.append(abs(-sh * float(dp_lr[0]) + ch * float(dp_lr[1])) * 100.0)

        b_st = la_bid if d.xpos[la_bid, 2] < d.xpos[ra_bid, 2] else ra_bid
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b_st, vel6, 0)
        R_f = d.xmat[b_st].reshape(3, 3)
        v_sole = vel6[3:6] + np.cross(vel6[0:3], R_f @ np.array([0.020, 0.0, -0.022]))
        slips_mm.append(float(np.linalg.norm(v_sole[:2]) * 1000.0))

    pos_arr = np.array(pos_xy)
    if len(pos_arr) < 900:
        return {"completed": False, "fell_step": len(pos_arr), "pd_oob_count": pd_oob}

    p0_walk, p1_walk = pos_arr[149], pos_arr[349]
    walk_disp_m = float(np.linalg.norm(p1_walk - p0_walk))
    turn_center = pos_arr[349]
    turn_rad_drift_m = float(np.max(np.linalg.norm(pos_arr[350:550] - turn_center[None, :], axis=-1)))
    stop_disp_last2s_m = float(np.linalg.norm(pos_arr[899] - pos_arr[799]))
    return {
        "completed": (not fell and len(pos_xy) == 900),
        "pd_oob_count": pd_oob,
        "tok_head_p50_ms": round(float(np.percentile(lat_ms, 50)), 4),
        "tok_head_p95_ms": round(float(np.percentile(lat_ms, 95)), 4),
        "stage1_stand_0_3s_slip_mm_s": round(float(np.mean(slips_mm[25:150])), 2),
        "stage2_walk_3_7s_disp_m": round(walk_disp_m, 3),
        "stage2_walk_3_7s_dx_m": round(float(p1_walk[0] - p0_walk[0]), 3),
        "stage2_walk_3_7s_footW_cm": round(float(np.mean(fws_cm[165:350])), 2),
        "stage2_walk_3_7s_pitch_std_deg": round(float(np.std(pitches_deg[165:350])), 2),
        "stage3_turn_7_11s_peak_yaw_deg": round(float(np.max(yaws_deg[350:550])), 2),
        "stage3_turn_7_11s_end_yaw_deg": round(float(yaws_deg[549]), 2),
        "stage3_turn_7_11s_radial_drift_m": round(turn_rad_drift_m, 3),
        "stage4_squat_11_15s_min_pelvis_z_m": round(float(np.min(z_pelvis[550:750])), 3),
        "stage4_squat_11_15s_end_pelvis_z_m": round(float(z_pelvis[749]), 3),
        "stage5_stop_15_18s_slip_mm_s": round(float(np.mean(slips_mm[800:900])), 2),
        "stage5_stop_last2s_drift_m": round(stop_disp_last2s_m, 3),
        "mean_step_jump_deg": round(float(np.mean(step_jumps_deg)), 3),
    }


def eval_cpu_smoke_latency(tok_onnx_path: Path) -> dict[str, Any]:
    tok_sess = _make_ort_session(tok_onnx_path)
    head_sess = _make_ort_session(HEAD_ONNX_PATH)
    c = _load_deploy_arrays()
    dq, asc, kp, kd, eff, q_lo, q_hi = (
        c["default_q"],
        c["ascale"],
        c["kp"],
        c["kd"],
        c["eff"],
        c["q_lo"],
        c["q_hi"],
    )
    qd0 = np.zeros(22, dtype=np.float32)
    rng = np.random.default_rng(20261121)
    hist = np.zeros((1, 10, 80), dtype=np.float32)
    lat_ms = []
    oob = 0
    max_tau = 0.0
    for step_i in range(250):
        obs80 = rng.normal(0.0, 0.15, size=(1, 1, 80)).astype(np.float32)
        hist = np.concatenate((hist[:, 1:], obs80), axis=1)
        m_in = rng.normal(0.0, 0.25, size=(1, 100)).astype(np.float32)
        t0 = time.perf_counter()
        tk = tok_sess.run(["token"], {"modal_input": m_in})[0]
        act = head_sess.run(["action"], {"obs_hist": hist, "token": tk})[0][0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        q_safe = apply_torque_safe_pd_np(dq + asc * act, dq, qd0, q_lo, q_hi, kp, kd, eff, q_init=dq, step_i=step_i, engage_steps=5)
        tau = kp * (q_safe - dq) - kd * qd0
        assert np.all(np.isfinite(q_safe)) and np.all(np.isfinite(tau))
        oob += int(np.sum((np.abs(tau) - eff) > 1e-4))
        max_tau = max(max_tau, float(np.max(np.abs(tau))))
        if step_i >= 50:
            lat_ms.append(dt_ms)
    return {
        "p50_ms": round(float(np.percentile(lat_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(lat_ms, 95)), 4),
        "oob": oob,
        "max_tau_nm": round(max_tau, 2),
    }


def evaluate_all(new_onnx: Path, old_onnx: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_old = _RunnerContext(old_onnx)
    ctx_new = _RunnerContext(new_onnx)

    print("[eval] Running nominal 2ms sweep (OLD vs NEW)...", flush=True)
    nom_old = eval_nominal_2ms_sweep(ctx_old)
    nom_new = eval_nominal_2ms_sweep(ctx_new)

    print("[eval] Running perturbed stance + 10ms delay 2ms suite (OLD vs NEW)...", flush=True)
    pert_old = eval_perturbed_delayed_2ms(ctx_old)
    pert_new = eval_perturbed_delayed_2ms(ctx_new)

    print("[eval] Running 17.5s continuous L<->R reversal sequence (OLD vs NEW)...", flush=True)
    rev_old = eval_reversal_sequence(ctx_old)
    rev_new = eval_reversal_sequence(ctx_new)

    mp4_path = out_dir / "q1_strafe_16s_1x2.mp4"
    qc_path = out_dir / "q1_strafe_qc.png"
    print(f"[eval] Rendering 1.0x video {mp4_path}...", flush=True)
    render_reversal_video(rev_old, rev_new, out_dir, mp4_path, qc_path)

    print("[eval] Running Card 04 18s regression benchmark (OLD vs NEW)...", flush=True)
    c04_old = run_card04_18s_eval(old_onnx)
    c04_new = run_card04_18s_eval(new_onnx)

    print("[eval] Running single-thread CPU smoke & latency check...", flush=True)
    smoke_old = eval_cpu_smoke_latency(old_onnx)
    smoke_new = eval_cpu_smoke_latency(new_onnx)

    rev_old_summary = {k: v for k, v in rev_old.items() if k not in ("qpos", "ref_qpos")}
    rev_new_summary = {k: v for k, v in rev_new.items() if k not in ("qpos", "ref_qpos")}

    report = {
        "old_onnx": str(old_onnx),
        "new_onnx": str(new_onnx),
        "smoke_latency": {"old": smoke_old, "new": smoke_new},
        "card04_18s": {"old": c04_old, "new": c04_new},
        "nominal_2ms": {"old": nom_old, "new": nom_new},
        "perturbed_delayed_2ms": {"old": pert_old, "new": pert_new},
        "reversal_17_5s_2ms": {"old": rev_old_summary, "new": rev_new_summary},
        "video_mp4": str(mp4_path),
        "qc_png": str(qc_path),
    }
    (out_dir / "eval_strafe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] Saved {out_dir / 'eval_strafe_report.json'}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-onnx", type=Path, required=True)
    ap.add_argument(
        "--old-onnx",
        type=Path,
        default=DEPLOY_ROOT / "onnx/q1_tok_planner_cmd.onnx",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    evaluate_all(args.new_onnx, args.old_onnx, args.out_dir)


if __name__ == "__main__":
    main()
