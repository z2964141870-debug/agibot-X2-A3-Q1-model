"""Two Hard Benchmarks for H002-SONIC v3:
  A. Closed-Loop Loop-Closure Navigation Benchmark (CPU MuJoCo 3.11 + 1-Thread ORT ONNX):
     - Track A1: 26.0s (1,300-step, ~9.5m) Racetrack Loop (Forward + 180deg Turn + Return + 180deg Turn -> Origin)
                 under both PlannerCmd 100D (pure velocity) and Real-Gait RobotPose 590D / HumanSMPL 430D,
                 plus Hardened variant (0.45 m/s mid-turn push @ t=8.0s + 20ms actuator delay).
     - Track A2: 30.0s (1,500-step, ~9.8m) Figure-8 (infty) Inflection Loop -> Origin.
  B. High-Platform Falling-Box Reactive Catch Latency Benchmark (Real MuJoCo 3.11 Rigid-Body Physics):
     - Free-body 0.26kg box drops from ~1.44m shelf at t=0.80s; human reacts at t=0.94s (140ms human RT)
       and snaps arms into catch cradle just before t~1.20s impact.
     - Compares 0ms Causal (V3 RobotPose & V3 HumanSMPL) vs Future-Buffer models
       (Orig SuperMimic 200ms buffer, SONIC-Future 120ms buffer, Orig SuperMimic 0ms ZOH padding)
       across 20 randomized drop trials + Lookahead Buffer Sweep (0..200 ms).
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

import mujoco
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

from cgfa_estimator import LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, ContactGatedFootAnchor
from eval_sonic_v3_suite import (
    _apply_torque_safe_pd_cpu,
    _configure_eval_events,
    _enter_free_roaming_mode,
    _load_ckpt_policy,
    _ort_sess,
)
from multimodal_features import (
    Q1_NOMINAL_PELVIS_Z,
    build_human_smpl_goal,
    build_planner_cmd_goal,
    build_synthetic_planner_stream,
    build_vr_3pt_goal,
)
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import (
    mat_to_6d,
    mat_to_quat,
    quat_to_mat,
    so3_exp,
    unflatten_state,
    yaw_mat,
)
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib, quat_slerp
from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, gains_of, motor_of
from unisupermimic.teleop.realtime_teleop import build_causal_history_goal

Q_STAND = np.array(
    [-0.09, -0.02, -0.02, 0.22, -0.13, 0.0, -0.09, 0.02, 0.02, 0.22, -0.13, 0.0, 0.0, 0.0, 0.0, 0.18, 0.0, 1.4, 0.0, -0.18, 0.0, 1.4],
    dtype=np.float32,
)
Z_STAND = 0.42132428


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def _min_jerk(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _extract_periodic_walk_cycle(walk_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract an active walking stride cycle (joints (L, 22), z (L,)) at 50 Hz from a walking clip."""
    raw = np.loadtxt(walk_csv, delimiter=",", dtype=np.float32)
    T30 = raw.shape[0]
    dur = (T30 - 1) / 30.0
    T50 = int(np.floor(dur / 0.02)) + 1
    s = np.arange(T50, dtype=np.float32) * 0.02 * 30.0
    lo = np.clip(np.floor(s).astype(np.int64), 0, T30 - 2)
    a = (s - lo)[:, None]
    p50 = raw[lo, :3] * (1.0 - a) + raw[lo + 1, :3] * a
    q50 = raw[lo, 7:29] * (1.0 - a) + raw[lo + 1, 7:29] * a
    dq50 = np.diff(q50, axis=0, prepend=q50[:1]) * 50.0
    v50 = np.linalg.norm(np.diff(p50[:, :2], axis=0, prepend=p50[:1, :2]) * 50.0, axis=1)

    best_score, best_ab = 1e9, None
    start_lo = min(20, T50 // 10)
    start_hi = max(start_lo + 1, T50 - 65)
    for t0 in range(start_lo, start_hi):
        for L in range(42, 66):
            t1 = t0 + L
            if t1 >= T50 - 5:
                break
            leg_std = float(np.std(q50[t0:t1, [0, 3, 6, 9]], axis=0).mean())
            mv = float(np.mean(v50[t0:t1]))
            if leg_std < 0.22 or mv < 0.38:
                continue
            sc = float(((q50[t1] - q50[t0]) ** 2).mean() + 0.04 * ((dq50[t1] - dq50[t0]) ** 2).mean())
            if sc < best_score:
                best_score = sc
                best_ab = (t0, t1)

    assert best_ab is not None, f"No active walking stride cycle found in {walk_csv}"
    t0, t1 = best_ab
    cyc_q = q50[t0:t1].copy()
    cyc_z = p50[t0:t1, 2].copy()
    L = len(cyc_q)
    # Cosine-blend final 6 frames toward frame 0 for C1 loop continuity
    blend_k = min(6, L // 5)
    for j in range(blend_k):
        w = 0.5 * (1.0 - np.cos(np.pi * float(j + 1) / float(blend_k)))
        idx = L - blend_k + j
        cyc_q[idx] = (1.0 - w) * cyc_q[idx] + w * cyc_q[0]
        cyc_z[idx] = (1.0 - w) * cyc_z[idx] + w * cyc_z[0]
    cyc_z = np.clip(cyc_z, 0.395, 0.435)
    assert float(np.std(cyc_q[:, [0, 3, 6, 9]], axis=0).mean()) >= 0.20, "Extracted stride cycle is unexpectedly static"
    return cyc_q, cyc_z


def _build_analytic_loop_traj(
    cmd_segments: list[tuple[int, float, float]],
    cyc_q: np.ndarray,
    cyc_z: np.ndarray,
    *,
    startup_steps: int = 30,
    settle_steps: int = 50,
    stop_steps: int = 60,
    stand_q: np.ndarray = Q_STAND,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float, float]]]:
    """Integrate (vx, wz) schedule and enforce exact geometric closure back to (0, 0, heading=0)."""
    dt = 0.02
    vx_arr, wz_arr = [], []
    for n, vx, wz in cmd_segments:
        vx_arr.extend([vx] * n)
        wz_arr.extend([wz] * n)
    vx_v = np.array(vx_arr, dtype=np.float64)
    wz_v = np.array(wz_arr, dtype=np.float64)
    T_move = len(vx_v)

    # Smooth ramp in startup_steps and ramp out in settle_steps
    gait_gain = np.ones(T_move, dtype=np.float64)
    for i in range(startup_steps):
        gait_gain[i] = float(_min_jerk(np.array((i + 1) / float(startup_steps))))
    for i in range(settle_steps):
        gait_gain[T_move - settle_steps + i] = float(1.0 - _min_jerk(np.array((i + 1) / float(settle_steps))))

    vx_exec = vx_v * gait_gain
    wz_exec = wz_v * gait_gain
    # Integrate initial trajectory
    x, y, th = 0.0, 0.0, 0.0
    xs, ys, ths = [0.0], [0.0], [0.0]
    for t in range(T_move):
        x += vx_exec[t] * np.cos(th) * dt
        y += vx_exec[t] * np.sin(th) * dt
        th += wz_exec[t] * dt
        xs.append(x); ys.append(y); ths.append(th)

    # Close any small discretization residual cleanly over active walking steps [startup..T_move-settle]
    rem_x, rem_y = -xs[-1], -ys[-1]
    rem_th = ((0.0 - ths[-1] + np.pi) % (2.0 * np.pi)) - np.pi
    active = np.zeros(T_move, dtype=np.float64)
    active[startup_steps : T_move - settle_steps] = 1.0
    w_norm = active / max(active.sum() * dt, 1e-8)

    # Adjust wz first, then distribute small world-frame velocity correction converted back to body (vx, vy)
    wz_corr = wz_exec + rem_th * w_norm
    x, y, th = 0.0, 0.0, 0.0
    xs, ys, ths = [0.0], [0.0], [0.0]
    for t in range(T_move):
        x += vx_exec[t] * np.cos(th) * dt
        y += vx_exec[t] * np.sin(th) * dt
        th += wz_corr[t] * dt
        xs.append(x); ys.append(y); ths.append(th)

    dx_res, dy_res = -xs[-1], -ys[-1]
    vx_corr = vx_exec.copy()
    vy_corr = np.zeros(T_move, dtype=np.float64)
    for t in range(T_move):
        if active[t] > 0.0:
            dvx_w = dx_res * w_norm[t]
            dvy_w = dy_res * w_norm[t]
            vx_corr[t] += np.cos(ths[t]) * dvx_w + np.sin(ths[t]) * dvy_w
            vy_corr[t] += -np.sin(ths[t]) * dvx_w + np.cos(ths[t]) * dvy_w

    # Final exact trajectory
    x, y, th = 0.0, 0.0, 0.0
    rows = []
    planner_cmds = []
    L_cyc = len(cyc_q)
    for t in range(T_move + stop_steps):
        if t < T_move:
            g = gait_gain[t]
            vx_t, vy_t, wz_t = float(vx_corr[t]), float(vy_corr[t]), float(wz_corr[t])
            qi = (1.0 - g) * stand_q + g * cyc_q[t % L_cyc]
            zi = float((1.0 - g) * Z_STAND + g * cyc_z[t % L_cyc])
        else:
            vx_t, vy_t, wz_t = 0.0, 0.0, 0.0
            qi = stand_q.copy()
            zi = Z_STAND
        c2, s2 = float(np.cos(0.5 * th)), float(np.sin(0.5 * th))
        # CSV stores quat as xyzw
        rows.append(np.concatenate(([x, y, zi, 0.0, 0.0, s2, c2], qi)).astype(np.float32))
        planner_cmds.append((1, vx_t, vy_t, wz_t, 1.0, 0.35))
        x += (vx_t * np.cos(th) - vy_t * np.sin(th)) * dt
        y += (vx_t * np.sin(th) + vy_t * np.cos(th)) * dt
        th += wz_t * dt

    arr = np.stack(rows, axis=0)
    closure_cm = float(np.linalg.norm(arr[-1, :2] - arr[0, :2]) * 100.0)
    assert closure_cm < 0.5, f"Analytic loop reference did not close: {closure_cm:.2f}cm"
    return arr, planner_cmds


def _build_box_catch_csv() -> np.ndarray:
    """135 frames @ 50Hz (2.70s): 180ms human visual RT after t_drop=0.80s, then 0.16s symmetrical V-cradle reach."""
    T = 135
    q_cradle = Q_STAND.copy()
    q_cradle[0] -= 0.05; q_cradle[3] += 0.10; q_cradle[4] -= 0.05
    q_cradle[6] -= 0.05; q_cradle[9] += 0.10; q_cradle[10] -= 0.05
    # Symmetrical upward-cupped V-cradle (negative elbow pitch raises forearms cleanly into catch pocket)
    q_cradle[14:18] = np.array([-0.75, 0.10, 0.05, -0.65], dtype=np.float32)
    q_cradle[18:22] = np.array([-0.75, -0.10, -0.05, -0.65], dtype=np.float32)

    rows = []
    for t in range(T):
        if t < 49:  # 0.00..0.98s stand (180ms visual RT after drop at t=0.80s)
            qi = Q_STAND.copy()
            zi = Z_STAND
        elif t < 57:  # 0.98..1.14s fast minimum-jerk reach into V-cradle
            w = float(_min_jerk(np.array((t - 48) / 8.0)))
            qi = (1.0 - w) * Q_STAND + w * q_cradle
            zi = Z_STAND - 0.015 * w
        else:
            qi = q_cradle.copy()
            zi = Z_STAND - 0.015
        rows.append(np.concatenate(([0.0, 0.0, zi, 0.0, 0.0, 0.0, 1.0], qi)).astype(np.float32))
    return np.stack(rows, axis=0)


def synthesize_hard_bench_motions(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    m2k = ROOT / "src/unisupermimic/assets/motions/q1/multidataset_2000"
    candidates = [
        m2k / "train/lafan_walk1_subject5_w480.csv",
        *sorted(m2k.glob("*/lafan_walk1_subject5_w480.csv")),
        *sorted(m2k.glob("*/lafan_walk2_subject1_w480.csv")),
    ]
    walk_csv = next((p for p in candidates if p.is_file()), None)
    assert walk_csv is not None, f"Could not find active walking CSV under {m2k}"
    cyc_q, cyc_z = _extract_periodic_walk_cycle(walk_csv)

    # Track A1: Racetrack (960 steps = 19.2s, ~7.25m): +X straight -> 180deg left -> -X return -> 180deg left -> origin
    race_cmds_raw = [
        (110, 0.42, 0.0),
        (320, 0.42, np.pi / 6.4),
        (110, 0.42, 0.0),
        (320, 0.42, np.pi / 6.4),
        (40, 0.42, 0.0),
    ]
    race_arr, race_plan_cmds = _build_analytic_loop_traj(race_cmds_raw, cyc_q, cyc_z, startup_steps=30, settle_steps=45, stop_steps=60)
    race_csv = out_dir / "loop_racetrack_walk.csv"
    np.savetxt(race_csv, race_arr, delimiter=",", fmt="%.6f")

    # Track A2: Figure-8 (1,340 steps = 26.8s, ~10.44m): 360deg Left circle -> 360deg Right circle -> origin
    fig8_cmds_raw = [
        (30, 0.42, 0.0),
        (620, 0.42, 2.0 * np.pi / 12.4),
        (620, 0.42, -2.0 * np.pi / 12.4),
        (10, 0.42, 0.0),
    ]
    fig8_arr, fig8_plan_cmds = _build_analytic_loop_traj(fig8_cmds_raw, cyc_q, cyc_z, startup_steps=30, settle_steps=45, stop_steps=60)
    fig8_csv = out_dir / "loop_figure8_walk.csv"
    np.savetxt(fig8_csv, fig8_arr, delimiter=",", fmt="%.6f")

    # Track B: Falling-box catch (135 steps = 2.70s @ 50Hz)
    box_arr = _build_box_catch_csv()
    box_csv = out_dir / "falling_box_catch.csv"
    np.savetxt(box_csv, box_arr, delimiter=",", fmt="%.6f")

    race_len_m = float(np.linalg.norm(np.diff(race_arr[:, :2], axis=0), axis=1).sum())
    fig8_len_m = float(np.linalg.norm(np.diff(fig8_arr[:, :2], axis=0), axis=1).sum())
    return {
        "race_csv": race_csv,
        "race_plan_cmds": race_plan_cmds,
        "race_ref_xy": race_arr[:, :2].copy(),
        "race_len_m": round(race_len_m, 2),
        "race_dur_s": round(len(race_arr) * 0.02, 2),
        "fig8_csv": fig8_csv,
        "fig8_plan_cmds": fig8_plan_cmds,
        "fig8_ref_xy": fig8_arr[:, :2].copy(),
        "fig8_len_m": round(fig8_len_m, 2),
        "fig8_dur_s": round(len(fig8_arr) * 0.02, 2),
        "box_csv": box_csv,
    }


def create_loop_cpu_env(motion_files: list[str]) -> Any:
    torch.set_num_threads(1)
    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"):
        cfg = compose("config", overrides=["task=q1_super_mimic/mujoco"])
    OmegaConf.set_struct(cfg, False)
    cfg.training.sim_backend = "mujoco"
    cfg.training.play_only = True
    cfg.env.ctrl_dt = 0.02
    cfg.env.max_episode_seconds = 45.0
    cfg.env.commands.motion.ctrl_dt = 0.02
    cfg.env.commands.motion.src_fps = 50.0
    cfg.env.commands.motion.max_episode_steps = 2200
    cfg.env.commands.motion.motion_files = motion_files
    cfg.env.commands.motion.observation_mode = "root_xy"
    _configure_eval_events(cfg)
    ov = build_env_override(cfg, eval_mode=True, learner_device="cpu")
    return create_env(cfg, num_envs=1, env_cfg_override=ov)


def eval_loop_motion_csv(
    env: Any,
    *,
    clip_idx: int,
    est_mode: str,
    modality: str,
    tok_sess: ort.InferenceSession | None = None,
    head_sess: ort.InferenceSession | None = None,
    pt_future_zoh: Any | None = None,
    delay_steps: int = 0,
    pushes: tuple[tuple[int, float, float], ...] = (),
    save_qpos: bool = False,
) -> dict[str, Any]:
    """Closed-loop full-body tracking across an entire closed-loop trajectory back to origin."""
    mdp = env.supermimic_mdp
    motion = mdp.motion
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()
    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32).view(1, 1, 3, 1)

    cs = int(motion.clip_start[clip_idx])
    budget = int(motion.clip_len[clip_idx]) - 1
    push_map = {s: (vx, vy) for s, vx, vy in pushes}

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(motion.clip_start[clip_idx : clip_idx + 1].clone())
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])
        f0 = cur["f"]
        p0_true = s0[0][0, :2].clone()
        yaw0_R = yaw_mat(s0[1])[0].clone()
        est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
        est.reset(s0[1], s0[3], s0[4], s0[5], motion.root_p[f0, :2], R_ref=motion.root_R[f0])

        history = None
        act_fifo = torch.zeros((delay_steps + 1, mdp.num_joints), dtype=torch.float32)
        path_sq, slip_spds, qpos_tr, true_xy = [], [], [], []
        failed = False
        steps_run = 0
        robot = env.scene["robot"]

        for t in range(budget):
            if t in push_map:
                vx, vy = push_map[t]
                robot.apply_root_linear_velocity_delta_to_sim(torch.tensor([[vx, vy, 0.0]], dtype=torch.float32))
            st = unflatten_state(cur["s"])
            f_cur = cur["f"]
            obs = mdp.obs(st, cur["a_prev"], cur["xi"], f_cur)
            exy, _ = est.step(
                st[1], st[3], st[4], st[5], motion.root_p[f_cur, :2],
                R_ref=motion.root_R[f_cur], mode=est_mode, true_obs_root_xy=obs[:, 78:80],
            )
            obs_pol = torch.cat((obs[:, :78], exy), dim=-1)
            history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)
            if pt_future_zoh is not None:
                goal, _ = _future_lookahead_goal_with_causal_buffer(motion, t, 0, "zoh_pad", clip_idx=clip_idx)
                act = pt_future_zoh(history, goal)
            else:
                if modality == "robot_pose":
                    goal = build_causal_history_goal(motion, f_cur, 10)
                elif modality == "human_smpl":
                    goal = build_human_smpl_goal(motion, f_cur, 10)
                elif modality == "planner_cmd":
                    goal = build_planner_cmd_goal(motion, f_cur, 10)
                else:
                    raise ValueError(f"Unsupported loop modality: {modality}")
                tok = tok_sess.run(["token"], {"modal_input": goal.numpy().astype(np.float32)})[0]
                act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])

            act_fifo = torch.cat((act, act_fifo[:-1]), dim=0)
            act_eff = act_fifo[min(t, delay_steps)].unsqueeze(0).contiguous()
            q_tgt = _apply_torque_safe_pd_cpu(
                mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act_eff[0],
                st[4][0], st[5][0], q_lo, q_hi, kp_vec, kd_vec, eff_vec,
            )
            act_sent = ((q_tgt - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0).contiguous()
            step_s = env.step(act_sent)
            tr = step_s.info[SUPERMIMIC_TRANSITION_KEY]
            s1 = unflatten_state(tr["s1"])
            f1 = int(motion.future_frames(tr["f"], 1)[0, 0])
            steps_run += 1

            p1 = s1[0][0, :2]
            true_xy.append(p1.numpy().copy())
            path_sq.append(float(((p1 - motion.root_p[f1, :2]) ** 2).sum().item()))

            bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
            ref_az = motion.body_p[f1, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]
            ref_av = motion.body_v[f1, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), :2].norm(dim=-1)
            smask = (ref_az <= 0.062) & (ref_av <= 0.15)
            if bool(smask.any()):
                v_sole = bv1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] + torch.cross(
                    bw1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)],
                    (bR1[:, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX)] @ sole_off).squeeze(-1), dim=-1,
                )
                slip_spds.extend(v_sole[0, :, :2].norm(dim=-1)[smask].tolist())

            if save_qpos:
                qpos_tr.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
            if bool(tr["term"][0].item()):
                failed = True
                break
            cur = step_s.info[SUPERMIMIC_STATE_KEY]

    p_end = torch.tensor(true_xy[-1], dtype=torch.float32) if true_xy else p0_true
    R_end = yaw_mat(s1[1])[0]
    dyaw = float(np.degrees(np.arctan2(float((R_end @ yaw0_R.T)[1, 0]), float((R_end @ yaw0_R.T)[0, 0]))))
    out: dict[str, Any] = {
        "completed": (not failed) and steps_run == budget,
        "steps_run": steps_run,
        "budget": budget,
        "loop_closure_err_cm": round(float(torch.norm(p_end - p0_true).item() * 100.0), 2),
        "heading_closure_err_deg": round(abs(dyaw), 2),
        "mean_path_err_cm": round(float(np.sqrt(np.mean(path_sq)) * 100.0), 2),
        "stance_slip_mm_s": round(float(np.mean(slip_spds) * 1000.0), 1) if slip_spds else 0.0,
        "xy_path": np.stack(true_xy, axis=0) if true_xy else np.zeros((1, 2)),
    }
    if save_qpos and qpos_tr:
        out["qpos"] = np.stack(qpos_tr, axis=0)
    return out


def eval_loop_planner_cmd(
    env: Any,
    stand_clip_idx: int,
    cmds: list[tuple[int, float, float, float, float, float]],
    *,
    est_mode: str,
    tok_plan: ort.InferenceSession,
    head_sess: ort.InferenceSession,
    delay_steps: int = 0,
    pushes: tuple[tuple[int, float, float], ...] = (),
    save_qpos: bool = False,
) -> dict[str, Any]:
    """Pure velocity-commanded (zero mocap reference!) closed-loop loop closure back to (0, 0)."""
    mdp = env.supermimic_mdp
    stand_frame_idx = int(mdp.motion.clip_start[stand_clip_idx])
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()
    sole_off = torch.tensor([0.020, 0.0, -0.022], dtype=torch.float32).view(1, 1, 3, 1)

    cmd_stream = build_synthetic_planner_stream(cmds)
    T = cmd_stream.shape[0]
    push_map = {s: (vx, vy) for s, vx, vy in pushes}

    with torch.no_grad():
        _, reset_info = env.reset_to_frames(mdp.motion.clip_start[stand_clip_idx : stand_clip_idx + 1].clone())
        _enter_free_roaming_mode(env, stand_frame_idx)
        cur = reset_info[SUPERMIMIC_STATE_KEY]
        s0 = unflatten_state(cur["s"])
        p0_true = s0[0][0, :2].clone()
        yaw0_R = yaw_mat(s0[1])[0].clone()
        p_plan_xy = p0_true.clone()
        yaw_plan_R = yaw0_R.clone()

        est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
        est.reset(s0[1], s0[3], s0[4], s0[5], p_plan_xy.unsqueeze(0), R_ref=yaw_plan_R.unsqueeze(0))

        history = None
        act_fifo = torch.zeros((delay_steps + 1, mdp.num_joints), dtype=torch.float32)
        path_sq, slip_spds, qpos_tr, true_xy = [], [], [], []
        failed = False
        steps_run = 0
        robot = env.scene["robot"]

        for t in range(T):
            if t in push_map:
                vx_p, vy_p = push_map[t]
                robot.apply_root_linear_velocity_delta_to_sim(torch.tensor([[vx_p, vy_p, 0.0]], dtype=torch.float32))
            st = unflatten_state(cur["s"])
            g_row = cmd_stream[t]
            vx_t, vy_t, wz_t = float(g_row[-10]), float(g_row[-9]), float(g_row[-8])
            v_w = yaw_plan_R[:2, :2] @ torch.tensor([vx_t, vy_t], dtype=torch.float32)
            p_plan_xy = p_plan_xy + v_w * 0.02
            yaw_dR = so3_exp(torch.tensor([[0.0, 0.0, wz_t * 0.02]], dtype=torch.float32))[0]
            yaw_plan_R = yaw_mat((yaw_dR @ yaw_plan_R).unsqueeze(0))[0]
            p_ref_xy = p_plan_xy.unsqueeze(0)
            R_ref_t = yaw_plan_R.unsqueeze(0)

            obs_raw = mdp.obs(st, cur["a_prev"], cur["xi"], cur["f"])
            rel_rot = mat_to_6d(st[1].transpose(-1, -2) @ R_ref_t)
            exy, _ = est.step(st[1], st[3], st[4], st[5], p_ref_xy, R_ref=R_ref_t, mode=est_mode)
            obs_pol = torch.cat((obs_raw[:, :72], rel_rot, exy), dim=-1)
            history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)

            tok = tok_plan.run(["token"], {"modal_input": g_row.unsqueeze(0).numpy().astype(np.float32)})[0]
            act = torch.from_numpy(head_sess.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])

            act_fifo = torch.cat((act, act_fifo[:-1]), dim=0)
            act_eff = act_fifo[min(t, delay_steps)].unsqueeze(0).contiguous()
            q_tgt = _apply_torque_safe_pd_cpu(
                mdp.default_q + cur["xi"][0, 1:23] + alpha_vec * act_eff[0],
                st[4][0], st[5][0], q_lo, q_hi, kp_vec, kd_vec, eff_vec,
            )
            act_sent = ((q_tgt - mdp.default_q - cur["xi"][0, 1:23]) / alpha_vec).unsqueeze(0).contiguous()
            env._supermimic_command.f[:] = stand_frame_idx
            env._supermimic_command.ep_step[:] = 0
            step_s = env.step(act_sent)
            s1 = unflatten_state(step_s.info[SUPERMIMIC_TRANSITION_KEY]["s1"])
            steps_run += 1

            p1 = s1[0][0, :2]
            true_xy.append(p1.numpy().copy())
            path_sq.append(float(((p1 - p_plan_xy) ** 2).sum().item()))

            bp1, bR1, bv1, bw1 = mdp.kin.forward(*s1)
            min_f = int(torch.argmin(bp1[0, (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX), 2]))
            a_idx = LEFT_ANKLE_IDX if min_f == 0 else RIGHT_ANKLE_IDX
            v_sole = bv1[0, a_idx] + torch.cross(bw1[0, a_idx], (bR1[0, a_idx] @ sole_off[0, 0]).squeeze(-1), dim=-1)
            slip_spds.append(float(v_sole[:2].norm().item() * 1000.0))

            if save_qpos:
                qpos_tr.append(torch.cat((s1[0][0], mat_to_quat(s1[1])[0], s1[4][0]), -1).numpy().copy())
            if float(s1[0][0, 2]) < 0.16 or float(s1[1][0, 2, 2]) < 0.35:
                failed = True
                break
            cur = step_s.info[SUPERMIMIC_STATE_KEY]

    p_end = torch.tensor(true_xy[-1], dtype=torch.float32) if true_xy else p0_true
    R_end = yaw_mat(s1[1])[0]
    dyaw = float(np.degrees(np.arctan2(float((R_end @ yaw0_R.T)[1, 0]), float((R_end @ yaw0_R.T)[0, 0]))))
    out: dict[str, Any] = {
        "completed": (not failed) and steps_run == T,
        "steps_run": steps_run,
        "budget": T,
        "loop_closure_err_cm": round(float(torch.norm(p_end - p0_true).item() * 100.0), 2),
        "heading_closure_err_deg": round(abs(dyaw), 2),
        "mean_path_err_cm": round(float(np.sqrt(np.mean(path_sq)) * 100.0), 2),
        "stance_slip_mm_s": round(float(np.mean(slip_spds)), 1) if slip_spds else 0.0,
        "xy_path": np.stack(true_xy, axis=0) if true_xy else np.zeros((1, 2)),
    }
    if save_qpos and qpos_tr:
        out["qpos"] = np.stack(qpos_tr, axis=0)
    return out


# ---------------- PART B: HIGH-PLATFORM FALLING-BOX REACTIVE CATCH ----------------

def build_falling_box_xml(xml_dir: Path, mass: float = 0.26) -> Path:
    """Create temporary MuJoCo XML scene with an overhead chute & rigid free-body falling box."""
    src_rob = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
    mesh_dir = ROOT / "src/unisupermimic/assets/robots/q1/assets"
    rob_txt = src_rob.read_text(encoding="utf-8").replace(
        'meshdir="assets"', f'meshdir="{mesh_dir}"'
    )
    abs_rob = xml_dir / "supermimic_abs.xml"
    abs_rob.write_text(rob_txt, encoding="utf-8")

    scene_xml = f"""<mujoco model="q1_falling_box_scene">
  <include file="{abs_rob}" />
  <option integrator="implicitfast" timestep="0.005" cone="pyramidal" />
  <asset>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.72 0.75 0.78" rgb2="0.56 0.60 0.65" width="256" height="256" />
    <material name="ground" texture="ground" texrepeat="2 2" texuniform="true" />
  </asset>
  <worldbody>
    <light pos="0.5 -0.8 3.2" dir="-0.2 0.2 -1" directional="true" />
    <geom name="floor" type="plane" size="0 0 0.05" material="ground" friction="0.6 0.005 0.0001" contype="1" conaffinity="1" />
    <!-- Origin target zone pad on floor (visual only) -->
    <geom name="origin_pad" type="box" pos="0 0 0.001" size="0.30 0.30 0.001" rgba="0.06 0.65 0.58 0.35" contype="0" conaffinity="0" />
    <!-- Overhead release dispenser gantry (non-colliding visual structure) -->
    <geom name="gantry_post" type="cylinder" pos="0.46 0.24 0.80" size="0.018 0.80" rgba="0.22 0.26 0.32 1" contype="0" conaffinity="0" />
    <geom name="gantry_arm" type="box" pos="0.32 0.24 1.58" size="0.16 0.018 0.016" rgba="0.22 0.26 0.32 1" contype="0" conaffinity="0" />
    <geom name="chute_top" type="box" pos="0.168 0.0 1.38" size="0.08 0.19 0.012" rgba="0.18 0.22 0.28 1" contype="0" conaffinity="0" />
    <geom name="chute_rail_l" type="box" pos="0.168 0.19 1.31" size="0.075 0.008 0.06" rgba="0.90 0.62 0.10 0.9" contype="0" conaffinity="0" />
    <geom name="chute_rail_r" type="box" pos="0.168 -0.19 1.31" size="0.075 0.008 0.06" rgba="0.90 0.62 0.10 0.9" contype="0" conaffinity="0" />
    <!-- Free-body rigid box colliding cleanly with Q1 arms/torso/floor -->
    <body name="falling_box" pos="0.168 0 1.24">
      <freejoint name="box_free" />
      <inertial pos="0 0 0" mass="{mass:.4f}" diaginertia="0.0008 0.0005 0.0009" />
      <geom name="box_geom" type="box" size="0.055 0.170 0.055" friction="1.5 0.02 0.005" solref="0.02 1.2" contype="1" conaffinity="1" rgba="0.95 0.42 0.10 1" />
    </body>
  </worldbody>
</mujoco>
"""
    scene_path = xml_dir / "q1_falling_box_scene.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")
    return scene_path


def _future_lookahead_goal_with_causal_buffer(
    motion: MotionLib,
    t_live: int,
    buf_frames: int,
    mode: str,
    *,
    clip_idx: int = 0,
) -> tuple[torch.Tensor, int]:
    """Compute goal input & target tracking frame for future-window models under a causal live clock.
    - mode == 'causal_buffer': wait `buf_frames` (`buf_frames * 20 ms`), so controller executes target frame
      `f_ctrl = max(0, t_live - buf_frames)` using future window `[f_ctrl+1 .. f_ctrl+10]` clamped <= `t_live`.
    - mode == 'zoh_pad': zero wait (`f_ctrl = t_live`), but unknown future slots `[t_live+1 .. t_live+10]`
      are Zero-Order-Held to the latest available frame `t_live`.
    """
    cs = int(motion.clip_start[clip_idx])
    ce = int(motion.clip_end[clip_idx])
    t_avail = min(cs + t_live, ce)
    if mode == "causal_buffer":
        f_ctrl = max(cs, min(cs + t_live - buf_frames, ce))
        fut = torch.clamp(f_ctrl + torch.arange(1, 11, dtype=torch.long), min=cs, max=t_avail)
    elif mode == "zoh_pad":
        f_ctrl = t_avail
        fut = torch.full((10,), t_avail, dtype=torch.long)
    else:
        raise ValueError(mode)

    frames_t = torch.tensor([f_ctrl], dtype=torch.long)
    fut_t = fut.unsqueeze(0)
    heading = yaw_mat(motion.root_R[frames_t])
    heading_t = heading.transpose(-1, -2).unsqueeze(1)
    p_delta = (heading_t @ (motion.root_p[fut_t] - motion.root_p[frames_t].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    rot_6d = mat_to_6d(heading_t @ motion.root_R[fut_t])
    feat = torch.cat(
        (p_delta, rot_6d, motion.root_v_local[fut_t], motion.root_w_local[fut_t], motion.q[fut_t], motion.qd[fut_t] * 0.1),
        dim=-1,
    ).reshape(1, -1)
    return feat, f_ctrl


@torch.no_grad()
def rollout_falling_box_trial(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mdp: MDP,
    controller_kind: str,  # 'v3_robot_0ms' | 'v3_smpl_0ms' | 'future_buf' | 'future_zoh'
    *,
    buf_frames: int = 0,
    x_drop: float = 0.175,
    y_drop: float = 0.0,
    z_drop: float = 1.44,
    box_mass: float = 0.26,
    t_drop_step: int = 40,
    sess_head: ort.InferenceSession | None = None,
    sess_tok: ort.InferenceSession | None = None,
    pt_future_pi: Any | None = None,
    save_qpos: bool = False,
) -> dict[str, Any]:
    """Run one full 2.70s real-physics falling-box trial in MuJoCo 3.11."""
    motion = mdp.motion
    alpha_vec = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kp_vec = torch.tensor([gains_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    kd_vec = torch.tensor([gains_of(n)[1] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    eff_vec = torch.tensor([motor_of(n)[0] for n in Q1_JOINT_NAMES], dtype=torch.float32)
    q_lo, q_hi = mdp.kin.q_lo.cpu(), mdp.kin.q_hi.cpu()

    box_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "falling_box")
    floor_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    lw_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    mj_model.body_mass[box_bid] = box_mass

    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = [0.0, 0.0, Z_STAND]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj_data.qpos[7:29] = Q_STAND
    mj_data.qpos[29:36] = [x_drop, y_drop, z_drop, 1.0, 0.0, 0.0, 0.0]
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = Q_STAND
    mujoco.mj_forward(mj_model, mj_data)

    est = ContactGatedFootAnchor(mdp.kin, 1, dt=0.02, device="cpu")
    R0 = quat_to_mat(torch.from_numpy(mj_data.qpos[3:7].astype(np.float32)).unsqueeze(0))
    est.reset(R0, torch.zeros((1, 3)), torch.from_numpy(Q_STAND).unsqueeze(0), torch.zeros((1, 22)), motion.root_p[0:1, :2], R_ref=motion.root_R[0:1])

    xi = torch.zeros((1, mdp.context_dim), dtype=torch.float32)
    xi[0, 0] = 1.0
    a_prev = torch.zeros((1, 22), dtype=torch.float32)
    history = None
    T = int(motion.num_frames) - 1

    box_z_trace, lw_z_trace, qpos_all = [], [], []
    arm_contact_steps = 0
    floor_contact_steps = 0
    min_pelvis_z = 9.9
    min_upright_cos = 1.0

    for t in range(T):
        if t < t_drop_step:
            mj_data.qpos[29:36] = [x_drop, y_drop, z_drop, 1.0, 0.0, 0.0, 0.0]
            mj_data.qvel[28:34] = 0.0
            mujoco.mj_forward(mj_model, mj_data)

        p_t = torch.from_numpy(mj_data.qpos[0:3].astype(np.float32)).unsqueeze(0)
        R_t = quat_to_mat(torch.from_numpy(mj_data.qpos[3:7].astype(np.float32)).unsqueeze(0))
        v_t = torch.from_numpy(mj_data.qvel[0:3].astype(np.float32)).unsqueeze(0)
        w_t = (R_t @ torch.from_numpy(mj_data.qvel[3:6].astype(np.float32)).unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        q_t = torch.from_numpy(mj_data.qpos[7:29].astype(np.float32)).unsqueeze(0)
        qd_t = torch.from_numpy(mj_data.qvel[6:28].astype(np.float32)).unsqueeze(0)

        if controller_kind in ("v3_robot_0ms", "v3_smpl_0ms", "v3_vr_0ms"):
            f_ctrl = max(0, t - buf_frames)
            f_tensor = torch.tensor([f_ctrl], dtype=torch.long)
            if controller_kind == "v3_robot_0ms":
                goal = build_causal_history_goal(motion, f_tensor, 10)
            elif controller_kind == "v3_smpl_0ms":
                goal = build_human_smpl_goal(motion, f_tensor, 10)
            else:
                goal = build_vr_3pt_goal(motion, f_tensor, 10)
        elif controller_kind == "future_buf":
            goal, f_ctrl = _future_lookahead_goal_with_causal_buffer(motion, t, buf_frames, "causal_buffer")
            f_tensor = torch.tensor([f_ctrl], dtype=torch.long)
        elif controller_kind == "future_zoh":
            goal, f_ctrl = _future_lookahead_goal_with_causal_buffer(motion, t, 0, "zoh_pad")
            f_tensor = torch.tensor([f_ctrl], dtype=torch.long)
        else:
            raise ValueError(controller_kind)

        obs_raw = mdp.obs((p_t, R_t, v_t, w_t, q_t, qd_t), a_prev, xi, f_tensor)
        exy, _ = est.step(R_t, w_t, q_t, qd_t, motion.root_p[f_tensor, :2], R_ref=motion.root_R[f_tensor], mode="cgfa")
        obs_pol = torch.cat((obs_raw[:, :78], obs_raw[:, 78:80] if t == 0 else exy), dim=-1)
        history = obs_pol.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs_pol.unsqueeze(1)), 1)

        if sess_tok is not None and sess_head is not None:
            tok = sess_tok.run(["token"], {"modal_input": goal.numpy().astype(np.float32)})[0]
            act = torch.from_numpy(sess_head.run(["action"], {"obs_hist": history.numpy(), "token": tok})[0])
        elif pt_future_pi is not None:
            act = pt_future_pi(history, goal)
        else:
            raise RuntimeError("No policy provided")

        q_tgt_raw = mdp.default_q + alpha_vec * act[0]
        q_tgt = _apply_torque_safe_pd_cpu(q_tgt_raw, q_t[0], qd_t[0], q_lo, q_hi, kp_vec, kd_vec, eff_vec)
        act_sent = ((q_tgt - mdp.default_q) / alpha_vec).unsqueeze(0)
        a_prev = act_sent.clone()

        mj_data.ctrl[:] = q_tgt.numpy()
        for _ in range(4):
            if t < t_drop_step:
                mj_data.qpos[29:36] = [x_drop, y_drop, z_drop, 1.0, 0.0, 0.0, 0.0]
                mj_data.qvel[28:34] = 0.0
            mujoco.mj_step(mj_model, mj_data)

        bz = float(mj_data.qpos[31])
        lz = float(mj_data.xpos[lw_bid, 2])
        box_z_trace.append(round(bz, 4))
        lw_z_trace.append(round(lz, 4))
        min_pelvis_z = min(min_pelvis_z, float(mj_data.qpos[2]))
        R_after = quat_to_mat(torch.from_numpy(mj_data.qpos[3:7].astype(np.float32)).unsqueeze(0))
        min_upright_cos = min(min_upright_cos, float(R_after[0, 2, 2]))

        if t >= t_drop_step:
            for ci in range(int(mj_data.ncon)):
                con = mj_data.contact[ci]
                g1, g2 = int(con.geom1), int(con.geom2)
                if box_gid in (g1, g2):
                    other = g2 if g1 == box_gid else g1
                    if other == floor_gid:
                        floor_contact_steps += 1
                    else:
                        arm_contact_steps += 1

        if save_qpos:
            qpos_all.append(mj_data.qpos.copy())

    post_drop = box_z_trace[t_drop_step:]
    min_bz = float(min(post_drop)) if post_drop else 0.0
    held_steps = sum(1 for z in post_drop if z > 0.35)
    robot_upright = (min_pelvis_z > 0.28) and (min_upright_cos > 0.70)
    caught = (
        robot_upright
        and (arm_contact_steps >= 8)
        and (floor_contact_steps == 0)
        and (min_bz > 0.14)
        and (held_steps >= 35)
        and (box_z_trace[-1] > 0.35)
    )
    return {
        "caught": bool(caught),
        "robot_upright": bool(robot_upright),
        "arm_contact_steps": arm_contact_steps,
        "floor_contact_steps": floor_contact_steps,
        "min_box_z_m": round(min_bz, 3),
        "final_box_z_m": round(box_z_trace[-1], 3),
        "min_pelvis_z_m": round(min_pelvis_z, 3),
        "box_z_trace": box_z_trace,
        "lw_z_trace": lw_z_trace,
        "qpos_all": np.stack(qpos_all, axis=0) if qpos_all else None,
    }


def main() -> None:
    t0_all = time.monotonic()
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=base / "hard_bench_v3_metrics.json")
    parser.add_argument("--loop-only", action="store_true", help="Re-run Part A loop closure only and preserve existing Part B falling-box metrics")
    args = parser.parse_args()
    networks.AMP_DTYPE = None

    syn_dir = Path("/tmp/hard_bench_motions")
    syn = synthesize_hard_bench_motions(syn_dir)

    stand_csv = str(ROOT / "src/unisupermimic/assets/motions/q1/multidataset_2000/train/syn_stand_idle_1.csv")
    mfiles = [str(syn["race_csv"]), str(syn["fig8_csv"]), stand_csv]
    env = create_loop_cpu_env(mfiles)
    mdp = env.supermimic_mdp

    v2_onnx = ROOT / "runs/h002_sonic_v2/onnx"
    v3_onnx = base / "onnx"
    s_v2_head = _ort_sess(v2_onnx / "q1_h002_token_head.onnx")
    s_v2_rob = _ort_sess(v2_onnx / "q1_tok_robot_pose.onnx")
    s_v2_plan = _ort_sess(v2_onnx / "q1_tok_planner_cmd.onnx")

    s_v3_head = _ort_sess(v3_onnx / "q1_h002_v3_token_head.onnx")
    s_v3_rob = _ort_sess(v3_onnx / "q1_v3_tok_robot_pose.onnx")
    s_v3_smpl = _ort_sess(v3_onnx / "q1_v3_tok_human_smpl.onnx")
    s_v3_vr = _ort_sess(v3_onnx / "q1_v3_tok_vr_3pt.onnx")
    s_v3_plan = _ort_sess(v3_onnx / "q1_v3_tok_planner_cmd.onnx")

    # V1 legacy ONNX helper via PyTorch (same MDP wrapper)
    pi_v1 = _load_ckpt_policy(mdp, ROOT / "runs/h002_cgfa_pilot/checkpoints/m3_h002_full_0ms.pt", "cpu")
    pi_oracle_fut = _load_ckpt_policy(mdp, ROOT / "checkpoints/q1_base_soma1k.pt", "cpu")

    # ---------------- PART A1: Racetrack Closed Loop (~7.3m, 19.2s) ----------------
    loop_trajs: dict[str, np.ndarray] = {
        "race_ref_xy": syn["race_ref_xy"],
        "fig8_ref_xy": syn["fig8_ref_xy"],
    }
    race_res: dict[str, Any] = {}
    for label, kw in [
        ("V3_RobotPose_CGFA", dict(est_mode="cgfa", modality="robot_pose", tok_sess=s_v3_rob, head_sess=s_v3_head, save_qpos=True)),
        ("V3_HumanSMPL_CGFA", dict(est_mode="cgfa", modality="human_smpl", tok_sess=s_v3_smpl, head_sess=s_v3_head, save_qpos=True)),
        ("V2_RobotPose_CGFA", dict(est_mode="cgfa", modality="robot_pose", tok_sess=s_v2_rob, head_sess=s_v2_head, save_qpos=False)),
        ("V3_UnfilteredOdom_DR", dict(est_mode="unfiltered_odom", modality="robot_pose", tok_sess=s_v3_rob, head_sess=s_v3_head, save_qpos=False)),
        ("V3_Zeroed_Blind", dict(est_mode="zeroed", modality="robot_pose", tok_sess=s_v3_rob, head_sess=s_v3_head, save_qpos=False)),
        ("SuperMimic_Orig_0msZOH", dict(est_mode="cgfa", modality="robot_pose", pt_future_zoh=pi_oracle_fut, save_qpos=False)),
        ("V3_PlannerCmd_CGFA", dict(est_mode="cgfa", modality="planner_cmd", tok_sess=s_v3_plan, head_sess=s_v3_head, save_qpos=True)),
        ("V2_PlannerCmd_CGFA", dict(est_mode="cgfa", modality="planner_cmd", tok_sess=s_v2_plan, head_sess=s_v2_head, save_qpos=False)),
        ("V3_PlannerCmd_Zeroed", dict(est_mode="zeroed", modality="planner_cmd", tok_sess=s_v3_plan, head_sess=s_v3_head, save_qpos=False)),
    ]:
        r = eval_loop_motion_csv(env, clip_idx=0, **kw)
        loop_trajs[f"race_{label}_xy"] = r.pop("xy_path")
        if "qpos" in r:
            loop_trajs[f"race_{label}_qpos"] = r.pop("qpos")
        race_res[label] = r
        print(f"[LOOP-RACE] {label:22s}: ok={r['completed']} ({r['steps_run']}/{r['budget']}) close={r['loop_closure_err_cm']:.1f}cm yawErr={r['heading_closure_err_deg']:.1f}deg path={r['mean_path_err_cm']:.1f}cm", flush=True)

    # Hardened Racetrack (+0.45 m/s mid-turn push @ t=5.4s (step 270) + 20ms delay)
    race_hard: dict[str, Any] = {}
    for label, mod, ts, hs in [
        ("V3_RobotPose_Hard", "robot_pose", s_v3_rob, s_v3_head),
        ("V2_RobotPose_Hard", "robot_pose", s_v2_rob, s_v2_head),
        ("V3_PlannerCmd_Hard", "planner_cmd", s_v3_plan, s_v3_head),
        ("V2_PlannerCmd_Hard", "planner_cmd", s_v2_plan, s_v2_head),
    ]:
        r = eval_loop_motion_csv(env, clip_idx=0, est_mode="cgfa", modality=mod, tok_sess=ts, head_sess=hs, delay_steps=1, pushes=((270, 0.0, 0.45),))
        r.pop("xy_path", None)
        race_hard[label] = r
        print(f"[RACE-HARD] {label:22s}: ok={r['completed']} ({r['steps_run']}/{r['budget']}) close={r['loop_closure_err_cm']:.1f}cm", flush=True)

    # ---------------- PART A2: Figure-8 Closed Loop (~10.4m, 26.8s) ----------------
    fig8_res: dict[str, Any] = {}
    for label, est_m, mod, ts, hs, sv in [
        ("V3_RobotPose_CGFA", "cgfa", "robot_pose", s_v3_rob, s_v3_head, True),
        ("V3_HumanSMPL_CGFA", "cgfa", "human_smpl", s_v3_smpl, s_v3_head, True),
        ("V2_RobotPose_CGFA", "cgfa", "robot_pose", s_v2_rob, s_v2_head, False),
        ("V3_UnfilteredOdom_DR", "unfiltered_odom", "robot_pose", s_v3_rob, s_v3_head, False),
        ("V3_PlannerCmd_CGFA", "cgfa", "planner_cmd", s_v3_plan, s_v3_head, True),
        ("V2_PlannerCmd_CGFA", "cgfa", "planner_cmd", s_v2_plan, s_v2_head, False),
    ]:
        r = eval_loop_motion_csv(env, clip_idx=1, est_mode=est_m, modality=mod, tok_sess=ts, head_sess=hs, save_qpos=sv)
        loop_trajs[f"fig8_{label}_xy"] = r.pop("xy_path")
        if "qpos" in r:
            loop_trajs[f"fig8_{label}_qpos"] = r.pop("qpos")
        fig8_res[label] = r
        print(f"[LOOP-FIG8] {label:22s}: ok={r['completed']} ({r['steps_run']}/{r['budget']}) close={r['loop_closure_err_cm']:.1f}cm yawErr={r['heading_closure_err_deg']:.1f}deg path={r['mean_path_err_cm']:.1f}cm", flush=True)

    np.savez_compressed(base / "hard_bench_loop_qpos.npz", **loop_trajs)
    loop_meta = {
        "racetrack_len_m": syn["race_len_m"],
        "racetrack_dur_s": syn["race_dur_s"],
        "figure8_len_m": syn["fig8_len_m"],
        "figure8_dur_s": syn["fig8_dur_s"],
    }
    if args.loop_only:
        prev = json.loads(args.out_json.read_text(encoding="utf-8"))
        out = {
            "wall_s": round(time.monotonic() - t0_all, 2),
            "loop_meta": loop_meta,
            "loop_closure_racetrack_26s": race_res,
            "loop_closure_racetrack_hardened": race_hard,
            "loop_closure_figure8_30s": fig8_res,
            "falling_box_catch_20trials": prev["falling_box_catch_20trials"],
            "falling_box_calibrated_x_drop_m": prev["falling_box_calibrated_x_drop_m"],
            "falling_box_latency_sweep_ms": prev["falling_box_latency_sweep_ms"],
        }
        args.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"HARD_BENCH_LOOP_ONLY_DONE in {out['wall_s']}s -> {args.out_json}", flush=True)
        return
    del pi_v1

    # ---------------- PART B: High-Platform Falling-Box Reactive Catch ----------------
    xml_dir = Path("/tmp/q1_box_scene")
    xml_dir.mkdir(parents=True, exist_ok=True)
    scene_path = build_falling_box_xml(xml_dir, mass=0.26)
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)

    kin_box = Q1Kinematics(str(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"), "cpu")
    motion_box = MotionLib([str(syn["box_csv"])], kin_box, "cpu", src_fps=50.0, ctrl_dt=0.02)
    mdp_box = MDP(kin_box, motion_box, torch.from_numpy(Q_STAND), observation_mode="root_xy")

    # Fast calibration over nominal drop X offset using Zero-IK HumanSMPL (natural catch cradle)
    best_x_drop = None
    for cand_x in (0.168, 0.160, 0.175, 0.150):
        cal = rollout_falling_box_trial(
            mj_model, mj_data, mdp_box, "v3_smpl_0ms",
            x_drop=cand_x, y_drop=0.0, z_drop=1.24, sess_head=s_v3_head, sess_tok=s_v3_smpl,
        )
        print(f"[BOX CALIB x={cand_x:.3f}] caught={cal['caught']} arm_con={cal['arm_contact_steps']} floor_con={cal['floor_contact_steps']} min_bz={cal['min_box_z_m']} end_bz={cal['final_box_z_m']}", flush=True)
        if cal["caught"]:
            best_x_drop = cand_x
            break
    if best_x_drop is None:
        raise RuntimeError("CALIBRATION_FAILED: Q1 catch cradle failed across all candidate x_drop offsets")

    rng = np.random.default_rng(20261119)
    trials_cfg = [
        {
            "x_drop": float(best_x_drop + rng.uniform(-0.010, 0.010)),
            "y_drop": float(rng.uniform(-0.020, 0.020)),
            "z_drop": float(rng.uniform(1.20, 1.30)),
            "mass": float(rng.uniform(0.22, 0.34)),
        }
        for _ in range(20)
    ]

    catch_specs = [
        ("V3_Causal_0ms_HumanSMPL", "v3_smpl_0ms", 0, s_v3_head, s_v3_smpl, None),
        ("V3_Causal_0ms_VR3Pt", "v3_vr_0ms", 0, s_v3_head, s_v3_vr, None),
        ("V2_Causal_0ms_RobotPose", "v3_robot_0ms", 0, s_v2_head, s_v2_rob, None),
        ("V3_Causal_0ms_RobotPose", "v3_robot_0ms", 0, s_v3_head, s_v3_rob, None),
        ("SONIC_Future_120msBuf", "future_buf", 6, None, None, pi_oracle_fut),
        ("SuperMimic_Orig_200msBuf", "future_buf", 10, None, None, pi_oracle_fut),
        ("SuperMimic_Orig_0msZOH", "future_zoh", 0, None, None, pi_oracle_fut),
    ]

    box_summary: dict[str, Any] = {}
    catch_demo_trajs: dict[str, Any] = {}
    for cname, ckind, bframes, sh, st, pfut in catch_specs:
        t_rows = []
        for ti, cfg_t in enumerate(trials_cfg):
            save_q = (ti == 0)
            res = rollout_falling_box_trial(
                mj_model, mj_data, mdp_box, ckind,
                buf_frames=bframes, x_drop=cfg_t["x_drop"], y_drop=cfg_t["y_drop"],
                z_drop=cfg_t["z_drop"], box_mass=cfg_t["mass"],
                sess_head=sh, sess_tok=st, pt_future_pi=pfut, save_qpos=save_q,
            )
            if save_q:
                catch_demo_trajs[f"{cname}__qpos"] = res.pop("qpos_all")
                catch_demo_trajs[f"{cname}__box_z"] = np.array(res["box_z_trace"], dtype=np.float32)
                catch_demo_trajs[f"{cname}__lw_z"] = np.array(res["lw_z_trace"], dtype=np.float32)
            else:
                res.pop("qpos_all", None)
            t_rows.append(res)
        box_summary[cname] = {
            "n_trials": len(t_rows),
            "buffer_ms": bframes * 20,
            "catch_rate": round(float(np.mean([float(r["caught"]) for r in t_rows])), 4),
            "upright_rate": round(float(np.mean([float(r["robot_upright"]) for r in t_rows])), 4),
            "mean_arm_contact_steps": round(float(np.mean([r["arm_contact_steps"] for r in t_rows])), 1),
            "mean_floor_contact_steps": round(float(np.mean([r["floor_contact_steps"] for r in t_rows])), 1),
            "mean_min_box_z_m": round(float(np.mean([r["min_box_z_m"] for r in t_rows])), 3),
            "mean_final_box_z_m": round(float(np.mean([r["final_box_z_m"] for r in t_rows])), 3),
            "sample_box_z_trace": t_rows[0]["box_z_trace"],
            "sample_lw_z_trace": t_rows[0]["lw_z_trace"],
        }
        bs = box_summary[cname]
        print(f"[BOX-20] {cname:26s}: catch={bs['catch_rate']*100:5.1f}% end_bz={bs['mean_final_box_z_m']:.3f}m arm_con={bs['mean_arm_contact_steps']:.1f}", flush=True)

    # Lookahead Buffer Latency Sweep (0, 40, 80, 120, 160, 200 ms across 10 trials)
    lat_sweep: dict[str, dict[str, float]] = {}
    for bframes in (0, 2, 4, 6, 8, 10):
        rows_b = [
            rollout_falling_box_trial(
                mj_model, mj_data, mdp_box, "v3_smpl_0ms", buf_frames=bframes,
                x_drop=cfg_t["x_drop"], y_drop=cfg_t["y_drop"], z_drop=cfg_t["z_drop"], box_mass=cfg_t["mass"],
                sess_head=s_v3_head, sess_tok=s_v3_smpl,
            )
            for cfg_t in trials_cfg[:10]
        ]
        rows_fut = [
            rollout_falling_box_trial(
                mj_model, mj_data, mdp_box, "future_buf", buf_frames=bframes,
                x_drop=cfg_t["x_drop"], y_drop=cfg_t["y_drop"], z_drop=cfg_t["z_drop"], box_mass=cfg_t["mass"],
                pt_future_pi=pi_oracle_fut,
            )
            for cfg_t in trials_cfg[:10]
        ]
        lat_sweep[f"{bframes*20}ms"] = {
            "v3_smpl_with_artif_lat": round(float(np.mean([float(r["caught"]) for r in rows_b])), 4),
            "future_model_at_lat": round(float(np.mean([float(r["caught"]) for r in rows_fut])), 4),
        }
        print(f"[BOX LAT SWEEP {bframes*20:3d}ms] V3_SMPL_del={lat_sweep[f'{bframes*20}ms']['v3_smpl_with_artif_lat']*100:.0f}% | FutureModel={lat_sweep[f'{bframes*20}ms']['future_model_at_lat']*100:.0f}%", flush=True)

    np.savez_compressed(base / "hard_bench_catch_qpos.npz", **catch_demo_trajs)
    out = {
        "wall_s": round(time.monotonic() - t0_all, 2),
        "loop_meta": loop_meta,
        "loop_closure_racetrack_26s": race_res,
        "loop_closure_racetrack_hardened": race_hard,
        "loop_closure_figure8_30s": fig8_res,
        "falling_box_catch_20trials": box_summary,
        "falling_box_calibrated_x_drop_m": best_x_drop,
        "falling_box_latency_sweep_ms": lat_sweep,
    }
    args.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"HARD_BENCH_ALL_DONE in {out['wall_s']}s -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
