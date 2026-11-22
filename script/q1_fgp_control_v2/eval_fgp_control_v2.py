"""5-Suite MuJoCo Ablation & Verification for Q1 FGP Cloth Control V2 (`A0` vs `A1` vs `A2` vs `A3`).

Compares:
  - A0 (Prod Baseline) : Deployed V2 ONNX + old FGPSMPLStream + get_latest() + stale-planner fallback + 50Hz 0.2 clip
  - A1 (Smooth Baseline): Deployed V2 ONNX + smooth FGP (no dropouts/debias/vel_cond) + 500Hz 0.2 ramp
  - A2 (Adapter V2 Only): Deployed V2 ONNX + FGPSMPLStreamV2 + FGPHttpSourceV2 + Q1FGPFallbackGuard + 500Hz 0.2 ramp
  - A3 (Full Stack V2)  : FGP-hardened q1_tok_human_smpl.onnx + Adapter V2 + Q1FGPFallbackGuard + 500Hz 0.2 ramp

Across 5 Suites (2 ms physics, 1-thread CPU ORT, zero-torch inference):
  1. Today's real FGP recording (140.9 s with real arrival timing + all 12 real outage flickers)
  2. Held-out real FGP recording (live_20261122_141319 cal_id=2 -22deg belt offset & cal_id=5)
  3. 8 Hard & non-walk primitives (kick, syn_squat_deep, lock_step, turn_walk_360, vr3pt_arm_wave, lafan, phuma, stand)
  4. H002-SONIC v2 50 Held-Out (Nominal + 20-Stress) + 8 Raw Unseen Human BVH Zero-IK
  5. 40 s Continuous Switchable Planner<->HumanSMPL + 0.70x friction + 20ms delay + 400ms real disconnect stress
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

SMHP_ROOT = Path("/home/yu/projects/SuperMimicHP")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_CANDIDATE = Path("/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122")
REC_TODAY = Path("/home/yu/projects/Q1_deploy/real_logs/q1_switchable_record_20261122")
REC_HELDOUT = Path("/home/yu/projects/FGP_mocap/results/live_20261122_141319/smpl.jsonl")
DEFAULT_ASSETS = SMHP_ROOT / "src/unisupermimic/assets/robots/q1/assets"

for p in (
    str(Path(__file__).resolve().parent),
    str(DEPLOY_CANDIDATE / "code"),
    str(SMHP_ROOT / "runs/h002_sonic_v2"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from q1_cgfa_obs_np import (
    BONE_PAIRS,
    ContactGatedFootAnchorNP,
    LivePlannerControllerNP,
    PrebakedClipLibraryNP,
    build_human_smpl_430_np,
    build_obs_80_np,
)
from q1_fgp_client_v2 import FGPHttpSourceV2, Q1FGPFallbackGuard
from q1_fgp_smpl_v2 import FGPSMPLStreamV2
from q1_kinematics_np import Q1KinematicsNP, mat_to_quat_wxyz, yaw_angle_from_mat, yaw_mat_np
from q1_v2b_runtime import apply_torque_safe_pd_np
from sim_q1_stand_start import load_model

BONE_NAMES = [
    "pelvis->torso", "L_thigh", "L_shin", "R_thigh", "R_shin",
    "L_upper_arm", "L_forearm", "R_upper_arm", "R_forearm",
]


def _load_legacy_fgp_classes():
    """Import original Q1_deploy FGP client & stream without modifying Q1_deploy."""
    smpl_spec = importlib.util.spec_from_file_location(
        "legacy_q1_fgp_smpl", DEPLOY_CANDIDATE / "code/q1_fgp_smpl.py"
    )
    smpl_mod = importlib.util.module_from_spec(smpl_spec)
    sys.modules["legacy_q1_fgp_smpl"] = smpl_mod
    smpl_spec.loader.exec_module(smpl_mod)

    cli_src = (DEPLOY_CANDIDATE / "code/q1_fgp_client.py").read_text(encoding="utf-8")
    cli_src = cli_src.replace("from q1_fgp_smpl import", "from legacy_q1_fgp_smpl import")
    ns: dict[str, Any] = {}
    exec(compile(cli_src, "legacy_q1_fgp_client.py", "exec"), ns)
    return smpl_mod.FGPSMPLStream, ns["FGPHttpSource"]


class DualModalityOnnxController:
    """1-thread CPU ORT controller decoupling head, human_smpl tokenizer, and planner_cmd tokenizer."""

    def __init__(self, cfg: dict[str, Any], head_onnx: Path, smpl_onnx: Path, planner_onnx: Path):
        self.cfg = cfg
        self.default_q = np.asarray(cfg["default_q"], dtype=np.float32)
        self.action_scale = np.asarray(cfg["action_scale"], dtype=np.float32)
        self.kp = np.asarray(cfg["kp"], dtype=np.float32)
        self.kd = np.asarray(cfg["kd"], dtype=np.float32)
        self.eff = np.asarray(cfg["effort_limit_nm"], dtype=np.float32)
        self.q_lo = np.asarray(cfg["q_lo"], dtype=np.float32)
        self.q_hi = np.asarray(cfg["q_hi"], dtype=np.float32)
        self.hist_len = int(cfg["hist_len"])
        self.obs_dim = int(cfg["obs_dim"])
        self.engage_steps = int(cfg["torque_safe_pd"]["engage_steps"])
        self.dq_tau_scale = float(cfg["torque_safe_pd"]["dq_tau_scale"])

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.head = ort.InferenceSession(str(head_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
        self.toks = {
            "human_smpl": ort.InferenceSession(str(smpl_onnx), sess_options=opts, providers=["CPUExecutionProvider"]),
            "planner_cmd": ort.InferenceSession(str(planner_onnx), sess_options=opts, providers=["CPUExecutionProvider"]),
        }
        self.history: np.ndarray | None = None
        self.q_init: np.ndarray | None = None
        self.step_count = 0

    def reset(self, q_init: np.ndarray | None = None) -> None:
        self.history = None
        self.q_init = None if q_init is None else np.asarray(q_init, dtype=np.float32).copy()
        self.step_count = 0

    def step(
        self,
        obs_80: np.ndarray,
        modal_input: np.ndarray,
        q_cur: np.ndarray,
        qd_cur: np.ndarray,
        *,
        modality: str = "human_smpl",
    ) -> tuple[np.ndarray, np.ndarray]:
        o = np.asarray(obs_80, dtype=np.float32).reshape(1, 1, self.obs_dim)
        self.history = np.repeat(o, self.hist_len, axis=1) if self.history is None else np.concatenate((self.history[:, 1:, :], o), axis=1)
        m_in = np.asarray(modal_input, dtype=np.float32).reshape(1, -1)
        tok = self.toks[modality].run(["token"], {"modal_input": m_in})[0]
        act = self.head.run(["action"], {"obs_hist": self.history, "token": tok})[0][0]
        q_raw = self.default_q + self.action_scale * act
        qf = np.asarray(q_cur, dtype=np.float32)
        qdf = np.asarray(qd_cur, dtype=np.float32)
        if self.q_init is None:
            self.q_init = qf.copy()
        q_safe = apply_torque_safe_pd_np(
            q_raw, qf, qdf, self.q_lo, self.q_hi, self.kp, self.kd, self.eff,
            q_init=self.q_init, step_i=self.step_count,
            engage_steps=self.engage_steps, dq_tau_scale=self.dq_tau_scale,
        )
        tau_est = self.kp * (q_safe - qf) - self.kd * qdf
        self.step_count += 1
        return q_safe, tau_est


def _ang_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


def _vec_corr(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    return float(np.sum(ac * bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-9))


def _step_mujoco_with_ramp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    act_ids: list[int],
    prev_tgt: np.ndarray,
    desired_tgt: np.ndarray,
    default_q: np.ndarray,
    *,
    ramp_500hz: bool,
    substeps: int = 10,
) -> tuple[np.ndarray, float]:
    box_lo, box_hi = default_q - 2.0, default_q + 2.0
    if ramp_500hz:
        desired = np.clip(desired_tgt, box_lo, box_hi)
        cur = prev_tgt.copy()
        for _ in range(substeps):
            cur = cur + np.clip(desired - cur, -0.2, 0.2)
            cur = np.clip(cur, box_lo, box_hi)
            data.ctrl[act_ids] = cur
            mujoco.mj_step(model, data)
        return cur.astype(np.float32), float(np.max(np.abs(cur - prev_tgt)))
    cmd = np.clip(desired_tgt, prev_tgt - 0.2, prev_tgt + 0.2)
    cmd = np.clip(cmd, box_lo, box_hi).astype(np.float32)
    data.ctrl[act_ids] = cmd
    for _ in range(substeps):
        mujoco.mj_step(model, data)
    return cmd, float(np.max(np.abs(cmd - prev_tgt)))


# ---------------------------------------------------------------------------
# Suite 1: Today's real FGP recording (140.9 s + 12 real outage flickers)
# ---------------------------------------------------------------------------

def run_suite1_today_fgp(
    model: mujoco.MjModel,
    kin: Q1KinematicsNP,
    cfg: dict[str, Any],
    v2_ctrl: DualModalityOnnxController,
    fgpv2_ctrl: DualModalityOnnxController,
) -> dict[str, Any]:
    fgp_rows = [json.loads(line) for line in (REC_TODAY / "fgp.jsonl").open(encoding="utf-8")]
    pol_rows = [json.loads(line) for line in (REC_TODAY / "policy_50hz.jsonl").open(encoding="utf-8")]
    hs_pol = [r for r in pol_rows if r["mode"] == "human_smpl"]
    t_start = int(hs_pol[0]["sample_monotonic_ns"])
    null_starts = [
        int(r["sample_monotonic_ns"])
        for i, r in enumerate(hs_pol)
        if r["fgp"] is None and (i == 0 or hs_pol[i - 1]["fgp"] is not None)
    ]

    # Pre-build both raw and debiased reference streams indexed by source_ns for fair apples-to-apples scoring
    def _build_ref_map(debias: bool) -> dict[int, np.ndarray]:
        st = FGPSMPLStreamV2(0.02, enable_tilt_debias=debias, enable_vel_conditioning=debias)
        ref_map = {}
        for r in fgp_rows:
            for tk in st.push_sample(r["sample"], allow_sequence_skip=True):
                ref_map[int(tk.stamp_ns)] = tk.feature.reshape(10, 43)[-1]
        return ref_map

    ref_raw = _build_ref_map(False)
    ref_deb = _build_ref_map(True)
    raw_stamps = np.array(sorted(ref_raw.keys()), dtype=np.int64)
    deb_stamps = np.array(sorted(ref_deb.keys()), dtype=np.int64)

    def _nearest_ref(st_arr: np.ndarray, mp: dict[int, np.ndarray], stamp: int) -> np.ndarray:
        pos = int(np.searchsorted(st_arr, stamp))
        if pos == 0:
            return mp[int(st_arr[0])]
        if pos >= len(st_arr):
            return mp[int(st_arr[-1])]
        k = pos - 1 if abs(int(st_arr[pos - 1]) - stamp) <= abs(int(st_arr[pos]) - stamp) else pos
        return mp[int(st_arr[k])]

    LegacyStream, LegacyHttp = _load_legacy_fgp_classes()
    names = cfg["joint_names"]
    pose_ids = [model.joint(n).qposadr[0] for n in names]
    vel_ids = [model.joint(n).dofadr[0] for n in names]
    act_ids = [model.actuator(n).id for n in names]
    eff_lim = np.asarray(cfg["effort_limit_nm"], dtype=np.float32)
    pelvis = model.body("pelvis").id
    w_start = int(model.jnt_dofadr[model.body_jntadr[pelvis]]) + 3
    zeros3 = np.zeros(3, dtype=np.float32)

    # Also build smooth tick list once for A1
    s_smooth = FGPSMPLStreamV2(0.02, enable_tilt_debias=False, enable_vel_conditioning=False)
    smooth_ticks = []
    for r in fgp_rows:
        t_rel = (int(r["recorded_monotonic_ns"]) - t_start) * 1e-9
        tks = s_smooth.push_sample(r["sample"], allow_sequence_skip=True)
        if 0.0 <= t_rel <= 141.0:
            smooth_ticks.extend(tks)

    arms = {}
    for arm_name, ctrl, mode_kind in [
        ("A0_ProdOld", v2_ctrl, "old_async_flicker"),
        ("A1_SmoothV2", v2_ctrl, "smooth_raw"),
        ("A2_AdapterV2_V2ONNX", v2_ctrl, "v2_async_flicker"),
        ("A3_AdapterV2_FGPV2ONNX", fgpv2_ctrl, "v2_async_flicker"),
    ]:
        data = mujoco.MjData(model)
        data.qpos[0:3] = [0.0, 0.0, 0.42132428111574716]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[pose_ids] = ctrl.default_q
        data.ctrl[act_ids] = ctrl.default_q
        mujoco.mj_forward(model, data)
        for _ in range(100):
            mujoco.mj_step(model, data)

        cgfa = ContactGatedFootAnchorNP(kin, dt=0.02)
        planner_stream = LivePlannerControllerNP(dt=0.02, num_frames=10, smooth_slew=True)
        guard = Q1FGPFallbackGuard()
        prev_act = np.zeros(22, dtype=np.float32)
        prev_tgt = ctrl.default_q.copy()
        fgp_seg = None
        fgp_heading = None
        fgp_origin = None
        ctrl.reset(ctrl.default_q)
        guard.reset("planner")

        # Async stream simulator matching exact wall arrival timing + 12 real 30ms outage flickers
        use_legacy = mode_kind == "old_async_flicker"
        ramp_500 = mode_kind != "old_async_flicker"
        if use_legacy:
            src = LegacyHttp("http://127.0.0.1:8090", 0.02)
        elif mode_kind == "v2_async_flicker":
            src = FGPHttpSourceV2("http://127.0.0.1:8090", 0.02)
        else:
            src = None

        fgp_idx = 0
        if src is not None:
            while fgp_idx < len(fgp_rows) and int(fgp_rows[fgp_idx]["recorded_monotonic_ns"]) <= t_start:
                src.ingest(fgp_rows[fgp_idx], arrival_ns=int(fgp_rows[fgp_idx]["recorded_monotonic_ns"]))
                fgp_idx += 1
            src.dropped_ticks = 0

        total_ticks = len(smooth_ticks) if src is None else len(hs_pol)
        tilts, pitches, dtgts, sim_dz, hum_vxy, sim_vxy = [], [], [], [], [], []
        sim_b_l, raw_b_l, deb_b_l = [], [], []
        pd_oob = 0
        reanchor_cnt = 0
        null_ticks = 0
        fell_at = None

        for k in range(total_ticks):
            R = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
            w = data.qvel[w_start : w_start + 3].astype(np.float32)
            q = data.qpos[pose_ids].astype(np.float32)
            qd = data.qvel[vel_ids].astype(np.float32)

            if src is None:
                tick = smooth_ticks[k]
                live_fgp = True
            else:
                now_ns = int(hs_pol[k]["sample_monotonic_ns"])
                while fgp_idx < len(fgp_rows) and int(fgp_rows[fgp_idx]["recorded_monotonic_ns"]) <= now_ns:
                    fr = fgp_rows[fgp_idx]
                    src.ingest(fr, arrival_ns=int(fr["recorded_monotonic_ns"]))
                    fgp_idx += 1
                for ns0 in null_starts:
                    if 0 <= now_ns - ns0 < 20_000_000:
                        fake = {
                            "status": "outage",
                            "realtime_valid": False,
                            "sample": fgp_rows[max(0, fgp_idx - 1)]["sample"],
                            "session_id": fgp_rows[0]["session_id"],
                            "calibration_id": fgp_rows[0]["calibration_id"],
                            "error": None,
                        }
                        src.ingest(fake, arrival_ns=now_ns - 5_000_000)
                if use_legacy:
                    queued, _ = src.get_latest()
                    live_fgp = queued is not None and (0 <= now_ns - queued.arrival_ns <= 150_000_000)
                else:
                    queued, _, _ = src.pop_paced(now_ns=now_ns)
                    live_fgp = queued is not None
                tick = None if not live_fgp else queued.tick

            if use_legacy:
                if live_fgp:
                    if tick.segment != fgp_seg:
                        if fgp_seg is not None:
                            reanchor_cnt += 1
                        fgp_seg = tick.segment
                        fgp_heading = yaw_mat_np(R) @ yaw_mat_np(tick.root_rotation).T
                        fgp_origin = tick.root_xy.astype(np.float32).copy()
                        r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                        cgfa.reset(R, w, q, qd, np.zeros(2, dtype=np.float32), R_ref=r_ref)
                        ctrl.reset(q)
                    p_ref_xy = (fgp_heading[:2, :2] @ (tick.root_xy - fgp_origin)).astype(np.float32)
                    r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                    exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                    obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                    tgt, tau = ctrl.step(obs, tick.feature, q, qd, modality="human_smpl")
                else:
                    null_ticks += 1
                    # Reproduce A0 stale-planner fallback bug!
                    plan_in, p_ref_xy, r_ref = planner_stream.push_and_build(0.0, 0.0, 0.0, 1.0, 0.04, cgfa=cgfa, R_imu_aligned=R)
                    exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                    obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                    tgt, tau = ctrl.step(obs, plan_in, q, qd, modality="planner_cmd")
            else:
                branch, need_reanchor = guard.resolve_branch("human_smpl", live_fgp)
                if branch == "human_smpl":
                    if tick.segment != fgp_seg:
                        if fgp_seg is not None:
                            reanchor_cnt += 1
                        fgp_seg = tick.segment
                        fgp_heading = yaw_mat_np(R) @ yaw_mat_np(tick.root_rotation).T
                        fgp_origin = tick.root_xy.astype(np.float32).copy()
                        r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                        cgfa.reset(R, w, q, qd, np.zeros(2, dtype=np.float32), R_ref=r_ref)
                        ctrl.reset(q)
                    p_ref_xy = (fgp_heading[:2, :2] @ (tick.root_xy - fgp_origin)).astype(np.float32)
                    r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                    exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                    obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                    tgt, tau = ctrl.step(obs, tick.feature, q, qd, modality="human_smpl")
                else:
                    null_ticks += 1
                    if need_reanchor:
                        r_ref = R.copy()
                        cgfa.reset(R, w, q, qd, np.zeros(2, dtype=np.float32), R_ref=r_ref)
                        planner_stream.reset(yaw0_rad=yaw_angle_from_mat(R))
                        fgp_seg = None
                        ctrl.reset(q)
                    plan_in, p_ref_xy, r_ref = planner_stream.push_and_build(0.0, 0.0, 0.0, 1.0, 0.04, cgfa=cgfa, R_imu_aligned=R)
                    exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                    obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                    tgt, tau = ctrl.step(obs, plan_in, q, qd, modality="planner_cmd")

            pd_oob += int(np.sum((np.abs(tau) - eff_lim) > 1e-4))
            sent_tgt, step_d = _step_mujoco_with_ramp(
                model, data, act_ids, prev_tgt, tgt, ctrl.default_q, ramp_500hz=ramp_500
            )
            dtgts.append(step_d)
            prev_tgt = sent_tgt
            prev_act = (sent_tgt - ctrl.default_q) / ctrl.action_scale
            mujoco.mj_forward(model, data)

            Rp = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
            qp = data.qpos[pose_ids].astype(np.float32)
            tilt = math.degrees(math.acos(float(np.clip(Rp[2, 2], -1, 1))))
            pitch = math.degrees(math.asin(float(np.clip(-Rp[2, 0], -1, 1))))
            tilts.append(tilt)
            pitches.append(pitch)
            ht = yaw_mat_np(Rp).T
            bp, _, _, _ = kin.forward_single(zeros3, Rp, zeros3, zeros3, qp, np.zeros(22, dtype=np.float32))
            sim_dz.append(abs(float(bp[3, 2] - bp[6, 2])) * 100.0)
            sb = np.stack([(ht @ (bp[b] - bp[a])) / max(np.linalg.norm(ht @ (bp[b] - bp[a])), 1e-6) for a, b in BONE_PAIRS])
            sim_b_l.append(sb)
            stamp_q = int(tick.stamp_ns) if tick is not None else int(raw_stamps[min(k, len(raw_stamps) - 1)])
            fr_raw = _nearest_ref(raw_stamps, ref_raw, stamp_q)
            fr_deb = _nearest_ref(deb_stamps, ref_deb, stamp_q)
            raw_b_l.append(fr_raw[16:43].reshape(9, 3))
            deb_b_l.append(fr_deb[16:43].reshape(9, 3))
            hum_vxy.append(float(np.linalg.norm(fr_deb[10:12]) * 0.4213))
            sim_vxy.append(float(np.linalg.norm(data.qvel[:2])))
            if data.qpos[2] < 0.15 or tilt >= 65.0:
                fell_at = round(k * 0.02, 2)
                break

        sim_b = np.asarray(sim_b_l)
        raw_b = np.asarray(raw_b_l)
        deb_b = np.asarray(deb_b_l)
        sim_dz_a = np.asarray(sim_dz)

        def _metrics_vs(ref_b: np.ndarray) -> dict[str, Any]:
            t_e = float(np.median(_ang_deg(ref_b[:, 0], sim_b[:, 0])))
            t_c = _vec_corr(ref_b[:, 0], sim_b[:, 0])
            l_e = float(np.mean([np.median(_ang_deg(ref_b[:, b], sim_b[:, b])) for b in (1, 2, 3, 4)]))
            l_c = float(np.mean([_vec_corr(ref_b[:, b], sim_b[:, b]) for b in (1, 2, 3, 4)]))
            a_e = float(np.mean([np.median(_ang_deg(ref_b[:, b], sim_b[:, b])) for b in (5, 6, 7, 8)]))
            a_c = float(np.mean([_vec_corr(ref_b[:, b], sim_b[:, b]) for b in (5, 6, 7, 8)]))
            return {
                "torso_deg": round(t_e, 2), "torso_corr": round(t_c, 3),
                "leg_deg": round(l_e, 2), "leg_corr": round(l_c, 3),
                "arm_deg": round(a_e, 2), "arm_corr": round(a_c, 3),
            }

        lifts = {}
        for li, (s_idx, e_idx) in enumerate([(455, 605), (1335, 1410), (3070, 3110), (5880, 5930), (6010, 6100), (6340, 6370)]):
            if len(sim_dz_a) > s_idx:
                lifts[f"lift_{li+1}"] = round(float(sim_dz_a[s_idx : min(e_idx, len(sim_dz_a))].max()), 2)

        res = {
            "survived_s": round(len(tilts) * 0.02, 2),
            "budget_s": round(total_ticks * 0.02, 2),
            "fell_at_s": fell_at,
            "tilt_p50_deg": round(float(np.median(tilts)), 2),
            "tilt_p95_deg": round(float(np.percentile(tilts, 95)), 2),
            "tilt_max_deg": round(float(max(tilts)), 2),
            "pitch_p50_deg": round(float(np.median(pitches)), 2),
            "v_xy_corr": round(float(np.corrcoef(hum_vxy, sim_vxy)[0, 1]), 3) if len(sim_vxy) > 20 else None,
            "foot_dz_p95_cm": round(float(np.percentile(sim_dz_a, 95)), 2),
            "foot_dz_max_cm": round(float(sim_dz_a.max()), 2),
            "dtgt_p95_rad": round(float(np.percentile(dtgts, 95)), 4),
            "pd_oob": pd_oob,
            "reanchor_count": reanchor_cnt,
            "null_ticks": null_ticks,
            "vs_raw_fgp": _metrics_vs(raw_b),
            "vs_debiased_fgp": _metrics_vs(deb_b),
            "lift_peaks_cm": lifts,
        }
        arms[arm_name] = res
        print(
            f"[Suite1 {arm_name:24s}] surv={res['survived_s']:6.2f}/{res['budget_s']:6.2f}s tilt={res['tilt_p50_deg']:4.1f}/{res['tilt_p95_deg']:4.1f}/{res['tilt_max_deg']:4.1f}° "
            f"pitch={res['pitch_p50_deg']:+5.2f}° | deb: tor={res['vs_debiased_fgp']['torso_deg']:.1f}° arm={res['vs_debiased_fgp']['arm_deg']:.1f}°({res['vs_debiased_fgp']['arm_corr']:+.2f}) "
            f"leg={res['vs_debiased_fgp']['leg_deg']:.1f}°({res['vs_debiased_fgp']['leg_corr']:+.2f}) | vxy_r={res['v_xy_corr']} lifts={list(lifts.values())} oob={pd_oob}",
            flush=True,
        )
    return arms


# ---------------------------------------------------------------------------
# Suite 2: Held-out real FGP recording (cal_id=2 -22deg offset & cal_id=5)
# ---------------------------------------------------------------------------

def run_suite2_heldout_fgp(
    model: mujoco.MjModel,
    kin: Q1KinematicsNP,
    cfg: dict[str, Any],
    v2_ctrl: DualModalityOnnxController,
    fgpv2_ctrl: DualModalityOnnxController,
) -> dict[str, Any]:
    segments_spec = [
        ("cal2_bias_neg22deg", 2, 29400, 32500),
        ("cal5_dynamic_112s", 5, 37760, 41115),
    ]
    rows_by_cal: dict[int, list[dict]] = {2: [], 5: []}
    with REC_HELDOUT.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if 29400 <= idx <= 32500 or 37760 <= idx <= 41115:
                d = json.loads(line)
                cid = int(d.get("calibration_id", -1))
                if cid in rows_by_cal:
                    rows_by_cal[cid].append(d)

    names = cfg["joint_names"]
    pose_ids = [model.joint(n).qposadr[0] for n in names]
    vel_ids = [model.joint(n).dofadr[0] for n in names]
    act_ids = [model.actuator(n).id for n in names]
    eff_lim = np.asarray(cfg["effort_limit_nm"], dtype=np.float32)
    pelvis = model.body("pelvis").id
    w_start = int(model.jnt_dofadr[model.body_jntadr[pelvis]]) + 3
    zeros3 = np.zeros(3, dtype=np.float32)
    LegacyStream, _ = _load_legacy_fgp_classes()

    out = {}
    for seg_name, cid, _, _ in segments_spec:
        raw_list = rows_by_cal[cid]
        st_old = LegacyStream(0.02)
        st_v2 = FGPSMPLStreamV2(0.02)
        tk_old, tk_v2 = [], []
        for d in raw_list:
            tk_old.extend(st_old.push_sample(d, allow_sequence_skip=True))
            tk_v2.extend(st_v2.push_sample(d, allow_sequence_skip=True))

        seg_res = {}
        for variant, ctrl, tks in [
            ("A0_OldAdapter_V2ONNX", v2_ctrl, tk_old),
            ("A2_AdapterV2_V2ONNX", v2_ctrl, tk_v2),
            ("A3_AdapterV2_FGPV2ONNX", fgpv2_ctrl, tk_v2),
        ]:
            data = mujoco.MjData(model)
            data.qpos[0:3] = [0.0, 0.0, 0.42132428111574716]
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qpos[pose_ids] = ctrl.default_q
            data.ctrl[act_ids] = ctrl.default_q
            mujoco.mj_forward(model, data)
            for _ in range(100):
                mujoco.mj_step(model, data)
            cgfa = ContactGatedFootAnchorNP(kin, dt=0.02)
            prev_act = np.zeros(22, np.float32)
            prev_tgt = ctrl.default_q.copy()
            seg = None; heading = None; origin = None
            tilts, pitches, sim_b, hum_b = [], [], [], []
            fell_at = None; pd_oob = 0
            for k, tk in enumerate(tks):
                R = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
                w = data.qvel[w_start : w_start + 3].astype(np.float32)
                q = data.qpos[pose_ids].astype(np.float32)
                qd = data.qvel[vel_ids].astype(np.float32)
                if tk.segment != seg:
                    seg = tk.segment
                    heading = yaw_mat_np(R) @ yaw_mat_np(tk.root_rotation).T
                    origin = tk.root_xy.astype(np.float32).copy()
                    r_ref = heading @ tk.root_rotation
                    cgfa.reset(R, w, q, qd, np.zeros(2, np.float32), R_ref=r_ref)
                    ctrl.reset(q)
                p_ref_xy = heading[:2, :2] @ (tk.root_xy - origin)
                r_ref = heading @ tk.root_rotation
                exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                tgt, tau = ctrl.step(obs, tk.feature, q, qd, modality="human_smpl")
                pd_oob += int(np.sum((np.abs(tau) - eff_lim) > 1e-4))
                sent, _ = _step_mujoco_with_ramp(model, data, act_ids, prev_tgt, tgt, ctrl.default_q, ramp_500hz=True)
                prev_tgt = sent
                prev_act = (sent - ctrl.default_q) / ctrl.action_scale
                mujoco.mj_forward(model, data)
                Rp = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
                qp = data.qpos[pose_ids].astype(np.float32)
                tilt = math.degrees(math.acos(float(np.clip(Rp[2, 2], -1, 1))))
                pitch = math.degrees(math.asin(float(np.clip(-Rp[2, 0], -1, 1))))
                tilts.append(tilt); pitches.append(pitch)
                ht = yaw_mat_np(Rp).T
                bp, _, _, _ = kin.forward_single(zeros3, Rp, zeros3, zeros3, qp, np.zeros(22, np.float32))
                sb = np.stack([(ht @ (bp[b] - bp[a])) / max(np.linalg.norm(ht @ (bp[b] - bp[a])), 1e-6) for a, b in BONE_PAIRS])
                sim_b.append(sb)
                hum_b.append(tk.feature.reshape(10, 43)[-1, 16:43].reshape(9, 3))
                if data.qpos[2] < 0.15 or tilt >= 65.0:
                    fell_at = round(k * 0.02, 2)
                    break
            sb_a = np.asarray(sim_b); hb_a = np.asarray(hum_b)
            seg_res[variant] = {
                "survived_s": round(len(tilts) * 0.02, 2),
                "budget_s": round(len(tks) * 0.02, 2),
                "fell_at_s": fell_at,
                "tilt_p50_deg": round(float(np.median(tilts)), 2),
                "tilt_max_deg": round(float(max(tilts)), 2),
                "pitch_p50_deg": round(float(np.median(pitches)), 2),
                "arm_deg": round(float(np.mean([np.median(_ang_deg(hb_a[:, b], sb_a[:, b])) for b in (5, 6, 7, 8)])), 2),
                "arm_corr": round(float(np.mean([_vec_corr(hb_a[:, b], sb_a[:, b]) for b in (5, 6, 7, 8)])), 3),
                "leg_corr": round(float(np.mean([_vec_corr(hb_a[:, b], sb_a[:, b]) for b in (1, 2, 3, 4)])), 3),
                "pd_oob": pd_oob,
            }
            print(f"[Suite2 {seg_name:20s} | {variant:22s}] surv={seg_res[variant]['survived_s']}/{seg_res[variant]['budget_s']}s tilt={seg_res[variant]['tilt_p50_deg']:.1f}/{seg_res[variant]['tilt_max_deg']:.1f}° arm={seg_res[variant]['arm_deg']:.1f}°({seg_res[variant]['arm_corr']:+.2f}) leg_r={seg_res[variant]['leg_corr']:+.2f}", flush=True)
        out[seg_name] = seg_res
    return out


# ---------------------------------------------------------------------------
# Suite 3: 8 Hard & Non-Walk Primitives
# ---------------------------------------------------------------------------

def run_suite3_primitives(
    model: mujoco.MjModel,
    kin: Q1KinematicsNP,
    cfg: dict[str, Any],
    v2_ctrl: DualModalityOnnxController,
    fgpv2_ctrl: DualModalityOnnxController,
) -> dict[str, Any]:
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    m2k = json.loads((SMHP_ROOT / "runs/h002_sonic_v3/dataset_2000_manifest.json").read_text())
    kin_t = Q1Kinematics(str(SMHP_ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"), torch.device("cpu"))
    specs = []
    for kw in ("kick", "syn_squat_deep", "lock_step", "turn_walk_360"):
        for c in m2k["clips"]:
            if kw in c["csv_rel"].lower():
                mot = MotionLib([str(SMHP_ROOT / c["csv_rel"])], kin_t, torch.device("cpu"), src_fps=30.0)
                T = min(int(mot.num_frames), 250)
                specs.append((kw, mot.root_p.numpy()[:T], mot.root_R.numpy()[:T], mot.q.numpy()[:T], mot.body_p.numpy()[:T]))
                break
    for npz_name in ("static_stand_50hz", "vr3pt_arm_wave_50hz", "ho_phuma_walk_009_50hz", "ho_lafan_walk2_subject3_50hz"):
        lib = PrebakedClipLibraryNP(Path("/home/yu/projects/Q1_deploy/clips") / f"{npz_name}.npz")
        specs.append((npz_name, lib.root_p, lib.root_R, lib.q, lib.body_p))

    names = cfg["joint_names"]
    pose_ids = [model.joint(n).qposadr[0] for n in names]
    vel_ids = [model.joint(n).dofadr[0] for n in names]
    act_ids = [model.actuator(n).id for n in names]
    eff_lim = np.asarray(cfg["effort_limit_nm"], dtype=np.float32)
    pelvis = model.body("pelvis").id
    w_start = int(model.jnt_dofadr[model.body_jntadr[pelvis]]) + 3
    zeros3 = np.zeros(3, dtype=np.float32)

    out = {}
    for cname, rp, rR, q_ref, bp_ref in specs:
        T = len(rp)
        ref_dz_max = float(np.max(np.abs(bp_ref[:, 3, 2] - bp_ref[:, 6, 2])) * 100.0)
        res_c = {"ref_foot_dz_max_cm": round(ref_dz_max, 2)}
        for mname, ctrl in [("A2_V2ONNX", v2_ctrl), ("A3_FGPV2ONNX", fgpv2_ctrl)]:
            data = mujoco.MjData(model)
            data.qpos[0:3] = rp[0]
            data.qpos[3:7] = mat_to_quat_wxyz(rR[0])
            data.qpos[pose_ids] = q_ref[0]
            data.ctrl[act_ids] = q_ref[0]
            mujoco.mj_forward(model, data)
            cgfa = ContactGatedFootAnchorNP(kin, dt=0.02)
            R0 = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
            cgfa.reset(R0, np.zeros(3, np.float32), q_ref[0], np.zeros(22, np.float32), rp[0, :2], R_ref=rR[0])
            ctrl.reset(q_ref[0])
            prev_act = np.zeros(22, np.float32)
            prev_tgt = q_ref[0].copy()
            tilts, dzs, leg_es, arm_es, wrist_cms = [], [], [], [], []
            z_min = 1.0; pd_oob = 0
            for k in range(T):
                R = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
                w = data.qvel[w_start : w_start + 3].astype(np.float32)
                q = data.qpos[pose_ids].astype(np.float32)
                qd = data.qvel[vel_ids].astype(np.float32)
                feat = build_human_smpl_430_np(rp, rR, bp_ref, k, dt=0.02)
                exy, _ = cgfa.step(R, w, q, qd, rp[k, :2], R_ref=rR[k])
                obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=rR[k], cgfa_exy=exy)
                tgt, tau = ctrl.step(obs, feat, q, qd, modality="human_smpl")
                pd_oob += int(np.sum((np.abs(tau) - eff_lim) > 1e-4))
                sent, _ = _step_mujoco_with_ramp(model, data, act_ids, prev_tgt, tgt, ctrl.default_q, ramp_500hz=True)
                prev_tgt = sent
                prev_act = (sent - ctrl.default_q) / ctrl.action_scale
                mujoco.mj_forward(model, data)
                Rp = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
                qp = data.qpos[pose_ids].astype(np.float32)
                tilt = math.degrees(math.acos(float(np.clip(Rp[2, 2], -1, 1))))
                tilts.append(tilt)
                z_min = min(z_min, float(data.qpos[2]))
                ht = yaw_mat_np(Rp).T
                ht_ref = yaw_mat_np(rR[k]).T
                bp, _, _, _ = kin.forward_single(zeros3, Rp, zeros3, zeros3, qp, np.zeros(22, np.float32))
                dzs.append(abs(float(bp[3, 2] - bp[6, 2])) * 100.0)
                sb = np.stack([(ht @ (bp[b] - bp[a])) / max(np.linalg.norm(ht @ (bp[b] - bp[a])), 1e-6) for a, b in BONE_PAIRS])
                rb = feat.reshape(10, 43)[-1, 16:43].reshape(9, 3)
                leg_es.append(float(np.mean(_ang_deg(rb[1:5], sb[1:5]))))
                arm_es.append(float(np.mean(_ang_deg(rb[5:9], sb[5:9]))))
                lw_e = np.linalg.norm(ht @ (bp[10] - bp[7]) - ht_ref @ (bp_ref[k, 10] - bp_ref[k, 7]))
                rw_e = np.linalg.norm(ht @ (bp[13] - bp[7]) - ht_ref @ (bp_ref[k, 13] - bp_ref[k, 7]))
                wrist_cms.append(0.5 * (lw_e + rw_e) * 100.0)
                if data.qpos[2] < 0.15 or tilt >= 65.0:
                    break
            res_c[mname] = {
                "steps": f"{len(tilts)}/{T}",
                "completed": len(tilts) == T,
                "tilt_max_deg": round(float(max(tilts)), 2),
                "leg_err_med_deg": round(float(np.median(leg_es)), 2),
                "arm_err_med_deg": round(float(np.median(arm_es)), 2),
                "wrist_err_p50_cm": round(float(np.median(wrist_cms)), 2),
                "foot_dz_max_cm": round(float(max(dzs)), 2),
                "z_min_m": round(float(z_min), 3),
                "pd_oob": pd_oob,
            }
            print(f"[Suite3 {cname:26s} | {mname:12s}] {res_c[mname]['steps']} tiltMax={res_c[mname]['tilt_max_deg']:4.1f}° leg={res_c[mname]['leg_err_med_deg']:4.1f}° arm={res_c[mname]['arm_err_med_deg']:4.1f}° wri={res_c[mname]['wrist_err_p50_cm']:4.1f}cm footDz={res_c[mname]['foot_dz_max_cm']:4.1f}/{ref_dz_max:.1f}cm", flush=True)
        out[cname] = res_c
    return out


# ---------------------------------------------------------------------------
# Suite 4: H002-SONIC v2 50 Held-Out (Nom + 20-Stress) & 8 Raw Zero-IK BVH
# ---------------------------------------------------------------------------

def run_suite4_sonic_benchmarks(v2_smpl_onnx: Path, fgpv2_smpl_onnx: Path, head_onnx: Path) -> dict[str, Any]:
    import eval_sonic_suite as es

    v2_base = SMHP_ROOT / "runs/h002_sonic_v2"
    manifest = json.loads((v2_base / "dataset_250_manifest.json").read_text(encoding="utf-8"))
    v1_train_clips = [c for c in manifest["clips"] if c.get("v1_clip") and c["split"] == "train"]
    held_out_clips = [c for c in manifest["clips"] if c["split"] == "held_out"]
    all_eval_clips = v1_train_clips + held_out_clips
    motion_files = [str(SMHP_ROOT / c["csv_rel"]) for c in all_eval_clips]
    stand_ref_csv = str(SMHP_ROOT / "src/unisupermimic/assets/motions/q1/multidataset_250/train/syn_stand_idle_1.csv")
    if stand_ref_csv not in motion_files:
        motion_files.append(stand_ref_csv)
    stand_clip_idx = motion_files.index(stand_ref_csv)
    ho_offset = len(v1_train_clips)

    stress_indices = [
        i for i in range(ho_offset, len(all_eval_clips))
        if all_eval_clips[i].get("v1_clip") or all_eval_clips[i]["source"] in ("lafan1", "synthetic")
    ]
    for i in range(ho_offset, len(all_eval_clips)):
        if len(stress_indices) >= 20:
            break
        if i not in stress_indices:
            stress_indices.append(i)

    env = es.create_eval_env(motion_files)
    head_sess = es._ort_sess(head_onnx)
    tok_v2 = es._ort_sess(v2_smpl_onnx)
    tok_v3 = es._ort_sess(fgpv2_smpl_onnx)
    mj_model = es._find_mj_model(env)
    orig_fric = mj_model.geom_friction.copy()

    results = {}
    for label, tok_s in [("V2_HumanSMPL_ONNX", tok_v2), ("FGP_V2_HumanSMPL_ONNX", tok_v3)]:
        mj_model.geom_friction[:] = orig_fric
        nom_rows = [
            es.rollout_clip(env, clip_idx=i, est_mode="cgfa", modality="human_smpl", tok_sess=tok_s, head_sess=head_sess, deploy_safety=True, stress=False)
            for i in range(ho_offset, len(all_eval_clips))
        ]
        nom_agg = es.aggregate(nom_rows)

        mj_model.geom_friction[:, 0] *= 0.55
        str_rows = [
            es.rollout_clip(env, clip_idx=i, est_mode="cgfa", modality="human_smpl", tok_sess=tok_s, head_sess=head_sess, deploy_safety=True, stress=True)
            for i in stress_indices
        ]
        str_agg = es.aggregate(str_rows)

        mj_model.geom_friction[:] = orig_fric
        bvh_res = es.eval_zero_ik_bvh(env, stand_clip_idx, tok_s, head_sess)

        results[label] = {
            "held_out_50_nominal": nom_agg,
            "held_out_20_stress": str_agg,
            "zero_ik_raw_bvh_8": bvh_res,
        }
        print(
            f"[Suite4 {label:22s}] 50-HO surv={nom_agg['survival_rate']*100:5.1f}% ret={nom_agg['norm_return']:.3f} body={nom_agg['body_pos_rmse_cm']:.2f}cm slip={nom_agg['stance_slip_mm_s']:.1f} | "
            f"20-STR surv={str_agg['survival_rate']*100:5.1f}% ret={str_agg['norm_return']:.3f} | 8-BVH surv={bvh_res['survival_rate']*100:5.1f}% xy={bvh_res['mean_root_xy_rmse_cm']:.1f}cm",
            flush=True,
        )
    env.close()
    return results


# ---------------------------------------------------------------------------
# Suite 5: 40 s Continuous Switchable + 0.70x Friction + 20 ms Delay + 400 ms Real Disconnect
# ---------------------------------------------------------------------------

def run_suite5_stress_switch(
    model: mujoco.MjModel,
    kin: Q1KinematicsNP,
    cfg: dict[str, Any],
    v2_ctrl: DualModalityOnnxController,
    fgpv2_ctrl: DualModalityOnnxController,
) -> dict[str, Any]:
    fgp_rows = [json.loads(line) for line in (REC_TODAY / "fgp.jsonl").open(encoding="utf-8")]
    t_hs_start = int(next(r["recorded_monotonic_ns"] for r in fgp_rows if False) if False else 0)
    pol_rows = [json.loads(line) for line in (REC_TODAY / "policy_50hz.jsonl").open(encoding="utf-8")]
    hs_pol = [r for r in pol_rows if r["mode"] == "human_smpl"]
    t0_rec = int(hs_pol[0]["sample_monotonic_ns"])

    LegacyStream, LegacyHttp = _load_legacy_fgp_classes()
    names = cfg["joint_names"]
    pose_ids = [model.joint(n).qposadr[0] for n in names]
    vel_ids = [model.joint(n).dofadr[0] for n in names]
    act_ids = [model.actuator(n).id for n in names]
    eff_lim = np.asarray(cfg["effort_limit_nm"], dtype=np.float32)
    pelvis = model.body("pelvis").id
    w_start = int(model.jnt_dofadr[model.body_jntadr[pelvis]]) + 3

    orig_fric = model.geom_friction.copy()
    model.geom_friction[:, 0] *= 0.70
    results = {}
    try:
        total_steps = 2000  # 40.0 s
        for label, ctrl, is_old in [
            ("A0_ProdOld", v2_ctrl, True),
            ("A2_AdapterV2_V2ONNX", v2_ctrl, False),
            ("A3_AdapterV2_FGPV2ONNX", fgpv2_ctrl, False),
        ]:
            data = mujoco.MjData(model)
            data.qpos[0:3] = [0.0, 0.0, 0.42132428111574716]
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qpos[pose_ids] = ctrl.default_q
            data.ctrl[act_ids] = ctrl.default_q
            mujoco.mj_forward(model, data)
            for _ in range(100):
                mujoco.mj_step(model, data)

            cgfa = ContactGatedFootAnchorNP(kin, dt=0.02)
            planner = LivePlannerControllerNP(dt=0.02, num_frames=10, smooth_slew=True)
            guard = Q1FGPFallbackGuard()
            guard.reset("planner")
            ctrl.reset(ctrl.default_q)
            R0 = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
            cgfa.reset(R0, np.zeros(3, np.float32), ctrl.default_q, np.zeros(22, np.float32), np.zeros(2, np.float32), R_ref=R0)
            planner.reset(yaw0_rad=yaw_angle_from_mat(R0))

            src = LegacyHttp("http://127.0.0.1:8090", 0.02) if is_old else FGPHttpSourceV2("http://127.0.0.1:8090", 0.02)
            fgp_idx = 0
            while fgp_idx < len(fgp_rows) and int(fgp_rows[fgp_idx]["recorded_monotonic_ns"]) <= t0_rec:
                src.ingest(fgp_rows[fgp_idx], arrival_ns=int(fgp_rows[fgp_idx]["recorded_monotonic_ns"]))
                fgp_idx += 1

            prev_act = np.zeros(22, np.float32)
            prev_tgt = ctrl.default_q.copy()
            delayed_desired = ctrl.default_q.copy()
            fgp_seg = None; fgp_heading = None; fgp_origin = None
            tilts = []; step_jumps = []; pd_oob = 0; disc_tilts = []; fell_at = None

            for k in range(total_steps):
                t_s = k * 0.02
                R = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
                w = data.qvel[w_start : w_start + 3].astype(np.float32)
                q = data.qpos[pose_ids].astype(np.float32)
                qd = data.qvel[vel_ids].astype(np.float32)

                # Phase schedule: [0..5s] Planner -> [5..30s] HumanSMPL (with 4 flickers + 18.0..18.4s 400ms outage!) -> [30..40s] Planner
                want_smpl = 5.0 <= t_s < 30.0
                in_real_outage = 18.0 <= t_s < 18.40
                in_flicker = (int(round(t_s * 50)) in (350, 500, 750, 1200))

                now_ns = t0_rec + max(0, k - 250) * 20_000_000
                if want_smpl and not in_real_outage:
                    while fgp_idx < len(fgp_rows) and int(fgp_rows[fgp_idx]["recorded_monotonic_ns"]) <= now_ns:
                        fr = fgp_rows[fgp_idx]
                        src.ingest(fr, arrival_ns=int(fr["recorded_monotonic_ns"]))
                        fgp_idx += 1
                if in_real_outage or in_flicker:
                    fake = {
                        "status": "outage",
                        "realtime_valid": False,
                        "sample": fgp_rows[max(0, fgp_idx - 1)]["sample"],
                        "session_id": fgp_rows[0]["session_id"],
                        "calibration_id": fgp_rows[0]["calibration_id"],
                        "error": None,
                    }
                    src.ingest(fake, arrival_ns=now_ns)

                # Determine planner command if in planner phase
                if t_s < 2.0:
                    cmd = (0.0, 0.0, 0.0, 1.0, 0.04)
                elif t_s < 5.0:
                    cmd = (0.25, 0.0, 0.0, 1.0, 0.04)
                elif 30.0 <= t_s < 31.0:
                    cmd = (0.0, 0.0, 0.0, 1.0, 0.04)
                elif 31.0 <= t_s < 34.0:
                    cmd = (0.25, 0.0, 0.0, 1.0, 0.04)
                elif 34.0 <= t_s < 37.0:
                    cmd = (0.0, 0.0, 0.45, 1.0, 0.04)
                else:
                    cmd = (0.0, 0.0, 0.0, 1.0, 0.04)

                # Consume FGP tick when in HumanSMPL phase
                queued = None; live_fgp = False; tick = None
                if want_smpl:
                    if is_old:
                        queued, _ = src.get_latest()
                        live_fgp = (not in_real_outage) and queued is not None and (0 <= now_ns - queued.arrival_ns <= 150_000_000)
                    else:
                        queued, _, _ = src.pop_paced(now_ns=now_ns)
                        live_fgp = queued is not None
                    tick = None if not live_fgp else queued.tick

                if is_old:
                    if want_smpl and live_fgp:
                        if tick.segment != fgp_seg:
                            fgp_seg = tick.segment
                            fgp_heading = yaw_mat_np(R) @ yaw_mat_np(tick.root_rotation).T
                            fgp_origin = tick.root_xy.astype(np.float32).copy()
                            r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                            cgfa.reset(R, w, q, qd, np.zeros(2, np.float32), R_ref=r_ref)
                            ctrl.reset(q)
                        p_ref_xy = (fgp_heading[:2, :2] @ (tick.root_xy - fgp_origin)).astype(np.float32)
                        r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                        exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                        obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                        tgt, tau = ctrl.step(obs, tick.feature, q, qd, modality="human_smpl")
                    else:
                        if (not want_smpl) and fgp_seg is not None:
                            r_ref = R.copy()
                            cgfa.reset(R, w, q, qd, np.zeros(2, np.float32), R_ref=r_ref)
                            planner.reset(yaw0_rad=yaw_angle_from_mat(R))
                            fgp_seg = None
                            ctrl.reset(q)
                        plan_in, p_ref_xy, r_ref = planner.push_and_build(*cmd, cgfa=cgfa, R_imu_aligned=R)
                        exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                        obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                        tgt, tau = ctrl.step(obs, plan_in, q, qd, modality="planner_cmd")
                else:
                    branch, need_reanchor = guard.resolve_branch("human_smpl" if want_smpl else "planner", live_fgp)
                    if branch == "human_smpl":
                        if tick.segment != fgp_seg:
                            fgp_seg = tick.segment
                            fgp_heading = yaw_mat_np(R) @ yaw_mat_np(tick.root_rotation).T
                            fgp_origin = tick.root_xy.astype(np.float32).copy()
                            r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                            cgfa.reset(R, w, q, qd, np.zeros(2, np.float32), R_ref=r_ref)
                            # Keep obs_hist & engage_steps live during hot transition (never wipe history mid-flight!)
                        p_ref_xy = (fgp_heading[:2, :2] @ (tick.root_xy - fgp_origin)).astype(np.float32)
                        r_ref = (fgp_heading @ tick.root_rotation).astype(np.float32)
                        exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                        obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                        tgt, tau = ctrl.step(obs, tick.feature, q, qd, modality="human_smpl")
                    else:
                        if need_reanchor:
                            r_ref = R.copy()
                            cgfa.reset(R, w, q, qd, np.zeros(2, np.float32), R_ref=r_ref)
                            planner.reset(yaw0_rad=yaw_angle_from_mat(R))
                            fgp_seg = None
                        p_cmd = (0.0, 0.0, 0.0, 1.0, 0.04) if want_smpl else cmd
                        plan_in, p_ref_xy, r_ref = planner.push_and_build(*p_cmd, cgfa=cgfa, R_imu_aligned=R)
                        exy, _ = cgfa.step(R, w, q, qd, p_ref_xy, R_ref=r_ref)
                        obs = build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R, w_body=w, a_prev=prev_act, R_ref=r_ref, cgfa_exy=exy)
                        tgt, tau = ctrl.step(obs, plan_in, q, qd, modality="planner_cmd")

                pd_oob += int(np.sum((np.abs(tau) - eff_lim) > 1e-4))
                # 20 ms control delay injection + standard 50 Hz +-0.2 rad step clip (matching real teleop bridge!)
                eff_tgt = tgt if k == 0 else delayed_desired
                delayed_desired = tgt.copy()
                sent, sd = _step_mujoco_with_ramp(model, data, act_ids, prev_tgt, eff_tgt, ctrl.default_q, ramp_500hz=False)
                step_jumps.append(math.degrees(sd))
                prev_tgt = sent
                prev_act = (sent - ctrl.default_q) / ctrl.action_scale
                mujoco.mj_forward(model, data)

                Rp = data.xmat[pelvis].reshape(3, 3)
                tilt = math.degrees(math.acos(float(np.clip(Rp[2, 2], -1, 1))))
                tilts.append(tilt)
                if 18.0 <= t_s <= 20.0:
                    disc_tilts.append(tilt)
                if data.qpos[2] < 0.15 or tilt >= 65.0:
                    fell_at = round(t_s, 2)
                    break

            results[label] = {
                "completed_40s": len(tilts) == total_steps,
                "survived_s": round(len(tilts) * 0.02, 2),
                "fell_at_s": fell_at,
                "tilt_p50_deg": round(float(np.median(tilts)), 2),
                "tilt_p95_deg": round(float(np.percentile(tilts, 95)), 2),
                "tilt_max_deg": round(float(max(tilts)), 2),
                "disconnect_window_max_tilt_deg": round(float(max(disc_tilts, default=0.0)), 2),
                "step_jump_p99_deg": round(float(np.percentile(step_jumps, 99)), 2),
                "pd_oob": pd_oob,
            }
            print(
                f"[Suite5 {label:24s}] ok={results[label]['completed_40s']} surv={results[label]['survived_s']}/40.0s "
                f"tiltMax={results[label]['tilt_max_deg']:.1f}° discMaxTilt={results[label]['disconnect_window_max_tilt_deg']:.1f}° oob={pd_oob}",
                flush=True,
            )
    finally:
        model.geom_friction[:] = orig_fric
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--v2-smpl-onnx",
        type=Path,
        default=DEPLOY_CANDIDATE / "onnx/q1_tok_human_smpl.onnx",
    )
    ap.add_argument(
        "--fgpv2-smpl-onnx",
        type=Path,
        default=SMHP_ROOT / "runs/q1_fgp_human_smpl_v2_20261122/onnx/q1_tok_human_smpl.onnx",
    )
    ap.add_argument(
        "--head-onnx",
        type=Path,
        default=DEPLOY_CANDIDATE / "onnx/q1_h002_token_head.onnx",
    )
    ap.add_argument(
        "--planner-onnx",
        type=Path,
        default=DEPLOY_CANDIDATE / "onnx/q1_tok_planner_cmd.onnx",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "data/manifests/q1_fgp_control_v2_eval_20261122.json",
    )
    ap.add_argument("--skip-suite4", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    cfg = json.loads((DEPLOY_CANDIDATE / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
    model = load_model(DEFAULT_ASSETS)
    model.opt.timestep = 0.002
    kin = Q1KinematicsNP(DEPLOY_CANDIDATE / "robot_model/supermimic.xml")

    v2_ctrl = DualModalityOnnxController(cfg, args.head_onnx, args.v2_smpl_onnx, args.planner_onnx)
    fgpv2_ctrl = DualModalityOnnxController(cfg, args.head_onnx, args.fgpv2_smpl_onnx, args.planner_onnx)

    report: dict[str, Any] = {
        "date": "2026-11-22",
        "v2_smpl_onnx": str(args.v2_smpl_onnx),
        "fgpv2_smpl_onnx": str(args.fgpv2_smpl_onnx),
        "head_onnx": str(args.head_onnx),
        "suite1_today_fgp_141s": run_suite1_today_fgp(model, kin, cfg, v2_ctrl, fgpv2_ctrl),
        "suite2_heldout_real_fgp": run_suite2_heldout_fgp(model, kin, cfg, v2_ctrl, fgpv2_ctrl),
        "suite3_hard_primitives": run_suite3_primitives(model, kin, cfg, v2_ctrl, fgpv2_ctrl),
        "suite5_stress_switch_40s": run_suite5_stress_switch(model, kin, cfg, v2_ctrl, fgpv2_ctrl),
    }
    if not args.skip_suite4:
        report["suite4_sonic_v2_benchmarks"] = run_suite4_sonic_benchmarks(
            args.v2_smpl_onnx, args.fgpv2_smpl_onnx, args.head_onnx
        )
    report["total_eval_wall_s"] = round(time.monotonic() - t0, 2)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED_EVAL_REPORT -> {args.out_json} in {report['total_eval_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
