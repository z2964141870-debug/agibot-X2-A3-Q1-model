"""Build 250-clip Physics-Verified Multi-Dataset Q1 Benchmark (200 Train / 50 Held-Out).

Sources:
  - PHUMA (GRAB, EgoBody, fitness squat/lunge, humanml walks/turns, humman/idea400): 120 clips
  - BONES-SEED / SOMA (upright canonical pose): 90 clips
  - LAFAN1 (walk, aiming, obstacles): 26 clips
  - Synthetic SONIC Stand / Squat-Hold / In-Place Turn / Walk-Stop: 14 clips
Retains all 16 v1 clips (12 train + 4 held-out) cleanly in matching splits, plus 8 raw BVHs
in /home/yu/datasets/heldout_raw_bvh/ for zero-IK direct HumanSMPL testing.
"""

from __future__ import annotations

import concurrent.futures
import gc
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/yu/projects/SuperMimicHP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.x2_geometry import CollisionGeometry

GMR_ROOT = Path("/home/yu/models/cache/GMR")
Q1_XML = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
G1_XML = ROOT / "src/unisupermimic/assets/robots/g1/supermimic.xml"
SOMA_CANON = Path("/home/yu/datasets/soma_showcase/soma_base_skel_minimal.bvh")
SOMA_ARCHIVE = Path("/data/datasets_archive/soma_uniform.tar.gz")

OUT_ROOT = ROOT / "src/unisupermimic/assets/motions/q1/multidataset_250"
MANIFEST_OUT = ROOT / "runs/h002_sonic_v2/dataset_250_manifest.json"
HELD_OUT_RAW_BVH = Path("/home/yu/datasets/heldout_raw_bvh")

V1_HELD_OUT = {
    "stat_fit_squat",
    "soma_stat_3_eating_apple_R_002__A276",
    "dyn_hml_3",
    "walk2_subject1",
}

EXCLUDE_SOMA = re.compile(
    r"stair|ladder|box|chair|sit|lie|climb|fence|wall|table|shelf|shelves|bed|sofa|"
    r"handstand|pushup|horse_riding|looking_by_binoculars|pull_leash|bump_into_sb_bar|jump|injured",
    re.I,
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _save_q1_csv(dst: Path, qpos: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Q1 CSV columns: [px, py, pz, qx, qy, qz, qw, 22 joints]
    arr = np.c_[qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]]
    np.savetxt(dst, arr, delimiter=",", fmt="%.8f")


def _check_qpos_clean(qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    if len(qpos) < 90 or not np.isfinite(qpos).all():
        return False
    joints = qpos[:, 7:]
    if (joints < lower - 1e-3).any() or (joints > upper + 1e-3).any():
        return False
    if (qpos[:, 2] < 0.16).any() or (qpos[:, 2] > 0.60).any():
        return False
    return True


def _slice_source(src: dict[str, Any], s: int, e: int) -> dict[str, Any]:
    out_p = {k: v[s:e].copy() for k, v in src["p"].items()}
    xy0 = out_p["pelvis"][0, :2].copy()
    for v in out_p.values():
        v[:, :2] -= xy0
    out_R = {k: v[s:e].copy() for k, v in src["R"].items()}
    clr = src["clearance"][s:e].copy()
    floor = float(np.quantile(clr.min(axis=1), 0.01))
    for v in out_p.values():
        v[:, 2] -= floor
    return {
        "p": out_p,
        "R": out_R,
        "clearance": clr - floor,
        "joint_q": {k: v[s:e].copy() for k, v in src.get("joint_q", {}).items()},
        **({"source_model": src["source_model"]} if "source_model" in src else {}),
    }


_WORKER: dict[str, Any] = {}


def _init_worker() -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from tools.retarget_q1 import Retargeter

    _WORKER["solver"] = Retargeter(Q1_XML, iterations=4)


def _process_phuma_job(job: tuple[str, str, str, str]) -> dict[str, Any]:
    alias, cat, npy_path_str, out_csv_str = job
    gc.collect()
    try:
        from tools.retarget_q1 import load_g1

        d = np.load(npy_path_str, allow_pickle=True).item()
        rt, ro, q = d["root_trans"], d["root_ori"], d["dof_pos"]
        tmp_g1 = Path(f"/tmp/phuma_g1_tmp/{alias}.csv")
        tmp_g1.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(tmp_g1, np.c_[rt, ro, q], delimiter=",", fmt="%.8f")
        src = load_g1(tmp_g1, max_frames=260)
        tmp_g1.unlink(missing_ok=True)
        solver = _WORKER["solver"]
        qpos, info = solver.solve(src)
        corr = float(info.get("floor_correction_max_m", 0.0))
        if corr > 0.018 or not _check_qpos_clean(qpos, solver.lower, solver.upper):
            return {"ok": False, "alias": alias, "reason": f"qc_fail(corr={corr:.4f})"}
        dst = Path(out_csv_str)
        _save_q1_csv(dst, qpos)
        disp = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
        return {"ok": True, "alias": alias, "source": "phuma", "category": cat, "frames_30hz": len(qpos), "disp_xy_m": round(disp, 3), "floor_corr_m": round(corr, 5), "path": str(dst)}
    except Exception as exc:
        return {"ok": False, "alias": alias, "reason": f"{type(exc).__name__}: {exc}"}


def _process_soma_job(job: tuple[str, str, str, str]) -> dict[str, Any]:
    alias, cat, bvh_path_str, out_csv_str = job
    gc.collect()
    try:
        from tools.soma_source import load_soma

        src = load_soma(Path(bvh_path_str), SOMA_CANON, GMR_ROOT, max_frames=240)
        clr = src["clearance"]
        if not np.isfinite(clr).all() or len(clr) < 105:
            return {"ok": False, "alias": alias, "reason": "short_or_nan"}
        # Reject prolonged elevated support (>20cm for >=0.5s)
        both_high = (clr.min(axis=1) > 0.20).astype(np.int32)
        if len(both_high) >= 15 and np.any(np.convolve(both_high, np.ones(15, dtype=np.int32), mode="valid") >= 15):
            return {"ok": False, "alias": alias, "reason": "elevated"}
        solver = _WORKER["solver"]
        qpos, info = solver.solve(src)
        corr = float(info.get("floor_correction_max_m", 0.0))
        if corr > 0.018 or not _check_qpos_clean(qpos, solver.lower, solver.upper):
            return {"ok": False, "alias": alias, "reason": f"qc_fail(corr={corr:.4f})"}
        dst = Path(out_csv_str)
        _save_q1_csv(dst, qpos)
        disp = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
        return {"ok": True, "alias": alias, "source": "soma", "category": cat, "frames_30hz": len(qpos), "disp_xy_m": round(disp, 3), "floor_corr_m": round(corr, 5), "path": str(dst), "bvh": bvh_path_str}
    except Exception as exc:
        return {"ok": False, "alias": alias, "reason": f"{type(exc).__name__}: {exc}"}


def _process_lafan_job(job: tuple[str, str, int, int, str]) -> dict[str, Any]:
    alias, bvh_path_str, start_f, end_f, out_csv_str = job
    gc.collect()
    try:
        from tools.retarget_q1 import load_bvh

        src_full = load_bvh(Path(bvh_path_str), max_frames=end_f, gmr_root=GMR_ROOT)
        T = len(src_full["p"]["pelvis"])
        if T < end_f:
            start_f = max(0, T - 240)
            end_f = T
        if end_f - start_f < 110:
            return {"ok": False, "alias": alias, "reason": "too_short"}
        src = _slice_source(src_full, start_f, end_f)
        solver = _WORKER["solver"]
        qpos, info = solver.solve(src)
        corr = float(info.get("floor_correction_max_m", 0.0))
        if corr > 0.018 or not _check_qpos_clean(qpos, solver.lower, solver.upper):
            return {"ok": False, "alias": alias, "reason": f"qc_fail(corr={corr:.4f})"}
        dst = Path(out_csv_str)
        _save_q1_csv(dst, qpos)
        disp = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
        return {"ok": True, "alias": alias, "source": "lafan1", "category": "dynamic" if disp > 0.35 else "stationary", "frames_30hz": len(qpos), "disp_xy_m": round(disp, 3), "floor_corr_m": round(corr, 5), "path": str(dst)}
    except Exception as exc:
        return {"ok": False, "alias": alias, "reason": f"{type(exc).__name__}: {exc}"}


def build_synthetic_stand_squat_turn(stage_dir: Path) -> list[dict[str, Any]]:
    """Generate 14 kinematically consistent 6s (180 frames @ 30Hz) Q1 clips via Retargeter."""
    from tools.retarget_q1 import Retargeter
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import TRACKED_BODIES, default_joint_pos_list
    import torch

    kin = Q1Kinematics(str(Q1_XML), "cpu")
    solver = Retargeter(Q1_XML, iterations=6)
    m = solver.model
    d = mujoco.MjData(m)
    def_q = np.array(default_joint_pos_list(), dtype=np.float32)

    # Measure resting foot positions at nominal default_q
    qpos0 = np.r_[0.0, 0.0, kin.nominal_root_height, 1.0, 0.0, 0.0, 0.0, def_q]
    d.qpos[:] = qpos0
    mujoco.mj_kinematics(m, d)
    bp0, bR0, _, _ = kin.forward(
        torch.tensor([[0.0, 0.0, kin.nominal_root_height]]),
        torch.eye(3).unsqueeze(0),
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.tensor(def_q).unsqueeze(0),
        torch.zeros(1, 22),
    )
    bp0_np = bp0[0].numpy()
    bR0_np = bR0[0].numpy()
    # Adjust floor so resting ankle height = ~0.045m -> z_offset
    l_foot_col = CollisionGeometry(m, ("left_ankle_roll_link",))
    r_foot_col = CollisionGeometry(m, ("right_ankle_roll_link",))
    z_floor_off = min(l_foot_col.minimum(d), r_foot_col.minimum(d))
    z_base = kin.nominal_root_height - z_floor_off
    bp0_np[:, 2] -= z_floor_off

    specs = [
        # (alias, mode, amp_z, yaw_total_rad, speed_x, arm_mode)
        ("syn_stand_idle_1", "stand", 0.00, 0.0, 0.0, "neutral"),
        ("syn_stand_idle_2", "stand", 0.00, 0.0, 0.0, "wave"),
        ("syn_stand_reach_1", "stand", 0.01, 0.15, 0.0, "reach_front"),
        ("syn_squat_gentle_1", "squat", 0.06, 0.0, 0.0, "balance"),
        ("syn_squat_medium_1", "squat", 0.10, 0.0, 0.0, "balance"),
        ("syn_squat_deep_1", "squat", 0.13, 0.0, 0.0, "reach_front"),
        ("syn_squat_hold_1", "squat_hold", 0.11, 0.0, 0.0, "reach_front"),
        ("syn_turn_left_90", "turn", 0.00, np.pi / 2, 0.0, "neutral"),
        ("syn_turn_right_90", "turn", 0.00, -np.pi / 2, 0.0, "neutral"),
        ("syn_turn_180", "turn", 0.01, np.pi, 0.0, "balance"),
        ("syn_walk_stop_1", "walk_stop", 0.00, 0.0, 0.45, "swing"),
        ("syn_walk_stop_2", "walk_stop", 0.00, 0.35, 0.55, "swing"),
        # 2 held-out synthetic clips
        ("syn_ho_squat_hold", "squat_hold", 0.12, 0.0, 0.0, "wave"),
        ("syn_ho_turn_stop", "walk_stop", 0.00, -np.pi / 2, 0.40, "swing"),
    ]

    T = 180
    tt = np.linspace(0.0, 1.0, T)
    records = []
    for alias, mode, amp_z, yaw_tot, vx_peak, arm_mode in specs:
        p = {name: np.repeat(bp0_np[i : i + 1], T, axis=0).copy() for i, name in enumerate(TRACKED_BODIES)}
        R = {name: np.repeat(bR0_np[i : i + 1], T, axis=0).copy() for i, name in enumerate(TRACKED_BODIES)}

        # Compute pelvis root trajectory
        dz = np.zeros(T)
        yaw = np.zeros(T)
        px = np.zeros(T)
        if mode == "squat":
            dz = -amp_z * 0.5 * (1.0 - np.cos(4.0 * np.pi * tt))
        elif mode == "squat_hold":
            # Smooth down [0..0.25], hold [0.25..0.75], smooth up [0.75..1.0]
            ramp_down = np.clip(tt / 0.25, 0.0, 1.0)
            ramp_up = np.clip((1.0 - tt) / 0.25, 0.0, 1.0)
            env = np.minimum(0.5 * (1.0 - np.cos(np.pi * ramp_down)), 0.5 * (1.0 - np.cos(np.pi * ramp_up)))
            dz = -amp_z * env
        elif mode == "turn":
            yaw = yaw_tot * 0.5 * (1.0 - np.cos(np.pi * tt))
        elif mode == "walk_stop":
            # Move smoothly during [0..0.6] then stand still [0.6..1.0]
            s_phase = np.clip(tt / 0.6, 0.0, 1.0)
            px = vx_peak * 3.0 * (s_phase - np.sin(2.0 * np.pi * s_phase) / (2.0 * np.pi))
            yaw = yaw_tot * 0.5 * (1.0 - np.cos(np.pi * s_phase))

        for t_i in range(T):
            Rz = Rotation.from_euler("z", float(yaw[t_i])).as_matrix()
            # Move pelvis/torso/upper-body/knees wrt rotated frame; feet step if walk_stop or turn
            p["pelvis"][t_i, 0] += px[t_i]
            p["pelvis"][t_i, 2] += dz[t_i]
            p["torso_link"][t_i, 0] += px[t_i] + (0.04 if dz[t_i] < -0.05 else 0.0)
            p["torso_link"][t_i, 2] += dz[t_i]
            R["pelvis"][t_i] = Rz @ bR0_np[0]
            R["torso_link"][t_i] = Rz @ Rotation.from_euler("y", 0.18 * (-dz[t_i] / 0.12)).as_matrix() @ bR0_np[7]

            for side_i, side in enumerate(("left", "right")):
                sgn = 1.0 if side == "left" else -1.0
                # Arms
                arm_dx, arm_dz = 0.0, 0.0
                if arm_mode == "reach_front":
                    arm_dx = 0.12 * 0.5 * (1.0 - np.cos(2.0 * np.pi * tt[t_i]))
                    arm_dz = 0.08 * 0.5 * (1.0 - np.cos(2.0 * np.pi * tt[t_i]))
                elif arm_mode == "wave" and side == "right":
                    arm_dz = 0.16 * 0.5 * (1.0 - np.cos(4.0 * np.pi * tt[t_i]))
                elif arm_mode == "balance":
                    arm_dx = 0.08 * (-dz[t_i] / max(amp_z, 0.01))
                elif arm_mode == "swing" and tt[t_i] < 0.65:
                    arm_dx = 0.08 * sgn * np.sin(4.0 * np.pi * tt[t_i])

                for part in ("shoulder_roll_link", "elbow_link", "wrist_yaw_link"):
                    loc = bp0_np[TRACKED_BODIES.index(f"{side}_{part}")]
                    rel = Rz @ (loc * np.array([1.0, 1.0, 1.0]))
                    p[f"{side}_{part}"][t_i, :2] = rel[:2] + np.array([px[t_i], 0.0])
                    p[f"{side}_{part}"][t_i, 2] = loc[2] + dz[t_i]
                    if part in ("elbow_link", "wrist_yaw_link"):
                        p[f"{side}_{part}"][t_i] += Rz @ np.array([arm_dx, 0.0, arm_dz])
                    R[f"{side}_{part}"][t_i] = Rz @ bR0_np[TRACKED_BODIES.index(f"{side}_{part}")]

                # Knees & Ankles
                knee_idx = TRACKED_BODIES.index(f"{side}_knee_link")
                ank_idx = TRACKED_BODIES.index(f"{side}_ankle_roll_link")
                k_loc = bp0_np[knee_idx]
                a_loc = bp0_np[ank_idx]
                if mode in ("stand", "squat", "squat_hold"):
                    p[f"{side}_knee_link"][t_i, 0] = k_loc[0] + 0.05 * (-dz[t_i] / max(amp_z, 0.01))
                    p[f"{side}_knee_link"][t_i, 2] = k_loc[2] + 0.5 * dz[t_i]
                    p[f"{side}_ankle_roll_link"][t_i] = a_loc
                    R[f"{side}_ankle_roll_link"][t_i] = bR0_np[ank_idx]
                else:
                    # Smooth alternating stepping for turn or walk_stop
                    phase = 4.0 * np.pi * tt[t_i] + (0.0 if side == "left" else np.pi)
                    active = 1.0 if (mode == "turn" or tt[t_i] < 0.60) else 0.0
                    swing_h = 0.045 * max(0.0, np.sin(phase)) * active
                    r_a = Rz @ a_loc
                    p[f"{side}_ankle_roll_link"][t_i, 0] = r_a[0] + px[t_i]
                    p[f"{side}_ankle_roll_link"][t_i, 1] = r_a[1]
                    p[f"{side}_ankle_roll_link"][t_i, 2] = a_loc[2] + swing_h
                    R[f"{side}_ankle_roll_link"][t_i] = Rz @ bR0_np[ank_idx]
                    r_k = Rz @ k_loc
                    p[f"{side}_knee_link"][t_i, 0] = r_k[0] + px[t_i]
                    p[f"{side}_knee_link"][t_i, 1] = r_k[1]
                    p[f"{side}_knee_link"][t_i, 2] = k_loc[2] + 0.5 * swing_h

        clr = np.c_[
            np.maximum(p["left_ankle_roll_link"][:, 2] - bp0_np[TRACKED_BODIES.index("left_ankle_roll_link"), 2], 0.0),
            np.maximum(p["right_ankle_roll_link"][:, 2] - bp0_np[TRACKED_BODIES.index("right_ankle_roll_link"), 2], 0.0),
        ]
        src = {"p": p, "R": R, "clearance": clr, "joint_q": {}}
        qpos, info = solver.solve(src)
        corr = float(info.get("floor_correction_max_m", 0.0))
        dst = stage_dir / f"{alias}.csv"
        _save_q1_csv(dst, qpos)
        disp = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
        records.append({
            "ok": True,
            "alias": alias,
            "source": "synthetic",
            "category": "dynamic" if mode in ("turn", "walk_stop") else "stationary",
            "held_out": alias.startswith("syn_ho_"),
            "frames_30hz": len(qpos),
            "disp_xy_m": round(disp, 3),
            "floor_corr_m": round(corr, 5),
            "path": str(dst),
        })
        print(f"SYN_OK {alias:20s}: T={len(qpos)} disp={disp:.2f}m corr={corr:.5f}", flush=True)
    return records


def main() -> None:
    t_start = time.monotonic()
    stage = Path("/tmp/md250_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "train").mkdir(parents=True)
    (OUT_ROOT / "held_out").mkdir(parents=True)
    HELD_OUT_RAW_BVH.mkdir(parents=True, exist_ok=True)
    for old in HELD_OUT_RAW_BVH.glob("*.bvh"):
        old.unlink()

    all_records: list[dict[str, Any]] = []

    # 1. Seed all 16 v1 clips verbatim
    v1_map = []
    for ds, folder in (
        ("lafan1", "lafan1_showcase"),
        ("phuma", "phuma_showcase"),
        ("soma", "soma_showcase"),
    ):
        for p in sorted((ROOT / f"src/unisupermimic/assets/motions/q1/{folder}").glob("*.csv")):
            stem = p.stem
            cat = "stationary" if (stem.startswith("stat_") or stem.startswith("soma_stat_")) else "dynamic"
            ho = stem in V1_HELD_OUT
            dst = stage / f"{stem}.csv"
            shutil.copy2(p, dst)
            rows = np.loadtxt(dst, delimiter=",")
            disp = float(np.linalg.norm(rows[-1, :2] - rows[0, :2]))
            all_records.append({
                "ok": True,
                "alias": stem,
                "source": ds,
                "category": cat,
                "v1_clip": True,
                "held_out": ho,
                "frames_30hz": len(rows),
                "disp_xy_m": round(disp, 3),
                "floor_corr_m": 0.0,
                "path": str(dst),
            })
            v1_map.append(stem)
    print(f"Seeded {len(v1_map)} v1 clips.", flush=True)

    # Save v1 held-out BVHs (walk2_subject1 + soma_stat_3) into HELD_OUT_RAW_BVH
    for bvh_src, bvh_dst in (
        (Path("/home/yu/datasets/lafan1/walk2_subject1.bvh"), HELD_OUT_RAW_BVH / "lafan_walk2_subject1.bvh"),
        (Path("/home/yu/datasets/soma_showcase/eating_apple_R_002__A276.bvh"), HELD_OUT_RAW_BVH / "soma_stat_3_eating_apple_R_002__A276.bvh"),
    ):
        if bvh_src.is_file():
            shutil.copy2(bvh_src, bvh_dst)

    # 2. Gather PHUMA candidates (+114 target -> sample ~145 candidates for QC headroom)
    g1_model = mujoco.MjModel.from_xml_path(str(G1_XML))
    g1_data = mujoco.MjData(g1_model)
    l_foot = CollisionGeometry(g1_model, ("left_ankle_roll_link",))
    r_foot = CollisionGeometry(g1_model, ("right_ankle_roll_link",))

    phuma_specs = [
        ("humanml", "dynamic", "/home/yu/datasets/phuma/data/g1/humanml/**/*.npy", 48),
        ("fitness", "stationary", "/home/yu/datasets/phuma/data/g1/fitness/**/*.npy", 40),
        ("GRAB", "stationary", "/home/yu/datasets/phuma/data/g1/GRAB/**/*.npy", 26),
        ("EgoBody", "stationary", "/home/yu/datasets/phuma/data/g1/EgoBody/**/*.npy", 22),
        ("idea400", "stationary", "/home/yu/datasets/phuma/data/g1/idea400/**/*.npy", 16),
    ]
    v1_phuma_srcs = {"airplane_lift_chunk_0001.npy", "000_chunk_0000.npy", "Perform_Ballet_clip_10_chunk_0000.npy", "000005_chunk_0000.npy", "000006_chunk_0000.npy", "000007_chunk_0000.npy"}
    phuma_jobs = []
    for sub, cat, pat, n_cand in phuma_specs:
        cands = sorted(glob.glob(pat, recursive=True))
        step = max(1, len(cands) // 250)
        picked = 0
        for p in cands[::step]:
            if Path(p).name in v1_phuma_srcs:
                continue
            d = np.load(p, allow_pickle=True).item()
            rt, ro, q = d["root_trans"], d["root_ori"], d["dof_pos"]
            if not (110 <= len(rt) <= 300 and np.isfinite(rt).all() and np.isfinite(q).all()):
                continue
            g1_data.qpos[:] = np.r_[rt[0], ro[0, [3, 0, 1, 2]], q[0]]
            mujoco.mj_kinematics(g1_model, g1_data)
            z_min = min(l_foot.minimum(g1_data), r_foot.minimum(g1_data))
            if abs(z_min) > 0.11:
                continue
            disp = float(np.linalg.norm(rt[-1, :2] - rt[0, :2]))
            if sub == "humanml" and disp < 0.35:
                continue
            alias = f"phuma_{sub.lower()}_{picked:03d}"
            cat_actual = "dynamic" if disp > 0.35 else "stationary"
            phuma_jobs.append((alias, cat_actual, p, str(stage / f"{alias}.csv")))
            picked += 1
            if picked >= n_cand:
                break
    print(f"Prepared {len(phuma_jobs)} PHUMA retarget jobs.", flush=True)

    # 3. Extract SOMA candidates (+84 target -> extract ~105 candidates sequentially)
    soma_tmp_bvh = Path("/tmp/soma_bvh_250")
    if soma_tmp_bvh.exists():
        shutil.rmtree(soma_tmp_bvh)
    soma_tmp_bvh.mkdir(parents=True)
    v1_soma_bvh = {p.name for p in Path("/home/yu/datasets/soma_showcase").glob("*.bvh")}
    soma_jobs = []
    seen_prefix = set()
    n_stat_c, n_dyn_c = 0, 0
    with tarfile.open(SOMA_ARCHIVE, "r|gz") as tf:
        for idx, m in enumerate(tf):
            if not m.isfile() or not m.name.endswith(".bvh") or "_M.bvh" in m.name:
                continue
            if not (380_000 <= m.size <= 3_200_000):
                continue
            bn = m.name.split("/")[-1]
            if bn in v1_soma_bvh or EXCLUDE_SOMA.search(bn):
                continue
            pref = re.sub(r"_[0-9]+__A[0-9]+.*", "", bn)
            if pref in seen_prefix:
                continue
            is_dyn = any(k in bn for k in ("walk", "turn", "jog_ff"))
            if is_dyn and n_dyn_c >= 65:
                continue
            if (not is_dyn) and n_stat_c >= 62:
                continue
            raw = tf.extractfile(m).read()
            dst_bvh = soma_tmp_bvh / bn
            dst_bvh.write_bytes(raw)
            seen_prefix.add(pref)
            cat = "dynamic" if is_dyn else "stationary"
            alias = f"soma_{'dyn' if is_dyn else 'stat'}_{len(soma_jobs):03d}_{Path(bn).stem[:24]}"
            soma_jobs.append((alias, cat, str(dst_bvh), str(stage / f"{alias}.csv")))
            if is_dyn:
                n_dyn_c += 1
            else:
                n_stat_c += 1
            if n_stat_c >= 62 and n_dyn_c >= 65:
                break
    print(f"Extracted {len(soma_jobs)} SOMA candidate BVHs ({n_stat_c} stat, {n_dyn_c} dyn).", flush=True)

    # 4. Prepare LAFAN1 candidates (+22 target -> 30 windows)
    lafan_specs = [
        ("walk1_subject2.bvh", 0), ("walk1_subject2.bvh", 240),
        ("walk1_subject5.bvh", 0), ("walk1_subject5.bvh", 240),
        ("walk2_subject3.bvh", 0), ("walk2_subject3.bvh", 240),
        ("walk2_subject4.bvh", 0), ("walk3_subject2.bvh", 0), ("walk3_subject2.bvh", 240),
        ("walk3_subject3.bvh", 0), ("walk3_subject4.bvh", 0),
        ("walk3_subject5.bvh", 0), ("aiming1_subject1.bvh", 0),
        ("aiming1_subject1.bvh", 240), ("aiming1_subject4.bvh", 0),
        ("aiming2_subject2.bvh", 0), ("aiming2_subject2.bvh", 240),
        ("aiming2_subject3.bvh", 0), ("aiming2_subject5.bvh", 0),
        ("obstacles1_subject1.bvh", 0), ("obstacles1_subject2.bvh", 0),
        ("obstacles1_subject5.bvh", 0), ("obstacles2_subject1.bvh", 0),
        ("obstacles2_subject2.bvh", 0), ("obstacles2_subject5.bvh", 0),
        ("obstacles3_subject3.bvh", 0), ("obstacles4_subject2.bvh", 0),
        ("obstacles5_subject2.bvh", 0), ("ground1_subject1.bvh", 0),
        ("ground2_subject2.bvh", 0),
    ]
    lafan_jobs = []
    for bn, s0 in lafan_specs:
        bp = Path("/home/yu/datasets/lafan1") / bn
        if not bp.is_file():
            continue
        alias = f"lafan_{bp.stem}_w{s0}"
        lafan_jobs.append((alias, str(bp), s0, s0 + 240, str(stage / f"{alias}.csv")))
    print(f"Prepared {len(lafan_jobs)} LAFAN1 window jobs.", flush=True)

    # 5. Execute all retarget jobs in parallel with 6 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=6, initializer=_init_worker) as pool:
        phuma_res = list(pool.map(_process_phuma_job, phuma_jobs))
        soma_res = list(pool.map(_process_soma_job, soma_jobs))
        lafan_res = list(pool.map(_process_lafan_job, lafan_jobs))

    phuma_ok = [r for r in phuma_res if r["ok"]][:114]
    soma_ok = [r for r in soma_res if r["ok"]][:84]
    lafan_ok = [r for r in lafan_res if r["ok"]][:22]
    print(
        f"Accepted new clips: PHUMA={len(phuma_ok)}/{len(phuma_jobs)}, "
        f"SOMA={len(soma_ok)}/{len(soma_jobs)}, LAFAN1={len(lafan_ok)}/{len(lafan_jobs)}",
        flush=True,
    )

    # 6. Synthetic Stand / Squat-Hold / Turn-Stop (14 clips)
    syn_ok = build_synthetic_stand_squat_turn(stage)

    # 7. Assemble exact 200 Train / 50 Held-Out split
    new_phuma_ho = phuma_ok[:22]
    new_phuma_tr = phuma_ok[22:]
    new_soma_ho = soma_ok[:16]
    new_soma_tr = soma_ok[16:]
    new_lafan_ho = lafan_ok[:6]
    new_lafan_tr = lafan_ok[6:]

    for r in new_phuma_ho + new_soma_ho + new_lafan_ho:
        r["held_out"] = True
    for r in new_phuma_tr + new_soma_tr + new_lafan_tr:
        r["held_out"] = False

    # Retain 3 more SOMA + 3 more LAFAN1 raw held-out BVHs for Zero-IK HumanSMPL testing (8 total)
    for r in new_soma_ho[:3]:
        bvh_p = Path(r["bvh"])
        if bvh_p.is_file():
            shutil.copy2(bvh_p, HELD_OUT_RAW_BVH / f"{r['alias']}.bvh")
    for alias_bvh, src_name in (
        ("lafan_aiming1_subject4", "aiming1_subject4.bvh"),
        ("lafan_obstacles1_subject2", "obstacles1_subject2.bvh"),
        ("lafan_walk3_subject3", "walk3_subject3.bvh"),
    ):
        src_p = Path("/home/yu/datasets/lafan1") / src_name
        if src_p.is_file():
            shutil.copy2(src_p, HELD_OUT_RAW_BVH / f"{alias_bvh}.bvh")

    combined = all_records + new_phuma_ho + new_phuma_tr + new_soma_ho + new_soma_tr + new_lafan_ho + new_lafan_tr + syn_ok
    held_out_list = [r for r in combined if r.get("held_out")]
    train_list = [r for r in combined if not r.get("held_out")]

    # Ensure exact 50 held_out and 200 train if any source had slight shortfall
    while len(held_out_list) > 50:
        mv = held_out_list.pop()
        if not mv.get("v1_clip"):
            mv["held_out"] = False
            train_list.append(mv)
    while len(held_out_list) < 50 and train_list:
        for i in range(len(train_list) - 1, -1, -1):
            if not train_list[i].get("v1_clip"):
                mv = train_list.pop(i)
                mv["held_out"] = True
                held_out_list.append(mv)
                break
    train_list = train_list[:200]

    final_records = []
    for split_name, lst in (("train", train_list), ("held_out", held_out_list)):
        for r in lst:
            dst_csv = OUT_ROOT / split_name / f"{r['alias']}.csv"
            shutil.copy2(Path(r["path"]), dst_csv)
            r_clean = {k: v for k, v in r.items() if k not in ("bvh", "path", "ok")}
            r_clean["split"] = split_name
            r_clean["csv_rel"] = str(dst_csv.relative_to(ROOT))
            r_clean["sha256"] = _sha256(dst_csv)
            final_records.append(r_clean)

    # 8. Final MotionLib 50Hz validation on ALL 250 CSVs
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    kin = Q1Kinematics(str(Q1_XML), "cpu")
    all_csvs = sorted(str(ROOT / r["csv_rel"]) for r in final_records)
    ml = MotionLib(all_csvs, kin, "cpu", src_fps=30.0)
    assert len(all_csvs) == len(final_records)
    assert bool((ml.q >= kin.q_lo - 2e-3).all() and (ml.q <= kin.q_hi + 2e-3).all()), "Joint limit violation in dataset!"

    by_source = {}
    for r in final_records:
        by_source.setdefault(r["source"], {"train": 0, "held_out": 0})
        by_source[r["source"]][r["split"]] += 1

    manifest = {
        "total_clips": len(final_records),
        "train_clips": sum(1 for r in final_records if r["split"] == "train"),
        "held_out_clips": sum(1 for r in final_records if r["split"] == "held_out"),
        "total_50hz_frames": int(ml.num_frames),
        "by_source": by_source,
        "held_out_raw_bvh": sorted(p.name for p in HELD_OUT_RAW_BVH.glob("*.bvh")),
        "wall_s": round(time.monotonic() - t_start, 2),
        "clips": final_records,
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "clips"}, indent=2), flush=True)

    # Cleanup stage & tmp BVHs
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(soma_tmp_bvh, ignore_errors=True)
    shutil.rmtree("/tmp/phuma_g1_tmp", ignore_errors=True)
    print("DATASET_250_READY", flush=True)


if __name__ == "__main__":
    main()
