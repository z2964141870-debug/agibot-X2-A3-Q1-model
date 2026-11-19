"""Build 2,000-clip Physics-Verified Multi-Dataset Q1 Benchmark (1,800 Train / 200 Held-Out).

Sources:
  - Seeds all 250 v2 clips verbatim (200 Train / 50 Held-Out) so v1(12) -> v2(200) -> v3(1,800) are strictly nested.
  - Expands across all 15 PHUMA subsets, BONES-SEED/SOMA archive, LAFAN1 multi-window BVHs, and Synthetic primitives
    (stand, squat, squat_hold, turns, walk_stop, and lateral sidestep) using 14 Mink-IK workers + /dev/shm staging.
  - Stages 24 unseen raw human BVHs (12 SOMA + 12 LAFAN1) in /home/yu/datasets/heldout_raw_bvh_24/ with pre-cached
    .50hz.npz files for zero-IK direct HumanSMPL evaluation under MuJoCo 3.11.
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
from collections import defaultdict
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.x2_geometry import CollisionGeometry

GMR_ROOT = Path("/home/yu/models/cache/GMR")
Q1_XML = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
G1_XML = ROOT / "src/unisupermimic/assets/robots/g1/supermimic.xml"
SOMA_CANON = Path("/home/yu/datasets/soma_showcase/soma_base_skel_minimal.bvh")
SOMA_ARCHIVE = Path("/data/datasets_archive/soma_uniform.tar.gz")
V2_MANIFEST = ROOT / "runs/h002_sonic_v2/dataset_250_manifest.json"
V2_RAW_BVH = Path("/home/yu/datasets/heldout_raw_bvh")

OUT_ROOT = ROOT / "src/unisupermimic/assets/motions/q1/multidataset_2000"
MANIFEST_OUT = ROOT / "runs/h002_sonic_v3/dataset_2000_manifest.json"
HELD_OUT_RAW_BVH_24 = Path("/home/yu/datasets/heldout_raw_bvh_24")

SHM_STAGE = Path("/dev/shm/md2k_stage")
SHM_PHUMA = Path("/dev/shm/phuma_2k_tmp")
SHM_SOMA = Path("/dev/shm/soma_bvh_2k")

EXCLUDE_SOMA = re.compile(
    r"stair|ladder|box|chair|sit|lie|climb|fence|wall|table|shelf|shelves|bed|sofa|"
    r"handstand|pushup|horse_riding|looking_by_binoculars|pull_leash|bump_into_sb_bar|jump|injured",
    re.I,
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _save_q1_csv(dst: Path, qpos: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.c_[qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]]
    np.savetxt(dst, arr, delimiter=",", fmt="%.8f")


def _check_qpos_clean(qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    if len(qpos) < 85 or not np.isfinite(qpos).all():
        return False
    joints = qpos[:, 7:]
    if (joints < lower - 1.5e-3).any() or (joints > upper + 1.5e-3).any():
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


def _process_phuma_job(job: tuple[str, str, str, str, str]) -> dict[str, Any]:
    alias, subset, cat, npy_path_str, out_csv_str = job
    gc.collect()
    try:
        from tools.retarget_q1 import load_g1

        d = np.load(npy_path_str, allow_pickle=True).item()
        rt, ro, q = d["root_trans"], d["root_ori"], d["dof_pos"]
        tmp_g1 = SHM_PHUMA / f"{alias}.csv"
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
        return {
            "ok": True,
            "alias": alias,
            "source": "phuma",
            "subset": subset,
            "category": cat,
            "frames_30hz": len(qpos),
            "disp_xy_m": round(disp, 3),
            "floor_corr_m": round(corr, 5),
            "path": str(dst),
        }
    except Exception as exc:
        return {"ok": False, "alias": alias, "reason": f"{type(exc).__name__}: {exc}"}


def _process_soma_job(job: tuple[str, str, str, str]) -> dict[str, Any]:
    alias, cat, bvh_path_str, out_csv_str = job
    gc.collect()
    try:
        from tools.soma_source import load_soma

        src = load_soma(Path(bvh_path_str), SOMA_CANON, GMR_ROOT, max_frames=240)
        clr = src["clearance"]
        if not np.isfinite(clr).all() or len(clr) < 95:
            return {"ok": False, "alias": alias, "reason": "short_or_nan"}
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
        return {
            "ok": True,
            "alias": alias,
            "source": "soma",
            "category": cat,
            "frames_30hz": len(qpos),
            "disp_xy_m": round(disp, 3),
            "floor_corr_m": round(corr, 5),
            "path": str(dst),
            "bvh": bvh_path_str,
        }
    except Exception as exc:
        return {"ok": False, "alias": alias, "reason": f"{type(exc).__name__}: {exc}"}


def _process_lafan_bvh_multi(job: tuple[str, list[tuple[str, int, int, str, bool]]]) -> list[dict[str, Any]]:
    bvh_path_str, windows = job
    gc.collect()
    out: list[dict[str, Any]] = []
    try:
        from tools.retarget_q1 import load_bvh

        max_end = max(e for _, _, e, _, _ in windows)
        src_full = load_bvh(Path(bvh_path_str), max_frames=max_end, gmr_root=GMR_ROOT)
        T = len(src_full["p"]["pelvis"])
        solver = _WORKER["solver"]
        for alias, s0, e0, out_csv_str, is_ho in windows:
            if T < s0 + 110:
                continue
            ee = min(T, e0)
            src = _slice_source(src_full, s0, ee)
            qpos, info = solver.solve(src)
            corr = float(info.get("floor_correction_max_m", 0.0))
            if corr > 0.018 or not _check_qpos_clean(qpos, solver.lower, solver.upper):
                continue
            dst = Path(out_csv_str)
            _save_q1_csv(dst, qpos)
            disp = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
            out.append({
                "ok": True,
                "alias": alias,
                "source": "lafan1",
                "category": "dynamic" if disp > 0.35 else "stationary",
                "held_out": is_ho,
                "frames_30hz": len(qpos),
                "disp_xy_m": round(disp, 3),
                "floor_corr_m": round(corr, 5),
                "path": str(dst),
                "bvh": bvh_path_str,
            })
        return out
    except Exception:
        return out


def build_synthetic_expansion(stage_dir: Path, n_new: int = 46) -> list[dict[str, Any]]:
    """Generate 46 new parameterized 6s (180-frame @ 30Hz) Q1 primitives including lateral sidestepping."""
    from tools.retarget_q1 import Retargeter
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import TRACKED_BODIES, default_joint_pos_list
    import torch

    kin = Q1Kinematics(str(Q1_XML), "cpu")
    solver = Retargeter(Q1_XML, iterations=6)
    m = solver.model
    d = mujoco.MjData(m)
    def_q = np.array(default_joint_pos_list(), dtype=np.float32)

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
    l_foot_col = CollisionGeometry(m, ("left_ankle_roll_link",))
    r_foot_col = CollisionGeometry(m, ("right_ankle_roll_link",))
    z_floor_off = min(l_foot_col.minimum(d), r_foot_col.minimum(d))
    bp0_np[:, 2] -= z_floor_off

    specs: list[tuple[str, str, float, float, float, float, str, bool]] = []
    # (alias, mode, amp_z, yaw_tot, vx_peak, vy_peak, arm_mode, held_out)
    for k, amp in enumerate((0.04, 0.055, 0.075, 0.09, 0.11, 0.125, 0.135, 0.08)):
        specs.append((f"syn3_squat_{k:02d}", "squat", amp, 0.0, 0.0, 0.0, "balance" if k % 2 == 0 else "reach_front", False))
    for k, amp in enumerate((0.07, 0.09, 0.115, 0.13, 0.085, 0.105)):
        specs.append((f"syn3_sqhold_{k:02d}", "squat_hold", amp, 0.0, 0.0, 0.0, "wave" if k % 2 else "reach_front", False))
    for k, deg in enumerate((45, -45, 75, -75, 120, -120, 150, -150, 180, -180)):
        specs.append((f"syn3_turn_{k:02d}", "turn", 0.01 if k % 2 else 0.0, float(np.radians(deg)), 0.0, 0.0, "balance", False))
    for k, (vx, deg) in enumerate([(0.25, 0), (0.35, 20), (0.45, -25), (0.55, 15), (0.65, -20), (0.30, 45), (0.50, -45), (0.60, 0)]):
        specs.append((f"syn3_walkstop_{k:02d}", "walk_stop", 0.0, float(np.radians(deg)), vx, 0.0, "swing", False))
    for k, vy in enumerate((0.20, -0.20, 0.28, -0.28, 0.34, -0.34, 0.24, -0.24)):
        specs.append((f"syn3_sidestep_{k:02d}", "sidestep", 0.0, 0.0, 0.0, vy, "balance", False))
    # 6 held-out synthetic primitives covering each movement family
    specs.extend([
        ("syn3_ho_squat_deep", "squat_hold", 0.13, 0.0, 0.0, 0.0, "reach_front", True),
        ("syn3_ho_turn_135", "turn", 0.01, float(np.radians(135)), 0.0, 0.0, "balance", True),
        ("syn3_ho_walk_fast_stop", "walk_stop", 0.0, float(np.radians(25)), 0.58, 0.0, "swing", True),
        ("syn3_ho_sidestep_l", "sidestep", 0.0, 0.0, 0.0, 0.26, "balance", True),
        ("syn3_ho_sidestep_r", "sidestep", 0.0, 0.0, 0.0, -0.26, "balance", True),
        ("syn3_ho_stand_reach", "stand", 0.01, -0.20, 0.0, 0.0, "wave", True),
    ])
    specs = specs[:n_new]

    T = 180
    tt = np.linspace(0.0, 1.0, T)
    records = []
    for alias, mode, amp_z, yaw_tot, vx_peak, vy_peak, arm_mode, is_ho in specs:
        p = {name: np.repeat(bp0_np[i : i + 1], T, axis=0).copy() for i, name in enumerate(TRACKED_BODIES)}
        R = {name: np.repeat(bR0_np[i : i + 1], T, axis=0).copy() for i, name in enumerate(TRACKED_BODIES)}

        dz = np.zeros(T)
        yaw = np.zeros(T)
        px = np.zeros(T)
        py = np.zeros(T)
        if mode == "squat":
            dz = -amp_z * 0.5 * (1.0 - np.cos(4.0 * np.pi * tt))
        elif mode == "squat_hold":
            ramp_down = np.clip(tt / 0.25, 0.0, 1.0)
            ramp_up = np.clip((1.0 - tt) / 0.25, 0.0, 1.0)
            env = np.minimum(0.5 * (1.0 - np.cos(np.pi * ramp_down)), 0.5 * (1.0 - np.cos(np.pi * ramp_up)))
            dz = -amp_z * env
        elif mode == "turn":
            yaw = yaw_tot * 0.5 * (1.0 - np.cos(np.pi * tt))
        elif mode == "walk_stop":
            s_phase = np.clip(tt / 0.65, 0.0, 1.0)
            px = vx_peak * 3.0 * (s_phase - np.sin(2.0 * np.pi * s_phase) / (2.0 * np.pi))
            yaw = yaw_tot * 0.5 * (1.0 - np.cos(np.pi * s_phase))
        elif mode == "sidestep":
            s_phase = np.clip(tt / 0.70, 0.0, 1.0)
            py = vy_peak * 2.5 * (s_phase - np.sin(2.0 * np.pi * s_phase) / (2.0 * np.pi))

        for t_i in range(T):
            Rz = Rotation.from_euler("z", float(yaw[t_i])).as_matrix()
            p["pelvis"][t_i, 0] += px[t_i]
            p["pelvis"][t_i, 1] += py[t_i]
            p["pelvis"][t_i, 2] += dz[t_i]
            p["torso_link"][t_i, 0] += px[t_i] + (0.04 if dz[t_i] < -0.05 else 0.0)
            p["torso_link"][t_i, 1] += py[t_i]
            p["torso_link"][t_i, 2] += dz[t_i]
            R["pelvis"][t_i] = Rz @ bR0_np[0]
            R["torso_link"][t_i] = Rz @ Rotation.from_euler("y", 0.18 * (-dz[t_i] / 0.12)).as_matrix() @ bR0_np[7]

            for side in ("left", "right"):
                sgn = 1.0 if side == "left" else -1.0
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
                    rel = Rz @ loc
                    p[f"{side}_{part}"][t_i, 0] = rel[0] + px[t_i]
                    p[f"{side}_{part}"][t_i, 1] = rel[1] + py[t_i]
                    p[f"{side}_{part}"][t_i, 2] = loc[2] + dz[t_i]
                    if part in ("elbow_link", "wrist_yaw_link"):
                        p[f"{side}_{part}"][t_i] += Rz @ np.array([arm_dx, 0.0, arm_dz])
                    R[f"{side}_{part}"][t_i] = Rz @ bR0_np[TRACKED_BODIES.index(f"{side}_{part}")]

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
                    phase = 4.0 * np.pi * tt[t_i] + (0.0 if side == "left" else np.pi)
                    active = 1.0 if (mode == "turn" or tt[t_i] < 0.68) else 0.0
                    swing_h = 0.045 * max(0.0, np.sin(phase)) * active
                    r_a = Rz @ a_loc
                    p[f"{side}_ankle_roll_link"][t_i, 0] = r_a[0] + px[t_i]
                    p[f"{side}_ankle_roll_link"][t_i, 1] = r_a[1] + py[t_i]
                    p[f"{side}_ankle_roll_link"][t_i, 2] = a_loc[2] + swing_h
                    R[f"{side}_ankle_roll_link"][t_i] = Rz @ bR0_np[ank_idx]
                    r_k = Rz @ k_loc
                    p[f"{side}_knee_link"][t_i, 0] = r_k[0] + px[t_i]
                    p[f"{side}_knee_link"][t_i, 1] = r_k[1] + py[t_i]
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
            "category": "dynamic" if mode in ("turn", "walk_stop", "sidestep") else "stationary",
            "held_out": is_ho,
            "frames_30hz": len(qpos),
            "disp_xy_m": round(disp, 3),
            "floor_corr_m": round(corr, 5),
            "path": str(dst),
        })
    print(f"SYN_OK: generated {len(records)} new synthetic Q1 clips.", flush=True)
    return records


def main() -> None:
    t_start = time.monotonic()
    for p in (SHM_STAGE, SHM_PHUMA, SHM_SOMA):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "train").mkdir(parents=True)
    (OUT_ROOT / "held_out").mkdir(parents=True)
    HELD_OUT_RAW_BVH_24.mkdir(parents=True, exist_ok=True)
    for old in list(HELD_OUT_RAW_BVH_24.glob("*.bvh")) + list(HELD_OUT_RAW_BVH_24.glob("*.npz")):
        old.unlink()

    # 1. Seed all 250 v2 clips verbatim (200 train / 50 held_out)
    v2_manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    seeded_records: list[dict[str, Any]] = []
    seen_csv_sha: set[str] = set()
    v2_soma_prefixes: set[str] = set()
    for c in v2_manifest["clips"]:
        src_csv = ROOT / c["csv_rel"]
        dst_csv = SHM_STAGE / f"{c['alias']}.csv"
        shutil.copy2(src_csv, dst_csv)
        sha = c.get("sha256") or _sha256(dst_csv)
        seen_csv_sha.add(sha)
        if c["source"] == "soma":
            pref = re.sub(r"^soma_(?:dyn|stat)_[0-9]{3}_", "", c["alias"])
            pref = re.sub(r"_[0-9]+__A[0-9]+.*", "", pref)
            v2_soma_prefixes.add(pref[:18])
        rec = {
            "ok": True,
            "alias": c["alias"],
            "source": c["source"],
            "category": c["category"],
            "v1_clip": bool(c.get("v1_clip", False)),
            "v2_clip": True,
            "held_out": (c["split"] == "held_out"),
            "frames_30hz": int(c["frames_30hz"]),
            "disp_xy_m": float(c["disp_xy_m"]),
            "floor_corr_m": float(c.get("floor_corr_m", 0.0)),
            "path": str(dst_csv),
            "sha256": sha,
        }
        seeded_records.append(rec)
    print(f"Seeded {len(seeded_records)} v2 clips (200 train / 50 held_out).", flush=True)

    # Copy existing 8 raw held-out BVHs into heldout_raw_bvh_24
    for bp in sorted(V2_RAW_BVH.glob("*.bvh")):
        shutil.copy2(bp, HELD_OUT_RAW_BVH_24 / bp.name)

    # 2. Gather PHUMA candidates across all 15 G1 subsets (~1,550 candidates -> target +1,080 accepted)
    g1_model = mujoco.MjModel.from_xml_path(str(G1_XML))
    g1_data = mujoco.MjData(g1_model)
    l_foot = CollisionGeometry(g1_model, ("left_ankle_roll_link",))
    r_foot = CollisionGeometry(g1_model, ("right_ankle_roll_link",))

    phuma_specs = [
        ("humanml", "dynamic", 380, 0.30),
        ("fitness", "stationary", 330, 0.0),
        ("idea400", "stationary", 200, 0.0),
        ("game_motion", "dynamic", 150, 0.0),
        ("music", "stationary", 100, 0.0),
        ("EgoBody", "stationary", 90, 0.0),
        ("kungfu", "dynamic", 85, 0.0),
        ("perform", "stationary", 85, 0.0),
        ("haa500", "stationary", 85, 0.0),
        ("aist", "dynamic", 80, 0.0),
        ("GRAB", "stationary", 80, 0.0),
        ("custom", "stationary", 55, 0.0),
        ("humman", "stationary", 55, 0.0),
        ("dance", "dynamic", 40, 0.0),
        ("animation", "stationary", 40, 0.0),
    ]
    phuma_jobs: list[tuple[str, str, str, str, str]] = []
    for sub, default_cat, n_cand, min_disp in phuma_specs:
        cands = sorted(glob.glob(f"/home/yu/datasets/phuma/data/g1/{sub}/**/*.npy", recursive=True))
        if not cands:
            continue
        stride = max(1, len(cands) // (n_cand * 2))
        picked = 0
        # Shift starting index by 1 to naturally avoid exact v2 stride hits (still SHA-deduped afterward)
        for p in cands[1::stride]:
            d = np.load(p, allow_pickle=True).item()
            rt, ro, q = d["root_trans"], d["root_ori"], d["dof_pos"]
            if not (95 <= len(rt) <= 320 and np.isfinite(rt).all() and np.isfinite(q).all()):
                continue
            g1_data.qpos[:] = np.r_[rt[0], ro[0, [3, 0, 1, 2]], q[0]]
            mujoco.mj_kinematics(g1_model, g1_data)
            z_min = min(l_foot.minimum(g1_data), r_foot.minimum(g1_data))
            if abs(z_min) > 0.12:
                continue
            disp = float(np.linalg.norm(rt[-1, :2] - rt[0, :2]))
            if min_disp > 0.0 and disp < min_disp:
                continue
            alias = f"phuma3_{sub.lower()}_{picked:04d}"
            cat_actual = "dynamic" if disp > 0.35 else "stationary"
            phuma_jobs.append((alias, sub, cat_actual, p, str(SHM_STAGE / f"{alias}.csv")))
            picked += 1
            if picked >= n_cand:
                break
    print(f"Prepared {len(phuma_jobs)} PHUMA retarget jobs across 15 subsets.", flush=True)

    # 3. Extract SOMA candidates from /data/datasets_archive/soma_uniform.tar.gz (~640 BVHs -> target +410)
    v1_soma_bvh = {p.name for p in Path("/home/yu/datasets/soma_showcase").glob("*.bvh")}
    soma_jobs: list[tuple[str, str, str, str]] = []
    prefix_counts: dict[str, int] = defaultdict(int)
    n_stat_c, n_dyn_c = 0, 0
    with tarfile.open(SOMA_ARCHIVE, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".bvh") or "_M.bvh" in m.name:
                continue
            if not (350_000 <= m.size <= 3_500_000):
                continue
            bn = m.name.split("/")[-1]
            if bn in v1_soma_bvh or EXCLUDE_SOMA.search(bn):
                continue
            pref = re.sub(r"_[0-9]+__A[0-9]+.*", "", bn)
            if any(pref.startswith(vp) for vp in v2_soma_prefixes if vp):
                continue
            max_per_pref = 2 if (n_stat_c + n_dyn_c) < 420 else 3
            if prefix_counts[pref] >= max_per_pref:
                continue
            is_dyn = any(k in bn for k in ("walk", "turn", "jog_ff", "sidestep", "run"))
            if is_dyn and n_dyn_c >= 330:
                continue
            if (not is_dyn) and n_stat_c >= 330:
                continue
            raw = tf.extractfile(m).read()
            dst_bvh = SHM_SOMA / f"s{len(soma_jobs):04d}_{bn}"
            dst_bvh.write_bytes(raw)
            prefix_counts[pref] += 1
            cat = "dynamic" if is_dyn else "stationary"
            alias = f"soma3_{'dyn' if is_dyn else 'stat'}_{len(soma_jobs):04d}_{Path(bn).stem[:22]}"
            soma_jobs.append((alias, cat, str(dst_bvh), str(SHM_STAGE / f"{alias}.csv")))
            if is_dyn:
                n_dyn_c += 1
            else:
                n_stat_c += 1
            if n_stat_c >= 330 and n_dyn_c >= 330:
                break
    print(f"Extracted {len(soma_jobs)} SOMA BVHs into /dev/shm ({n_stat_c} stat, {n_dyn_c} dyn).", flush=True)

    # 4. Prepare LAFAN1 multi-window jobs (~290 windows -> target +214, clean BVH split for 8 new held-out BVHs)
    v2_lafan_windows = {c["alias"] for c in v2_manifest["clips"] if c["source"] == "lafan1"}
    v2_ho_bvh_stems = {"aiming1_subject4", "obstacles1_subject2", "walk2_subject1", "walk3_subject3"}
    new_ho_lafan_bvhs = {
        "dance1_subject2",
        "dance2_subject4",
        "fight1_subject3",
        "multipleActions1_subject2",
        "obstacles2_subject1",
        "push1_subject2",
        "run1_subject5",
        "walk1_subject5",
    }
    lafan_jobs: list[tuple[str, list[tuple[str, int, int, str, bool]]]] = []
    for bp in sorted(Path("/home/yu/datasets/lafan1").glob("*.bvh")):
        stem = bp.stem
        if stem.startswith(("fallAndGetUp", "pushAndFall", "ground")):
            continue
        is_ho_bvh = (stem in new_ho_lafan_bvhs) or (stem in v2_ho_bvh_stems)
        max_w = 3 if is_ho_bvh else 5
        windows = []
        for wi in range(max_w):
            s0 = wi * 240
            alias = f"lafan_{stem}_w{s0}"
            if alias in v2_lafan_windows or (stem == "walk4_subject1" and s0 == 0):
                continue
            # Put windows from v2_ho_bvh_stems at s0 >= 240 into held_out too or skip so v2_ho BVHs don't leak into train!
            if stem in v2_ho_bvh_stems and not is_ho_bvh:
                continue
            windows.append((alias, s0, s0 + 240, str(SHM_STAGE / f"{alias}.csv"), is_ho_bvh))
        if windows:
            lafan_jobs.append((str(bp), windows))
    print(f"Prepared {len(lafan_jobs)} LAFAN1 multi-window BVH jobs.", flush=True)

    # 5. Execute all Mink-IK retarget jobs in parallel across 14 CPU workers
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=14, max_tasks_per_child=60, initializer=_init_worker
    ) as pool:
        phuma_res = list(pool.map(_process_phuma_job, phuma_jobs))
        soma_res = list(pool.map(_process_soma_job, soma_jobs))
        lafan_res_nested = list(pool.map(_process_lafan_bvh_multi, lafan_jobs))

    lafan_res = [item for sub in lafan_res_nested for item in sub]

    def dedup_accept(res_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted = []
        for r in res_list:
            if not r.get("ok"):
                continue
            p = Path(r["path"])
            sha = _sha256(p)
            if sha in seen_csv_sha:
                continue
            seen_csv_sha.add(sha)
            r["sha256"] = sha
            accepted.append(r)
        return accepted

    phuma_ok = dedup_accept(phuma_res)
    soma_ok = dedup_accept(soma_res)
    lafan_ok = dedup_accept(lafan_res)
    print(
        f"Accepted deduplicated new clips: PHUMA={len(phuma_ok)}, SOMA={len(soma_ok)}, LAFAN1={len(lafan_ok)}",
        flush=True,
    )

    # 6. Synthetic expansion (+46 clips -> 60 total Synthetic)
    syn_ok = build_synthetic_expansion(SHM_STAGE, n_new=46)
    for r in syn_ok:
        r["sha256"] = _sha256(Path(r["path"]))

    # 7. Assemble exact 1,800 Train / 200 Held-Out split (round-robin across all 15 PHUMA subsets)
    rng = np.random.default_rng(20261119)
    by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in phuma_ok:
        by_sub[r.get("subset", "unknown")].append(r)
    phuma_rr: list[dict[str, Any]] = []
    sub_keys = sorted(by_sub.keys())
    while len(phuma_rr) < 1080 and any(by_sub[k] for k in sub_keys):
        for k in sub_keys:
            if by_sub[k] and len(phuma_rr) < 1080:
                phuma_rr.append(by_sub[k].pop(0))
    phuma_ok = phuma_rr
    soma_ok = soma_ok[:410]
    lafan_ho_pool = [r for r in lafan_ok if r.get("held_out")]
    lafan_tr_pool = [r for r in lafan_ok if not r.get("held_out")]

    # New Held-Out target = 150 (90 PHUMA + 34 SOMA + 20 LAFAN1 + 6 Synthetic)
    phuma_idx = np.arange(len(phuma_ok))
    rng.shuffle(phuma_idx)
    phuma_ho = [phuma_ok[i] for i in phuma_idx[:90]]
    phuma_tr = [phuma_ok[i] for i in phuma_idx[90:]]

    soma_idx = np.arange(len(soma_ok))
    rng.shuffle(soma_idx)
    soma_ho = [soma_ok[i] for i in soma_idx[:34]]
    soma_tr = [soma_ok[i] for i in soma_idx[34:]]

    lafan_ho = lafan_ho_pool[:20]
    lafan_tr = (lafan_ho_pool[20:] + lafan_tr_pool)[: max(0, 214 - len(lafan_ho))]

    for r in phuma_ho + soma_ho + lafan_ho:
        r["held_out"] = True
    for r in phuma_tr + soma_tr + lafan_tr:
        r["held_out"] = False

    # Copy 8 new unseen SOMA BVHs + 8 new unseen LAFAN1 BVHs into HELD_OUT_RAW_BVH_24
    for r in soma_ho[:8]:
        bvh_p = Path(r["bvh"])
        if bvh_p.is_file():
            shutil.copy2(bvh_p, HELD_OUT_RAW_BVH_24 / f"{r['alias']}.bvh")
    for stem in sorted(new_ho_lafan_bvhs):
        src_p = Path("/home/yu/datasets/lafan1") / f"{stem}.bvh"
        if src_p.is_file():
            shutil.copy2(src_p, HELD_OUT_RAW_BVH_24 / f"lafan_{stem}.bvh")

    new_clips = phuma_ho + phuma_tr + soma_ho + soma_tr + lafan_ho + lafan_tr + syn_ok
    ho_all = [r for r in seeded_records if r["held_out"]] + [r for r in new_clips if r.get("held_out")]
    tr_all = [r for r in seeded_records if not r["held_out"]] + [r for r in new_clips if not r.get("held_out")]

    while len(ho_all) > 200:
        mv = ho_all.pop()
        if mv.get("v2_clip"):
            ho_all.insert(0, mv)
            break
        mv["held_out"] = False
        tr_all.append(mv)
    while len(ho_all) < 200 and tr_all:
        for i in range(len(tr_all) - 1, -1, -1):
            if not tr_all[i].get("v2_clip"):
                mv = tr_all.pop(i)
                mv["held_out"] = True
                ho_all.append(mv)
                break
    tr_all = tr_all[:1800]

    final_records: list[dict[str, Any]] = []
    for split_name, lst in (("train", tr_all), ("held_out", ho_all)):
        for r in lst:
            dst_csv = OUT_ROOT / split_name / f"{r['alias']}.csv"
            shutil.copy2(Path(r["path"]), dst_csv)
            r_clean = {k: v for k, v in r.items() if k not in ("bvh", "path", "ok")}
            r_clean["split"] = split_name
            r_clean["csv_rel"] = str(dst_csv.relative_to(ROOT))
            final_records.append(r_clean)

    # 8. Final MotionLib 50Hz verification on ALL CSVs
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    kin = Q1Kinematics(str(Q1_XML), "cpu")
    all_csvs = sorted(str(ROOT / r["csv_rel"]) for r in final_records)
    ml = MotionLib(all_csvs, kin, "cpu", src_fps=30.0)
    assert len(all_csvs) == len(final_records)
    assert bool((ml.q >= kin.q_lo - 2e-3).all() and (ml.q <= kin.q_hi + 2e-3).all()), "Joint limit violation!"

    # 9. Pre-cache all 24 raw BVHs -> .50hz.npz so .venv MuJoCo 3.11 reads directly
    from multimodal_features import load_bvh_as_mock_motion

    cached_bvhs = []
    for p in sorted(HELD_OUT_RAW_BVH_24.glob("*.bvh")):
        kind = "lafan1" if p.name.startswith("lafan_") else "soma"
        m_bvh, h_nom, _, arm_l = load_bvh_as_mock_motion(p, kind=kind, save_cache=True)
        cached_bvhs.append({"name": p.name, "kind": kind, "frames_50hz": m_bvh.num_frames, "h_nom": round(h_nom, 3), "arm_len": round(arm_l, 3)})

    by_source: dict[str, dict[str, int]] = {}
    for r in final_records:
        by_source.setdefault(r["source"], {"train": 0, "held_out": 0})
        by_source[r["source"]][r["split"]] += 1

    manifest = {
        "total_clips": len(final_records),
        "train_clips": sum(1 for r in final_records if r["split"] == "train"),
        "held_out_clips": sum(1 for r in final_records if r["split"] == "held_out"),
        "total_50hz_frames": int(ml.num_frames),
        "by_source": by_source,
        "held_out_raw_bvh_24": cached_bvhs,
        "wall_s": round(time.monotonic() - t_start, 2),
        "clips": final_records,
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "clips"}, indent=2), flush=True)

    shutil.rmtree(SHM_STAGE, ignore_errors=True)
    shutil.rmtree(SHM_PHUMA, ignore_errors=True)
    shutil.rmtree(SHM_SOMA, ignore_errors=True)
    print("DATASET_2000_READY", flush=True)


if __name__ == "__main__":
    main()
