#!/usr/bin/env python3
"""smplx2smpl24.py -- AMASS (SMPL-X topology) npz -> SMPL-24 (p16_pool motionlib schema) bridge.

Topology mapping (AMASS SMPL-X 'poses' layout, (T,165) = 55 joints x 3):
    [0:66]   22 body joints incl. root/global orient  -> SAME joint order as SMPL-24[:22]
    [66:69]  jaw, [69:75] eyes, [75:165] finger joints -> DROPPED (fingers assumed zero)
SMPL-24 joints 22/23 (L/R_Hand) have no SMPL-X counterpart:
    --hand zero  (default): zero rotation; hand orientation is inherited from the wrist
                           rotation via the kinematic chain ("腕旋转近似" in the chained sense,
                           avoids double-applying the wrist rotation)
    --hand wrist: copy wrist rotvec onto hand joints (explicit variant, kept for compat)
Resampling: per-joint SLERP + linear translation to --fps (default 30).
Verification FK: SMPL_male.pkl, pure numpy LBS (chumpy-shimmed pickle load, scipy-sparse aware).
Convention NOTE: fk_joints(betas=0) matches FGP SMPLight (used by p16_pool) to 0.000 mm
root-relative -- pool 'smpl_joints' therefore use betas=0 template FK by default.

Output pkl per motion (joblib, compress=3), p16_pool/m2-schema isomorphic:
    {name: {"pose_aa": (T,32,3) f32   [24 body + 8 zero hand joints; flatten [:, :24] -> (T,72)],
            "smpl_joints": (T,24,3) f32  root-relative FK,
            "root_rot": (T,4) f32      root rotation quat, wxyz,
            "root_trans_offset": (T,3) f32  AMASS transl resampled,
            "dof": (T,29) f32 zeros,
            "fps": 30,
            ...extras: pose_aa_72 (T,72), betas10, src_fps, gender, source}}

Usage:
  smplx2smpl24.py selftest [--smpl-model P] [--report R.json]
  smplx2smpl24.py convert --input FILE_OR_DIR --out-dir D [--fps 30] [--hand zero|wrist]
                          [--jobs N] [--verify N] [--smpl-model P] [--report R.json]
  smplx2smpl24.py verify --input X.npz [--smpl-model P]   (sanity metrics, no output write)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import types

import numpy as np

DEFAULT_SMPL = "/opt/data/private/yuanqi_ops/script/tools/FGP/models/SMPL_male.pkl"
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINT_NAMES = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle",
               "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head",
               "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
               "l_hand", "r_hand"]

# --------------------------------------------------------------------------- SMPL model


def _install_chumpy_shim():
    """Minimal chumpy stand-in so SMPL pkls unpickle without chumpy installed."""
    ch = types.ModuleType("chumpy")

    class Ch(object):
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                for k, v in state.items():
                    try:
                        if isinstance(v, (str, bytes, int, float)) or v is None:
                            self.__dict__[k] = v
                        else:
                            self.__dict__[k] = np.asarray(v)
                    except Exception:
                        pass

        def __array__(self, dtype=None):
            return np.asarray(self.__dict__.get("x"), dtype=dtype)

    ch.Ch = Ch
    chch = types.ModuleType("chumpy.ch")
    chch.Ch = Ch
    ch.ch = chch
    sys.modules["chumpy"] = ch
    sys.modules["chumpy.ch"] = chch
    reo = types.ModuleType("chumpy.reordering")

    class Reorder(object):
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

        def __array__(self, dtype=None):
            for key in ("arg", "a", "x"):
                v = self.__dict__.get(key)
                if v is not None:
                    return np.asarray(v, dtype=dtype)
            return np.zeros(0)

    for nm in ("MatrixViewer", "SimpleReordering", "MKLMatrixViewer", "Reordering"):
        reo.__dict__[nm] = Reorder
    ch.reordering = reo
    sys.modules["chumpy.reordering"] = reo


def load_smpl(path=DEFAULT_SMPL):
    import scipy.sparse as sp

    try:
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
    except Exception:
        _install_chumpy_shim()
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")

    def toarr(x):
        if sp.issparse(x):
            return np.asarray(x.todense())
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (int, float, str, bytes)) or x is None:
            return x
        for k in ("x", "a", "arg"):
            v = getattr(x, k, None)
            if isinstance(v, np.ndarray):
                return v
        try:
            a = np.asarray(x)
            if a.size and a.dtype != object:
                return a
        except Exception:
            pass
        return None

    arr = {k: toarr(v) for k, v in d.items()}
    m = {
        "v_template": np.asarray(arr["v_template"], float),
        "shapedirs": np.asarray(arr["shapedirs"], float),
        "posedirs": np.asarray(arr["posedirs"], float),
        "J_regressor": np.asarray(arr["J_regressor"], float),
        "weights": np.asarray(arr["weights"], float),
    }
    kt = arr["kintree_table"].astype(np.int64)
    parents = kt[0].copy()
    parents[0] = -1
    if parents.tolist() != SMPL_PARENTS:
        raise ValueError("kintree_table mismatch vs SMPL-24 expectation: %s" % parents.tolist())
    m["parents"] = parents
    m["template_height_m"] = float(m["v_template"][:, 1].max() - m["v_template"][:, 1].min())
    return m


_MODEL_CACHE = {}


def get_model(path=DEFAULT_SMPL):
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = load_smpl(path)
    return _MODEL_CACHE[path]

# --------------------------------------------------------------------------- rotations / FK


def rotvec_to_mat(r):
    """(..., 3) axis-angle -> (..., 3, 3), vectorized Rodrigues."""
    r = np.asarray(r, float)
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    un = np.where(th > 1e-8, r / np.maximum(th, 1e-8), np.array([0.0, 0.0, 1.0]))
    x, y, z = un[..., 0], un[..., 1], un[..., 2]
    zr = np.zeros_like(x)
    K = np.stack([zr, -z, y, z, zr, -x, -y, x, zr], axis=-1).reshape(*r.shape[:-1], 3, 3)
    I = np.tile(np.eye(3), (*r.shape[:-1], 1, 1))
    return I + np.sin(th)[..., None] * K + (1.0 - np.cos(th))[..., None] * (K @ K)


def fk_joints(model, pose_aa, betas=None, want_G=False):
    """pose_aa (T,72)|(24,3)rotvec-flat -> joints (T,24,3) [absolute, pelvis at rest J0],
    optional homogeneous skinning transforms G (T,24,4,4) (rest pose == identity)."""
    p = np.asarray(pose_aa, float).reshape(-1, 72)
    T = p.shape[0]
    betas = np.zeros(10) if betas is None else np.asarray(betas, float).ravel()[:10]
    v_sh = model["v_template"] + np.einsum("vdk,k->vd", model["shapedirs"], betas)
    J = model["J_regressor"] @ v_sh  # (24,3)
    R = rotvec_to_mat(p.reshape(T, 24, 3))  # (T,24,3,3)
    A_t = J[None] - np.einsum("tjmn,jn->tjm", R, J)  # J_j - R_j @ J_j  (pivot form)
    G = np.zeros((T, 24, 4, 4))
    G[:, :, 3, 3] = 1.0
    G[:, :, :3, :3] = R
    G[:, :, :3, 3] = A_t
    parents = model["parents"]
    for j in range(1, 24):
        G[:, j] = np.einsum("tij,tjk->tik", G[:, parents[j]], G[:, j])
    joints = np.einsum("tjmn,jn->tjm", G[:, :, :3, :3], J) + G[:, :, :3, 3]
    return (joints, G, J) if want_G else (joints, J)


def lbs_vertices_sampled(model, pose_aa, betas, G, frame_idx):
    """Full LBS vertices for a few sampled frames (height check)."""
    p = np.asarray(pose_aa, float).reshape(-1, 72)[frame_idx]
    R = rotvec_to_mat(p.reshape(-1, 24, 3))
    feat = (R[:, 1:] - np.eye(3)).reshape(len(p), -1)  # (t,207)
    v_sh = model["v_template"] + np.einsum("vdk,k->vd", model["shapedirs"], betas)
    v_posed = v_sh + (model["posedirs"].reshape(-1, 207) @ feat.T).T.reshape(len(p), -1, 3)
    W = model["weights"]  # (V,24)
    out = np.zeros_like(v_posed)
    Gs = G[frame_idx]
    for j in range(24):
        Rg = Gs[:, j, :3, :3]
        tg = Gs[:, j, :3, 3]
        out += W[:, j, None] * (np.einsum("tmn,tvn->tvm", Rg, v_posed) + tg[:, None, :])
    return out

# --------------------------------------------------------------------------- resampling


def resample_motion(poses, transl, src_fps, dst_fps):
    """poses (T,J*3) SLERP + transl (T,3) linear interp from src_fps to dst_fps."""
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial.transform import Slerp

    poses = np.asarray(poses, float)
    transl = np.asarray(transl, float)
    T = poses.shape[0]
    if T < 2 or abs(src_fps - dst_fps) < 1e-9:
        return poses.copy(), transl.copy()
    t_src = np.arange(T) / float(src_fps)
    keep = np.concatenate([[True], np.diff(t_src) > 1e-12])
    t_src = t_src[keep]
    poses, transl = poses[keep], transl[keep]
    duration = t_src[-1]
    n_out = max(2, int(round(duration * dst_fps)) + 1)
    t_out = np.clip(np.arange(n_out) / float(dst_fps), t_src[0], t_src[-1])
    J = poses.shape[1] // 3
    quats = Rot.from_rotvec(poses.reshape(-1, 3)).as_quat().reshape(-1, J, 4)
    out_q = np.empty((n_out, J, 4))
    for j in range(J):
        slerp = Slerp(t_src, Rot.from_quat(quats[:, j, :]))
        out_q[:, j, :] = slerp(t_out).as_quat()
    out_pose = Rot.from_quat(out_q.reshape(-1, 4)).as_rotvec().reshape(n_out, J * 3)
    out_tr = np.stack([np.interp(t_out, t_src, transl[:, k]) for k in range(3)], axis=1)
    return out_pose, out_tr

# --------------------------------------------------------------------------- conversion


def _scalar(x):
    try:
        return x.item() if hasattr(x, "item") else x
    except Exception:
        return x


def convert_npz(path, fps=30.0, hand="zero", model=None):
    """AMASS npz -> (payload dict, info dict). Raises on malformed input."""
    d = np.load(path, allow_pickle=True)
    poses = np.asarray(d["poses"], float)
    if poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError("poses %s unsupported (need >=66 cols)" % (poses.shape,))
    if "transl" in d.files:
        transl = np.asarray(d["transl"], float)
    elif "trans" in d.files:
        transl = np.asarray(d["trans"], float)
    else:
        raise ValueError("no transl/trans key")
    if transl.ndim != 2 or transl.shape[0] != poses.shape[0] or transl.shape[1] != 3:
        raise ValueError("transl %s mismatched vs poses %s" % (transl.shape, poses.shape))
    betas = np.asarray(d["betas"], float).ravel() if "betas" in d.files else np.zeros(10)
    src_fps = float(_scalar(d["mocap_framerate"])) if "mocap_framerate" in d.files else 30.0
    gender = ""
    if "gender" in d.files:
        try:
            gender = str(_scalar(d["gender"]))
        except Exception:
            gender = ""
    body = poses[:, :66]
    body_r, trans_r = resample_motion(body, transl, src_fps, fps)
    T = body_r.shape[0]
    pose24 = np.zeros((T, 72), np.float32)
    pose24[:, :66] = body_r
    if hand == "wrist":  # copy wrist (joints 20/21) onto hand (22/23)
        pose24[:, 66:69] = body_r[:, 57:60]
        pose24[:, 69:72] = body_r[:, 60:63]
    # p16_pool / SMPLight convention: template (betas=0) FK, root-relative
    model = get_model() if model is None else model
    joints, _ = fk_joints(model, pose24, betas=None)
    joints_rel = (joints - joints[:, 0:1]).astype(np.float32)
    from scipy.spatial.transform import Rotation as Rot
    root_quat = Rot.from_rotvec(pose24[:, 0:3].astype(float)).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    pose32 = np.concatenate([pose24.reshape(T, 24, 3), np.zeros((T, 8, 3), np.float32)], axis=1)
    name = os.path.splitext(os.path.basename(path))[0]
    payload = {name: {
        "pose_aa": pose32.astype(np.float32),
        "smpl_joints": joints_rel,
        "root_rot": root_quat,
        "root_trans_offset": trans_r.astype(np.float32),
        "dof": np.zeros((T, 29), np.float32),
        "fps": int(round(fps)),
        "pose_aa_72": pose24,
        "betas10": betas[:10].astype(np.float32),
        "src_fps": float(src_fps),
        "gender": gender,
        "source": os.path.abspath(path),
    }}
    info = {"name": name, "frames_in": int(poses.shape[0]), "frames_out": int(T),
            "src_fps": src_fps, "gender": gender}
    return payload, info

# --------------------------------------------------------------------------- verification


def _angle_at(joints, a, b, c):
    v1 = joints[:, a] - joints[:, b]
    v2 = joints[:, c] - joints[:, b]
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(np.einsum("tn,tn->t", v1, v2) / np.maximum(n1 * n2, 1e-9), -1, 1)
    return np.arccos(cos)


def verify_pose(model, pose_aa72, transl, betas10=None, mesh_frames=8):
    """Coarse plausibility metrics for converted pose (absolute frame incl. transl)."""
    p = np.asarray(pose_aa72, float).reshape(-1, 72)
    T = p.shape[0]
    joints, G = fk_joints(model, p, betas=None, want_G=True)[:2]
    abs_j = joints + np.asarray(transl, float).reshape(-1, 3)[:, None, :]
    per_frame_h = abs_j[:, :, 1].max(1) - abs_j[:, :, 1].min(1)
    height_joints = float(np.percentile(per_frame_h, 95)) + 0.12  # head-top allowance
    idx = np.unique(np.linspace(0, T - 1, min(mesh_frames, T)).astype(int))
    verts = lbs_vertices_sampled(model, p, betas10 if betas10 is not None else np.zeros(10), G, idx)
    mesh_h = float(np.percentile(verts[:, :, 1].max(1) - verts[:, :, 1].min(1), 95))
    knee = np.concatenate([_angle_at(abs_j, 1, 4, 7), _angle_at(abs_j, 2, 5, 8)])
    elbow = np.concatenate([_angle_at(abs_j, 16, 18, 20), _angle_at(abs_j, 17, 19, 21)])
    rot_mag = np.linalg.norm(p.reshape(T, 24, 3), axis=-1)
    m = {
        "frames": T,
        "nan": bool(np.isnan(p).any() or np.isnan(np.asarray(transl)).any()),
        "height_joints_m": round(height_joints, 3),
        "height_mesh_m": round(mesh_h, 3),
        "height_in_1p5_1p9": bool(1.5 <= mesh_h <= 1.9),
        "knee_angle_rad": [round(float(knee.min()), 3), round(float(knee.max()), 3)],
        "elbow_angle_rad": [round(float(elbow.min()), 3), round(float(elbow.max()), 3)],
        "knee_fold_suspicious": int(np.count_nonzero(knee < 0.15)),
        "joint_rot_gt_3p2rad": int(np.count_nonzero(rot_mag > 3.2)),
        "root_y_min_max_m": [round(float(abs_j[:, 0, 1].min()), 3), round(float(abs_j[:, 0, 1].max()), 3)],
        "template_height_m": round(model["template_height_m"], 3),
    }
    m["warnings"] = []
    if m["nan"]:
        m["warnings"].append("NaN")
    if not (1.45 <= mesh_h <= 1.95):
        m["warnings"].append("mesh height %.2f outside 1.45-1.95" % mesh_h)
    if m["knee_fold_suspicious"] > 0.05 * T:
        m["warnings"].append("knee folded <0.15rad in %d frames" % m["knee_fold_suspicious"])
    if m["joint_rot_gt_3p2rad"] > 0.02 * T:
        m["warnings"].append("joint rotvec >3.2rad in %d frames" % m["joint_rot_gt_3p2rad"])
    return m

# --------------------------------------------------------------------------- selftest


def selftest(args):
    from scipy.spatial.transform import Rotation as Rot

    model = get_model(args.smpl_model)
    rng = np.random.default_rng(7)
    res = {"model": args.smpl_model, "template_height_m": round(model["template_height_m"], 4)}
    ok = True

    def check(name, val, thr, pass_cond=None, unit=""):
        nonlocal ok
        good = (val <= thr) if pass_cond is None else pass_cond
        ok = ok and bool(good)
        res[name] = {"value": val, "threshold": thr, "unit": unit, "pass": bool(good)}

    # --- FK self-consistency
    ji, J0 = fk_joints(model, np.zeros(72))
    check("fk_identity_vs_regressor_mm", float(np.abs(ji - J0).max() * 1000), 1e-6, unit="mm")
    jz, _ = fk_joints(model, np.pad([[0.0, 0.0, np.pi / 2]], ((0, 0), (0, 69))))
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    expect = (J0 - J0[0]) @ Rz.T + J0[0]
    check("fk_root_yaw90_mm", float(np.abs(jz - expect).max() * 1000), 1e-6, unit="mm")

    # --- Test A: exact 2:1 downsample, known-answer FK (60fps -> 30fps)
    T = 121
    t60 = np.arange(T) / 60.0
    posesA = np.zeros((T, 165))
    posesA[:, 0:3] = np.stack([np.zeros(T), np.zeros(T), np.deg2rad(90) * (t60 / t60[-1])], 1)
    posesA[:, 48:51] = np.stack([np.zeros(T), np.zeros(T), 1.2 * (t60 / t60[-1])], 1)
    posesA[:, 60:63] = np.stack([np.zeros(T), 0.3 * np.sin(2 * np.pi * t60), np.zeros(T)], 1)
    translA = np.stack([0.5 * t60, np.zeros(T) + 0.02 * np.sin(2 * np.pi * t60), 0.1 * t60], 1)
    betasA = (rng.normal(0, 0.5, 10)).astype(np.float32)
    tmp = "/tmp/amass_bridge_selftest_A.npz"
    np.savez(tmp, poses=posesA, transl=translA, betas=betasA, mocap_framerate=60)
    payload, info = convert_npz(tmp, fps=30, hand="zero", model=model)
    out = payload[info["name"]]
    p30 = out["pose_aa_72"].astype(float)
    check("A_frames_out", float(p30.shape[0] == 61), 1.0, pass_cond=bool(p30.shape[0] == 61))
    gt72 = np.zeros((61, 72))
    gt72[:, 0:66] = posesA[::2, :66]
    joints_out, _ = fk_joints(model, p30, betas=betasA)
    joints_gt, _ = fk_joints(model, gt72, betas=betasA)
    check("A_fk_joint_err_mm", float(np.abs(joints_out - joints_gt).max() * 1000), 0.5, unit="mm")
    check("A_trans_err_mm", float(np.abs(out["root_trans_offset"] - translA[::2]).max() * 1000), 0.5, unit="mm")
    check("A_hands_zero", float(np.abs(p30[:, 66:72]).max()), 0.0, unit="rad")
    vm = verify_pose(model, p30, out["root_trans_offset"], betas10=betasA)
    res["A_verify"] = vm  # arms-up pose: bbox height ~2.0m by construction (info only)

    # --- Test B: constant-angular-velocity single-axis rotations, analytic ground truth (59fps -> 30fps)
    T = 97
    src = 59.0
    tB = np.arange(T) / src
    axes = rng.normal(size=(22, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    omegas = rng.uniform(0.2, 1.2, 22)
    posesB = np.zeros((T, 165))
    for j in range(22):
        posesB[:, 3 * j:3 * j + 3] = axes[j][None, :] * (omegas[j] * tB[:, None])
    translB = np.stack([0.5 * tB, 1.0 + 0.05 * np.sin(2 * np.pi * tB), np.zeros(T)], 1)
    tmpB = "/tmp/amass_bridge_selftest_B.npz"
    np.savez(tmpB, poses=posesB, transl=translB, betas=np.zeros(10), mocap_framerate=59)
    payloadB, infoB = convert_npz(tmpB, fps=30, hand="zero", model=model)
    outB = payloadB[infoB["name"]]
    p30B = outB["pose_aa_72"].astype(float)
    nB = p30B.shape[0]
    t30 = np.clip(np.arange(nB) / 30.0, 0.0, tB[-1])  # converter clips to source duration
    gt_rotvec = np.stack([axes[j][None, :] * (omegas[j] * t30[:, None]) for j in range(22)], 1)
    err_rot = np.abs((p30B[:, :66].reshape(nB, 22, 3) + np.pi) % (2 * np.pi) - np.pi - gt_rotvec)
    ang_err_deg = np.degrees(np.linalg.norm(err_rot, axis=-1)).max()
    check("B_analytic_rotvec_err_deg", float(ang_err_deg), 0.05, unit="deg")
    gt_tr = np.stack([0.5 * t30, 1.0 + 0.05 * np.sin(2 * np.pi * t30), np.zeros(nB)], 1)
    check("B_trans_err_mm", float(np.abs(outB["root_trans_offset"] - gt_tr).max() * 1000), 2.0, unit="mm")

    # --- passthrough 30fps + arms-down height sanity
    posesC = posesA[::2].copy()
    posesC[:, 48:66] = 0.0  # arms down: standing height must be in 1.5-1.9 m
    np.savez("/tmp/amass_bridge_selftest_C.npz", poses=posesC, transl=translA[::2],
             betas=betasA, mocap_framerate=30)
    payloadC, infoC = convert_npz("/tmp/amass_bridge_selftest_C.npz", fps=30, hand="zero", model=model)
    outC = payloadC[infoC["name"]]
    check("C_passthrough_exact", float(np.abs(outC["pose_aa_72"][:, :66] - posesC[:, :66]).max()), 1e-5, unit="rad")
    vmC = verify_pose(model, outC["pose_aa_72"], outC["root_trans_offset"], betas10=betasA)
    res["C_verify"] = vmC
    check("C_height_mesh_1p5_1p9", vmC["height_mesh_m"], 1.0, pass_cond=vmC["height_in_1p5_1p9"])

    # --- hand=wrist variant
    payloadW, infoW = convert_npz(tmp, fps=30, hand="wrist", model=model)
    outW = payloadW[infoW["name"]]["pose_aa_72"]
    check("W_hand_wrist_copy", float(np.abs(outW[:, 66:69] - outW[:, 57:60]).max()
                                     + np.abs(outW[:, 69:72] - outW[:, 60:63]).max()), 1e-6, unit="rad")

    res["PASS"] = bool(ok)
    rep = args.report or os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_test_report.json")
    json.dump(res, open(rep, "w"), indent=1)
    print("== smplx2smpl24 selftest ==")
    for k, v in res.items():
        if isinstance(v, dict) and "value" in v:
            print("  %-32s %12.6f %-3s (thr %s) %s" % (k, v["value"], v.get("unit", ""), v["threshold"], "PASS" if v["pass"] else "FAIL"))
    print("A_verify:", json.dumps(res.get("A_verify", {}), ensure_ascii=False))
    print("OVERALL:", "PASS" if ok else "FAIL", "->", rep)
    return 0 if ok else 1

# --------------------------------------------------------------------------- batch convert


def _out_name(npz_path, root=None):
    rel = os.path.relpath(npz_path, root) if root else os.path.basename(npz_path)
    name = os.path.splitext(rel)[0].replace("/", "__")
    return name


def convert_batch(args):
    import joblib

    model = get_model(args.smpl_model)
    if os.path.isfile(args.input):
        files, root = [args.input], None
    else:
        files, root = [], args.input
        for dp, _, fns in os.walk(args.input):
            files += [os.path.join(dp, f) for f in fns if f.endswith(".npz")]
        files.sort()
    if not files:
        print("no npz under", args.input)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    def one(f):
        name = _out_name(f, root)
        dest = os.path.join(args.out_dir, name + ".pkl")
        row = {"name": name, "src": f, "pkl": dest}
        try:
            if os.path.exists(dest):
                row["exists"] = True
                d = joblib.load(dest)[name]
                row["frames"] = int(d["pose_aa"].shape[0])
                return row
            payload, info = convert_npz(f, fps=args.fps, hand=args.hand, model=model)
            payload = {name: payload[info["name"]]}  # key == pkl filename stem (p16 convention)
            joblib.dump(payload, dest, compress=3)
            row.update({"frames": info["frames_out"], "frames_in": info["frames_in"],
                        "src_fps": info["src_fps"], "gender": info["gender"]})
            return row
        except Exception as e:
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
            return row

    rows = joblib.Parallel(n_jobs=args.jobs, verbose=5)(joblib.delayed(one)(f) for f in files)
    bad = [r for r in rows if r.get("error")]
    # verification sample (first N successes incl. different datasets)
    verify_rows = []
    for r in rows:
        if len(verify_rows) >= args.verify:
            break
        if r.get("error") or r.get("exists"):
            continue
        try:
            import joblib as jb
            loaded = jb.load(r["pkl"])
            p = loaded[r["name"]] if r["name"] in loaded else loaded[list(loaded)[0]]
            vr = verify_pose(model, p["pose_aa_72"], p["root_trans_offset"], betas10=p["betas10"])
            vr.update({"name": r["name"], "frames": vr["frames"]})
            verify_rows.append(vr)
        except Exception as e:
            verify_rows.append({"name": r["name"], "error": str(e)[:120]})
    summary = {
        "input": args.input, "out_dir": args.out_dir, "fps": args.fps, "hand": args.hand,
        "n": len(rows), "ok": len(rows) - len(bad), "bad": len(bad),
        "total_frames_out": int(sum(r.get("frames", 0) for r in rows)),
        "seconds": round(time.time() - t0, 1),
        "verify_sample": verify_rows,
        "bad_rows": bad[:50],
    }
    rep = args.report or os.path.join(args.out_dir, "convert_report.json")
    json.dump(summary, open(rep, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("n", "ok", "bad", "total_frames_out", "seconds")}))
    for v in verify_rows:
        print("VERIFY", json.dumps(v, ensure_ascii=False))
    return 0 if not bad else 2


def verify_cli(args):
    model = get_model(args.smpl_model)
    d = np.load(args.input, allow_pickle=True)
    poses = np.asarray(d["poses"], float)
    tr = np.asarray(d["transl" if "transl" in d.files else "trans"], float)
    betas = np.asarray(d["betas"], float).ravel()[:10] if "betas" in d.files else np.zeros(10)
    body, tr_r = resample_motion(poses[:, :66], tr, float(_scalar(d["mocap_framerate"])), 30.0)
    p72 = np.zeros((body.shape[0], 72))
    p72[:, :66] = body
    m = verify_pose(model, p72, tr_r, betas10=betas)
    print(json.dumps(m, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("selftest")
    sp.add_argument("--smpl-model", default=DEFAULT_SMPL)
    sp.add_argument("--report", default=None)
    sp.set_defaults(fn=selftest)
    cp = sub.add_parser("convert")
    cp.add_argument("--input", required=True)
    cp.add_argument("--out-dir", required=True)
    cp.add_argument("--fps", type=float, default=30.0)
    cp.add_argument("--hand", choices=["zero", "wrist"], default="zero")
    cp.add_argument("--jobs", type=int, default=8)
    cp.add_argument("--verify", type=int, default=3)
    cp.add_argument("--smpl-model", default=DEFAULT_SMPL)
    cp.add_argument("--report", default=None)
    cp.set_defaults(fn=convert_batch)
    vp = sub.add_parser("verify")
    vp.add_argument("--input", required=True)
    vp.add_argument("--smpl-model", default=DEFAULT_SMPL)
    vp.set_defaults(fn=verify_cli)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
