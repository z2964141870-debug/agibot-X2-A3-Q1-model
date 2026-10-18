#!/usr/bin/env python3
"""HP25 Exp-2a: root-displacement identifiability upper bound (CPU-only).

Contract: analysis/EXP_CONTRACTS.md section 2 (Exp-2a); criteria pre-registered
in section 2.4 and in this experiment's registration.json. Reads saved HP11
baseline trajectories only; builds leak-free features from allowed inputs only
(arrived reference joints, robot joint history, last action, relative anchor
orientation, gravity direction). Predicts future root displacement (true and
reference) for H in {0.1, 0.5, 2.0, 4.9} s with ridge regression (alpha=1,
train-only normalization) versus hold / constant-velocity baselines, plus one
positive control that appends the current true root displacement-so-far.

Single run: no hyperparameter tuning on validation; no re-run after COMPLETE.
Zero GPU, zero simulation. Inputs read-only; writes only its own output dir.

v2 changes vs HP24 (single registered bug-fix re-registration, see HP25
registration predecessor block):
- labels, const-vel baseline and positive-control root feature are expressed in
  the sim anchor (root heading) frame at time t (removes cross-subject world
  heading bias; the transform is a labeling convention only, never a model
  input);
- standardized features are clipped to [-5, 5] with train-only statistics
  (bounds the unbounded linear extrapolation observed on fall segments in
  HP24: zzx fall at tick 283 produced out-of-distribution features and
  meter-scale predictions).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time

import numpy as np

POLICY_DT = 0.02
N_JOINTS = 29
VEL_WINDOW_TICKS = 5            # 100 ms velocity proxy
REF_LAGS_TICKS = (2, 3, 5, 10)  # ~30 Hz frame lags 1, 2, 3, 6
HORIZONS_S = (0.1, 0.5, 2.0, 4.9)
RIDGE_ALPHA = 1.0
Z_CLIP = 5.0
TRAIN_KEYS = ("zj_walk", "zj_warmUp", "fjw_walk")
VAL_KEYS = ("zls", "yy", "zzx")
BASE_DIM = 4 * N_JOINTS + N_JOINTS + N_JOINTS + 3 + 3  # 180
SEGMENTS = {
    "zj_walk": "G_0113_zj_train__183926_zj_walk",
    "zj_warmUp": "G_0113_zj_train__192333_zj_warmUp",
    "fjw_walk": "G_20251226_fjw__fjw_140008_walk",
    "zls": "G_20251226_zls__zls_155013_walk",
    "yy": "G_20251229_yy__yy_125106_walk",
    "zzx": "G_20251229_zzx__zzx_111505_walk",
}
SEGMENT_GROUP = {
    "zj_walk": "cloth", "zj_warmUp": "cloth", "fjw_walk": "cloth",
    "zls": "cloth", "yy": "cloth", "zzx": "cloth",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if angle > np.pi - 1e-6:
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        axis = A[:, k] / axis[k]
        return axis * angle
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (angle / (2.0 * np.sin(angle)))


def relative_orientation_rotvec(q_ref, q_sim) -> np.ndarray:
    R = quat_to_rot(q_ref).T @ quat_to_rot(q_sim)
    return rot_to_rotvec(R)


def build_feature_block(ref_q, q_state, action, q_ref_anchor, q_sim_anchor, t, positive_control_root=None):
    """Leak-free by construction: root arrays are never passed in.

    Layout (180 dims base):
      4x29 arrived-reference joint relative diffs, tick lags (2,3,5,10)
      1x29 robot joint velocity proxy q[t] - q[t-5]
      1x29 last action a[t-1]
      3    gravity direction in sim anchor frame
      3    relative anchor orientation rotvec (ref^T sim)
    positive_control_root: optional 3 dims of current TRUE root
    displacement-so-far in anchor frame (positive control only).
    """
    blocks = []
    for lag in REF_LAGS_TICKS:
        blocks.append(ref_q[t - lag] - ref_q[t])
    blocks.append(q_state[t] - q_state[t - VEL_WINDOW_TICKS])
    blocks.append(action[t - 1])
    R_sim = quat_to_rot(q_sim_anchor[t])
    blocks.append(R_sim.T @ np.array([0.0, 0.0, -1.0]))
    blocks.append(relative_orientation_rotvec(q_ref_anchor[t], q_sim_anchor[t]))
    if positive_control_root is not None:
        blocks.append(np.asarray(positive_control_root, dtype=np.float64))
    return np.concatenate(blocks)


def load_segment(data_root: str, key: str) -> dict:
    stem = SEGMENTS[key]
    grp = SEGMENT_GROUP[key]
    d = os.path.join(data_root, grp)
    npz = np.load(os.path.join(d, stem + ".inputs.npz"))
    obs = npz["observations"]
    act = npz["actions"]
    assert obs.shape == (499, 1570), (key, "observations", obs.shape)
    assert act.shape == (499, N_JOINTS), (key, "actions", act.shape)
    with open(os.path.join(d, stem + ".timeseries.json")) as f:
        ts = json.load(f)
    assert abs(float(ts["fps"]) - 50.0) < 1e-9, (key, ts["fps"])
    m = ts["motions"][0]
    arrays = {
        "reference_q_29": np.asarray(m["reference_q_29"], dtype=np.float64),
        "q_state_29": np.asarray(m["q_state_29"], dtype=np.float64),
        "raw_action_29": np.asarray(m["raw_action_29"], dtype=np.float64),
        "ref_anchor_quat_wxyz": np.asarray(m["ref_anchor_quat_wxyz"], dtype=np.float64),
        "sim_anchor_quat_wxyz": np.asarray(m["sim_anchor_quat_wxyz"], dtype=np.float64),
        "sim_root_pos_w": np.asarray(m["sim_root_pos_w"], dtype=np.float64),
        "ref_root_pos_w": np.asarray(m["ref_root_pos_w"], dtype=np.float64),
        "time_s": np.asarray(m["time_s"], dtype=np.float64),
    }
    n = 499
    for name, a in arrays.items():
        assert a.shape[0] == n, (key, name, a.shape)
    assert abs(arrays["time_s"][1] - arrays["time_s"][0] - POLICY_DT) < 1e-9
    return {"key": key, "obs": obs, "actions": act, "arrays": arrays}


def max_lag() -> int:
    return max(max(REF_LAGS_TICKS), VEL_WINDOW_TICKS, 1)


def sample_dataset(segs: dict, h_ticks: int, h_s: float, positive_control: bool):
    X, y_true, y_ref, cv_true, cv_ref, keys = [], [], [], [], [], []
    for key, seg in segs.items():
        a = seg["arrays"]
        ref_q, q_state, action = a["reference_q_29"], a["q_state_29"], a["raw_action_29"]
        q_ref_a, q_sim_a = a["ref_anchor_quat_wxyz"], a["sim_anchor_quat_wxyz"]
        sim_root, ref_root = a["sim_root_pos_w"], a["ref_root_pos_w"]
        for t in range(max_lag(), len(q_state) - h_ticks):
            R_t = quat_to_rot(q_sim_a[t])
            pc = R_t.T @ (sim_root[t] - sim_root[0]) if positive_control else None
            X.append(build_feature_block(ref_q, q_state, action, q_ref_a, q_sim_a, t, pc))
            y_true.append(R_t.T @ (sim_root[t + h_ticks] - sim_root[t]))
            y_ref.append(R_t.T @ (ref_root[t + h_ticks] - ref_root[t]))
            v_true = R_t.T @ (sim_root[t] - sim_root[t - VEL_WINDOW_TICKS]) / (VEL_WINDOW_TICKS * POLICY_DT)
            v_ref = R_t.T @ (ref_root[t] - ref_root[t - VEL_WINDOW_TICKS]) / (VEL_WINDOW_TICKS * POLICY_DT)
            cv_true.append(v_true * h_s)
            cv_ref.append(v_ref * h_s)
            keys.append(key)
    return (np.asarray(X), np.asarray(y_true), np.asarray(y_ref),
            np.asarray(cv_true), np.asarray(cv_ref), keys)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = np.clip((X - mu) / sd, -Z_CLIP, Z_CLIP)
    y_mean = y.mean(axis=0)
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - y_mean))
    return {"mu": mu, "sd": sd, "w": w, "y_mean": y_mean}


def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = np.clip((X - model["mu"]) / model["sd"], -Z_CLIP, Z_CLIP)
    return Xs @ model["w"] + model["y_mean"]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def masked_rmse(pred, y, keys, subset):
    idx = [i for i, k in enumerate(keys) if k in subset]
    return rmse(pred[idx], y[idx]), len(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    reg = json.load(open(args.registration))

    # ---- step 0: input SHA + field verification ----
    verification = []
    ok = True
    for entry in reg["inputs"]:
        p = entry["path"]
        if not os.path.exists(p):
            verification.append({"path": p, "status": "MISSING"})
            ok = False
            continue
        got = sha256_file(p)
        status = "OK" if got == entry["sha256"] else "SHA_MISMATCH"
        if status != "OK":
            ok = False
        verification.append({"path": p, "status": status, "sha256": got})
    print("input verification:", "PASS" if ok else "FAIL", "entries:", len(verification))
    if not ok:
        json.dump({"status": "FAILED_INPUT_VERIFICATION", "verification": verification},
                  open(os.path.join(args.out_dir, "result.json"), "w"), indent=2)
        return 2

    segs = {k: load_segment(args.data_root, k) for k in SEGMENTS}

    results = {"label_sets": {}}
    csv_rows = []
    verdict_inputs = {}
    pc_results = {}

    for pc in (False, True):
        label = "positive_control" if pc else "main"
        for y_name, cv_name in (("true", "cv_true"), ("ref", "cv_ref")):
            table = {}
            for h_s in HORIZONS_S:
                h_ticks = int(round(h_s / POLICY_DT))
                X, yt, yr, cvt, cvr, keys = sample_dataset(segs, h_ticks, h_s, pc)
                y = yt if y_name == "true" else yr
                cv = cvt if y_name == "true" else cvr
                tr_idx = [i for i, k in enumerate(keys) if k in TRAIN_KEYS]
                va_idx = [i for i, k in enumerate(keys) if k in VAL_KEYS]
                model = ridge_fit(X[tr_idx], y[tr_idx], RIDGE_ALPHA)
                pred = ridge_predict(model, X)
                hold = np.zeros_like(y)
                entry = {
                    "n_train": len(tr_idx), "n_val": len(va_idx),
                    "rmse_hold": rmse(hold[va_idx], y[va_idx]),
                    "rmse_cv": rmse(cv[va_idx], y[va_idx]),
                    "rmse_ridge": rmse(pred[va_idx], y[va_idx]),
                    "r2_ridge": r2(pred[va_idx], y[va_idx]),
                    "segments": {},
                }
                for k in VAL_KEYS:
                    m, n = masked_rmse(pred, y, keys, {k})
                    entry["segments"][k] = {"rmse_ridge": m, "n": n}
                improvement = (entry["rmse_cv"] - entry["rmse_ridge"]) / entry["rmse_cv"] * 100.0
                entry["improvement_vs_cv_pct"] = improvement
                table[str(h_s)] = entry
                results["label_sets"][label + "/" + y_name] = table
                csv_rows.append({
                    "label_set": label + "/" + y_name, "horizon_s": h_s,
                    "segment": "ALL_VAL", "rmse_hold_m": round(entry["rmse_hold"], 6),
                    "rmse_cv_m": round(entry["rmse_cv"], 6),
                    "rmse_ridge_m": round(entry["rmse_ridge"], 6),
                    "improvement_vs_cv_pct": round(improvement, 3),
                    "r2_ridge": round(entry["r2_ridge"], 5),
                    "n_val": entry["n_val"],
                })
                for k in VAL_KEYS:
                    csv_rows.append({
                        "label_set": label + "/" + y_name, "horizon_s": h_s,
                        "segment": k, "rmse_hold_m": "",
                        "rmse_cv_m": "",
                        "rmse_ridge_m": round(entry["segments"][k]["rmse_ridge"], 6),
                        "improvement_vs_cv_pct": "", "r2_ridge": "",
                        "n_val": entry["segments"][k]["n"],
                    })
            if not pc:
                verdict_inputs[y_name] = table
            if pc and y_name == "true":
                pc_results["true"] = table
            if not pc and y_name in ("true", "ref"):
                plot_path = os.path.join(args.out_dir, "rmse_by_horizon_" + y_name + ".png")
                make_plot(table, path=plot_path, title=y_name.capitalize() + " root displacement (val pooled, anchor frame)")

    # positive control gate: ridge_pc must at least halve CV error for H >= 0.5 s
    pc_gate = {}
    for h_s in ("0.5", "2.0", "4.9"):
        e = pc_results["true"][h_s]
        pc_gate[h_s] = {"rmse_ridge": e["rmse_ridge"], "rmse_cv": e["rmse_cv"],
                        "passed": bool(e["rmse_ridge"] < 0.5 * e["rmse_cv"])}
    pc_pass = all(v["passed"] for v in pc_gate.values())

    main_true = verdict_inputs["true"]
    best_h = max(main_true, key=lambda h: main_true[h]["improvement_vs_cv_pct"])
    best_impr = main_true[best_h]["improvement_vs_cv_pct"]
    max_r2 = max(main_true[h]["r2_ridge"] for h in main_true)
    if not pc_pass:
        verdict = "IMPLEMENTATION_DEFECT_POSITIVE_CONTROL_FAILED"
    elif best_impr < 5.0 and max_r2 < 0.2:
        verdict = "H2_CONFIRMED_ROOT_NOT_RECOVERABLE_FROM_ALLOWED_INPUTS"
    elif best_impr >= 10.0:
        verdict = "INDIRECT_CUES_H2_DOWNGRADED"
    else:
        verdict = "INTERMEDIATE_BAND_5_TO_10_PCT_REPORT_ONLY"

    result = {
        "experiment": reg["experiment"],
        "status": "COMPLETE" if pc_pass else "FAILED_POSITIVE_CONTROL",
        "verdict": verdict,
        "criteria_source": "EXP_CONTRACTS.md section 2.4 (pre-registered)",
        "main_readout": {
            "label_set": "true",
            "best_horizon_s": float(best_h),
            "best_improvement_vs_cv_pct": best_impr,
            "max_r2_true": max_r2,
            "per_horizon": {h: {k: main_true[h][k] for k in
                                ("rmse_hold", "rmse_cv", "rmse_ridge", "r2_ridge",
                                 "improvement_vs_cv_pct", "n_val")}
                            for h in main_true},
        },
        "ref_label_readout": {
            h: {k: verdict_inputs["ref"][h][k] for k in
                ("rmse_hold", "rmse_cv", "rmse_ridge", "r2_ridge",
                 "improvement_vs_cv_pct")} for h in verdict_inputs["ref"]
        },
        "positive_control": {"gate": pc_gate, "passed": pc_pass},
        "feature_dim_base": BASE_DIM,
        "label_frame": "sim anchor (root heading) frame at t; transform is labeling convention only",
        "z_clip": Z_CLIP,
        "script_version": "v2 (HP25): anchor-frame labels + clipped z-scores; HP24 evidence preserved",
        "horizons_s": list(HORIZONS_S),
        "split": {"train": list(TRAIN_KEYS), "val": list(VAL_KEYS)},
        "ridge_alpha": RIDGE_ALPHA,
        "normalization": "train-only z-score clipped to [-5,5]",
        "input_verification": verification,
        "elapsed_s": round(time.time() - t_start, 3),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out_dir, "identifiability_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("verdict:", verdict)
    print("best_improvement_pct:", round(best_impr, 3), "at H =", best_h, "s; max_r2_true:", round(max_r2, 5))
    print("positive_control_pass:", pc_pass)
    print("outputs:", args.out_dir)
    return 0 if pc_pass else 3


def make_plot(table: dict, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = sorted(table, key=float)
    x = [float(h) for h in hs]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, key in (("hold", "rmse_hold"), ("const-vel", "rmse_cv"), ("ridge(allowed inputs)", "rmse_ridge")):
        ax.plot(x, [table[h][key] for h in hs], marker="o", label=name)
    ax.set_xlabel("horizon H (s)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
