#!/usr/bin/env python3
"""HP26 Exp-2a v3: fall-excluded per-segment identifiability with linear+MLP readers.

Lead-approved contract revision v2 of section 2 (Exp-2a), executed as HP26:
1. fall frames are OOD contamination: samples whose feature or label window
   intersects a fall zone are excluded; fall zones come from the HP11 baseline
   reference metrics.json fall_tick fields (zzx tick 283), not re-derived;
2. readers: ridge (alpha=1) AND a small numpy MLP (2x128 ReLU, Adam, early
   stopped on a held-out slice of TRAIN segments only). H2 is confirmed only if
   BOTH readers fail the pre-registered recovery bars;
3. the positive control is scoped to the same non-fall validation samples.
Everything else is unchanged from HP25 (anchor-frame labels, clipped z-scores,
single run, no tuning on validation segments, zero GPU/simulation).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.a3 import root_identifiability_trial as v2  # noqa: E402

HORIZONS_S = v2.HORIZONS_S
POLICY_DT = v2.POLICY_DT
TRAIN_KEYS = v2.TRAIN_KEYS
VAL_KEYS = v2.VAL_KEYS
MLP_HIDDEN = (128, 128)
MLP_LR = 1e-3
MLP_EPOCHS = 400
MLP_BATCH = 256
MLP_PATIENCE = 25
SEED = 0


def load_fall_ticks(data_root: str, registration) -> dict:
    ticks = {}
    for entry in registration["fall_metadata"]:
        path = entry["path"]
        data = json.load(open(path))
        motion = data["motions"][0]
        ticks[entry["segment"]] = {"fall": bool(motion.get("fall")),
                                   "fall_tick": motion.get("fall_tick"),
                                   "sha256": entry["sha256"],
                                   "actual_sha256": v2.sha256_file(path)}
    return ticks


def fall_excluded(t: int, h_ticks: int, fall_tick, max_lag: int) -> bool:
    if fall_tick is None:
        return False
    return t + h_ticks >= fall_tick or t - max_lag >= fall_tick


def sample_dataset_v3(segs: dict, h_ticks: int, h_s: float, positive_control: bool,
                      fall_ticks: dict):
    X, y_true, y_ref, cv_true, cv_ref, keys = [], [], [], [], [], []
    excluded = {k: 0 for k in segs}
    total = {k: 0 for k in segs}
    max_lag = v2.max_lag()
    for key, seg in segs.items():
        a = seg["arrays"]
        ref_q, q_state, action = a["reference_q_29"], a["q_state_29"], a["raw_action_29"]
        q_ref_a, q_sim_a = a["ref_anchor_quat_wxyz"], a["sim_anchor_quat_wxyz"]
        sim_root, ref_root = a["sim_root_pos_w"], a["ref_root_pos_w"]
        fall_tick = fall_ticks.get(key, {}).get("fall_tick")
        for t in range(max_lag, len(q_state) - h_ticks):
            total[key] += 1
            if fall_excluded(t, h_ticks, fall_tick, max_lag):
                excluded[key] += 1
                continue
            R_t = v2.quat_to_rot(q_sim_a[t])
            pc = R_t.T @ (sim_root[t] - sim_root[0]) if positive_control else None
            X.append(v2.build_feature_block(ref_q, q_state, action, q_ref_a, q_sim_a, t, pc))
            y_true.append(R_t.T @ (sim_root[t + h_ticks] - sim_root[t]))
            y_ref.append(R_t.T @ (ref_root[t + h_ticks] - ref_root[t]))
            v_true = R_t.T @ (sim_root[t] - sim_root[t - v2.VEL_WINDOW_TICKS]) / (v2.VEL_WINDOW_TICKS * POLICY_DT)
            v_ref = R_t.T @ (ref_root[t] - ref_root[t - v2.VEL_WINDOW_TICKS]) / (v2.VEL_WINDOW_TICKS * POLICY_DT)
            cv_true.append(v_true * h_s)
            cv_ref.append(v_ref * h_s)
            keys.append(key)
    return (np.asarray(X), np.asarray(y_true), np.asarray(y_ref),
            np.asarray(cv_true), np.asarray(cv_ref), keys, excluded, total)


def _init_net(rng, dims):
    weights, biases = [], []
    for fan_in, fan_out in zip(dims, dims[1:]):
        weights.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out))
    return weights, biases


def _forward(weights, biases, X):
    activations = [X]
    for layer, (W, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ W + b
        activations.append(z if layer == len(weights) - 1 else np.maximum(z, 0.0))
    return activations


def mlp_fit(X: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = np.clip((X - mu) / sd, -v2.Z_CLIP, v2.Z_CLIP)
    y_mean = y.mean(axis=0)
    Y = y - y_mean
    holdout = np.arange(len(Xs)) % 10 == 9  # deterministic train-only early-stop slice
    fit_idx = np.arange(len(Xs))[~holdout]
    hold_idx = np.arange(len(Xs))[holdout]
    dims = (Xs.shape[1],) + MLP_HIDDEN + (y.shape[1],)
    weights, biases = _init_net(rng, dims)
    mW = [np.zeros_like(W) for W in weights]
    vW = [np.zeros_like(W) for W in weights]
    mB = [np.zeros_like(b) for b in biases]
    vB = [np.zeros_like(b) for b in biases]
    best = None
    best_epoch = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    for epoch in range(MLP_EPOCHS):
        order = rng.permutation(len(fit_idx))
        for start in range(0, len(order), MLP_BATCH):
            batch = fit_idx[order[start:start + MLP_BATCH]]
            activations = _forward(weights, biases, Xs[batch])
            error = activations[-1] - Y[batch]
            grad = error / len(batch)
            for layer in range(len(weights) - 1, -1, -1):
                gW = activations[layer].T @ grad
                gB = grad.sum(axis=0)
                grad = (grad @ weights[layer].T) * (activations[layer] > 0)
                step += 1
                for param, grad_param, m, v in ((weights, gW, mW, vW), (biases, gB, mB, vB)):
                    m[layer] = beta1 * m[layer] + (1 - beta1) * grad_param
                    v[layer] = beta2 * v[layer] + (1 - beta2) * grad_param ** 2
                    m_hat = m[layer] / (1 - beta1 ** step)
                    v_hat = v[layer] / (1 - beta2 ** step)
                    param[layer] -= MLP_LR * m_hat / (np.sqrt(v_hat) + eps)
        hold_pred = _forward(weights, biases, Xs[hold_idx])[-1]
        hold_loss = float(np.mean((hold_pred - Y[hold_idx]) ** 2))
        if best is None or hold_loss < best - 1e-12:
            best = hold_loss
            best_epoch = epoch
            snapshot = ([W.copy() for W in weights], [b.copy() for b in biases])
        elif epoch - best_epoch >= MLP_PATIENCE:
            break
    return {"mu": mu, "sd": sd, "y_mean": y_mean, "weights": snapshot[0], "biases": snapshot[1],
            "best_epoch": best_epoch, "best_holdout_mse": best, "epochs_run": epoch + 1}


def mlp_predict(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = np.clip((X - model["mu"]) / model["sd"], -v2.Z_CLIP, v2.Z_CLIP)
    return _forward(model["weights"], model["biases"], Xs)[-1] + model["y_mean"]


def read_out(table, reader, key):
    return table[str(key)]["readers"][reader]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    reg = json.load(open(args.registration))

    verification, ok = [], True
    for entry in reg["inputs"] + reg["fall_metadata"]:
        path = entry["path"]
        if not os.path.exists(path):
            verification.append({"path": path, "status": "MISSING"})
            ok = False
            continue
        got = v2.sha256_file(path)
        status = "OK" if got == entry["sha256"] else "SHA_MISMATCH"
        ok = ok and status == "OK"
        verification.append({"path": path, "status": status, "sha256": got})
    print("input verification:", "PASS" if ok else "FAIL")
    if not ok:
        json.dump({"status": "FAILED_INPUT_VERIFICATION", "verification": verification},
                  open(os.path.join(args.out_dir, "result.json"), "w"), indent=2)
        return 2

    segs = {k: v2.load_segment(args.data_root, k) for k in v2.SEGMENTS}
    fall_ticks = load_fall_ticks(args.data_root, reg)
    zzx = fall_ticks.get("zzx", {})
    if not zzx.get("fall") or zzx.get("fall_tick") != 283:
        raise ValueError("Registered zzx fall metadata changed; contract premise broken")

    results = {"label_sets": {}}
    csv_rows, verdict_inputs, pc_results = [], {}, {}
    exclusion_report = {}
    for pc in (False, True):
        label = "positive_control" if pc else "main"
        for y_name in ("true", "ref"):
            table = {}
            for h_s in HORIZONS_S:
                h_ticks = int(round(h_s / POLICY_DT))
                X, yt, yr, cvt, cvr, keys, excluded, total = sample_dataset_v3(
                    segs, h_ticks, h_s, pc, fall_ticks)
                if not pc:
                    exclusion_report[str(h_s)] = {"excluded": excluded, "total": total}
                y = yt if y_name == "true" else yr
                cv = cvt if y_name == "true" else cvr
                tr_idx = [i for i, k in enumerate(keys) if k in TRAIN_KEYS]
                va_idx = [i for i, k in enumerate(keys) if k in VAL_KEYS]
                ridge = v2.ridge_fit(X[tr_idx], y[tr_idx], v2.RIDGE_ALPHA)
                ridge_pred = v2.ridge_predict(ridge, X)
                mlp = mlp_fit(X[tr_idx], y[tr_idx])
                mlp_pred = mlp_predict(mlp, X)
                hold = np.zeros_like(y)
                entry = {"n_train": len(tr_idx), "n_val": len(va_idx),
                         "readers": {}, "segments": {}}
                for reader, pred in (("ridge", ridge_pred), ("mlp", mlp_pred)):
                    entry["readers"][reader] = {
                        "rmse_hold": v2.rmse(hold[va_idx], y[va_idx]),
                        "rmse_cv": v2.rmse(cv[va_idx], y[va_idx]),
                        "rmse_reader": v2.rmse(pred[va_idx], y[va_idx]),
                        "r2_reader": v2.r2(pred[va_idx], y[va_idx]),
                        "improvement_vs_cv_pct": (v2.rmse(cv[va_idx], y[va_idx]) -
                                                  v2.rmse(pred[va_idx], y[va_idx])) /
                                                 v2.rmse(cv[va_idx], y[va_idx]) * 100.0,
                    }
                    for k in VAL_KEYS:
                        m, n = v2.masked_rmse(pred, y, keys, {k})
                        entry["segments"].setdefault(k, {})[reader] = {"rmse": m, "n": n}
                if not pc:
                    entry["mlp_training"] = {k: mlp[k] for k in ("best_epoch", "best_holdout_mse", "epochs_run")}
                table[str(h_s)] = entry
                for reader in ("ridge", "mlp"):
                    r = entry["readers"][reader]
                    csv_rows.append({"label_set": label + "/" + y_name, "reader": reader,
                        "horizon_s": h_s, "segment": "ALL_VAL",
                        "rmse_cv_m": round(r["rmse_cv"], 6), "rmse_reader_m": round(r["rmse_reader"], 6),
                        "improvement_vs_cv_pct": round(r["improvement_vs_cv_pct"], 3),
                        "r2_reader": round(r["r2_reader"], 5), "n_val": entry["n_val"]})
            results["label_sets"][label + "/" + y_name] = table
            if not pc:
                verdict_inputs[y_name] = table
            if pc and y_name == "true":
                pc_results["true"] = table

    pc_gate = {}
    for h_s in ("0.5", "2.0", "4.9"):
        e = read_out(pc_results["true"], "ridge", h_s)
        pc_gate[h_s] = {"rmse_ridge": e["rmse_reader"], "rmse_cv": e["rmse_cv"],
                        "passed": bool(e["rmse_reader"] < 0.5 * e["rmse_cv"])}
    pc_pass = all(v["passed"] for v in pc_gate.values())

    main_true = verdict_inputs["true"]
    reader_bars = {}
    for reader in ("ridge", "mlp"):
        best_impr = max(read_out(main_true, reader, h)["improvement_vs_cv_pct"] for h in main_true)
        max_r2 = max(read_out(main_true, reader, h)["r2_reader"] for h in main_true)
        reader_bars[reader] = {"best_improvement_vs_cv_pct": best_impr, "max_r2": max_r2}
    nonlinear_failed = (reader_bars["mlp"]["best_improvement_vs_cv_pct"] < 5.0 and
                        reader_bars["mlp"]["max_r2"] < 0.2)
    linear_failed = (reader_bars["ridge"]["best_improvement_vs_cv_pct"] < 5.0 and
                     reader_bars["ridge"]["max_r2"] < 0.2)
    any_strong = any(reader_bars[r]["best_improvement_vs_cv_pct"] >= 10.0 for r in ("ridge", "mlp"))
    if not pc_pass:
        verdict = "IMPLEMENTATION_DEFECT_POSITIVE_CONTROL_FAILED"
    elif linear_failed and nonlinear_failed:
        verdict = "H2_CONFIRMED_ROOT_NOT_RECOVERABLE_EVEN_BY_NONLINEAR_READER"
    elif any_strong:
        verdict = "INDIRECT_CUES_H2_DOWNGRADED"
    else:
        verdict = "INTERMEDIATE_BAND_5_TO_10_PCT_REPORT_ONLY"

    result = {
        "experiment": reg["experiment"], "status": "COMPLETE" if pc_pass else "FAILED_POSITIVE_CONTROL",
        "verdict": verdict,
        "criteria_source": "EXP_CONTRACTS.md section 2 revision v2 (lead-approved HP26); bars pre-registered",
        "fall_scope": {"source": "HP11 baseline reference metrics.json fall_tick",
                       "zzx_fall_tick": zzx.get("fall_tick"),
                       "exclusion_rule": "sample excluded iff [t-max_lag, t+h] intersects [fall_tick, end)",
                       "exclusions_by_horizon": exclusion_report},
        "reader_bars": reader_bars,
        "main_readout": {"h2_confirmed_requires_both_readers_failed": True,
            "per_horizon": {h: {"ridge": read_out(main_true, "ridge", h),
                                "mlp": read_out(main_true, "mlp", h)} for h in main_true}},
        "positive_control": {"gate": pc_gate, "passed": pc_pass,
                             "scope": "same non-fall validation samples as main"},
        "split": {"train": list(TRAIN_KEYS), "val": list(VAL_KEYS)},
        "ridge_alpha": v2.RIDGE_ALPHA, "z_clip": v2.Z_CLIP,
        "mlp": {"hidden": list(MLP_HIDDEN), "lr": MLP_LR, "epochs": MLP_EPOCHS,
                "batch": MLP_BATCH, "patience": MLP_PATIENCE, "seed": SEED,
                "early_stop": "held-out deterministic 10% of TRAIN samples only"},
        "script_version": "v3 (HP26): fall-excluded, per-segment, ridge+MLP readers, PC scoped",
        "input_verification": verification,
        "elapsed_s": round(time.time() - t_start, 3),
        "host": socket.gethostname(), "numpy": np.__version__,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out_dir, "identifiability_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("verdict:", verdict)
    print("reader_bars:", json.dumps(reader_bars))
    print("positive_control_pass:", pc_pass)
    return 0 if pc_pass else 3


if __name__ == "__main__":
    sys.exit(main())
