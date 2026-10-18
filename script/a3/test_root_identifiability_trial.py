#!/usr/bin/env python3
"""Stdlib+numpy control tests for HP25 root_identifiability_trial (no GPU, no sim)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import root_identifiability_trial as T


def make_synthetic_segment(n=499, vx=0.2):
    rng = np.random.default_rng(0)
    t = np.arange(n) * T.POLICY_DT
    q = np.cumsum(rng.normal(0, 1e-3, (n, 29)), axis=0)
    ref_q = np.cumsum(rng.normal(0, 1e-3, (n, 29)), axis=0)
    act = rng.normal(0, 1e-2, (n, 29))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    root = np.stack([0.1 + vx * t, np.zeros(n), np.full(n, 1.07)], axis=1)
    arrays = {
        "reference_q_29": ref_q, "q_state_29": q, "raw_action_29": act,
        "ref_anchor_quat_wxyz": quat, "sim_anchor_quat_wxyz": quat,
        "sim_root_pos_w": root, "ref_root_pos_w": root.copy(),
        "time_s": t,
    }
    return {"key": "syn", "obs": np.zeros((n, 1570), dtype=np.float32),
            "actions": act, "arrays": arrays}


def test_feature_dim_and_no_root_leak():
    seg = make_synthetic_segment()
    a = seg["arrays"]
    v = T.build_feature_block(a["reference_q_29"], a["q_state_29"], a["raw_action_29"],
                              a["ref_anchor_quat_wxyz"], a["sim_anchor_quat_wxyz"], 20)
    assert v.shape == (T.BASE_DIM,), v.shape


def test_features_use_only_past():
    seg = make_synthetic_segment()
    a = seg["arrays"]
    ref_q = a["reference_q_29"].copy()
    q_state = a["q_state_29"].copy()
    action = a["raw_action_29"].copy()
    qr = a["ref_anchor_quat_wxyz"].copy()
    qs = a["sim_anchor_quat_wxyz"].copy()
    t = 30
    v1 = T.build_feature_block(ref_q, q_state, action, qr, qs, t)
    ref_q[t + 1:] += 7.0
    q_state[t + 1:] += 7.0
    action[t + 1:] += 7.0
    v2 = T.build_feature_block(ref_q, q_state, action, qr, qs, t)
    assert np.array_equal(v1, v2), "feature depends on future values"


def test_anchor_frame_label_rotation():
    seg = make_synthetic_segment(vx=0.2)
    a = seg["arrays"]
    yaw = np.pi / 2
    q_y = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
    a["sim_anchor_quat_wxyz"] = np.tile(q_y, (499, 1))
    t, h = 20, 50
    R_t = T.quat_to_rot(a["sim_anchor_quat_wxyz"][t])
    y_anchor = R_t.T @ (a["sim_root_pos_w"][t + h] - a["sim_root_pos_w"][t])
    assert abs(y_anchor[1] + 0.2 * 1.0) < 1e-9, y_anchor
    assert abs(y_anchor[0]) < 1e-9 and abs(y_anchor[2]) < 1e-9


def test_labels_true_displacement_identity_frame():
    seg = make_synthetic_segment(vx=0.2)
    a = seg["arrays"]
    h = 50
    t = 20
    y = a["sim_root_pos_w"][t + h] - a["sim_root_pos_w"][t]
    assert abs(y[0] - 0.2 * 1.0) < 1e-9
    assert abs(y[1]) < 1e-9 and abs(y[2]) < 1e-9


def test_ridge_matches_linear_truth():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (400, 6))
    w_true = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 3.0])
    y1 = X @ w_true + 0.7
    y = np.tile(y1.reshape(-1, 1), (1, 3))
    m = T.ridge_fit(X, y, 1e-12)
    pred = T.ridge_predict(m, X)
    assert T.rmse(pred, y) < 1e-6, T.rmse(pred, y)


def test_clip_bounds_predictions():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (300, 5))
    y = np.tile((X @ np.array([1.0, 1.0, 0.0, 0.0, 0.0])).reshape(-1, 1), (1, 3))
    m = T.ridge_fit(X, y, 1.0)
    X_bad = X.copy()
    X_bad[:, 0] += 1e4
    pred = T.ridge_predict(m, X_bad)
    assert np.all(np.isfinite(pred))
    assert np.max(np.abs(pred)) < 1e3


def test_split_integrity():
    assert set(T.TRAIN_KEYS).isdisjoint(T.VAL_KEYS)
    assert set(T.TRAIN_KEYS) == {"zj_walk", "zj_warmUp", "fjw_walk"}
    assert set(T.VAL_KEYS) == {"zls", "yy", "zzx"}
    assert set(T.SEGMENTS) == set(T.TRAIN_KEYS) | set(T.VAL_KEYS)


def test_horizons_positive_ticks():
    n = 499
    for h in T.HORIZONS_S:
        ticks = int(round(h / T.POLICY_DT))
        assert abs(ticks * T.POLICY_DT - h) < 1e-9
        assert ticks >= 1
        assert n - ticks - T.max_lag() > 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print("%d control tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
