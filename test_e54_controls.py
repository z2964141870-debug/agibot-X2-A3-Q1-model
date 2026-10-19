"""E54 controls (torch-free): manifest math, error-form zero point, verdict logic."""

import copy
import hashlib
import json
import numpy as np
from pathlib import Path

OPS = Path("/opt/data/private/yuanqi_ops")
import sys
sys.path.insert(0, str(OPS))
import e54_protocol as P  # noqa: E402
import e54_channel_stats as S  # noqa: E402

STATS = P.TRAIN_ROOT.parent.parent / "experiments" / P.NS / P.STATS_NAME
STATS = OPS / ("data/experiments/" + P.NS + "/" + P.STATS_NAME)


def approx(a, b, tol):
    assert abs(a - b) <= tol, (a, b, tol)


def test_manifest_math():
    manifest = json.loads(STATS.read_text())
    assert len(manifest) == 10, len(manifest)
    for name, row in manifest.items():
        assert len(row["mu"]) == 6 and len(row["sigma"]) == 6
        assert all(v > 1e-6 for v in row["sigma"]), (name, row["sigma"])
        sample = S.stats_from_trace({"ref_anchor_pos_w": np.asarray(
            json.loads(Path(row["source_file"]).read_text())["motions"][0]["ref_anchor_pos_w"]),
            "ref_anchor_quat_wxyz": np.asarray(
            json.loads(Path(row["source_file"]).read_text())["motions"][0]["ref_anchor_quat_wxyz"]),
            "motion_name": name})
        assert np.allclose(sample["mu"], row["mu"]) and np.allclose(sample["sigma"], row["sigma"])
    return "manifest recompute determinism OK"


def test_standardization_bounds():
    manifest = json.loads(STATS.read_text())
    worst = 0.0
    for name, row in manifest.items():
        blob = json.loads(Path(row["source_file"]).read_text())["motions"][0]
        p = np.asarray(blob["ref_anchor_pos_w"], dtype=float)
        q = np.asarray(blob["ref_anchor_quat_wxyz"], dtype=float)
        R = S.heading_rot(q)
        disp = S.rot_transpose_vec(R, p - p[0])
        vel_raw = np.zeros_like(p)
        vel_raw[1:] = (p[1:] - p[:-1]) / S.DT
        vel = S.rot_transpose_vec(R, vel_raw)
        X = np.concatenate([disp, vel], axis=1)
        z = np.clip((X - np.asarray(row["mu"])) / np.asarray(row["sigma"]), -5.0, 5.0)
        assert np.abs(z).max() <= 5.0 + 1e-9
        worst = max(worst, float(np.abs(z).max()))
    return "standardization bounds OK (max |z|=%.3f)" % worst


def test_errorform_zero():
    # sim == ref pointwise -> error channels identically zero (numpy mirror of the term)
    rng = np.random.default_rng(0)
    T = 50
    p = np.cumsum(rng.normal(scale=0.01, size=(T, 3)), axis=0)
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    R = S.heading_rot(q)
    dt = S.DT
    origin_ref = origin_sim = p[0]
    prev_ref = prev_sim = p[0]
    worst = 0.0
    for t in range(1, T):
        err_disp = (p[t] - origin_ref) - (p[t] - origin_sim)
        err_vel = ((p[t] - prev_ref) - (p[t] - prev_sim)) / dt
        rt = R[t].T
        worst = max(worst, float(np.abs(rt @ err_disp).max()), float(np.abs(rt @ err_vel).max()))
        prev_ref = prev_sim = p[t]
    assert worst == 0.0, worst
    # constant drift 0.1 m on sim -> |err_disp| = 0.1 regardless of rotation
    rt = R[10].T
    drift = p[10] + np.array([0.1, 0.0, 0.0])
    err = rt @ ((p[10] - p[0]) - (drift - p[0]))
    approx(float(np.linalg.norm(err)), 0.1, 1e-9)
    return "error-form zero point + bounded drift OK"


def test_verdict():
    good = {"wdr": {"max": 1.10}, "cloth6": 0.18}
    tie = {"P": copy.deepcopy(good), "Q": copy.deepcopy(good)}
    out, sel = P.verdict(tie)
    assert out["P"]["both_pass"] and sel["winner"] == "P"
    split = {"P": {"wdr": {"max": 1.05}, "cloth6": 0.17},
             "Q": {"wdr": {"max": 1.08}, "cloth6": 0.175}}
    out, sel = P.verdict(split)
    assert sel["winner"] == "P"
    split2 = {"P": {"wdr": {"max": 1.12}, "cloth6": 0.175},
              "Q": {"wdr": {"max": 1.05}, "cloth6": 0.17}}
    out, sel = P.verdict(split2)
    assert sel["winner"] == "Q"
    single = {"P": {"wdr": {"max": 1.05}, "cloth6": 0.17},
              "Q": {"wdr": {"max": 1.30}, "cloth6": 0.20}}
    out, sel = P.verdict(single)
    assert sel["passing"] == ["P"] and sel["winner"] is None
    none = {"P": {"wdr": {"max": 1.30}, "cloth6": 0.20}, "Q": {"wdr": {"max": 1.40}, "cloth6": 0.19}}
    out, sel = P.verdict(none)
    assert sel["passing"] == []
    approx(P.EFF_MAX_BOUND, 1.1608, 1e-9)
    approx(P.CLOTH6_BOUND, 0.183447, 1e-9)
    return "verdict matrix + pinned bounds (1.1608 / 0.183447) OK"


def test_stats_file_stable():
    before = hashlib.sha256(STATS.read_bytes()).hexdigest()
    first = json.loads(STATS.read_text())
    rebuilt = {}
    for name, row in first.items():
        rebuilt[name] = S.stats_from_trace({"ref_anchor_pos_w": np.asarray(
            json.loads(Path(row["source_file"]).read_text())["motions"][0]["ref_anchor_pos_w"]),
            "ref_anchor_quat_wxyz": np.asarray(
            json.loads(Path(row["source_file"]).read_text())["motions"][0]["ref_anchor_quat_wxyz"]),
            "motion_name": name})
    assert set(rebuilt) == set(first)
    assert all(np.allclose(rebuilt[k]["mu"], first[k]["mu"]) and
               np.allclose(rebuilt[k]["sigma"], first[k]["sigma"]) for k in first)
    return "stats file sha %s stable" % before[:12]


if __name__ == "__main__":
    for check in (test_manifest_math, test_standardization_bounds, test_errorform_zero,
                  test_verdict, test_stats_file_stable):
        print("E54 CONTROL", check())
    print("E54 CONTROLS ALL PASS")
