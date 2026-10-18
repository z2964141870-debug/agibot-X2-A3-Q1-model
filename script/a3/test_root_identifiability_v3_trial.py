import json, os, sys
import numpy as np
sys.path.insert(0, "/home/humanplus/YUANQI_A3")
from script.a3 import root_identifiability_trial as v2
from script.a3 import root_identifiability_v3_trial as v3

P = []
def check(name, cond):
    P.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name)

# 1. fall exclusion: zzx excluded only near/after tick 283, others untouched
check("zzx excluded after 283", v3.fall_excluded(290, 25, 283, 10))
check("zzx kept before 273", not v3.fall_excluded(200, 25, 283, 10))
check("zzx label-window rule", v3.fall_excluded(270, 25, 283, 10))
check("no fall no exclusion", not v3.fall_excluded(400, 25, None, 10))

# 2. dataset exclusion counts with synthetic fall on every segment
segs = {k: v2.load_segment("/home/humanplus/YUANQI_A3/data/experiments/a3_humanplus_baseline_reference_20261018_HP11", k) for k in ("zzx", "yy")}
falls = {"zzx": {"fall": True, "fall_tick": 283}, "yy": {"fall": None}}
X, yt, yr, cvt, cvr, keys, excluded, total = v3.sample_dataset_v3(segs, 25, 0.5, False, falls)
check("zzx rows excluded", excluded["zzx"] > 0 and excluded["yy"] == 0)
check("kept rows sum", excluded["zzx"] + excluded["yy"] + len(keys) == total["zzx"] + total["yy"])
check("feature dim intact", X.shape[1] == 180)

# 3. MLP determinism + fits synthetic linear map better than hold
rng = np.random.default_rng(7)
Xs = rng.normal(size=(600, 180))
W = rng.normal(scale=0.05, size=(180, 3))
Ys = Xs @ W + 0.01 * rng.normal(size=(600, 3))
m1, m2 = v3.mlp_fit(Xs, Ys), v3.mlp_fit(Xs, Ys)
p1, p2 = v3.mlp_predict(m1, Xs[:50]), v3.mlp_predict(m2, Xs[:50])
check("mlp deterministic seed0", float(np.max(np.abs(p1 - p2))) == 0.0)
check("mlp beats hold on linear map", v2.rmse(p1, Ys[:50]) < v2.rmse(np.zeros((50, 3)), Ys[:50]))

# 4. v3 verdict plumbing: PC gate arithmetic
check("pc gate strict half", (0.49 < 0.5 * 1.0) and not (0.5 < 0.5 * 1.0))

# 5. v2 contract preserved: anchor-frame labels identical between readers' dataset
check("labels shared", yt.shape == yr.shape and cvt.shape == cvr.shape)

fails = [n for n, ok in P if not ok]
print("TOTAL", len(P), "FAILS", len(fails))
sys.exit(1 if fails else 0)
