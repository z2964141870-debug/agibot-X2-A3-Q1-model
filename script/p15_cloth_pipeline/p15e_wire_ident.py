#!/usr/bin/env python3
# p15e v2 — fixed SMPL-24 indices; 11 wires x 12 segments Pearson matrix
import sys, os, json
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R

BULK = "/opt/data/private/yuanqi_ops/data/smb_bulk"
SC = "session_20260903_113804"
PRE = "/opt/data/private/yuanqi_ops/data/humanplus_preview"

by = {}
for line in open(os.path.join(BULK, SC, "imu_samples.jsonl")):
    d = json.loads(line)
    if d.get("k") != "i" or "i" not in d: continue
    r = d.get("r")
    for row in d["i"]:
        if len(row) < 11 or row[0] is None or row[3] is None: continue
        try:
            ch = (r, int(row[0])); u = float(row[3]); q = [float(v) for v in row[7:11]]
        except (TypeError, ValueError): continue
        if sum(abs(x) for x in q) < 1e-6: continue
        by.setdefault(ch, ([], []))
        by[ch][0].append(u); by[ch][1].append(q)

with h5py.File(os.path.join(PRE, SC, "annotation.hdf5"), "r") as f:
    kp = f["body_motion/body_keypoints"][:].astype(np.float32)
T_gt = len(kp)

def seg(a, b):
    v = kp[:, b] - kp[:, a]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

SEGS = {
    "head": seg(12, 15), "torso": seg(0, 12),
    "uarmL": seg(16, 18), "uarmR": seg(17, 19),
    "farmL": seg(18, 20), "farmR": seg(19, 21),
    "thighL": seg(1, 4), "thighR": seg(2, 5),
    "shinL": seg(4, 7), "shinR": seg(5, 8),
    "footL": seg(7, 10), "footR": seg(8, 11),
}
names = list(SEGS)

SYNC = 3
live = sorted([k for k in by if len(by[k][0]) > 1000], key=str)
W = len(live)
grid = np.zeros((T_gt, W, 3, 3), np.float32)
for i, ch in enumerate(live):
    uts, qs = by[ch]
    o = np.argsort(uts); uts = np.asarray(uts)[o]; qs = np.asarray(qs)[o]
    rm = R.from_quat(qs[:, [1,2,3,0]]).as_matrix()
    idx = np.round((uts - uts[0]) / 1e9 * 30.0).astype(int) - SYNC
    m = (idx >= 0) & (idx < T_gt)
    grid[idx[m], i] = rm[m].astype(np.float32)

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / ((np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + 1e-12))

M = np.zeros((W, len(names)))
for i in range(W):
    v = np.linalg.norm(grid[:, i].reshape(T_gt, -1), axis=1) > 0.5
    for s, name in enumerate(names):
        best = -2.0
        for ax in range(3):
            for sgn in (1, -1):
                c = pearson(grid[v, i, :, ax] * sgn, SEGS[name][v])
                if c > best: best = c
        M[i, s] = best

print("wire   " + " ".join("%8s" % n for n in names))
for i, ch in enumerate(live):
    print("%s%-3d " % (ch[0], ch[1]) + " ".join("%8.3f" % M[i, s] for s in range(len(names))))
am = [names[int(np.argmax(M[i]))] for i in range(W)]
print("\nargmax:", dict(zip([f"{c[0]}{c[1]}" for c in live], am)))
dup = len(am) != len(set(am))
print("duplicates:", dup)
hcol = names.index("head")
print("head top3:", sorted([(f"{live[i][0]}{live[i][1]}", round(float(M[i, hcol]), 3)) for i in range(W)], key=lambda t: -t[1])[:3])
json.dump({"matrix": {f"{live[i][0]}{live[i][1]}": {names[s]: float(M[i, s]) for s in range(len(names))} for i in range(W)},
           "argmax": dict(zip([f"{c[0]}{c[1]}" for c in live], am)),
           "method": "per-axis Pearson vs SMPL24 GT segments, -3f sync, no d2b, trial scene"},
          open("/opt/data/private/yuanqi_ops/data/experiments/p15b_fgp_pilot/WIRE_IDENT.json", "w"), indent=1)
print("saved")
