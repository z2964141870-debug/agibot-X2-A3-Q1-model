#!/usr/bin/env python3
# p15f_grid.py — wire mapping grid on trial scene: head candidate (u4|l5) x order (u_first|l_first) -> B-seg MPJPE
import sys, os, json, time
import numpy as np, h5py, torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/opt/data/private/yuanqi_ops/script")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main/Aplus/tools")
import p15b_fgp_pipeline as P
P.cano = P.load_cano()
P.lip = P.load_lip()
P.s1, P.s2 = P.lip.net_s1, P.lip.net_s2
print("models loaded", flush=True)

BULK = "/opt/data/private/yuanqi_ops/data/smb_bulk"
SC = "session_20260903_113804"
PRE = "/opt/data/private/yuanqi_ops/data/humanplus_preview"
SYNC = 3

def parse_wires():
    by = {}
    for line in open(os.path.join(BULK, SC, "imu_samples.jsonl")):
        d = json.loads(line)
        if d.get("k") != "i" or "i" not in d: continue
        r = d.get("r")
        for row in d["i"]:
            if len(row) < 11 or row[0] is None or row[3] is None: continue
            try:
                ch = (r, int(row[0])); u = float(row[3])
                acc = [float(v) for v in row[4:7]]; q = [float(v) for v in row[7:11]]
            except (TypeError, ValueError): continue
            if sum(abs(x) for x in q) < 1e-6: continue
            by.setdefault(ch, ([], [], []))
            by[ch][0].append(u); by[ch][1].append(acc); by[ch][2].append(q)
    return {k: v for k, v in by.items() if len(v[0]) > 1000}

def make_feed(wires, head, u_first):
    if u_first:
        order = [("u", i) for i in range(6)] + [("l", i) for i in (0, 1, 2, 4, 5)]
    else:
        order = [("l", i) for i in (0, 1, 2, 4, 5)] + [("u", i) for i in range(6)]
    order = [c for c in order if c != head]
    assert len(order) == 10
    with h5py.File(os.path.join(PRE, SC, "annotation.hdf5"), "r") as f:
        T_gt = f["body_motion/body_keypoints"].shape[0]
    rot = np.zeros((T_gt, 10, 3, 3), np.float32); acc = np.zeros((T_gt, 10, 3), np.float32)
    for i, ch in enumerate(order):
        uts, accs, qs = wires[ch]
        o = np.argsort(uts); uts = np.asarray(uts)[o]
        accs = np.asarray(accs)[o]; qs = np.asarray(qs)[o]
        rm = R.from_quat(qs[:, [1,2,3,0]]).as_matrix()
        idx = np.round((uts - uts[0]) / 1e9 * 30.0).astype(int) - SYNC
        m = (idx >= 0) & (idx < T_gt)
        rot[idx[m], i] = rm[m].astype(np.float32); acc[idx[m], i] = accs[m].astype(np.float32)
    return rot, acc

def run_b(rot, acc):
    A = torch.from_numpy(acc.astype(np.float32)); Rt = torch.from_numpy(rot.astype(np.float32))
    with torch.no_grad():
        acc_d, rot_d = P.run_cano(A, Rt, imu_ids=list(range(10)), chunk_len=512)
        poses = P.lip_batch(acc_d, rot_d)
    with h5py.File(os.path.join(PRE, SC, "annotation.hdf5"), "r") as f:
        gt = f["body_motion/body_keypoints"][:].astype(np.float32)[:len(poses)]
    gr = gt - gt[:, 0:1]
    sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main/Aplus/tools")
    from smpl_light import SMPLight
    sm = SMPLight(); jf = np.zeros((len(poses), 24, 3), np.float32)
    with torch.no_grad():
        for s in range(0, len(poses), 4096):
            Rb = torch.from_numpy(poses[s:s+4096].astype(np.float32))
            _, j = sm.forward_kinematics(Rb, calc_joint=True)
            jf[s:s+len(j)] = j.numpy()
    jr = jf - jf[:, 0:1]
    V = jr[::20].reshape(-1, 3); W_ = gr[::20].reshape(-1, 3)
    H = V.T @ W_; U, S2, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T)); Rw = U @ np.diag([1,1,dd]) @ Vt
    jra = jr @ Rw.T
    sc = (jra[::20] * gr[::20]).sum() / (jra[::20] ** 2).sum(); jra = jra * sc
    return float(np.linalg.norm(jra - gr, axis=2).mean()), float(sc)

wires = parse_wires()
print("live wires:", sorted([f"{k[0]}{k[1]}" for k in wires]))
results = {}
for head in [("u", 4), ("l", 5)]:
    for u_first in (True, False):
        tag = "drop_%s%d_%s" % (head[0], head[1], "u1st" if u_first else "l1st")
        t0 = time.time()
        rot, acc = make_feed(wires, head, u_first)
        mp, sc = run_b(rot, acc)
        results[tag] = {"mpjpe": mp, "scale": sc, "secs": round(time.time()-t0)}
        print(tag, "MPJPE %.4f scale %.3f (%.0fs)" % (mp, sc, time.time()-t0), flush=True)
json.dump(results, open("/opt/data/private/yuanqi_ops/data/experiments/p15b_fgp_pilot/WIRE_GRID.json", "w"), indent=1)
best = min(results, key=lambda k: results[k]["mpjpe"])
print("BEST:", best, results[best]["mpjpe"])
print("saved WIRE_GRID.json")
