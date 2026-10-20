"""p15c batch synth: all preview sessions -> p15c_data/*.npz with provenance."""
import sys, os, glob, json, time, traceback
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d

OUT_DIR = "/opt/data/private/yuanqi_ops/p15c_data"
os.makedirs(OUT_DIR, exist_ok=True)
IDX = {"pelvis": 0, "neck": 12, "l_shoulder": 16, "l_elbow": 18, "l_wrist": 20,
       "r_shoulder": 17, "r_elbow": 19, "r_wrist": 21, "l_hip": 1, "r_hip": 2}
tp = np.load("/opt/data/private/yuanqi_ops/data/smb_weave_staging/session_20261018_152655/tpose_quat_means.npz")
M = {i: tp["M%d" % i].reshape(3, 3) for i in range(6)}

def seg_frame(a, b, c):
    x = b - a
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    t = c - a
    y = t - x * (t * x).sum(-1, keepdims=True)
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)
    return np.stack([x, y, z_ := np.cross(x, y)], axis=-1)

def synth(path, out, max_frames=16000):
    with h5py.File(path, "r") as f:
        G = f["body_motion/body_keypoints"][:max_frames].astype(np.float64)
    T = len(G)
    if T < 900:
        return "skip_short_%d" % T
    if not np.isfinite(G).all():
        return "skip_nan"
    fps = 30.0
    R_back = seg_frame(G[:, IDX["pelvis"]], G[:, IDX["neck"]], G[:, IDX["r_shoulder"]])
    R_uarm_l = seg_frame(G[:, IDX["l_shoulder"]], G[:, IDX["l_elbow"]], G[:, IDX["l_wrist"]] + np.array([0, 0, .1]))
    R_uarm_r = seg_frame(G[:, IDX["r_shoulder"]], G[:, IDX["r_elbow"]], G[:, IDX["r_wrist"]] + np.array([0, 0, .1]))
    R_farm_l = seg_frame(G[:, IDX["l_elbow"]], G[:, IDX["l_wrist"]], G[:, IDX["l_shoulder"]] + np.array([0, 0, .1]))
    R_farm_r = seg_frame(G[:, IDX["r_elbow"]], G[:, IDX["r_wrist"]], G[:, IDX["r_shoulder"]] + np.array([0, 0, .1]))
    R_waist = seg_frame(G[:, IDX["l_hip"]], G[:, IDX["r_hip"]], G[:, IDX["neck"]])
    CH = [("back", R_back, IDX["neck"], 0, 0), ("waist", R_waist, IDX["pelvis"], 9, 1),
          ("uarm_l", R_uarm_l, IDX["l_elbow"], 4, 2), ("uarm_r", R_uarm_r, IDX["r_elbow"], 6, 3),
          ("farm_l", R_farm_l, IDX["l_wrist"], 5, 4), ("farm_r", R_farm_r, IDX["r_wrist"], 1, 5)]
    g_w = np.array([0.0, 0.0, -9.81])
    acc6, rot6 = {}, {}
    for name, Rb, aj, slot, mi in CH:
        Rwd = Rb @ (Rb[0].T @ M[mi])
        P = gaussian_filter1d(G[:, aj], 5.0, axis=0)
        a = np.gradient(np.gradient(P, 1 / fps, axis=0), 1 / fps, axis=0)
        acc_dev = np.clip(gaussian_filter1d(np.einsum("tji,tj->ti", Rwd.transpose(0, 2, 1), a - g_w), 6, axis=0), -60, 60)
        acc6[slot] = acc_dev.astype(np.float32)
        rot6[slot] = Rwd.astype(np.float32)
    S = sorted(acc6)
    np.savez_compressed(out, acc=np.stack([acc6[s] for s in S]), rot=np.stack([rot6[s] for s in S]),
                        slots=json.dumps(S), provenance="synthe_from_preview_gt", source=os.path.basename(os.path.dirname(path)))
    return "ok_%d" % T

sess = sorted(glob.glob("/opt/data/private/yuanqi_ops/data/humanplus_preview/session_*/annotation.hdf5"))
print("sessions:", len(sess), flush=True)
t0 = time.time()
ok = 0
for i, p in enumerate(sess):
    sname = os.path.basename(os.path.dirname(p))
    out = os.path.join(OUT_DIR, sname + ".npz")
    if os.path.exists(out):
        continue
    try:
        st = synth(p, out)
    except Exception as e:
        st = "ERR_" + type(e).__name__
        traceback.print_exc()
    if st.startswith("ok"):
        ok += 1
    if i % 10 == 0 or i == len(sess) - 1:
        print("[%4.0fs] %d/%d ok=%d %s" % (time.time() - t0, i + 1, len(sess), ok, st), flush=True)
print("DONE ok=%d/%d" % (ok, len(sess)), flush=True)
