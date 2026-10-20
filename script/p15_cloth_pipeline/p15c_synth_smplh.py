"""p15c_synth_smplh.py — v3: SMPL-H FK source, twist intact, analytic accel + IMU noise."""
import os, sys, glob, json, time, traceback
import numpy as np
import h5py
import torch
sys.path.insert(0, "/opt/data/private/yuanqi_ops/tools/FGP_Full/FGP-main")
import articulate as art

OUT_DIR = "/opt/data/private/yuanqi_ops/p15c_data_v3"
os.makedirs(OUT_DIR, exist_ok=True)
JL = np.array([
[0,0,0],[58.5813,-82.28,-17.6641],[-60.3097,-90.5133,-13.5425],[4.4394,124.4036,-38.3852],
[43.4514,-386.4695,8.037],[-43.2566,-383.6879,-4.843],[4.4884,137.9564,26.8203],
[-14.7903,-426.8745,-37.428],[19.0555,-420.0456,-34.5617],[-2.2646,56.0324,2.855],
[41.0544,-60.2859,122.0424],[-34.8399,-62.1055,130.3233],[-13.3902,211.6355,-33.4676],
[71.7025,113.9997,-18.8982],[-82.9537,112.4724,-23.7074],[10.1132,88.9373,50.4099],
[122.9214,45.2051,-19.046],[-113.2283,46.8532,-8.4721],[255.3319,-15.649,-22.9465],
[-260.1275,-14.3692,-31.2687],[265.7092,12.6981,-7.3747],[-269.1084,6.7937,-6.0268],
[86.6905,-10.636,-15.5943],[-88.7537,-8.6516,-10.1071]]) / 1000.0
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
tp = np.load("/opt/data/private/yuanqi_ops/data/smb_weave_staging/session_20261018_152655/tpose_quat_means.npz")
M = {i: tp["M%d" % i].reshape(3, 3) for i in range(6)}
CH = [(9, 12, 0, 0), (0, 0, 9, 1), (16, 18, 4, 2), (17, 19, 6, 3), (18, 20, 5, 4), (19, 21, 1, 5)]
# (joint_for_rot, joint_for_accel_pos, slot, M index) — accel anchor follows old-line convention
FPS = 30.0
SIG_W = 0.02 * 9.81     # white noise sigma ~0.02g
BIAS0 = 0.015 * 9.81    # initial bias range +-0.015g
WALK = 0.0003 * 9.81    # bias random walk per frame
CLIP = 8.0 * 9.81       # 8g range

def fk(pose):
    T = len(pose)
    aa = pose[:, :66].reshape(T, 22, 3)
    Rl = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(aa).float()).reshape(T, 22, 3, 3).numpy()
    gR = np.zeros((T, 22, 3, 3)); pos = np.zeros((T, 22, 3))
    gR[:, 0] = Rl[:, 0]; pos[:, 0] = 0
    for c in sorted(range(1, 22), key=lambda x: PARENT[x]):
        p = PARENT[c]
        gR[:, c] = gR[:, p] @ Rl[:, c]
        pos[:, c] = pos[:, p] + np.einsum("tij,j->ti", gR[:, p], JL[c])
    return gR, pos

def synth(path, out, max_frames=16000):
    with h5py.File(path, "r") as f:
        pose = f["body_motion/smplh_pose"][:max_frames].astype(np.float64)
    T = len(pose)
    if T < 900:
        return "skip_short_%d" % T
    if not np.isfinite(pose).all():
        return "skip_nan"
    gR, pos = fk(pose)
    g_w = np.array([0.0, 0.0, -9.81])
    acc6, rot6, sat = {}, {}, 0
    for jr, ja, slot, mi in CH:
        Rb = gR[:, jr]
        Rwd = Rb @ (Rb[0].T @ M[mi])
        P = pos[:, ja]
        a = np.gradient(np.gradient(P, 1 / FPS, axis=0), 1 / FPS, axis=0)
        sf = np.einsum("tji,tj->ti", Rwd.transpose(0, 2, 1), a - g_w)
        bias = np.random.uniform(-BIAS0, BIAS0, 3) + np.cumsum(np.random.normal(0, WALK, (T, 3)), axis=0)
        meas = sf + bias + np.random.normal(0, SIG_W, (T, 3))
        meas = np.clip(meas, -CLIP, CLIP)
        sat += float((np.abs(sf) > CLIP).mean())
        acc6[slot] = meas.astype(np.float32)
        rot6[slot] = Rwd.astype(np.float32)
    S = sorted(acc6)
    np.savez_compressed(out, acc=np.stack([acc6[s] for s in S]), rot=np.stack([rot6[s] for s in S]),
                        slots=json.dumps(S), provenance="synth_from_preview_smplh_pose",
                        source=os.path.basename(os.path.dirname(path)))
    return "ok_%d_sat%.3f" % (T, sat / 6)

sess = sorted(glob.glob("/opt/data/private/yuanqi_ops/data/humanplus_preview/session_*/annotation.hdf5"))
print("sessions:", len(sess), flush=True)
np.random.seed(17)
t0 = time.time(); ok = 0
for i, p in enumerate(sess):
    sname = os.path.basename(os.path.dirname(p))
    out = os.path.join(OUT_DIR, sname + ".npz")
    if os.path.exists(out):
        continue
    try:
        st = synth(p, out)
    except Exception as e:
        st = "ERR_" + type(e).__name__; traceback.print_exc()
    if st.startswith("ok"):
        ok += 1
    if i % 20 == 0 or i == len(sess) - 1:
        print("[%4.0fs] %d/%d ok=%d %s" % (time.time() - t0, i + 1, len(sess), ok, st), flush=True)
print("DONE ok=%d/%d" % (ok, len(sess)), flush=True)
