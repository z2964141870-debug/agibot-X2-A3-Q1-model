"""P1.5b step1 v5: pure matmul relative rotation mapping."""
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

H = "/opt/data/private/yuanqi_ops/data/humanplus_preview/session_20260901_144355/annotation.hdf5"
with h5py.File(H, "r") as f:
    rot = f["imu/body/rot"][:].astype(np.float64)
    acc = f["imu/body/acc"][:].astype(np.float64)
    kp = f["body_motion/body_keypoints"][:].astype(np.float64)
T = len(rot)

def smooth(x, w=15):
    k = np.ones(w) / w
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, x)

rel = np.einsum("tnji,tnjk->tnik", np.swapaxes(rot[:-1], -1, -2), rot[1:])   # (T-1,10,3,3)
rv = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(T - 1, 10, 3)
sig = smooth(rv)
print("SIG |w| p50/p95:", np.round(np.percentile(np.linalg.norm(sig, axis=-1), [50, 95]), 4))

IDX = {"pelvis": 0, "spine3": 9, "neck": 12,
       "l_hip": 1, "l_knee": 4, "l_ankle": 7,
       "r_hip": 2, "r_knee": 5, "r_ankle": 8,
       "l_shoulder": 16, "l_elbow": 18, "l_wrist": 20,
       "r_shoulder": 17, "r_elbow": 19, "r_wrist": 21}
G = kp
PAIRS = [("pelvis", "spine3", "torso"), ("spine3", "neck", "chest"),
         ("l_hip", "l_knee", "l_thigh"), ("l_knee", "l_ankle", "l_shin"),
         ("r_hip", "r_knee", "r_thigh"), ("r_knee", "r_ankle", "r_shin"),
         ("l_shoulder", "l_elbow", "l_uarm"), ("l_elbow", "l_wrist", "l_farm"),
         ("r_shoulder", "r_elbow", "r_uarm"), ("r_elbow", "r_wrist", "r_farm")]
bones = {}
for a, b, nm in PAIRS:
    v = G[:, IDX[b]] - G[:, IDX[a]]
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    bones[nm] = smooth(np.cross(v[:-1], v[1:]))
bl = list(bones)
N = min(sig.shape[0], min(len(x) for x in bones.values()))
cor = np.zeros((10, len(bl)))
for i in range(10):
    A = sig[:N, i] - sig[:N, i].mean(0)
    for j, nm in enumerate(bl):
        B = bones[nm][:N]
        B = B - B.mean(0)
        den = np.linalg.norm(A) * np.linalg.norm(B)
        cor[i, j] = float((A * B).sum() / den) if den > 1e-9 else 0.0
np.set_printoptions(precision=2, suppress=True, linewidth=180)
print("cols:", bl)
print(cor)
ri, cj = linear_sum_assignment(cor)
print("SIGNED", {int(i): bl[j] for i, j in zip(ri, cj)}, "mean %.3f" % cor[ri, cj].mean())
ri2, cj2 = linear_sum_assignment(-np.abs(cor))
print("ABS   ", {int(i): bl[j] for i, j in zip(ri2, cj2)}, "mean|%.3f|" % np.abs(cor[ri2, cj2]).mean())
