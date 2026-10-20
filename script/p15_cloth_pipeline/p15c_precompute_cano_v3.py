"""p15c precompute: per-session CLOTHO canonicalize cache -> p15c_cano/*.npz (float16)."""
import sys, os, glob, json, time, traceback
import numpy as np
import torch
sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/tools/FGP_Full/FGP-main")
import articulate as art
import p15b_fgp_pipeline as P
from cano_eval import run_imu_canonicalize_on_sequence2

OUT = "/opt/data/private/yuanqi_ops/p15c_cano_v3"
os.makedirs(OUT, exist_ok=True)
P.cano = P.load_cano()
fs = sorted(glob.glob("/opt/data/private/yuanqi_ops/p15c_data_v3/*.npz"))
print("sessions:", len(fs), flush=True)
t0 = time.time()
for i, p in enumerate(fs):
    sname = os.path.basename(p)[:-4]
    out = os.path.join(OUT, sname + ".npz")
    if os.path.exists(out):
        continue
    try:
        z = np.load(p, allow_pickle=True)
        T = z["acc"].shape[1]
        A = z["acc"].transpose(1, 0, 2).copy()
        Rm = z["rot"].transpose(1, 0, 2, 3).copy()
        r6 = art.math.rotation_matrix_to_r6d(torch.from_numpy(Rm).float().reshape(-1, 3, 3)).reshape(T, 6, 6)
        x = torch.cat([torch.from_numpy(A).float(), r6.float()], -1).to(P.DEV)
        with torch.no_grad():
            o = run_imu_canonicalize_on_sequence2(P.cano, x, [0, 1, 5, 6, 4, 9], 512, K_out=10)
        acc_d = o[..., :3].cpu().numpy().astype(np.float16)
        rot_d = P.project_to_so3(art.math.r6d_to_rotation_matrix(o[..., 3:].reshape(-1, 6))).reshape(T, 10, 3, 3).cpu().numpy().astype(np.float16)
        gt_rot = Rm.astype(np.float16)  # [T,6,3,3] bone GT (device world)
        np.savez_compressed(out, acc_d=acc_d, rot_d=rot_d, gt_rot=gt_rot, slots=json.dumps([0, 1, 5, 6, 4, 9]))
        if i % 8 == 0 or i == len(fs) - 1:
            print("[%3.0fs] %d/%d %s T=%d" % (time.time() - t0, i + 1, len(fs), sname, T), flush=True)
    except Exception as e:
        traceback.print_exc()
        print("ERR %s %s" % (sname, type(e).__name__), flush=True)
print("CANO DONE", flush=True)
