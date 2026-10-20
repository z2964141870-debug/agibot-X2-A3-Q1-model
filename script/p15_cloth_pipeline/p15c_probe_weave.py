"""p15c_probe_weave.py — gate-2 recon: v1 Lip54 on weave REAL imu, DIAGNOSTIC metrics. (DO NOT RUN before lead approves option B.)"""
import sys, os, json, math
import numpy as np
import torch
sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/tools/FGP_Full/FGP-main")
import articulate as art
import p15b_fgp_pipeline as P
from cano_eval import run_imu_canonicalize_on_sequence2

SLOT6 = [0, 1, 5, 6, 4, 9]
ROOT = 5
GT2JOINT = [9, 0, 16, 17, 18, 19]
STG = "/opt/data/private/yuanqi_ops/data/smb_weave_staging/session_20261018_152655"

def pack6(acc_d, rot_d):
    print('pack6 in:', tuple(acc_d.shape), tuple(rot_d.shape), flush=True)
    T = len(acc_d)
    acc = acc_d[:, SLOT6].clamp(-60, 60)
    rot = art.math.normalize_rotation_matrix(rot_d[:, SLOT6].float()).reshape(T, 6, 3, 3)
    acc_rooted = torch.cat([acc[:, :ROOT] - acc[:, ROOT:ROOT + 1], acc[:, ROOT:ROOT + 1]], 1) @ rot[:, ROOT]
    rot_rooted = torch.cat([rot[:, ROOT].transpose(1, 2).unsqueeze(1).expand(-1, 5, -1, -1) @ rot[:, :ROOT], rot[:, ROOT:ROOT + 1]], 1)
    x = torch.cat([(acc_rooted / 30.0).reshape(T, -1),
                   art.math.rotation_matrix_to_r6d(rot_rooted.reshape(-1, 3, 3)).reshape(T, -1)], -1)
    return x.float(), rot_rooted

def main():
    src = __import__("numpy").load(STG + "/weave10_grid.npz")
    acc_t = torch.from_numpy(src["acc"]).float()
    rot_t = torch.from_numpy(src["rot"]).float()
    P.cano = P.load_cano()
    r6 = art.math.rotation_matrix_to_r6d(rot_t.reshape(-1, 3, 3)).reshape(len(rot_t), 10, 6)
    x10 = torch.cat([acc_t, r6.float()], -1).to(P.DEV)
    with torch.no_grad():
        out = run_imu_canonicalize_on_sequence2(P.cano, x10, list(range(10)), 512, K_out=10)
    acc_d = out[..., :3].cpu()
    rot_d = P.project_to_so3(art.math.r6d_to_rotation_matrix(out[..., 3:].reshape(-1, 6))).reshape(len(out), 10, 3, 3).cpu()
    x6, rr6 = pack6(acc_d, rot_d)
    net = P.__dict__  # placeholder; real load below
    src2 = open("/opt/data/private/yuanqi_ops/p15c_train_v1.py").read().replace('if __name__ == "__main__":\n    main()', "")
    g = {}
    exec(compile(src2, "tv1", "exec"), g)
    net = g["Lip54"]()
    net.load_state_dict(torch.load("/opt/data/private/yuanqi_ops/p15c_lip54_v1.pt", map_location=P.DEV))
    net = net.to(P.DEV)
    net.eval()
    poses = []
    with torch.no_grad():
        for s in range(0, len(x6), 1024):
            r6o = net(x6[s:s + 1024].unsqueeze(0).to(P.DEV))
            R = art.math.r6d_to_rotation_matrix(r6o.reshape(-1, 24, 6)).reshape(1, -1, 24, 3, 3)[0]
            poses.append(R.cpu())
    R_all = torch.cat(poses)  # [T,24,3,3]
    met = P.physical_metrics(R_all.numpy())
    print("WEAVE-PROBE v1 real6:", json.dumps({k: round(float(v), 4) for k, v in met.items()}))
    print("BASELINES all=0.480/0.037 real6=0.445/0.046 neutral4=0.680/0.052 (joint_speed_p50/jerk_mean)")

if __name__ == "__main__":
    main()
