"""p15c_probe_v3.py — final dual-cut physical acceptance with v3 model."""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/tools/FGP_Full/FGP-main")
import articulate as art
import p15b_fgp_pipeline as P
from cano_eval import run_imu_canonicalize_on_sequence2

STG = "/opt/data/private/yuanqi_ops/data/smb_weave_staging/session_20261018_152655"
SLOT6 = [0, 1, 5, 6, 4, 9]
ROOT = 5
src = np.load(STG + "/weave10_grid.npz")
acc, rot = src["acc"], src["rot"]
T = len(acc)
stds = acc.std(axis=(0, 2))
print("SLOT_HEALTH accstd:", {i: round(float(stds[i]), 3) for i in range(10)}, flush=True)

def pack6(acc_d, rot_d, healthy=False):
    sl = list(SLOT6)
    if healthy:
        sl = [1 if s == 0 else s for s in sl]
    Tn = len(acc_d)
    a = acc_d[:, sl].clamp(-60, 60)
    r = art.math.normalize_rotation_matrix(rot_d[:, sl].float()).reshape(Tn, 6, 3, 3)
    ar = torch.cat([a[:, :ROOT] - a[:, ROOT:ROOT + 1], a[:, ROOT:ROOT + 1]], 1) @ r[:, ROOT]
    rr = torch.cat([r[:, ROOT].transpose(1, 2).unsqueeze(1).expand(-1, 5, -1, -1) @ r[:, :ROOT], r[:, ROOT:ROOT + 1]], 1)
    x = torch.cat([(ar / 30.0).reshape(Tn, -1),
                   art.math.rotation_matrix_to_r6d(rr.reshape(-1, 3, 3)).reshape(Tn, -1)], -1)
    return x.float()

src2 = open("/opt/data/private/yuanqi_ops/p15c_train_v1.py").read().replace('if __name__ == "__main__":\n    main()', "")
g = {}
exec(compile(src2, "tv1", "exec"), g)
net = g["Lip54"]().to(P.DEV)
net.load_state_dict(torch.load("/opt/data/private/yuanqi_ops/p15c_lip54_v3.pt", map_location=P.DEV))
net.eval()
P.cano = P.load_cano()
a_t = torch.from_numpy(acc).float(); r_t = torch.from_numpy(rot).float()
r6 = art.math.rotation_matrix_to_r6d(r_t.reshape(-1, 3, 3)).reshape(T, 10, 6)
x10 = torch.cat([a_t, r6.float()], -1).to(P.DEV)
with torch.no_grad():
    out = run_imu_canonicalize_on_sequence2(P.cano, x10, list(range(10)), 512, K_out=10)
acc_d = out[..., :3].cpu()
rot_d = P.project_to_so3(art.math.r6d_to_rotation_matrix(out[..., 3:].reshape(-1, 6))).reshape(T, 10, 3, 3).cpu()
for tag, healthy in [("FULL", False), ("HEALTHY(slot0->1)", True)]:
    x6 = pack6(acc_d, rot_d, healthy)
    poses = []
    with torch.no_grad():
        for s in range(0, len(x6), 1024):
            rr = net(x6[s:s + 1024].unsqueeze(0).to(P.DEV))
            poses.append(art.math.r6d_to_rotation_matrix(rr.reshape(-1, 24, 6)).reshape(-1, 24, 3, 3).cpu())
    met = P.physical_metrics(torch.cat(poses).numpy())
    print("%s v3: %s" % (tag, json.dumps({k: round(float(v), 4) for k, v in met.items()})), flush=True)
print("GATES full-vs-neutral4=0.680/0.052 healthy-vs-real6=0.445/0.046")
