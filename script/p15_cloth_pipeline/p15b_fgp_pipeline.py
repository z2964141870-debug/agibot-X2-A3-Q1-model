"""P1.5b FGP full pipeline driver (weave-ready).
Chain: imu9[T,K,9] -> CLOTHO -> acc/rot[T,10] -> root-relative pack -> LIP(LSTM) -> pose_mat[T,24,3,3].
Modes: --selftest (random shapes) | --denoised acc.npy rot.npy | --imu9 x.npy --ids ...
"""
import sys, argparse, time
import numpy as np
import torch

FGP = "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main"
sys.path.insert(0, FGP)
import articulate as art
from neural_backbone import IMUCanonicalization
from Aplus.models import EasyLSTM
from PoseReconstruction.LIP_model import BiPoser
from cano_eval import run_imu_canonicalize_on_sequence2, project_to_so3

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CK = "/opt/data/private/yuanqi_ops/script/tools/FGP/checkpoints"
LIP_ACC_SCALE, J = 30.0, 24

def load_cano(size="B"):
    common = dict(cin_state=9, cin_cond=9, cout=9, max_T=512, t_embed_dim=256)
    m = IMUCanonicalization(pretrained_ae=None, depth=6, nhead=4, ffn=512, dropout=0.1,
                            enc_d_model=384, enc_layers=4, enc_nhead=8, **common).to(DEV)
    st = torch.load(CK + "/canonicalization/U100hL7.07h_residual.pth", map_location="cpu")
    m.load_state_dict(st["model"] if "model" in st else st, strict=True)
    return m.to(DEV).eval()

def load_lip():
    s1 = EasyLSTM(n_input=90, n_hidden=256, n_output=J * 3, n_lstm_layer=2,
                  bidirectional=False, output_type="seq", dropout=0.2).to(DEV)
    s2 = EasyLSTM(n_input=90 + J * 3, n_hidden=256, n_output=J * 6, n_lstm_layer=2,
                  bidirectional=False, output_type="seq", dropout=0.2).to(DEV)
    bp = BiPoser(net_s1=s1, net_s2=s2, export_mode=True).to(DEV)
    st = torch.load(CK + "/LIP/LIP_10_real_denoised_7.07_10imu_10.pth", map_location="cpu")
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    missing, unexpected = bp.load_state_dict(sd, strict=False)
    print("LIP load missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    return bp.eval()

@torch.no_grad()
def run_cano(acc, rmat, imu_ids, chunk_len=512):
    """acc[T,K,3] rmat[T,K,3,3] -> denoised acc_d[T,10,3] rot_d[T,10,3,3]."""
    r6 = art.math.rotation_matrix_to_r6d(rmat.reshape(-1, 3, 3)).reshape(*rmat.shape[:2], 6)
    x = torch.cat([acc, r6], dim=-1).float()
    out = run_imu_canonicalize_on_sequence2(cano, x.to(DEV), imu_ids, chunk_len, K_out=10)
    acc_d = out[..., :3].cpu()
    rot_d = project_to_so3(art.math.r6d_to_rotation_matrix(out[..., 3:].reshape(-1, 6))).reshape(len(out), 10, 3, 3).cpu()
    return acc_d, rot_d

@torch.no_grad()
def lip_batch(acc_d, rot_d, warm=64):
    """Stream LIP over T frames -> pose_mat[T,24,3,3] (numpy)."""
    T = len(acc_d)
    poses = np.zeros((T, J, 3, 3), np.float32)
    h1 = c1 = h2 = c2 = None
    for t in range(T):
        acc = acc_d[t].clamp(-60, 60)
        rot = art.math.normalize_rotation_matrix(rot_d[t].float())
        acc_rooted = torch.cat([acc[:9] - acc[9:10], acc[9:10]], 0) @ rot[9]
        rot_rooted = torch.cat([rot[9].T @ rot[:9], rot[9:10]], 0)
        x = torch.cat([(acc_rooted / LIP_ACC_SCALE).reshape(-1),
                       art.math.rotation_matrix_to_r6d(rot_rooted.reshape(-1, 3, 3)).reshape(-1)]).view(1, 1, -1).to(DEV)
        if t == 0:
            h0, c0 = s1.h_0.repeat(1, 1, 1), s1.c_0.repeat(1, 1, 1)
            o1, h1, c1 = lip.net_s1(x, h0, c0)
            h0, c0 = s2.h_0.repeat(1, 1, 1), s2.c_0.repeat(1, 1, 1)
            o2, h2, c2 = lip.net_s2(torch.cat([x, o1], -1), h0, c0)
        else:
            o1, h1, c1 = lip.net_s1(x, h1, c1)
            o2, h2, c2 = lip.net_s2(torch.cat([x, o1], -1), h2, c2)
        r6 = o2.view(-1, 6)
        poses[t] = art.math.r6d_to_rotation_matrix(r6).reshape(J, 3, 3).cpu().numpy()
    return poses

def physical_metrics(poses):
    rel = np.einsum("tnji,tnjk->tnik", np.swapaxes(poses[:-1], -1, -2), poses[1:])
    from scipy.spatial.transform import Rotation
    ang = Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude().reshape(poses.shape[0] - 1, poses.shape[1])
    jerk = float(np.abs(np.diff(ang, axis=0)).mean())
    return {"joint_speed_p50": float(np.median(ang)), "jerk_mean": jerk}


def art_rotvec(R):
    return torch.from_numpy(R).float() if isinstance(R, np.ndarray) else R
    # note: use scipy-free approximation below if art lacks this

cano = None; lip = None; s1 = None; s2 = None

def main():
    global cano, lip, s1, s2
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cano = load_cano(); print("CANO loaded %.1fs" % (time.time() - t0))
    lip = load_lip(); s1, s2 = lip.net_s1, lip.net_s2
    if a.selftest:
        T, K = 600, 4
        acc = torch.randn(T, K, 3)
        r = torch.tensor(np.eye(3, dtype=np.float32))
        rmat = r.repeat(T, K, 1, 1)
        acc_d, rot_d = run_cano(acc, rmat, imu_ids=[0, 1, 5, 6], chunk_len=256)
        print("CANO out", tuple(acc_d.shape), tuple(rot_d.shape))
        poses = lip_batch(acc_d, rot_d)
        print("LIP out", poses.shape, "finite", bool(np.isfinite(poses).all()))
        print("METRICS", physical_metrics(poses))
    print("DONE %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
