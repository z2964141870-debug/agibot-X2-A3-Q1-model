"""p15c_train_lip54.py — LIP-54 variant training entry (skeleton, E55-preemption aware).
n_input=54 = 6 ch x (acc3 + r6d6); rest mirrors LIP BiPoser export contract."""
import sys, os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/tools/FGP_Full/FGP-main")
import articulate as art
from Aplus.models import EasyLSTM
from PoseReconstruction.LIP_model import BiPoser

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SLOT6 = [0, 1, 5, 6, 4, 9]          # purple_6 slots for our 6 channels
ROOT_SLOT = 9                        # waist/root (3-config consensus)

def pack_input(acc_d, rot_d):
    """[T,10,3],[T,10,3,3] canonical -> x[T,54]: root-relative acc(6x3) + r6d(6x6), same packing as LIP-90 minus legs."""
    acc = acc_d[:, SLOT6].clamp(-60, 60)
    rot = art.math.normalize_rotation_matrix(rot_d[:, SLOT6].float())
    acc_rooted = torch.cat([acc[:, :5] - acc[:, 5:6], acc[:, 5:6]], 1) @ rot[:, 5]
    rot_rooted = torch.cat([rot[:, 5].transpose(1, 2).unsqueeze(1).repeat(1, 5, 1, 1) @ rot[:, :5], rot[:, 5:6]], 1)
    x = torch.cat([(acc_rooted / 30.0).reshape(len(acc_d), -1),
                   art.math.rotation_matrix_to_r6d(rot_rooted.reshape(-1, 3, 3)).reshape(len(acc_d), -1)], -1)
    return x  # [T, 15+36=51]... NOTE: 5 arms + root = 6 ch -> acc 6x3=18, r6d 6x6=36 -> 54

class Lip54Net(nn.Module):
    """EasyLSTM(n_input=54) -> joint r6d heads; BiPoser-compatible export."""
    def __init__(self, hidden=512, layers=2):
        super().__init__()
        self.lstm = EasyLSTM(54, hidden, layers, dropout=0.2)
        self.head = nn.Linear(hidden, 24 * 6)  # SMPL 24 joints r6d (joint_name_list order)

    def forward(self, x, h=None):
        y, h = self.lstm(x, h)
        return self.head(y), h

def build_loader(npz_list, win=128, stride=64):
    """Synthetic npz windows -> (x[T,54], gt_rot[T,24,3,3] placeholder)."""
    for p in npz_list:
        z = np.load(p, allow_pickle=True)
        A = z["acc"].transpose(1, 0, 2); Rm = z["rot"].transpose(1, 0, 2, 3)
        # cano at pack time in the full pipeline; skeleton yields raw
        yield A, Rm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", default="/opt/data/private/yuanqi_ops/p15c_data/*.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--holdout_frac", type=float, default=0.25)
    ap.add_argument("--out", default="/opt/data/private/yuanqi_ops/data/experiments/p15c_lip54")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "E55_PREEMPTION.txt"), "w").write(
        "E55 amendment approval => immediately yield this slot; P1.5c unconditionally pauses.\n")
    print("p15c LIP-54 skeleton ready; holdout=%.2f; device=%s" % (a.holdout_frac, DEV))
    print("pack check:", pack_input(torch.randn(8, 10, 3), art.math.r6d_to_rotation_matrix(
        torch.randn(8, 10, 6)).clamp(-1, 1)).shape)

if __name__ == "__main__":
    main()
