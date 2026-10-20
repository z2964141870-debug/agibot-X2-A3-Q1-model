"""p15c_train_v1.py — LIP-54 trial: session-level 75/25, dual loss reported separately."""
import sys, os, glob, json, time, math, random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/tools/FGP_Full/FGP-main")
import articulate as art
from Aplus.models import EasyLSTM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 17
WIN = 128
SLOT6 = [0, 1, 5, 6, 4, 9]
ROOT = 5  # index of slot9 within SLOT6
BONE2JOINT = [0, 9, 16, 18, 17, 19]  # waist->pelvis, back->spine3, uarm_l->l_shoulder, farm_l->l_elbow, uarm_r->r_shoulder, farm_r->r_elbow
# CH order in synth: back(0), waist(9), uarm_l(4), uarm_r(6), farm_l(5), farm_r(1) -> gt_rot channel order
GT2JOINT = [9, 0, 16, 17, 18, 19]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_sessions():
    fs = sorted(glob.glob("/opt/data/private/yuanqi_ops/p15c_cano/*.npz"))
    k = int(len(fs) * 0.75)
    return fs[:k], fs[k:]

class WinDS(torch.utils.data.Dataset):
    def __init__(self, files, win=WIN, per_session=96):
        self.items = []
        for p in files:
            z = np.load(p, allow_pickle=True)
            T = z["acc_d"].shape[0]
            if T < win + 2:
                continue
            self.items.append((p, T, per_session))
        self.win = win
    def __len__(self):
        return sum(c for _, _, c in self.items)
    def __getitem__(self, i):
        # sample session weighted by count
        j = random.randrange(len(self.items))
        p, T, _ = self.items[j]
        s = random.randrange(0, T - self.win - 1)
        z = np.load(p, allow_pickle=True)
        sl = slice(s, s + self.win)
        acc_d = z["acc_d"][sl][:, SLOT6].astype(np.float32)   # [win,6,3]
        rot_d = z["rot_d"][sl][:, SLOT6].astype(np.float32)   # [win,6,3,3]
        gt = z["gt_rot"][sl].astype(np.float32)               # [win,6,3,3] device-world
        # canonical GT: waist(device) frame -> bone-relative rotations
        Rw = torch.from_numpy(gt[:, 1])                        # waist bone device->world
        gt_canon = torch.einsum("tij,tnjk->tnik", Rw.transpose(1, 2), torch.from_numpy(gt))
        # pack x: root-relative acc + r6d of rot
        acc = torch.from_numpy(acc_d)
        rot = torch.from_numpy(rot_d)
        acc_rooted = torch.cat([acc[:, :ROOT] - acc[:, ROOT:ROOT + 1], acc[:, ROOT:ROOT + 1]], 1) @ rot[:, ROOT]
        rot_rooted = torch.cat([rot[:, ROOT].transpose(1, 2).unsqueeze(1).expand(-1, 5, -1, -1) @ rot[:, :ROOT], rot[:, ROOT:ROOT + 1]], 1)
        x = torch.cat([(acc_rooted / 30.0).reshape(self.win, -1),
                       art.math.rotation_matrix_to_r6d(rot_rooted.reshape(-1, 3, 3)).reshape(self.win, -1)], -1)
        return x.float(), gt_canon.float(), rot_rooted.float()

class Lip54(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = EasyLSTM(54, 512, 512, 2, dropout=0.2)
        self.head = nn.Linear(512, 24 * 6)
        self.calib = nn.Parameter(torch.eye(3).repeat(6, 1, 1))
    def forward(self, x, h=None):
        y = self.lstm(x)
        r6 = self.head(y).view(-1, 24, 6)
        return r6

def main():
    tr, ho = load_sessions()
    print("sessions train=%d holdout=%d dev=%s" % (len(tr), len(ho), DEV), flush=True)
    dtr = WinDS(tr); dho = WinDS(ho, per_session=24)
    dl = torch.utils.data.DataLoader(dtr, batch_size=32, shuffle=True, num_workers=2)
    net = Lip54().to(DEV)
    opt = torch.optim.Adam([
        {"params": net.lstm.parameters()}, {"params": net.head.parameters()},
        {"params": [net.calib], "lr": 3e-3}], lr=1e-3)
    W_SMOOTH = 0.01
    for ep in range(5):
        net.train(); tl = ts = n = 0; t0 = time.time()
        for x, gt, _ in dl:
            x, gt = x.to(DEV), gt.to(DEV)
            r6 = net(x)
            R = art.math.r6d_to_rotation_matrix(r6.reshape(-1, 24, 6)).reshape(x.shape[0], x.shape[1], 24, 3, 3)
            out = R[:, :, GT2JOINT]                   # [B,win,6,3,3]
            cal = torch.einsum("nip,btnpq->btniq", net.calib, gt)
            l_rot = (out - cal).pow.sum(-1).mean() if False else (out - cal).pow(2).sum((-1, -2)).mean()
            spd = (R[:, 1:] - R[:, :-1]).pow(2).sum((-1, -2)).mean()
            loss = l_rot + W_SMOOTH * spd
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(l_rot); ts += float(spd); n += 1
        # holdout angle error (deg)
        net.eval(); errs = []
        with torch.no_grad():
            for i in range(min(len(dho), 48)):
                x, gt, _ = dho[i * 7 % len(dho)]
                r6 = net(x.unsqueeze(0).to(DEV))
                R = art.math.r6d_to_rotation_matrix(r6.reshape(-1, 24, 6)).reshape(x.shape[0], 24, 3, 3)[:, GT2JOINT]
                cal = torch.einsum("nip,tnpq->tniq", net.calib, gt.to(DEV))
                ang = torch.acos((((R * cal).sum((-1, -2)) - 1) / 2).clamp(-1, 1))
                errs.append(float(ang.median()) * 180 / math.pi)
        print("EP%d l_rot=%.4f l_smooth=%.4f holdout_med_deg=%.1f (%.0fs)" % (
            ep, tl / max(n, 1), ts / max(n, 1), float(np.median(errs)), time.time() - t0), flush=True)
    torch.save(net.state_dict(), "/opt/data/private/yuanqi_ops/p15c_lip54_v1.pt")
    print("V1 DONE", flush=True)

if __name__ == "__main__":
    main()
