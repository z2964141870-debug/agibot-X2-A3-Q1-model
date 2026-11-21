"""Stage A v3: Decoupled L<->R-canonicalized side landing reflex (0..11) + settled roll (12..149)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.distributions import Normal

RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
for p in (str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from command_support import sample_record
from common import EFF, HI, KD, KP, LO, Plant, SCALE, STAND

PERM_22 = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 18, 19, 20, 21, 14, 15, 16, 17]
SIGN_22 = [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1]


def mirror_obs_74(obs: torch.Tensor, perm_t: torch.Tensor, sign_t: torch.Tensor) -> torch.Tensor:
    """Exact XZ-plane L<->R mirror for (..., 74) recovery observations (ONNX-clean index_select)."""
    q = obs[..., :22].index_select(-1, perm_t) * sign_t
    qd = obs[..., 22:44].index_select(-1, perm_t) * sign_t
    w = obs[..., 44:47] * obs.new_tensor([-1.0, 1.0, -1.0])
    up = obs[..., 47:50] * obs.new_tensor([1.0, -1.0, 1.0])
    prev = obs[..., 50:72].index_select(-1, perm_t) * sign_t
    hv = obs[..., 72:74]
    return torch.cat((q, qd, w, up, prev, hv), dim=-1)


def canonicalize_obs(obs: torch.Tensor, side: torch.Tensor, perm_t: torch.Tensor, sign_t: torch.Tensor) -> torch.Tensor:
    obs_m = mirror_obs_74(obs, perm_t, sign_t)
    mask = (side > 0.5).unsqueeze(-1)
    if obs.ndim == 3:
        mask = mask.unsqueeze(1)
    return torch.where(mask, obs, obs_m)


def unmirror_action(target_canon: torch.Tensor, side: torch.Tensor, perm_t: torch.Tensor, sign_t: torch.Tensor) -> torch.Tensor:
    t_m = target_canon.index_select(-1, perm_t) * sign_t
    return torch.where((side > 0.5)[:, None], target_canon, t_m)


def symmetrize_goal(faceup_cmd0: np.ndarray) -> np.ndarray:
    g = np.array(faceup_cmd0, dtype=np.float32).copy()
    g_sym = 0.5 * (g + g[PERM_22] * np.array(SIGN_22, dtype=np.float32))
    # Keep arms wide and clear of hips at supine arrival to prevent elbow2 vs hip1 brush.
    g_sym[14:18] = [-0.15, 0.65, 0.0, 0.45]
    g_sym[18:22] = [-0.15, -0.65, 0.0, 0.45]
    return np.clip(g_sym, LO, HI).astype(np.float32)


class SideLandingNet(nn.Module):
    """Module 1: L<->R-canonicalized 240 ms (ticks 0..11) landing impact & clearance reflex."""

    def __init__(self):
        super().__init__()
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(LO, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(HI, dtype=torch.float32))
        self.register_buffer("perm_22", torch.tensor(PERM_22, dtype=torch.long))
        self.register_buffer("sign_22", torch.tensor(SIGN_22, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(75, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 22),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, obs_last: torch.Tensor, age: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        obs_c = canonicalize_obs(obs_last, side, self.perm_22, self.sign_22)
        return torch.cat((obs_c.clamp(-12.0, 12.0), (age / 12.0).clamp(0.0, 1.0)[:, None]), dim=1)

    def raw_mean(self, obs_last: torch.Tensor, age: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(obs_last, age, side))

    def target_from_action(
        self, raw: torch.Tensor, obs_last: torch.Tensor, age: torch.Tensor, side: torch.Tensor
    ) -> torch.Tensor:
        obs_c = canonicalize_obs(obs_last, side, self.perm_22, self.sign_22)
        q_c = obs_c[:, :22] + self.stand
        qd_c = obs_c[:, 22:44] * 10.0
        target_c = (q_c + 1.15 * raw.tanh() - 0.03 * qd_c).clamp(self.lo, self.hi)
        return unmirror_action(target_c, side, self.perm_22, self.sign_22)

    def forward(self, obs_last: torch.Tensor, age: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        return self.target_from_action(self.raw_mean(obs_last, age, side), obs_last, age, side)


class SideSettledRollNet(nn.Module):
    """Module 2: L<->R-canonicalized settled side-to-supine roll (ticks 12..149)."""

    def __init__(self, faceup_cmd0: np.ndarray):
        super().__init__()
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(LO, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(HI, dtype=torch.float32))
        self.register_buffer("perm_22", torch.tensor(PERM_22, dtype=torch.long))
        self.register_buffer("sign_22", torch.tensor(SIGN_22, dtype=torch.float32))
        self.register_buffer("goal_cmd", torch.tensor(symmetrize_goal(faceup_cmd0), dtype=torch.float32))

        self.encoder = nn.Sequential(nn.Linear(74, 64), nn.ELU())
        self.actor = nn.Sequential(
            nn.Linear(257, 192),
            nn.ELU(),
            nn.Linear(192, 128),
            nn.ELU(),
            nn.Linear(128, 22),
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def features(self, rec_obs_hist: torch.Tensor, roll_age: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        hist4_c = canonicalize_obs(rec_obs_hist[:, -4:, :], side, self.perm_22, self.sign_22)
        enc = self.encoder(hist4_c.clamp(-12.0, 12.0)).flatten(1)
        return torch.cat((enc, (roll_age / 130.0).clamp(0.0, 1.0)[:, None]), dim=1)

    def raw_mean(self, rec_obs_hist: torch.Tensor, roll_age: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        return self.actor(self.features(rec_obs_hist, roll_age, side))

    def target_from_action(
        self,
        raw: torch.Tensor,
        rec_obs_hist: torch.Tensor,
        roll_age: torch.Tensor,
        side: torch.Tensor,
        origin_world: torch.Tensor,
    ) -> torch.Tensor:
        obs_c = canonicalize_obs(rec_obs_hist[:, -1, :], side, self.perm_22, self.sign_22)
        qd_c = obs_c[:, 22:44] * 10.0
        origin_m = origin_world.index_select(-1, self.perm_22) * self.sign_22
        origin_c = torch.where((side > 0.5)[:, None], origin_world, origin_m)

        phase = ((roll_age + 1.0) / 90.0).clamp(0.0, 1.0)
        w_goal = 0.5 - 0.5 * torch.cos(torch.pi * phase)
        base_c = origin_c + w_goal[:, None] * (self.goal_cmd[None] - origin_c)
        scale_roll = 0.75 * (roll_age / 5.0).clamp(0.0, 1.0)
        target_c = (base_c + scale_roll[:, None] * raw.tanh() - 0.02 * qd_c).clamp(self.lo, self.hi)
        return unmirror_action(target_c, side, self.perm_22, self.sign_22)


class SideV3Combined(nn.Module):
    """Combined Stage A v3 ONNX-clean side-to-supine module (ticks 0..11 landing + 12+ roll)."""

    def __init__(self, faceup_cmd0: np.ndarray):
        super().__init__()
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(LO, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(HI, dtype=torch.float32))
        self.landing = SideLandingNet()
        self.roll = SideSettledRollNet(faceup_cmd0)

    def step_components(
        self,
        rec_obs_hist: torch.Tensor,
        age: torch.Tensor,
        side: torch.Tensor,
        origin_tk12: torch.Tensor,
    ):
        t_land = self.landing(rec_obs_hist[:, -1, :], age, side)
        roll_age = (age - 12.0).clamp(min=0.0)
        t_roll = self.roll.target_from_action(
            self.roll.raw_mean(rec_obs_hist, roll_age, side),
            rec_obs_hist,
            roll_age,
            side,
            origin_tk12,
        )
        return torch.where((age < 12.0)[:, None], t_land, t_roll).clamp(self.lo, self.hi)

    def forward(self, rec_obs_hist: torch.Tensor, rec_state: torch.Tensor):
        x = rec_obs_hist[:, -1]
        q = x[:, :22] + self.stand
        begin = (rec_state[:, 0] == 0) & (rec_state[:, 1] != 10)
        age = torch.where(begin, torch.zeros_like(rec_state[:, 0]), rec_state[:, 0])
        side = torch.where(begin, (x[:, 48] < 0).float(), rec_state[:, 2]).clamp(0.0, 1.0)
        # Latch origin_tk12 right when age transitions to 12.0.
        origin = torch.where((begin | (age == 12.0))[:, None], q, rec_state[:, 4:])
        target = self.step_components(rec_obs_hist, age, side, origin)

        w_norm = torch.linalg.vector_norm(x[:, 44:47] * 4.0, dim=1)
        quiet_now = (-x[:, 47] > 0.84) & (w_norm < 1.45)
        quiet_cnt = torch.where(
            quiet_now, torch.where(begin, torch.zeros_like(rec_state[:, 3]), rec_state[:, 3]) + 1.0, 0.0
        )
        next_state = torch.cat((torch.stack((age + 1.0, age * 0.0 + 10.0, side, quiet_cnt), 1), origin), 1)
        return target, next_state


def _update_sent_500hz(obs_row, target_row, sent_row, kp32, kd32, eff32, stand32, lo32, hi32, margin):
    q_o = obs_row[:22].astype(np.float32) + stand32
    qd_o = obs_row[22:44].astype(np.float32) * 10.0
    raw = np.clip(target_row.astype(np.float32), lo32 - margin, hi32 + margin)
    tau = np.clip(kp32 * (raw - q_o) - kd32 * qd_o, -eff32, eff32)
    des = np.clip(q_o + (tau + kd32 * qd_o) / kp32, stand32 - 2.0, stand32 + 2.0)
    for _ in range(10):
        sent_row = sent_row + np.clip(des - sent_row, -0.2, 0.2)
    return sent_row


def save_curve(out: Path, filename: str, csv_name: str, title: str, rows: list[dict], val_rows: list[dict], keys_spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    x = np.array([r["steps"] / 1e6 for r in rows])
    for ax, (k, ttl) in zip(axes.flat, keys_spec):
        if k.startswith("val:") and val_rows:
            vx = np.array([r["steps"] / 1e6 for r in val_rows])
            if "soft_12" in val_rows[0]:
                ax.plot(vx, [r["soft_12"] for r in val_rows], "s-", label="soft_12/32")
                ax.axhline(28, color="green", ls="--", lw=1)
            else:
                ax.plot(vx, [r["supine"] for r in val_rows], "o--", label="supine/32")
                ax.plot(vx, [r["simultaneous_soft"] for r in val_rows], "s-", label="e2e_soft/32")
                ax.axhline(28, color="green", ls="--", lw=1)
            ax.legend(fontsize=8)
        elif k in rows[0]:
            y = np.array([r[k] for r in rows])
            ax.plot(x, y, lw=0.7, alpha=0.6)
            if len(y) >= 20:
                sm = np.convolve(y, np.ones(20) / 20, mode="valid")
                ax.plot(x[19:], sm, lw=2.0)
        ax.set_title(ttl)
        ax.set_xlabel("Million 20 ms steps")
        ax.grid(alpha=0.25)
    for ax, v in ((axes[0, 2], 12.0), (axes[1, 0], 10.0), (axes[1, 1], 2.0)):
        ax.axhline(v, color="black", ls="--", lw=1)
    fig.suptitle(title)
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)

    with (out / csv_name).open("w", newline="") as fp:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def eval_landing(env: Plant, model: SideLandingNet, val_rows: np.ndarray, device: torch.device):
    n = len(val_rows)
    assert n <= env.n
    pad = np.tile(val_rows[0], (env.n, 1))
    pad[:n] = val_rows
    obs = env.reset(pad)
    side = torch.from_numpy((obs[:, 48] < 0).astype(np.float32)).to(device)
    pks = np.zeros((n, 3), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for tk in range(12):
            o_t = torch.from_numpy(obs.astype(np.float32)).to(device)
            ag_t = torch.full((env.n,), float(tk), dtype=torch.float32, device=device)
            tgt = model(o_t, ag_t, side).cpu().numpy()
            obs, met = env.step(tgt)
            pks = np.maximum(pks, met[:n, [4, 5, 6]])
    model.train()
    up_x = -obs[:n, 47]
    soft_ok = (pks[:, 0] <= 12.0) & (pks[:, 1] <= 10.0) & (pks[:, 2] <= 2.0) & (up_x > -0.60)
    viol = np.maximum(pks / np.array([12.0, 10.0, 2.0]) - 1.0, 0.0)
    return dict(
        soft_12=int(soft_ok.sum()),
        soft_only=int(((pks[:, 0] <= 12.0) & (pks[:, 1] <= 10.0) & (pks[:, 2] <= 2.0)).sum()),
        max_qd=float(pks[:, 0].max()),
        max_ground_mm=float(pks[:, 1].max()),
        max_self_mm=float(pks[:, 2].max()),
        max_rel_violation=float(viol.max()),
        bad_self=[int(j) for j in np.flatnonzero(pks[:, 2] > 2.0)],
        bad_qd=[int(j) for j in np.flatnonzero(pks[:, 0] > 12.0)],
        facedown=[int(j) for j in np.flatnonzero(up_x <= -0.60)],
    )


def train_phase_a1(
    out: Path,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    device: torch.device,
    seed: int,
    updates: int,
    lr: float = 3e-4,
    resume_ckpt: dict | None = None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = Plant(64, out / "scene_a1", delay=11)
    val_env = Plant(max(32, len(val_rows)), out / "val_scene_a1", delay=11)

    model = SideLandingNet().to(device)
    critic = nn.Sequential(nn.Linear(88, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    logstd = nn.Parameter(torch.full((22,), -1.6, device=device))
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["landing"], strict=True)
        if "critic_a1" in resume_ckpt:
            critic.load_state_dict(resume_ckpt["critic_a1"], strict=True)
        if "logstd_a1" in resume_ckpt:
            logstd.data.copy_(resume_ckpt["logstd_a1"].to(device).clamp(min=-2.3))
        elif "logstd" in resume_ckpt:
            logstd.data.copy_(resume_ckpt["logstd"].to(device).clamp(min=-2.3))
        else:
            logstd.data.fill_(-2.1)
    optimizer = torch.optim.Adam([*model.parameters(), *critic.parameters(), logstd], lr=lr)

    def refresh_hard_pool(cur_model: SideLandingNet, tag: str) -> np.ndarray:
        was_tr = cur_model.training
        cur_model.eval()
        scores = []
        with torch.no_grad():
            for s0 in range(0, len(train_rows), val_env.n):
                b = train_rows[s0 : s0 + val_env.n]
                m = len(b)
                pad = np.tile(b[0], (val_env.n, 1))
                pad[:m] = b
                o_b = val_env.reset(pad)
                sd_b = torch.from_numpy((o_b[:, 48] < 0).astype(np.float32)).to(device)
                pks_b = np.zeros((m, 3), dtype=np.float64)
                for tk in range(12):
                    ag_b = torch.full((val_env.n,), float(tk), dtype=torch.float32, device=device)
                    tgt = cur_model(torch.from_numpy(o_b.astype(np.float32)).to(device), ag_b, sd_b).cpu().numpy()
                    o_b, met_b = val_env.step(tgt)
                    pks_b = np.maximum(pks_b, met_b[:m, [4, 5, 6]])
                scores.append(
                    pks_b[:, 2]
                    + 0.5 * np.maximum(0.0, pks_b[:, 0] - 11.2)
                    + 0.5 * np.maximum(0.0, pks_b[:, 1] - 9.2)
                )
        if was_tr:
            cur_model.train()
        diff_arr = np.concatenate(scores, axis=0)
        hp = np.argsort(diff_arr)[-64:]
        print(
            f"[A1 {tag}] Hard-pool top-64/{len(train_rows)} difficulty range: [{diff_arr[hp[0]]:.2f}, {diff_arr[hp[-1]]:.2f}] (gt2={int((diff_arr>2).sum())}/{len(train_rows)})",
            flush=True,
        )
        return hp

    metric_scale = torch.tensor([1, 1, 0.01, 1, 0.1, 0.1, 0.1, 1, 0.2, 1, 1, 1, 1], dtype=torch.float32, device=device)
    train_log, val_log = [], []
    v0 = eval_landing(val_env, model, val_rows, device)
    v0["update"] = 0
    v0["steps"] = 0
    val_log.append(v0)
    with (out / "val_a1.jsonl").open("a") as fp:
        fp.write(json.dumps(v0) + "\n")
    best_score = (v0["soft_12"], -v0["max_rel_violation"])
    torch.save(
        dict(landing=model.state_dict(), critic_a1=critic.state_dict(), logstd=logstd.detach(), update=0, val=v0),
        out / "best_landing.pt",
    )
    print(
        f"[A1 upd=  0/{updates}] baseline | VAL soft12={v0['soft_12']}/32 peaks=({v0['max_qd']:.2f},{v0['max_ground_mm']:.2f},{v0['max_self_mm']:.2f}) bad_s={v0['bad_self']} bad_qd={v0['bad_qd']} fd={v0['facedown']}",
        flush=True,
    )
    if updates <= 0:
        env.close()
        val_env.close()
        return torch.load(out / "best_landing.pt", map_location="cpu", weights_only=False)
    hard_pool = refresh_hard_pool(model, "init")
    start_t = time.monotonic()

    for update in range(1, updates + 1):
        if update in (80, 160, 240):
            hard_pool = refresh_hard_pool(model, f"upd{update}")
        use_hard = rng.random(64) < 0.70
        idxs = np.where(use_hard, rng.choice(hard_pool, size=64, replace=True), rng.integers(len(train_rows), size=64))
        batch = train_rows[idxs].copy()
        noise = rng.random(64) < 0.30
        if noise.any():
            batch[noise, 7:29] = np.clip(batch[noise, 7:29] + rng.uniform(-0.025, 0.025, (noise.sum(), 22)), LO, HI)
            batch[noise, 57:] = batch[noise, 7:29]

        obs_np = env.reset(batch)
        side_t = torch.from_numpy((obs_np[:, 48] < 0).astype(np.float32)).to(device)
        ep_peaks = np.zeros((64, 3), dtype=np.float64)

        obs_l, age_l, sd_l, cf_l, act_l, lp_l, val_l, rew_l, done_l = [], [], [], [], [], [], [], [], []
        batch_stats = []
        with torch.no_grad():
            for tk in range(12):
                o_t = torch.from_numpy(obs_np.astype(np.float32)).to(device)
                ag_t = torch.full((64,), float(tk), dtype=torch.float32, device=device)
                af = model.features(o_t, ag_t, side_t)
                priv = torch.from_numpy(env.metrics.astype(np.float32)).to(device) * metric_scale
                cf = torch.cat((af, priv), dim=1)
                mean = model.raw_mean(o_t, ag_t, side_t)
                dist = Normal(mean, logstd.clamp(-3.0, -0.4).exp())
                act = dist.sample()
                tgt = model.target_from_action(act, o_t, ag_t, side_t)

                obs_l.append(o_t)
                age_l.append(ag_t)
                sd_l.append(side_t)
                cf_l.append(cf)
                act_l.append(act)
                lp_l.append(dist.log_prob(act).sum(1))
                val_l.append(critic(cf).squeeze(1))

                obs_np, met = env.step(tgt.cpu().numpy())
                ep_peaks = np.maximum(ep_peaks, met[:, [4, 5, 6]])
                up_x = -obs_np[:, 47]

                self_pen = (
                    12.0 * np.maximum(met[:, 6] - 0.10, 0.0) ** 2
                    + 20.0 * np.maximum(met[:, 6] - 0.50, 0.0)
                    + 35.0 * (met[:, 6] > 1.3)
                )
                r = (
                    -self_pen
                    - 0.5 * np.maximum(met[:, 4] - 9.5, 0.0) ** 2
                    - 4.0 * np.maximum(met[:, 4] - 11.0, 0.0)
                    - 15.0 * (met[:, 4] > 11.8)
                    - 0.6 * np.maximum(met[:, 5] - 7.5, 0.0) ** 2
                    - 4.0 * np.maximum(met[:, 5] - 9.0, 0.0)
                    - 15.0 * (met[:, 5] > 9.8)
                    + 1.0 * up_x
                    - 3.0 * np.maximum(0.0, -up_x - 0.35)
                )
                is_end = tk == 11
                if is_end:
                    clean_ep = (
                        (ep_peaks[:, 0] <= 11.8)
                        & (ep_peaks[:, 1] <= 9.6)
                        & (ep_peaks[:, 2] <= 1.50)
                        & (up_x > -0.45)
                    )
                    r += np.where(clean_ep, 25.0 + 5.0 * (up_x > 0.80), 0.0)
                r = np.clip(r, -120.0, 30.0)
                rew_l.append(torch.tensor(r * 0.1, dtype=torch.float32, device=device))
                done_l.append(torch.full((64,), is_end, dtype=torch.bool, device=device))
                batch_stats.append([float(r.mean()), float(met[:, 4].max()), float(met[:, 5].max()), float(met[:, 6].max())])

            gae = torch.zeros(64, dtype=torch.float32, device=device)
            advs = []
            following = torch.zeros(64, dtype=torch.float32, device=device)
            for t in reversed(range(12)):
                alive = (~done_l[t]).float()
                gae = rew_l[t] + 0.99 * alive * following - val_l[t] + 0.99 * 0.95 * alive * gae
                advs.append(gae)
                following = val_l[t]
            advantage = torch.cat(advs[::-1])
            returns = advantage + torch.cat(val_l)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        o_b, ag_b, sd_b, cf_b, act_b, lp_b = map(torch.cat, (obs_l, age_l, sd_l, cf_l, act_l, lp_l))
        lr_now = lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * update / updates)))
        for pg_opt in optimizer.param_groups:
            pg_opt["lr"] = lr_now
        pg_w = 0.0 if (resume_ckpt is not None and "critic_a1" not in resume_ckpt and update <= 4) else 1.0
        for _ in range(4):
            for ix in torch.randperm(len(act_b), device=device).split(256):
                dist = Normal(model.raw_mean(o_b[ix], ag_b[ix], sd_b[ix]), logstd.clamp(-3.0, -0.4).exp())
                ratio = (dist.log_prob(act_b[ix]).sum(1) - lp_b[ix]).exp()
                pg = torch.minimum(ratio * advantage[ix], ratio.clamp(0.8, 1.2) * advantage[ix])
                loss = -pg_w * pg.mean() + 0.3 * (critic(cf_b[ix]).squeeze(1) - returns[ix]).square().mean() - pg_w * 0.001 * dist.entropy().sum(1).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()

        bs = np.array(batch_stats)
        row = dict(
            update=update,
            steps=update * 64 * 12,
            elapsed_s=round(time.monotonic() - start_t, 2),
            reward=float(bs[:, 0].mean()),
            peak_qd=float(bs[:, 1].max()),
            ground_mm=float(bs[:, 2].max()),
            self_mm=float(bs[:, 3].max()),
        )
        train_log.append(row)
        with (out / "metrics_a1.jsonl").open("a") as fp:
            fp.write(json.dumps(row) + "\n")

        if update == 1 or update % 40 == 0 or update == updates:
            v = eval_landing(val_env, model, val_rows, device)
            v["update"] = update
            v["steps"] = row["steps"]
            val_log.append(v)
            with (out / "val_a1.jsonl").open("a") as fp:
                fp.write(json.dumps(v) + "\n")
            key = (v["soft_12"], -v["max_rel_violation"])
            ckpt = dict(landing=model.state_dict(), critic_a1=critic.state_dict(), logstd=logstd.detach(), update=update, val=v)
            if key > best_score:
                best_score = key
                torch.save(ckpt, out / "best_landing.pt")
            print(
                f"[A1 upd={update:3d}/{updates}] r={row['reward']:+.2f} batch=({row['peak_qd']:.1f},{row['ground_mm']:.1f},{row['self_mm']:.1f}) | VAL soft12={v['soft_12']}/32 peaks=({v['max_qd']:.2f},{v['max_ground_mm']:.2f},{v['max_self_mm']:.2f}) bad_s={v['bad_self']} bad_qd={v['bad_qd']} fd={v['facedown']}",
                flush=True,
            )

    save_curve(
        out,
        "train_curve_a1.png",
        "train_curve_a1.csv",
        "Q1 Stage A1 Side Landing Reflex (Ticks 0..11, L<->R Canonicalized)",
        train_log,
        val_log,
        [
            ("reward", "A1 Mean tick reward"),
            ("val:soft_12", "Unseen val soft_12 / 32"),
            ("peak_qd", "Batch max qd (rad/s)"),
            ("ground_mm", "Batch max ground pen (mm)"),
            ("self_mm", "Batch max arm-leg pen (mm)"),
            ("val:soft_12_b", "Validation soft_12 / 32"),
        ],
    )
    env.close()
    val_env.close()
    return torch.load(out / "best_landing.pt", map_location="cpu", weights_only=False)


def rollout_landing_12(val_env: Plant, landing: SideLandingNet, rows: np.ndarray, device: torch.device):
    """Run best_landing on all rows for 12 ticks; return settled tk12 states, histories, side, peaks, and up_x."""
    kp32, kd32, eff32, stand32, lo32, hi32 = [x.astype(np.float32) for x in (KP, KD, EFF, STAND, LO, HI)]
    margin = 0.5 * eff32 / kp32
    n = len(rows)
    out_rows, out_hists, out_sides, out_pks, out_upx = [], [], [], [], []
    landing.eval()
    with torch.no_grad():
        for s0 in range(0, n, val_env.n):
            b = rows[s0 : s0 + val_env.n]
            m = len(b)
            pad = np.tile(b[0], (val_env.n, 1))
            pad[:m] = b
            obs = val_env.reset(pad)
            hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
            sd_np = (obs[:, 48] < 0).astype(np.float32)
            sd_t = torch.from_numpy(sd_np).to(device)
            sent = pad[:, 57:].astype(np.float32).copy()
            pks = np.zeros((m, 3), dtype=np.float64)
            for tk in range(12):
                o_t = torch.from_numpy(obs.astype(np.float32)).to(device)
                ag_t = torch.full((val_env.n,), float(tk), dtype=torch.float32, device=device)
                tgt = landing(o_t, ag_t, sd_t).cpu().numpy()
                for j in range(m):
                    sent[j] = _update_sent_500hz(obs[j], tgt[j], sent[j], kp32, kd32, eff32, stand32, lo32, hi32, margin)
                obs, met = val_env.step(tgt)
                hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                pks = np.maximum(pks, met[:m, [4, 5, 6]])
            for j in range(m):
                out_rows.append(np.r_[val_env.state[j], sent[j].astype(np.float64)].copy())
                out_hists.append(hist[j].copy())
                out_sides.append(float(sd_np[j]))
                out_pks.append(pks[j].copy())
                out_upx.append(float(-obs[j, 47]))
    return np.stack(out_rows), np.stack(out_hists), np.array(out_sides, np.float32), np.array(out_pks), np.array(out_upx)


def eval_e2e_v3(
    env: Plant,
    landing: SideLandingNet,
    roll_net: SideSettledRollNet,
    val_rows: np.ndarray,
    roll_fit,
    device: torch.device,
):
    """Exact end-to-end validation over ticks 0..149 including age==12 prone->roll.npz routing."""
    n = len(val_rows)
    assert n <= env.n
    pad = np.tile(val_rows[0], (env.n, 1))
    pad[:n] = val_rows
    obs = env.reset(pad)
    hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
    side_np = (obs[:, 48] < 0).astype(np.float32)
    side_t = torch.from_numpy(side_np).to(device)
    org12 = torch.zeros((env.n, 22), dtype=torch.float32, device=device)

    roll_keys = roll_fit["keys"].astype(np.float32)
    roll_dur = float(roll_fit["duration"])
    prone_mode = np.zeros(n, dtype=bool)
    prone_tick = np.zeros(n, dtype=np.int32)
    prone_org = np.zeros((n, 22), dtype=np.float32)

    goal_np = roll_net.goal_cmd.cpu().numpy()
    ep_peaks = np.zeros((n, 3), dtype=np.float64)
    quiet_cnt = np.zeros(n, dtype=np.int32)
    reached = np.zeros(n, dtype=bool)
    reach_peaks = np.zeros((n, 3), dtype=np.float64)
    reach_qerr = np.full(n, 9.9, dtype=np.float64)
    reach_upz = np.ones(n, dtype=np.float64)
    landing.eval()
    roll_net.eval()

    with torch.no_grad():
        for tk in range(220):
            h_t = torch.from_numpy(hist).to(device)
            if tk < 12:
                ag_t = torch.full((env.n,), float(tk), dtype=torch.float32, device=device)
                tgt = landing(h_t[:, -1, :], ag_t, side_t).cpu().numpy()
            else:
                if tk == 12:
                    org12 = h_t[:, -1, :22] + torch.tensor(STAND, dtype=torch.float32, device=device)
                for j in range(n):
                    if not reached[j] and not prone_mode[j] and -obs[j, 47] < -0.75:
                        prone_mode[j] = True
                        prone_tick[j] = 0
                        prone_org[j] = obs[j, :22].astype(np.float32) + STAND.astype(np.float32)
                rag_t = torch.full((env.n,), float(tk - 12), dtype=torch.float32, device=device)
                tgt = roll_net.target_from_action(
                    roll_net.raw_mean(h_t, rag_t, side_t), h_t, rag_t, side_t, org12
                ).cpu().numpy()
                for j in range(n):
                    if prone_mode[j]:
                        ph = min((prone_tick[j] + 1) / roll_dur, 5.0)
                        ix = min(int(ph), 4)
                        prv = prone_org[j] if ix == 0 else roll_keys[ix - 1]
                        bl = 0.5 - 0.5 * math.cos(math.pi * (ph - ix))
                        tgt[j] = np.clip(prv + bl * (roll_keys[ix] - prv), LO, HI)
                        prone_tick[j] += 1

            obs, met = env.step(tgt)
            hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
            for j in range(n):
                if not reached[j]:
                    ep_peaks[j] = np.maximum(ep_peaks[j], met[j, [4, 5, 6]])
                    up_x = -obs[j, 47]
                    up_z = -obs[j, 49]
                    w_norm = float(np.linalg.norm(obs[j, 44:47] * 4.0))
                    q_err = float(np.abs(env.state[j, 7:29] - goal_np).max())
                    if prone_mode[j]:
                        quiet_now = (up_x > 0.80) and (w_norm < 1.70)
                        need_k = 10
                    else:
                        quiet_now = (up_x > 0.84) and (w_norm < 1.45)
                        need_k = 8
                    if quiet_now:
                        quiet_cnt[j] += 1
                    else:
                        quiet_cnt[j] = 0
                    if quiet_cnt[j] >= need_k:
                        reached[j] = True
                        reach_peaks[j] = ep_peaks[j]
                        reach_qerr[j] = q_err
                        reach_upz[j] = abs(up_z)
            if reached.all():
                break

    roll_net.train()
    soft_ok = reached & (reach_peaks[:, 0] <= 12.0) & (reach_peaks[:, 1] <= 10.0) & (reach_peaks[:, 2] <= 2.0)
    viol = np.maximum(ep_peaks / np.array([12.0, 10.0, 2.0]) - 1.0, 0.0)
    return dict(
        supine=int(reached.sum()),
        simultaneous_soft=int(soft_ok.sum()),
        flat_supine=int((reached & (reach_upz <= 0.28)).sum()),
        max_q_err=float(reach_qerr[reached].max()) if reached.any() else 9.9,
        total=n,
        max_qd=float(ep_peaks[:, 0].max()),
        max_ground_mm=float(ep_peaks[:, 1].max()),
        max_self_mm=float(ep_peaks[:, 2].max()),
        max_rel_violation=float(viol.max()),
        per_trial_self_gt2=[int(j) for j in np.flatnonzero(ep_peaks[:, 2] > 2.0)],
        per_trial_qd_gt12=[int(j) for j in np.flatnonzero(ep_peaks[:, 0] > 12.0)],
        per_trial_gnd_gt10=[int(j) for j in np.flatnonzero(ep_peaks[:, 1] > 10.0)],
        prone_routed=int(prone_mode.sum()),
    )


def export_routed_pools(
    out: Path,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    cmd0: np.ndarray,
    roll_fit,
    ckpt_path: Path,
    device: torch.device,
):
    """Roll out best SideV3Combined with dynamic (age>=12 & up_x<-0.75)->roll.npz routing."""
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lnet = SideLandingNet().to(device).eval()
    lnet.load_state_dict(ck["landing"])
    rnet = SideSettledRollNet(cmd0).to(device).eval()
    rnet.load_state_dict(ck["roll"])

    roll_env = Plant(max(64, len(val_rows)), out / "routed_scene", delay=11)
    kp32, kd32, eff32, stand32, lo32, hi32 = [x.astype(np.float32) for x in (KP, KD, EFF, STAND, LO, HI)]
    margin = 0.5 * eff32 / kp32
    roll_keys = roll_fit["keys"].astype(np.float32)
    roll_dur = float(roll_fit["duration"])
    goal_np = rnet.goal_cmd.cpu().numpy()

    manifest = dict(ckpt=str(ckpt_path), ckpt_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest())
    for split, src_rows, dst_name in [
        ("train", train_rows, "side_to_supine_train.npz"),
        ("val", val_rows, "side_to_supine_val.npz"),
    ]:
        captured = []
        reached = 0
        prone_cnt = 0
        flat_cnt = 0
        all_pks = []
        for s0 in range(0, len(src_rows), roll_env.n):
            b = src_rows[s0 : s0 + roll_env.n]
            m = len(b)
            pad = np.tile(b[0], (roll_env.n, 1))
            pad[:m] = b
            obs = roll_env.reset(pad)
            hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
            side_np = (obs[:, 48] < 0).astype(np.float32)
            sd_t = torch.from_numpy(side_np).to(device)
            sent = pad[:, 57:].astype(np.float32).copy()
            org12_b = torch.zeros((roll_env.n, 22), dtype=torch.float32, device=device)
            pm = np.zeros(m, dtype=bool)
            ptk = np.zeros(m, dtype=np.int32)
            porg = np.zeros((m, 22), dtype=np.float32)
            qcnt = np.zeros(m, dtype=np.int32)
            pks = np.zeros((m, 3), dtype=np.float64)
            got = [None] * m
            flat_ok = np.zeros(m, dtype=bool)
            with torch.no_grad():
                for tk in range(250):
                    h_t = torch.from_numpy(hist).to(device)
                    if tk < 12:
                        ag_t = torch.full((roll_env.n,), float(tk), dtype=torch.float32, device=device)
                        tgt = lnet(h_t[:, -1, :], ag_t, sd_t).cpu().numpy()
                    else:
                        if tk == 12:
                            org12_b = h_t[:, -1, :22] + torch.tensor(STAND, dtype=torch.float32, device=device)
                        for j in range(m):
                            if got[j] is None and not pm[j] and -obs[j, 47] < -0.75:
                                pm[j] = True
                                ptk[j] = 0
                                porg[j] = obs[j, :22].astype(np.float32) + STAND.astype(np.float32)
                        rag_t = torch.full((roll_env.n,), float(tk - 12), dtype=torch.float32, device=device)
                        tgt = rnet.target_from_action(
                            rnet.raw_mean(h_t, rag_t, sd_t), h_t, rag_t, sd_t, org12_b
                        ).cpu().numpy()
                        for j in range(m):
                            if pm[j]:
                                ph = min((ptk[j] + 1) / roll_dur, 5.0)
                                ix = min(int(ph), 4)
                                prv = porg[j] if ix == 0 else roll_keys[ix - 1]
                                bl = 0.5 - 0.5 * math.cos(math.pi * (ph - ix))
                                tgt[j] = np.clip(prv + bl * (roll_keys[ix] - prv), LO, HI)
                                ptk[j] += 1
                    for j in range(m):
                        if got[j] is None:
                            sent[j] = _update_sent_500hz(obs[j], tgt[j], sent[j], kp32, kd32, eff32, stand32, lo32, hi32, margin)
                    obs, met = roll_env.step(tgt)
                    hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                    for j in range(m):
                        if got[j] is None:
                            pks[j] = np.maximum(pks[j], met[j, [4, 5, 6]])
                            ux = -obs[j, 47]
                            uz = -obs[j, 49]
                            wn = float(np.linalg.norm(obs[j, 44:47] * 4.0))
                            if pm[j]:
                                is_q = (ux > 0.80) and (wn < 1.70)
                                nk = 10
                            else:
                                is_q = (ux > 0.84) and (wn < 1.45)
                                nk = 8
                            qcnt[j] = qcnt[j] + 1 if is_q else 0
                            if qcnt[j] >= nk:
                                got[j] = np.r_[roll_env.state[j], sent[j].astype(np.float64)].copy()
                                flat_ok[j] = (not pm[j]) and (abs(uz) <= 0.28)
                    if all(g is not None for g in got):
                        break
            for j in range(m):
                all_pks.append(pks[j])
                if got[j] is not None:
                    reached += 1
                    captured.append(got[j])
            prone_cnt += int(pm.sum())
            flat_cnt += int(flat_ok.sum())
        np.savez(out / dst_name, rows=np.stack(captured))
        pa = np.array(all_pks)
        soft_n = int(((pa[:, 0] <= 12.0) & (pa[:, 1] <= 10.0) & (pa[:, 2] <= 2.0)).sum())
        manifest[split] = dict(
            reached=reached,
            total=len(src_rows),
            soft=soft_n,
            flat_supine=flat_cnt,
            prone_routed=prone_cnt,
            peaks_max=pa.max(0).tolist(),
        )
        print(f"SideV3Routed {split}: reached={reached}/{len(src_rows)} soft={soft_n}/{len(src_rows)} flat_supine={flat_cnt} prone={prone_cnt} peaks=({pa[:,0].max():.2f},{pa[:,1].max():.2f},{pa[:,2].max():.2f})", flush=True)
    roll_env.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    side_train_path = args.side_train if args.side_train is not None else (args.pools / "side_train.npz")
    if args.export_only is not None:
        train_rows = np.load(side_train_path)["rows"]
        val_rows = np.load(args.pools / "side_val.npz")["rows"]
        cmd0 = sample_record("q1_official_standup_faceup_20261120")["command_position"][0]
        roll_fit = np.load(args.roll)
        export_routed_pools(args.out.resolve(), train_rows, val_rows, cmd0, roll_fit, args.export_only.resolve(), device)
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), out / "train_side_v3.py")
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    train_rows = np.load(side_train_path)["rows"]
    val_rows = np.load(args.pools / "side_val.npz")["rows"]
    cmd0 = sample_record("q1_official_standup_faceup_20261120")["command_position"][0]
    roll_fit = np.load(args.roll)
    resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume is not None else None
    if args.resume_landing is not None:
        lck = torch.load(args.resume_landing, map_location="cpu", weights_only=False)
        resume_ckpt = dict(resume_ckpt) if resume_ckpt is not None else {}
        resume_ckpt["landing"] = lck["landing"]
        if "critic_a1" in lck:
            resume_ckpt["critic_a1"] = lck["critic_a1"]
        else:
            resume_ckpt.pop("critic_a1", None)
        if "logstd" in lck:
            resume_ckpt["logstd_a1"] = lck["logstd"]
        else:
            resume_ckpt.pop("logstd_a1", None)

    # Phase A1: Landing Reflex (ticks 0..11)
    best_a1 = train_phase_a1(out, train_rows, val_rows, device, args.seed, args.updates_a1, lr=args.lr, resume_ckpt=resume_ckpt)

    landing_net = SideLandingNet().to(device)
    landing_net.load_state_dict(best_a1["landing"])
    landing_net.eval()

    # Roll out Phase A1 to generate settled tick-12 pools
    roll_env = Plant(max(32, len(val_rows)), out / "val_scene_a2", delay=11)
    tr12_rows, tr12_hists, tr12_sides, tr12_pks, tr12_upx = rollout_landing_12(roll_env, landing_net, train_rows, device)
    va12_rows, va12_hists, va12_sides, va12_pks, va12_upx = rollout_landing_12(roll_env, landing_net, val_rows, device)
    print(
        f"[A1->A2 Handoff] Train tk12: n={len(tr12_rows)} peaks=({tr12_pks[:,0].max():.2f},{tr12_pks[:,1].max():.2f},{tr12_pks[:,2].max():.2f}) facedown={(tr12_upx<-0.6).sum()} | Val tk12: peaks=({va12_pks[:,0].max():.2f},{va12_pks[:,1].max():.2f},{va12_pks[:,2].max():.2f}) facedown={(va12_upx<-0.6).sum()}",
        flush=True,
    )

    keep_tr = tr12_upx >= -0.60
    a2_rows = tr12_rows[keep_tr]
    a2_hists = tr12_hists[keep_tr]
    a2_sides = tr12_sides[keep_tr]

    # Phase A2: Settled Roll (ticks 12..149)
    torch.manual_seed(args.seed + 100)
    rng = np.random.default_rng(args.seed + 100)
    env = Plant(64, out / "scene_a2", delay=11)
    roll_net = SideSettledRollNet(cmd0).to(device)
    critic = nn.Sequential(nn.Linear(270, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    logstd = nn.Parameter(torch.full((22,), -2.0, device=device))
    if resume_ckpt is not None:
        roll_net.load_state_dict(resume_ckpt["roll"], strict=True)
        if "critic_a2" in resume_ckpt:
            critic.load_state_dict(resume_ckpt["critic_a2"], strict=True)
        if "logstd_a2" in resume_ckpt:
            logstd.data.copy_(resume_ckpt["logstd_a2"].to(device).clamp(min=-2.3))
        else:
            logstd.data.fill_(-2.1)
    optimizer = torch.optim.Adam([*roll_net.parameters(), *critic.parameters(), logstd], lr=args.lr)

    histories = np.zeros((64, 10, 74), np.float32)
    roll_age = torch.zeros(64, dtype=torch.float32, device=device)
    side_t = torch.zeros(64, dtype=torch.float32, device=device)
    org12 = torch.zeros((64, 22), dtype=torch.float32, device=device)
    ep_steps = np.zeros(64, dtype=np.int32)
    quiet_streak = np.zeros(64, dtype=np.int32)
    ep_peaks = np.zeros((64, 3), dtype=np.float64)
    prev_up_x = np.zeros(64, dtype=np.float64)

    def refresh_a2_hard(tag: str) -> np.ndarray:
        was_tr = roll_net.training
        roll_net.eval()
        scores = []
        with torch.no_grad():
            for s0 in range(0, len(a2_rows), roll_env.n):
                b = a2_rows[s0 : s0 + roll_env.n]
                m = len(b)
                pad = np.tile(b[0], (roll_env.n, 1))
                pad[:m] = b
                roll_env.reset(pad)
                hh = np.zeros((roll_env.n, 10, 74), dtype=np.float32)
                hh[:m] = a2_hists[s0 : s0 + m]
                if m < roll_env.n:
                    hh[m:] = a2_hists[s0]
                sd = np.zeros(roll_env.n, dtype=np.float32)
                sd[:m] = a2_sides[s0 : s0 + m]
                sd_b = torch.from_numpy(sd).to(device)
                o12 = torch.from_numpy((hh[:, -1, :22] + STAND).astype(np.float32)).to(device)
                pks = np.zeros((m, 3), dtype=np.float64)
                qc = np.zeros(m, dtype=np.int32)
                done_m = np.zeros(m, dtype=bool)
                for tk in range(120):
                    h_t = torch.from_numpy(hh).to(device)
                    ag_t = torch.full((roll_env.n,), float(tk), dtype=torch.float32, device=device)
                    tgt = roll_net.target_from_action(roll_net.raw_mean(h_t, ag_t, sd_b), h_t, ag_t, sd_b, o12).cpu().numpy()
                    o, met = roll_env.step(tgt)
                    hh = np.concatenate((hh[:, 1:], o[:, None, :].astype(np.float32)), axis=1)
                    active = ~done_m
                    if not active.any():
                        break
                    pks[active] = np.maximum(pks[active], met[:m][active][:, [4, 5, 6]])
                    ux = -o[:m, 47]
                    wn = np.linalg.norm(o[:m, 44:47] * 4.0, axis=1)
                    for j in np.flatnonzero(active):
                        if ux[j] < -0.75:
                            done_m[j] = True
                        elif ux[j] > 0.84 and wn[j] < 1.45:
                            qc[j] += 1
                            if qc[j] >= 8:
                                done_m[j] = True
                        else:
                            qc[j] = 0
                sc = (
                    pks[:, 2]
                    + 0.5 * np.maximum(0.0, pks[:, 0] - 11.2)
                    + 0.5 * np.maximum(0.0, pks[:, 1] - 9.2)
                    + 3.0 * (~done_m)
                )
                scores.append(sc)
        if was_tr:
            roll_net.train()
        arr = np.concatenate(scores, axis=0)
        hp = np.argsort(arr)[-64:]
        print(
            f"[A2 {tag}] Hard-pool top-64/{len(a2_rows)} range: [{arr[hp[0]]:.2f}, {arr[hp[-1]]:.2f}] (gt2={int((arr>2).sum())}/{len(a2_rows)})",
            flush=True,
        )
        return hp

    a2_hard_pool = refresh_a2_hard("init")

    def reset_a2(mask: np.ndarray):
        idx_m = np.flatnonzero(mask)
        picks = np.where(
            rng.random(len(idx_m)) < 0.50,
            rng.choice(a2_hard_pool, size=len(idx_m), replace=True),
            rng.integers(len(a2_rows), size=len(idx_m)),
        )
        batch = np.zeros((64, 79), dtype=np.float64)
        batch[idx_m] = a2_rows[picks]
        env.reset(batch, mask.astype(np.int32))
        histories[idx_m] = a2_hists[picks]
        m_t = torch.from_numpy(idx_m).to(device)
        roll_age[m_t] = 0.0
        side_t[m_t] = torch.from_numpy(a2_sides[picks]).to(device)
        org12[m_t] = torch.from_numpy((a2_hists[picks, -1, :22] + STAND).astype(np.float32)).to(device)
        ep_steps[idx_m] = 0
        quiet_streak[idx_m] = 0
        ep_peaks[idx_m] = 0.0
        prev_up_x[idx_m] = -a2_hists[picks, -1, 47]

    reset_a2(np.ones(64, dtype=bool))
    metric_scale = torch.tensor([1, 1, 0.01, 1, 0.1, 0.1, 0.1, 1, 0.2, 1, 1, 1, 1], dtype=torch.float32, device=device)
    goal_np = roll_net.goal_cmd.cpu().numpy()

    train_log, val_log = [], []
    start_t = time.monotonic()
    v0 = eval_e2e_v3(roll_env, landing_net, roll_net, val_rows, roll_fit, device)
    v0["update"] = 0
    v0["steps"] = 0
    val_log.append(v0)
    with (out / "val_a2.jsonl").open("a") as fp:
        fp.write(json.dumps(v0) + "\n")
    best_e2e_score = (v0["simultaneous_soft"], v0["supine"], v0["flat_supine"], -v0["max_rel_violation"])
    torch.save(
        dict(
            landing=landing_net.state_dict(),
            roll=roll_net.state_dict(),
            critic_a2=critic.state_dict(),
            logstd_a2=logstd.detach(),
            update=0,
            val=v0,
            a1_val=best_a1["val"],
        ),
        out / "best_side_v3.pt",
    )
    print(
        f"[A2 upd=  0/{args.updates_a2}] baseline | E2E VAL soft={v0['simultaneous_soft']}/32 sup={v0['supine']}/32 flat={v0['flat_supine']}/32 qerr={v0['max_q_err']:.2f} (prone={v0['prone_routed']}) peaks=({v0['max_qd']:.2f},{v0['max_ground_mm']:.2f},{v0['max_self_mm']:.2f}) bad_s={v0['per_trial_self_gt2']} bad_qd={v0['per_trial_qd_gt12']}",
        flush=True,
    )

    for update in range(1, args.updates_a2 + 1):
        if update in (120, 240, 360):
            a2_hard_pool = refresh_a2_hard(f"upd{update}")
        hs, rgs, sds, orgs, cfs, acts, lps, vals, rews, dones = [], [], [], [], [], [], [], [], [], []
        batch_stats = []
        with torch.no_grad():
            for _ in range(32):
                h = torch.from_numpy(histories.copy()).to(device)
                af = roll_net.features(h, roll_age, side_t)
                priv = torch.from_numpy(env.metrics.astype(np.float32)).to(device) * metric_scale
                cf = torch.cat((af, priv), dim=1)
                mean = roll_net.raw_mean(h, roll_age, side_t)
                dist = Normal(mean, logstd.clamp(-3.0, -0.5).exp())
                act = dist.sample()
                tgt = roll_net.target_from_action(act, h, roll_age, side_t, org12)

                hs.append(h)
                rgs.append(roll_age.clone())
                sds.append(side_t.clone())
                orgs.append(org12.clone())
                cfs.append(cf)
                acts.append(act)
                lps.append(dist.log_prob(act).sum(1))
                vals.append(critic(cf).squeeze(1))

                last_sent = STAND + 4.0 * SCALE * histories[:, -1, 50:72]
                obs, met = env.step(tgt.cpu().numpy())
                ep_steps += 1
                roll_age += 1.0
                ep_peaks = np.maximum(ep_peaks, met[:, [4, 5, 6]])

                up_x = -obs[:, 47]
                up_z = -obs[:, 49]
                w_norm = np.linalg.norm(obs[:, 44:47] * 4.0, axis=1)
                qd_max_obs = np.abs(obs[:, 22:44] * 10.0).max(axis=1)
                q_err_mse = np.mean((env.state[:, 7:29] - goal_np) ** 2, axis=1)
                quiet_now = (up_x > 0.84) & (np.abs(up_z) < 0.30) & (w_norm < 1.45) & (qd_max_obs < 4.5)
                quiet_streak = np.where(quiet_now, quiet_streak + 1, 0)
                supine_goal = (up_x > 0.55).astype(np.float64) * (
                    2.5 * np.exp(-2.0 * q_err_mse) - 1.5 * np.maximum(np.abs(up_z) - 0.20, 0.0)
                )

                r = (
                    3.0 * up_x
                    + 10.0 * np.clip(up_x - prev_up_x, -0.3, 0.3)
                    - 3.0 * np.maximum(0.0, -up_x - 0.25)
                    + supine_goal
                    + 3.5 * quiet_now
                )
                r -= 0.45 * np.maximum(met[:, 4] - 9.5, 0.0) ** 2 + 3.5 * np.maximum(met[:, 4] - 11.0, 0.0) + 12.0 * (met[:, 4] > 11.8)
                r -= 0.55 * np.maximum(met[:, 5] - 7.5, 0.0) ** 2 + 3.5 * np.maximum(met[:, 5] - 9.0, 0.0) + 12.0 * (met[:, 5] > 9.8)
                r -= 8.00 * np.maximum(met[:, 6] - 0.20, 0.0) ** 2 + 15.0 * np.maximum(met[:, 6] - 0.80, 0.0) + 25.0 * (met[:, 6] > 1.5)
                r -= 0.04 * np.mean((tgt.cpu().numpy() - last_sent) ** 2, axis=1)

                success_done = quiet_streak >= 8
                soft_ep = (ep_peaks[:, 0] <= 11.7) & (ep_peaks[:, 1] <= 9.6) & (ep_peaks[:, 2] <= 1.60)
                r += np.where(success_done, np.where(soft_ep, 22.0, 2.0), 0.0)
                blowup = (met[:, 4] > 45.0) | (met[:, 6] > 30.0)
                done = success_done | blowup | (ep_steps >= 138)
                r = np.clip(r - 15.0 * blowup, -100.0, 30.0)

                prev_up_x = up_x.copy()
                histories = np.concatenate((histories[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                rews.append(torch.tensor(r * 0.1, dtype=torch.float32, device=device))
                dones.append(torch.tensor(done, device=device))
                batch_stats.append([float(r.mean()), float(quiet_now.mean()), float(met[:, 4].max()), float(met[:, 5].max()), float(met[:, 6].max())])
                if done.any():
                    reset_a2(done)

            af_n = roll_net.features(torch.from_numpy(histories.copy()).to(device), roll_age, side_t)
            priv_n = torch.from_numpy(env.metrics.astype(np.float32)).to(device) * metric_scale
            following = critic(torch.cat((af_n, priv_n), dim=1)).squeeze(1)
            gae = torch.zeros_like(following)
            advs = []
            for t in reversed(range(32)):
                alive = (~dones[t]).float()
                gae = rews[t] + 0.99 * alive * following - vals[t] + 0.99 * 0.95 * alive * gae
                advs.append(gae)
                following = vals[t]
            advantage = torch.cat(advs[::-1])
            returns = advantage + torch.cat(vals)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        h_b, rg_b, sd_b, org_b, cf_b, act_b, lp_b = map(torch.cat, (hs, rgs, sds, orgs, cfs, acts, lps))
        lr_now = args.lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * update / args.updates_a2)))
        for pg_opt in optimizer.param_groups:
            pg_opt["lr"] = lr_now
        pg_w = 0.0 if (resume_ckpt is not None and "critic_a2" not in resume_ckpt and update <= 4) else 1.0
        for _ in range(4):
            for ix in torch.randperm(len(act_b), device=device).split(512):
                dist = Normal(roll_net.raw_mean(h_b[ix], rg_b[ix], sd_b[ix]), logstd.clamp(-3.0, -0.5).exp())
                ratio = (dist.log_prob(act_b[ix]).sum(1) - lp_b[ix]).exp()
                pg = torch.minimum(ratio * advantage[ix], ratio.clamp(0.8, 1.2) * advantage[ix])
                loss = -pg_w * pg.mean() + 0.3 * (critic(cf_b[ix]).squeeze(1) - returns[ix]).square().mean() - pg_w * 0.001 * dist.entropy().sum(1).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(roll_net.parameters(), 1.0)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()

        bs = np.array(batch_stats)
        row = dict(
            update=update,
            steps=update * 64 * 32,
            elapsed_s=round(time.monotonic() - start_t, 2),
            reward=float(bs[:, 0].mean()),
            quiet_fraction=float(bs[:, 1].mean()),
            peak_qd=float(bs[:, 2].max()),
            ground_mm=float(bs[:, 3].max()),
            self_mm=float(bs[:, 4].max()),
        )
        train_log.append(row)
        with (out / "metrics_a2.jsonl").open("a") as fp:
            fp.write(json.dumps(row) + "\n")

        if update == 1 or update % 50 == 0 or update == args.updates_a2:
            v = eval_e2e_v3(roll_env, landing_net, roll_net, val_rows, roll_fit, device)
            v["update"] = update
            v["steps"] = row["steps"]
            val_log.append(v)
            with (out / "val_a2.jsonl").open("a") as fp:
                fp.write(json.dumps(v) + "\n")
            key = (v["simultaneous_soft"], v["supine"], v["flat_supine"], -v["max_rel_violation"])
            ckpt = dict(
                landing=landing_net.state_dict(),
                roll=roll_net.state_dict(),
                critic_a2=critic.state_dict(),
                logstd_a2=logstd.detach(),
                update=update,
                val=v,
                a1_val=best_a1["val"],
            )
            if key > best_e2e_score:
                best_e2e_score = key
                torch.save(ckpt, out / "best_side_v3.pt")
            print(
                f"[A2 upd={update:3d}/{args.updates_a2}] r={row['reward']:+.2f} batch=({row['peak_qd']:.1f},{row['ground_mm']:.1f},{row['self_mm']:.1f}) | E2E VAL soft={v['simultaneous_soft']}/32 sup={v['supine']}/32 flat={v['flat_supine']}/32 qerr={v['max_q_err']:.2f} (prone={v['prone_routed']}) peaks=({v['max_qd']:.2f},{v['max_ground_mm']:.2f},{v['max_self_mm']:.2f}) bad_s={v['per_trial_self_gt2']} bad_qd={v['per_trial_qd_gt12']}",
                flush=True,
            )

    save_curve(
        out,
        "train_curve_a2.png",
        "train_curve_a2.csv",
        "Q1 Stage A2 Settled Side-to-Supine Roll (Ticks 12..149, L<->R Canonicalized)",
        train_log,
        val_log,
        [
            ("reward", "A2 Sampled reward"),
            ("quiet_fraction", "A2 Quiet supine fraction"),
            ("peak_qd", "Batch max qd (rad/s)"),
            ("ground_mm", "Batch max ground pen (mm)"),
            ("self_mm", "Batch max arm-leg pen (mm)"),
            ("val:e2e", "Unseen E2E val supine & soft / 32"),
        ],
    )

    # Export best combined ONNX (SideV3Combined) and check ORT parity < 1e-5.
    best_ck = torch.load(out / "best_side_v3.pt", map_location="cpu", weights_only=False)
    combined = SideV3Combined(cmd0).eval()
    combined.landing.load_state_dict(best_ck["landing"])
    combined.roll.load_state_dict(best_ck["roll"])
    sample = torch.zeros((1, 10, 74))
    sample[:, :, 48] = -0.95
    onnx_path = out / "side_v3_expert.onnx"
    torch.onnx.export(
        combined,
        (sample, torch.zeros((1, 26), dtype=torch.float32)),
        str(onnx_path),
        input_names=["rec_obs_hist", "rec_state"],
        output_names=["joint_target", "next_state"],
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng_p = np.random.default_rng(3)
    ort_diff = 0.0
    for st_val in [[0.0]*26, [12.0, 10.0, 1.0, 0.0] + list(STAND), [25.0, 10.0, 0.0, 2.0] + list(STAND)]:
        th = torch.tensor(rng_p.normal(0, 0.2, (1, 10, 74)), dtype=torch.float32)
        th[:, :, 48] = -0.95 if st_val[2] > 0.5 else 0.95
        ts = torch.tensor([st_val], dtype=torch.float32)
        with torch.no_grad():
            pt_y, pt_s = combined(th, ts)
        ox_y, ox_s = sess.run(["joint_target", "next_state"], {"rec_obs_hist": th.numpy(), "rec_state": ts.numpy()})
        ort_diff = max(ort_diff, float(np.max(np.abs(pt_y.numpy() - ox_y))), float(np.max(np.abs(pt_s.numpy() - ox_s))))
    assert ort_diff < 1e-5, f"Stage A v3 ONNX parity failed: {ort_diff}"

    # Roll out END-TO-END best model to generate side_to_supine_train.npz and side_to_supine_val.npz inside out/.
    roll_best = SideSettledRollNet(cmd0).to(device).eval()
    roll_best.load_state_dict(best_ck["roll"])
    kp32, kd32, eff32, stand32, lo32, hi32 = [x.astype(np.float32) for x in (KP, KD, EFF, STAND, LO, HI)]
    margin = 0.5 * eff32 / kp32
    roll_keys = roll_fit["keys"].astype(np.float32)
    roll_dur = float(roll_fit["duration"])

    for split, src_rows, dst_name in [
        ("train", train_rows, "side_to_supine_train.npz"),
        ("val", val_rows, "side_to_supine_val.npz"),
    ]:
        captured = []
        reached = 0
        all_pks = []
        for s0 in range(0, len(src_rows), roll_env.n):
            b = src_rows[s0 : s0 + roll_env.n]
            m = len(b)
            pad = np.tile(b[0], (roll_env.n, 1))
            pad[:m] = b
            obs = roll_env.reset(pad)
            hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
            side_np = (obs[:, 48] < 0).astype(np.float32)
            sd_t = torch.from_numpy(side_np).to(device)
            sent = pad[:, 57:].astype(np.float32).copy()
            org12_b = torch.zeros((roll_env.n, 22), dtype=torch.float32, device=device)
            pm = np.zeros(m, dtype=bool)
            ptk = np.zeros(m, dtype=np.int32)
            porg = np.zeros((m, 22), dtype=np.float32)
            qcnt = np.zeros(m, dtype=np.int32)
            pks = np.zeros((m, 3), dtype=np.float64)
            got = [None] * m
            with torch.no_grad():
                for tk in range(220):
                    h_t = torch.from_numpy(hist).to(device)
                    if tk < 12:
                        ag_t = torch.full((roll_env.n,), float(tk), dtype=torch.float32, device=device)
                        tgt = landing_net(h_t[:, -1, :], ag_t, sd_t).cpu().numpy()
                    else:
                        if tk == 12:
                            org12_b = h_t[:, -1, :22] + torch.tensor(STAND, dtype=torch.float32, device=device)
                        for j in range(m):
                            if got[j] is None and not pm[j] and -obs[j, 47] < -0.75:
                                pm[j] = True
                                ptk[j] = 0
                                porg[j] = obs[j, :22].astype(np.float32) + STAND.astype(np.float32)
                        rag_t = torch.full((roll_env.n,), float(tk - 12), dtype=torch.float32, device=device)
                        tgt = roll_best.target_from_action(
                            roll_best.raw_mean(h_t, rag_t, sd_t), h_t, rag_t, sd_t, org12_b
                        ).cpu().numpy()
                        for j in range(m):
                            if pm[j]:
                                ph = min((ptk[j] + 1) / roll_dur, 5.0)
                                ix = min(int(ph), 4)
                                prv = porg[j] if ix == 0 else roll_keys[ix - 1]
                                bl = 0.5 - 0.5 * math.cos(math.pi * (ph - ix))
                                tgt[j] = np.clip(prv + bl * (roll_keys[ix] - prv), LO, HI)
                                ptk[j] += 1
                    for j in range(m):
                        if got[j] is None:
                            sent[j] = _update_sent_500hz(obs[j], tgt[j], sent[j], kp32, kd32, eff32, stand32, lo32, hi32, margin)
                    obs, met = roll_env.step(tgt)
                    hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                    for j in range(m):
                        if got[j] is None:
                            pks[j] = np.maximum(pks[j], met[j, [4, 5, 6]])
                            ux = -obs[j, 47]
                            wn = float(np.linalg.norm(obs[j, 44:47] * 4.0))
                            tu, tw, nk = (0.80, 1.70, 10) if pm[j] else (0.84, 1.45, 8)
                            qcnt[j] = qcnt[j] + 1 if (ux > tu and wn < tw) else 0
                            if qcnt[j] >= nk:
                                got[j] = np.r_[roll_env.state[j], sent[j].astype(np.float64)].copy()
                    if all(g is not None for g in got):
                        break
            for j in range(m):
                all_pks.append(pks[j])
                if got[j] is not None:
                    reached += 1
                    captured.append(got[j])
        np.savez(out / dst_name, rows=np.stack(captured))
        pa = np.array(all_pks)
        print(f"SideV3->Supine {split}: reached={reached}/{len(src_rows)} peaks=({pa[:,0].max():.2f},{pa[:,1].max():.2f},{pa[:,2].max():.2f})", flush=True)

    summary = dict(
        best_e2e_val=best_ck["val"],
        best_e2e_update=int(best_ck["update"]),
        best_a1_val=best_ck["a1_val"],
        ort_parity=ort_diff,
        onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("STAGE A V3 DONE:", json.dumps(summary), flush=True)
    env.close()
    roll_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pools",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/pools"),
    )
    p.add_argument(
        "--roll",
        type=Path,
        default=RUN_V1 / "joined_roll_dev/roll.npz",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/side_v2_03"),
    )
    p.add_argument("--side-train", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--resume-landing", type=Path, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--updates-a1", type=int, default=400)
    p.add_argument("--updates-a2", type=int, default=300)
    p.add_argument("--export-only", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=94203)
    run(p.parse_args())
