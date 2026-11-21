"""Stage B v3: Fast (~6.8 s) closed-loop neural supine-to-upright-stand policy under exact 2 ms physics."""
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
for p in reversed((str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT))):
    if p not in sys.path:
        sys.path.insert(0, p)

from command_support import sample_record
from common import EFF, HI, KD, KP, LO, Plant, SCALE, STAND


def build_base_cmd_table_v3(faceup_cmd: np.ndarray, horizon: int = 401) -> np.ndarray:
    """Precompute piecewise-speed faceup reference blending smoothly to STAND by tick 341."""
    L = len(faceup_cmd)
    table = np.zeros((horizon, 22), dtype=np.float32)

    def interp_raw(s: float) -> np.ndarray:
        s = min(max(s, 0.0), float(L - 1))
        i0 = int(math.floor(s))
        i1 = min(i0 + 1, L - 1)
        frac = s - i0
        return (1.0 - frac) * faceup_cmd[i0] + frac * faceup_cmd[i1]

    ref_end = interp_raw(295.0)
    s_cur = 0.0
    for tick in range(horizon):
        if tick < 15:
            table[tick] = faceup_cmd[0]
        elif tick <= 311:
            if tick > 15:
                rate = 1.10 if s_cur < 175.0 else (0.80 if s_cur < 230.0 else 0.95)
                s_cur = min(295.0, s_cur + rate)
            table[tick] = interp_raw(s_cur)
        elif tick < 341:
            a = 0.5 - 0.5 * math.cos(math.pi * (tick - 311) / 30.0)
            table[tick] = (1.0 - a) * ref_end + a * STAND
        else:
            table[tick] = STAND
    return np.clip(table, LO, HI).astype(np.float32)


class SupineRiseNetV3(nn.Module):
    """ONNX-compatible closed-loop supine-to-stand residual policy finishing smoothly by ~6.8 s."""

    def __init__(self, faceup_cmd: np.ndarray):
        super().__init__()
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(LO, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(HI, dtype=torch.float32))
        table = build_base_cmd_table_v3(faceup_cmd, 401)
        self.register_buffer("base_cmd", torch.tensor(table, dtype=torch.float32))

        self.encoder = nn.Sequential(nn.Linear(74, 48), nn.ELU())
        self.actor = nn.Sequential(
            nn.Linear(337, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 22),
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def _smooth_step_mask(self, x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
        t = ((x - x0) / max(1e-6, x1 - x0)).clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(torch.pi * t)

    def base_and_scale(self, phase_tick: torch.Tensor, origin: torch.Tensor):
        idx = phase_tick.long().clamp(0, self.base_cmd.shape[0] - 1)
        idx_next = (idx + 1).clamp(0, self.base_cmd.shape[0] - 1)
        cmd_now = self.base_cmd[idx]
        cmd_next = self.base_cmd[idx_next]

        blend = 0.5 - 0.5 * torch.cos(torch.pi * ((phase_tick + 1.0) / 15.0).clamp(0.0, 1.0))
        blend_next = 0.5 - 0.5 * torch.cos(torch.pi * ((phase_tick + 2.0) / 15.0).clamp(0.0, 1.0))
        early = (phase_tick < 15.0)[:, None]
        base_q = torch.where(early, origin + blend[:, None] * (cmd_now - origin), cmd_now)
        base_q_next = torch.where(
            (phase_tick + 1.0 < 15.0)[:, None],
            origin + blend_next[:, None] * (cmd_next - origin),
            cmd_next,
        )

        # Phase-adaptive residual authority: 0->0.45 (0..15), 0.45 (15..135), 0.45->0.80 (135..170),
        # 0.80 (170..330), 0.80->0.45 (330..365).
        s_init = 0.45 * self._smooth_step_mask(phase_tick, 0.0, 15.0)
        s_rise = 0.35 * self._smooth_step_mask(phase_tick, 135.0, 170.0)
        s_taper = -0.35 * self._smooth_step_mask(phase_tick, 330.0, 365.0)
        scale = s_init + s_rise + s_taper
        return base_q, base_q_next, scale

    def actor_features(self, rec_obs_hist: torch.Tensor, phase_tick: torch.Tensor, origin: torch.Tensor):
        q = rec_obs_hist[:, -1, :22] + self.stand
        base_q, base_q_next, _ = self.base_and_scale(phase_tick, origin)
        enc = self.encoder(rec_obs_hist[:, -6:, :].clamp(-12.0, 12.0)).flatten(1)
        p = (phase_tick / 345.0).clamp(0.0, 1.0)[:, None]
        phase_feat = torch.cat(
            (p, torch.sin(2.0 * torch.pi * p), torch.cos(2.0 * torch.pi * p), torch.sin(4.0 * torch.pi * p), torch.cos(4.0 * torch.pi * p)),
            1,
        )
        return torch.cat((enc, phase_feat, base_q - q, base_q_next - base_q), 1)

    def raw_mean(self, rec_obs_hist: torch.Tensor, phase_tick: torch.Tensor, origin: torch.Tensor):
        return self.actor(self.actor_features(rec_obs_hist, phase_tick, origin))

    def target_from_action(
        self, action: torch.Tensor, phase_tick: torch.Tensor, origin: torch.Tensor
    ):
        base_q, _, scale = self.base_and_scale(phase_tick, origin)
        return (base_q + scale[:, None] * action.tanh()).clamp(self.lo, self.hi)

    def forward(self, rec_obs_hist: torch.Tensor, rec_state: torch.Tensor):
        q = rec_obs_hist[:, -1, :22] + self.stand
        begin = rec_state[:, 1] != 6
        phase_tick = torch.where(begin, torch.zeros_like(rec_state[:, 3]), rec_state[:, 3])
        origin = torch.where(begin[:, None], q, rec_state[:, 4:])
        act = self.raw_mean(rec_obs_hist, phase_tick, origin)
        target = self.target_from_action(act, phase_tick, origin)
        next_pt = (phase_tick + 1.0).clamp(max=10000.0)
        z = phase_tick * 0.0
        next_state = torch.cat((torch.stack((next_pt, z + 6.0, z + 1.0, next_pt), 1), origin), 1)
        return target, next_state


def collect_transition_snapshots(
    env: Plant,
    starts: np.ndarray,
    model: SupineRiseNetV3,
    device: torch.device,
    target_tick: int = 160,
    filter_mode: str = "quadruped",
):
    """Roll out up to target_tick from floor starts and save valid transition states & 10-frame histories."""
    snaps_rows, snaps_hist, snaps_orig = [], [], []
    was_train = model.training
    model.eval()
    kp32, kd32, eff32, stand32, lo32, hi32 = [x.astype(np.float32) for x in (KP, KD, EFF, STAND, LO, HI)]
    margin = 0.5 * eff32 / kp32
    for s0 in range(0, len(starts), env.n):
        batch = starts[s0 : s0 + env.n]
        m = len(batch)
        pad = np.tile(batch[0], (env.n, 1))
        pad[:m] = batch
        obs = env.reset(pad)
        hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
        origin = torch.from_numpy((obs[:, :22] + STAND).astype(np.float32)).to(device)
        sent = pad[:, 57:].astype(np.float32).copy()
        with torch.no_grad():
            for tk in range(target_tick):
                pt = torch.full((env.n,), float(tk), dtype=torch.float32, device=device)
                h_t = torch.from_numpy(hist).to(device)
                tgt = model.target_from_action(model.raw_mean(h_t, pt, origin), pt, origin).cpu().numpy()
                for j in range(m):
                    q_o = obs[j, :22].astype(np.float32) + stand32
                    qd_o = obs[j, 22:44].astype(np.float32) * 10.0
                    raw = np.clip(tgt[j].astype(np.float32), lo32 - margin, hi32 + margin)
                    tau = np.clip(kp32 * (raw - q_o) - kd32 * qd_o, -eff32, eff32)
                    des = np.clip(q_o + (tau + kd32 * qd_o) / kp32, stand32 - 2.0, stand32 + 2.0)
                    for _ in range(10):
                        sent[j] = sent[j] + np.clip(des - sent[j], -0.2, 0.2)
                obs, met = env.step(tgt)
                hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
        orig_np = origin.cpu().numpy()
        for j in range(m):
            if filter_mode == "quadruped":
                valid = -obs[j, 47] < -0.55 and met[j, 0] > 0.16
            elif filter_mode == "rising":
                valid = met[j, 1] > 0.10 or met[j, 0] > 0.20
            else:
                raise ValueError(f"Unknown filter_mode: {filter_mode}")
            if valid:
                snaps_rows.append(np.r_[env.state[j], sent[j].astype(np.float64)].copy())
                snaps_hist.append(hist[j].copy())
                snaps_orig.append(orig_np[j].copy())
    if was_train:
        model.train()
    if not snaps_rows:
        return (
            np.zeros((0, 79), dtype=np.float64),
            np.zeros((0, 10, 74), dtype=np.float32),
            np.zeros((0, 22), dtype=np.float32),
        )
    return np.stack(snaps_rows), np.stack(snaps_hist), np.stack(snaps_orig)


def eval_validation(
    env: Plant,
    model: SupineRiseNetV3,
    val_groups: list[tuple[str, np.ndarray]],
    horizon: int,
    device: torch.device,
):
    all_rows = np.concatenate([g[1] for g in val_groups], axis=0)
    n_val = len(all_rows)
    assert n_val <= env.n
    pad = np.tile(all_rows[0], (env.n, 1))
    pad[:n_val] = all_rows
    obs = env.reset(pad)
    hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
    origin = torch.from_numpy((obs[:, :22] + STAND).astype(np.float32)).to(device)

    ep_peaks = np.zeros((n_val, 3), dtype=np.float64)
    quiet_streak = np.zeros(n_val, dtype=np.int32)
    confirmed = np.zeros(n_val, dtype=bool)
    was_train = model.training
    model.eval()
    with torch.no_grad():
        for tk in range(horizon):
            pt = torch.full((env.n,), float(tk), dtype=torch.float32, device=device)
            h_t = torch.from_numpy(hist).to(device)
            tgt = model.target_from_action(model.raw_mean(h_t, pt, origin), pt, origin).cpu().numpy()
            obs, met = env.step(tgt)
            hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
            for j in range(n_val):
                ep_peaks[j] = np.maximum(ep_peaks[j], met[j, [4, 5, 6]])
                qd_rms = float(np.sqrt(np.mean((obs[j, 22:44] * 10.0) ** 2)))
                upright = met[j, 0] >= 0.385 and met[j, 1] > 0.96 and met[j, 2] > 15.0 and met[j, 3] == 0.0
                quiet = upright and met[j, 7] < 0.18 and met[j, 8] < 0.70 and qd_rms < 0.85
                quiet_streak[j] = quiet_streak[j] + 1 if quiet else 0
                if quiet_streak[j] >= 25:
                    confirmed[j] = True
    if was_train:
        model.train()

    soft_ok = confirmed & (ep_peaks[:, 0] <= 12.0) & (ep_peaks[:, 1] <= 10.0) & (ep_peaks[:, 2] <= 2.0)
    viol = np.maximum(ep_peaks / np.array([12.0, 10.0, 2.0]) - 1.0, 0.0)
    by_src = {}
    offset = 0
    for name, arr in val_groups:
        sl = slice(offset, offset + len(arr))
        by_src[name] = dict(quiet=int(confirmed[sl].sum()), soft=int(soft_ok[sl].sum()), n=len(arr))
        offset += len(arr)
    return dict(
        quiet_stand=int(confirmed.sum()),
        simultaneous_soft=int(soft_ok.sum()),
        total=n_val,
        max_qd=float(ep_peaks[:, 0].max()),
        max_ground_mm=float(ep_peaks[:, 1].max()),
        max_self_mm=float(ep_peaks[:, 2].max()),
        max_rel_violation=float(viol.max()),
        by_source=by_src,
    )


def plot_curves(out: Path, train_rows: list[dict], val_rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    x = np.array([r["steps"] / 1e6 for r in train_rows])
    keys = [
        ("reward", "Sampled reward"),
        ("quiet_fraction", "Sampled quiet stand fraction"),
        ("peak_qd", "Batch max joint speed (rad/s)"),
        ("ground_mm", "Batch max ground pen (mm)"),
        ("self_mm", "Batch max arm-leg pen (mm)"),
        ("val_stand", "Unseen val stand & soft stand / 96"),
    ]
    for ax, (k, title) in zip(axes.flat, keys):
        if k == "val_stand" and val_rows:
            vx = np.array([r["steps"] / 1e6 for r in val_rows])
            ax.plot(vx, [r["quiet_stand"] for r in val_rows], "o--", label="quiet_stand/96")
            ax.plot(vx, [r["simultaneous_soft"] for r in val_rows], "s-", label="soft_stand/96")
            ax.axhline(90, color="green", ls="--", lw=1)
            ax.legend(fontsize=8)
        elif k in train_rows[0]:
            y = np.array([r[k] for r in train_rows])
            ax.plot(x, y, lw=0.6, alpha=0.55)
            if len(y) >= 40:
                sm = np.convolve(y, np.ones(40) / 40, mode="valid")
                ax.plot(x[39:], sm, lw=2.0)
        ax.set_title(title)
        ax.set_xlabel("Million 20 ms steps")
        ax.grid(alpha=0.25)
    for ax, v in ((axes[0, 2], 12.0), (axes[1, 0], 10.0), (axes[1, 1], 2.0)):
        ax.axhline(v, color="black", ls="--", lw=1)
    fig.suptitle("Q1 Stage B v3 Fast Supine-to-Stand PPO | Closed-loop 2 ms plant")
    fig.savefig(out / "train_curve.png", dpi=150)
    plt.close(fig)

    with (out / "train_curve.csv").open("w", newline="") as fp:
        fields = sorted(set().union(*(r.keys() for r in train_rows)))
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(train_rows)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), out / "train_supine_v3.py")
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    supine_tr = np.load(args.pools / "supine_train.npz")["rows"]
    p2s_tr = np.load(args.pools / "prone_to_supine_train.npz")["rows"]
    s2s_tr = np.load(args.side_pool_dir / "side_to_supine_train.npz")["rows"]
    full_sources = [supine_tr, p2s_tr, s2s_tr]

    val_groups = [
        ("supine", np.load(args.pools / "supine_val.npz")["rows"]),
        ("prone_to_supine", np.load(args.pools / "prone_to_supine_val.npz")["rows"]),
        ("side_to_supine", np.load(args.side_pool_dir / "side_to_supine_val.npz")["rows"]),
    ]

    faceup_cmd = sample_record("q1_official_standup_faceup_20261120")["command_position"]

    env = Plant(args.envs, out / "scene", delay=11)
    val_env = Plant(max(96, sum(len(g[1]) for g in val_groups)), out / "val_scene", delay=11)

    model = SupineRiseNetV3(faceup_cmd).to(device)
    critic = nn.Sequential(nn.Linear(357, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1)).to(device)
    logstd = nn.Parameter(torch.full((22,), -2.2, device=device))
    if args.resume is not None:
        prior = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(prior["actor"], strict=True)
        critic.load_state_dict(prior["critic"], strict=True)
        logstd.data.copy_(prior["logstd"].to(device).clamp(min=-2.4, max=args.max_logstd))
        print(f"Resumed V3 actor+critic+logstd from {args.resume} (prior update={prior.get('update')}).", flush=True)
    optimizer = torch.optim.Adam([*model.parameters(), *critic.parameters(), logstd], lr=args.lr)

    all_full = np.concatenate(full_sources, axis=0)
    snap160_rows, snap160_hists, snap160_origs = collect_transition_snapshots(
        val_env, all_full, model, device, target_tick=160, filter_mode="quadruped"
    )
    snap235_rows, snap235_hists, snap235_origs = collect_transition_snapshots(
        val_env, all_full, model, device, target_tick=235, filter_mode="rising"
    )
    print(
        f"Collected {len(snap160_rows)}/{len(all_full)} tick=160 quadruped & "
        f"{len(snap235_rows)}/{len(all_full)} tick=235 rising snapshots.",
        flush=True,
    )

    histories = np.zeros((args.envs, 10, 74), np.float32)
    origin = torch.zeros((args.envs, 22), dtype=torch.float32, device=device)
    phase_tick = torch.zeros(args.envs, dtype=torch.float32, device=device)
    max_ticks = np.zeros(args.envs, dtype=np.int32)
    ep_steps = np.zeros(args.envs, dtype=np.int32)
    faceplant_streak = np.zeros(args.envs, dtype=np.int32)
    prev_up_z = np.zeros(args.envs, dtype=np.float64)
    prev_z = np.zeros(args.envs, dtype=np.float64)

    def _add_mild_noise(arr: np.ndarray) -> np.ndarray:
        noise = rng.random(len(arr)) < 0.50
        if noise.any():
            arr[noise, 7:29] = np.clip(arr[noise, 7:29] + rng.uniform(-0.03, 0.03, (noise.sum(), 22)), LO, HI)
            arr[noise, 35:57] += rng.uniform(-0.15, 0.15, (noise.sum(), 22))
            arr[noise, 57:] = arr[noise, 7:29]
        return arr

    def reset_envs(mask: np.ndarray):
        idx_mask = np.flatnonzero(mask)
        u = rng.random(len(idx_mask))
        # 3-tier reset mix ending at age=400: polish emphasizes age=0 (0.50) & s2s transitions.
        if args.polish:
            tier = np.where(u < 0.50, 0, np.where(u < 0.65, 1, 2))
        else:
            tier = np.where(u < 0.35, 0, np.where(u < 0.65, 1, 2))
        if len(snap235_rows) == 0:
            tier = np.where(tier == 2, 1, tier)
        if len(snap160_rows) == 0:
            tier = np.where(tier == 1, 0, tier)
        full_indices = idx_mask[tier == 0]
        mid_indices = idx_mask[tier == 1]
        late_indices = idx_mask[tier == 2]
        batch = np.zeros((args.envs, 79), dtype=np.float64)

        if len(full_indices):
            src_probs = [0.25, 0.25, 0.50] if args.polish else None
            src_choices = rng.choice(len(full_sources), size=len(full_indices), p=src_probs)
            picks = [full_sources[int(si)][int(rng.integers(len(full_sources[int(si)])))] for si in src_choices]
            batch[full_indices] = _add_mild_noise(np.stack(picks))

        chosen_mid = chosen_late = None
        if len(mid_indices):
            chosen_mid = rng.integers(len(snap160_rows), size=len(mid_indices))
            batch[mid_indices] = _add_mild_noise(snap160_rows[chosen_mid].copy())
        if len(late_indices):
            chosen_late = rng.integers(len(snap235_rows), size=len(late_indices))
            batch[late_indices] = _add_mild_noise(snap235_rows[chosen_late].copy())

        frames = env.reset(batch, mask.astype(np.int32))
        if len(full_indices):
            histories[full_indices] = frames[full_indices, None, :].astype(np.float32)
            origin[torch.from_numpy(full_indices).to(device)] = torch.from_numpy(
                (frames[full_indices, :22] + STAND).astype(np.float32)
            ).to(device)
            phase_tick[torch.from_numpy(full_indices).to(device)] = 0.0
            max_ticks[full_indices] = 400
        if len(mid_indices):
            histories[mid_indices] = snap160_hists[chosen_mid]
            origin[torch.from_numpy(mid_indices).to(device)] = torch.from_numpy(snap160_origs[chosen_mid]).to(device)
            phase_tick[torch.from_numpy(mid_indices).to(device)] = 160.0
            max_ticks[mid_indices] = 240
        if len(late_indices):
            histories[late_indices] = snap235_hists[chosen_late]
            origin[torch.from_numpy(late_indices).to(device)] = torch.from_numpy(snap235_origs[chosen_late]).to(device)
            phase_tick[torch.from_numpy(late_indices).to(device)] = 235.0
            max_ticks[late_indices] = 165

        ep_steps[idx_mask] = 0
        faceplant_streak[idx_mask] = 0
        prev_up_z[idx_mask] = env.metrics[idx_mask, 1]
        prev_z[idx_mask] = env.metrics[idx_mask, 0]

    reset_envs(np.ones(args.envs, dtype=bool))
    metric_scale = torch.tensor([1, 1, 0.01, 1, 0.1, 0.1, 0.1, 1, 0.2, 1, 1, 1, 1], dtype=torch.float32, device=device)

    def critic_features(h, pt, org, met_np, state_np):
        af = model.actor_features(h, pt, org)
        priv = torch.from_numpy(met_np.astype(np.float32)).to(device) * metric_scale
        q0 = state_np[:, 3:7]
        true_up_x = -2.0 * (q0[:, 1] * q0[:, 3] - q0[:, 0] * q0[:, 2])
        true_up_y = -2.0 * (q0[:, 2] * q0[:, 3] + q0[:, 0] * q0[:, 1])
        true_up_z = met_np[:, 1]
        extra = np.column_stack((state_np[:, 2], true_up_x, true_up_y, true_up_z, state_np[:, 29:32])).astype(np.float32)
        return torch.cat((af, priv, torch.from_numpy(extra).to(device)), 1)

    train_log, val_log = [], []
    best_score = (-1, -1, -1e9)
    start_t = time.monotonic()

    def record_validation(upd: int, steps_done: int, stats_row: dict | None):
        nonlocal best_score
        v = eval_validation(val_env, model, val_groups, 400, device)
        v["update"] = upd
        v["steps"] = steps_done
        val_log.append(v)
        with (out / "val_metrics.jsonl").open("a") as fp:
            fp.write(json.dumps(v) + "\n")
        key = (v["simultaneous_soft"], v["quiet_stand"], -v["max_rel_violation"])
        ckpt = dict(actor=model.state_dict(), critic=critic.state_dict(), logstd=logstd.detach(), update=upd, val=v)
        torch.save(ckpt, out / f"checkpoint_{upd:05d}.pt")
        if key > best_score:
            best_score = key
            torch.save(ckpt, out / "best.pt")
        bs_str = " ".join(f"{k}={vv['quiet']}/{vv['soft']}" for k, vv in v["by_source"].items())
        if stats_row is not None:
            prefix = f"[SupineV3 upd={upd:4d}/{args.updates} t={stats_row['elapsed_s']:.0f}s] r={stats_row['reward']:+.2f} batch=({stats_row['peak_qd']:.1f},{stats_row['ground_mm']:.1f},{stats_row['self_mm']:.1f})"
        else:
            prefix = f"[SupineV3 upd={upd:4d}/{args.updates} BASELINE]"
        print(
            f"{prefix} | VAL soft={v['simultaneous_soft']}/96 quiet={v['quiet_stand']}/96 ({bs_str}) peaks=({v['max_qd']:.2f},{v['max_ground_mm']:.2f},{v['max_self_mm']:.2f})",
            flush=True,
        )

    if args.polish:
        record_validation(0, 0, None)

    for update in range(1, args.updates + 1):
        if (args.polish and update == 250) or (not args.polish and update in (450, 950)):
            r160, h160, o160 = collect_transition_snapshots(
                val_env, all_full, model, device, target_tick=160, filter_mode="quadruped"
            )
            r235, h235, o235 = collect_transition_snapshots(
                val_env, all_full, model, device, target_tick=235, filter_mode="rising"
            )
            if len(r160):
                snap160_rows, snap160_hists, snap160_origs = r160, h160, o160
            if len(r235):
                snap235_rows, snap235_hists, snap235_origs = r235, h235, o235
            print(
                f"[Refresh update={update}] Collected {len(snap160_rows)}/{len(all_full)} tick=160 & "
                f"{len(snap235_rows)}/{len(all_full)} tick=235 snapshots.",
                flush=True,
            )

        hs, pts, orgs, cfs, acts, oldlps, vals, rews, dones = [], [], [], [], [], [], [], [], []
        batch_stats = []
        with torch.no_grad():
            for _ in range(args.steps):
                h = torch.from_numpy(histories.copy()).to(device)
                cf = critic_features(h, phase_tick, origin, env.metrics, env.state)
                mean = model.raw_mean(h, phase_tick, origin)
                dist = Normal(mean, logstd.clamp(-3.0, args.max_logstd).exp())
                act = dist.sample()
                tgt = model.target_from_action(act, phase_tick, origin)
                base_q, _, _ = model.base_and_scale(phase_tick, origin)

                hs.append(h)
                pts.append(phase_tick.clone())
                orgs.append(origin.clone())
                cfs.append(cf)
                acts.append(act)
                oldlps.append(dist.log_prob(act).sum(1))
                vals.append(critic(cf).squeeze(1))

                last_sent = STAND + 4.0 * SCALE * histories[:, -1, 50:72]
                obs, met = env.step(tgt.cpu().numpy())
                age_now = phase_tick.cpu().numpy()
                phase_tick = phase_tick + 1.0
                ep_steps += 1

                z = met[:, 0]
                up_z = met[:, 1]
                foot_F = met[:, 2]
                body = met[:, 3]
                v_xy = met[:, 7]
                w_norm = met[:, 8]
                up_y = -obs[:, 48]

                q_true = env.state[:, 7:29]
                qd_rms = np.sqrt(np.mean(env.state[:, 35:57] ** 2, axis=1))
                base_np = base_q.cpu().numpy()
                q_err_sq = np.mean((q_true - base_np) ** 2, axis=1)
                stand_err_sq = np.mean((q_true - STAND) ** 2, axis=1)
                stand_rmse = np.sqrt(stand_err_sq)
                act_sq = np.mean(act.cpu().numpy() ** 2, axis=1)

                upright = (z >= 0.385) & (up_z > 0.96) & (foot_F > 15.0) & (body == 0.0)
                quiet = upright & (v_xy < 0.16) & (w_norm < 0.65) & (qd_rms < 0.80) & (stand_rmse < 0.22)
                feet_only = (foot_F > 15.0) & (body == 0.0)

                early = age_now < 175
                mid = (age_now >= 175) & (age_now < 295)
                late = age_now >= 255

                r_early = np.where((met[:, 6] > 0.25) | (met[:, 5] > 6.0), 0.1, 1.6) * np.exp(-2.5 * q_err_sq) - 0.10 * act_sq
                r_mid = (
                    2.5 * up_z
                    + 10.0 * np.clip(up_z - prev_up_z, -0.2, 0.2)
                    - 2.5 * np.maximum(0.0, -up_z - 0.10)
                    + (up_z > 0.15) * (10.0 * np.clip(z - prev_z, -0.1, 0.1) + 2.0 * np.clip((z - 0.22) / 0.20, 0.0, 1.0))
                    + 2.0 * feet_only
                    - 1.5 * (age_now > 210) * (up_y ** 2)
                    + 0.6 * np.exp(-1.5 * q_err_sq)
                )
                r_late = (
                    3.0 * np.exp(-10.0 * np.maximum(0.0, 1.0 - up_z))
                    + 3.0 * np.exp(-25.0 * (z - 0.420) ** 2)
                    + 2.0 * ((foot_F > 20.0) & (body == 0.0))
                    + 5.5 * quiet
                    + 2.5 * upright * np.exp(-4.0 * v_xy ** 2 - 1.2 * w_norm ** 2 - 0.6 * qd_rms ** 2)
                    + 2.0 * upright * np.exp(-3.0 * stand_err_sq)
                )

                r = early * r_early + mid * r_mid + late * r_late
                r -= 0.35 * np.maximum(met[:, 4] - 9.0, 0.0) ** 2 + 3.5 * np.maximum(met[:, 4] - 10.8, 0.0) + 10.0 * (met[:, 4] > 11.6)
                r -= 0.60 * np.maximum(met[:, 5] - 7.0, 0.0) ** 2 + 4.0 * np.maximum(met[:, 5] - 8.8, 0.0) + 10.0 * (met[:, 5] > 9.5)
                r -= 8.00 * np.maximum(met[:, 6] - 0.3, 0.0) ** 2 + 15.0 * np.maximum(met[:, 6] - 1.0, 0.0) + 25.0 * (met[:, 6] > 1.6)
                r -= (0.05 + 0.03 * late) * np.mean((tgt.cpu().numpy() - last_sent) ** 2, axis=1)

                fp_now = (age_now > 210) & (up_z < -0.35)
                faceplant_streak = np.where(fp_now, faceplant_streak + 1, 0)
                collapse = (
                    (faceplant_streak >= 8)
                    | ((age_now > 270) & ((z < 0.18) | (up_z < 0.10)))
                    | ((age_now > 300) & ((z < 0.25) | (up_z < 0.50) | (body > 0.5)))
                )
                blowup = (met[:, 4] > 45.0) | (met[:, 6] > 35.0)
                early_fail = collapse | blowup
                done = early_fail | (ep_steps >= max_ticks)
                r = np.clip(r - 15.0 * early_fail, -100.0, 30.0)

                prev_up_z = up_z.copy()
                prev_z = z.copy()
                histories = np.concatenate((histories[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                rews.append(torch.tensor(r * 0.1, dtype=torch.float32, device=device))
                dones.append(torch.tensor(done, device=device))
                batch_stats.append(
                    [float(r.mean()), float(quiet[late].mean() if late.any() else 0.0), float(met[:, 4].max()), float(met[:, 5].max()), float(met[:, 6].max())]
                )
                if done.any():
                    reset_envs(done)

            next_cf = critic_features(torch.from_numpy(histories.copy()).to(device), phase_tick, origin, env.metrics, env.state)
            following = critic(next_cf).squeeze(1)
            gae = torch.zeros_like(following)
            advs = []
            for t in reversed(range(args.steps)):
                alive = (~dones[t]).float()
                gae = rews[t] + 0.995 * alive * following - vals[t] + 0.995 * 0.95 * alive * gae
                advs.append(gae)
                following = vals[t]
            advantage = torch.cat(advs[::-1])
            returns = advantage + torch.cat(vals)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        h_b, pt_b, org_b, cf_b, act_b, lp_b = map(torch.cat, (hs, pts, orgs, cfs, acts, oldlps))
        lr_now = args.lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * update / args.updates)))
        for pg_opt in optimizer.param_groups:
            pg_opt["lr"] = lr_now
        for _ in range(4):
            for ix in torch.randperm(len(act_b), device=device).split(1152):
                dist = Normal(model.raw_mean(h_b[ix], pt_b[ix], org_b[ix]), logstd.clamp(-3.0, args.max_logstd).exp())
                ratio = (dist.log_prob(act_b[ix]).sum(1) - lp_b[ix]).exp()
                pg = torch.minimum(ratio * advantage[ix], ratio.clamp(0.8, 1.2) * advantage[ix])
                loss = -pg.mean() + 0.3 * (critic(cf_b[ix]).squeeze(1) - returns[ix]).square().mean() - 0.001 * dist.entropy().sum(1).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()

        bs = np.array(batch_stats)
        row = dict(
            update=update,
            steps=update * args.envs * args.steps,
            elapsed_s=round(time.monotonic() - start_t, 2),
            reward=float(bs[:, 0].mean()),
            quiet_fraction=float(bs[:, 1].mean()),
            peak_qd=float(bs[:, 2].max()),
            ground_mm=float(bs[:, 3].max()),
            self_mm=float(bs[:, 4].max()),
        )
        train_log.append(row)
        with (out / "metrics.jsonl").open("a") as fp:
            fp.write(json.dumps(row) + "\n")

        if update == 1 or update % args.eval_every == 0 or update == args.updates:
            record_validation(update, row["steps"], row)

    best_ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    export_model = SupineRiseNetV3(faceup_cmd).eval()
    export_model.load_state_dict(best_ckpt["actor"])
    sample = torch.zeros((1, 10, 74))
    sample[:, :, 49] = -1.0
    torch.onnx.export(
        export_model,
        (sample, torch.zeros(1, 26)),
        str(out / "supine_expert.onnx"),
        input_names=["rec_obs_hist", "rec_state"],
        output_names=["joint_target", "next_state"],
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(out / "supine_expert.onnx"), providers=["CPUExecutionProvider"])
    rng_t = np.random.default_rng(2)
    th = torch.tensor(rng_t.normal(0, 0.2, (1, 10, 74)), dtype=torch.float32)
    ts = torch.tensor([[0.0, 6.0, 1.0, 200.0] + list(STAND)], dtype=torch.float32)
    with torch.no_grad():
        pt_y, pt_s = export_model(th, ts)
    ox_y, ox_s = sess.run(["joint_target", "next_state"], {"rec_obs_hist": th.numpy(), "rec_state": ts.numpy()})
    ort_diff = max(float(np.max(np.abs(pt_y.numpy() - ox_y))), float(np.max(np.abs(pt_s.numpy() - ox_s))))
    assert ort_diff < 1e-5, f"Stage B v3 ONNX parity failed: {ort_diff}"
    plot_curves(out, train_log, val_log)

    summary = dict(
        best_val=best_ckpt["val"],
        best_update=int(best_ckpt["update"]),
        onnx_sha256=hashlib.sha256((out / "supine_expert.onnx").read_bytes()).hexdigest(),
        ort_parity=ort_diff,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("STAGE B V3 DONE:", json.dumps(summary), flush=True)
    env.close()
    val_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pools",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/pools"),
    )
    p.add_argument(
        "--side-pool-dir",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/side_v2_05_routed"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/supine_v2_03"),
    )
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--envs", type=int, default=96)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--max-logstd", type=float, default=-0.5)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--polish", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=94303)
    run(p.parse_args())
