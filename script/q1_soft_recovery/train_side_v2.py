"""Stage A: Train closed-loop neural side-to-supine recovery policy under exact 2 ms physics."""
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


class SideRollNet(nn.Module):
    """ONNX-compatible closed-loop side-to-supine controller with immediate bottom-arm clearance."""

    def __init__(self, faceup_cmd0: np.ndarray, residual_scale: float = 0.85):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(LO, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(HI, dtype=torch.float32))
        goal = np.array(faceup_cmd0, dtype=np.float32).copy()
        # Keep shoulders slightly abducted when arriving at supine to avoid hip-wrist brush.
        goal[15] = max(float(goal[15]), 0.25)
        goal[19] = min(float(goal[19]), -0.25)
        self.register_buffer("goal_cmd", torch.tensor(np.clip(goal, LO, HI), dtype=torch.float32))

        delta_r = np.zeros(22, dtype=np.float32)
        delta_r[[18, 19, 20, 21]] = [-0.70, -0.55, -0.55, -0.50]
        delta_r[[12, 13]] = [-0.35, -0.20]
        delta_r[[6, 7, 8, 9]] = [-0.20, 0.12, -0.04, -0.45]
        delta_r[1] = 0.04
        delta_l = np.zeros(22, dtype=np.float32)
        delta_l[[14, 15, 16, 17]] = [-0.70, 0.55, 0.55, -0.50]
        delta_l[[12, 13]] = [0.35, 0.20]
        delta_l[[0, 1, 2, 3]] = [-0.20, -0.12, 0.04, -0.45]
        delta_l[7] = -0.04
        self.register_buffer("delta_r", torch.tensor(delta_r, dtype=torch.float32))
        self.register_buffer("delta_l", torch.tensor(delta_l, dtype=torch.float32))

        self.encoder = nn.Sequential(nn.Linear(74, 64), nn.ELU())
        self.actor = nn.Sequential(
            nn.Linear(258, 192),
            nn.ELU(),
            nn.Linear(192, 128),
            nn.ELU(),
            nn.Linear(128, 22),
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def _base_and_ctx(self, rec_obs_hist: torch.Tensor, age: torch.Tensor, side: torch.Tensor, origin: torch.Tensor):
        side_sign = 2.0 * side - 1.0  # +1 = right down, -1 = left down
        right_down = (side > 0.5)[:, None]
        q_clear = (origin + torch.where(right_down, self.delta_r[None], self.delta_l[None])).clamp(self.lo, self.hi)

        # Immediate 12-DOF clearance ramp over ticks 0..1 (alpha=1 at age>=0.5).
        alpha_clear = ((age + 1.0) / 1.5).clamp(0.0, 1.0)
        p1 = origin + alpha_clear[:, None] * (q_clear - origin)

        # After tick 12, smoothly roll toward supine goal pose over ticks 12..100.
        phase = ((age + 1.0 - 12.0) / 88.0).clamp(0.0, 1.0)
        w_goal = 0.5 - 0.5 * torch.cos(torch.pi * phase)
        base_target = p1 + w_goal[:, None] * (self.goal_cmd[None] - p1)
        return base_target, side_sign

    def actor_features(self, rec_obs_hist: torch.Tensor, age: torch.Tensor, side: torch.Tensor):
        side_sign = 2.0 * side - 1.0
        enc = self.encoder(rec_obs_hist[:, -4:, :].clamp(-12.0, 12.0)).flatten(1)
        return torch.cat((enc, side_sign[:, None], (age[:, None] / 140.0).clamp(0.0, 1.0)), 1)

    def raw_mean(self, rec_obs_hist: torch.Tensor, age: torch.Tensor, side: torch.Tensor):
        return self.actor(self.actor_features(rec_obs_hist, age, side))

    def target_from_action(
        self, action: torch.Tensor, rec_obs_hist: torch.Tensor, age: torch.Tensor, side: torch.Tensor, origin: torch.Tensor
    ):
        base_target, _ = self._base_and_ctx(rec_obs_hist, age, side, origin)
        scale = self.residual_scale + 0.25 * (1.0 - ((age - 8.0) / 8.0).clamp(0.0, 1.0))
        return (base_target + scale[:, None] * action.tanh()).clamp(self.lo, self.hi)

    def forward(self, rec_obs_hist: torch.Tensor, rec_state: torch.Tensor):
        x = rec_obs_hist[:, -1]
        q = x[:, :22] + self.stand
        begin = (rec_state[:, 0] == 0) & (rec_state[:, 1] != 10)
        age = torch.where(begin, torch.zeros_like(rec_state[:, 0]), rec_state[:, 0])
        side = torch.where(begin, (x[:, 48] < 0).float(), rec_state[:, 2]).clamp(0.0, 1.0)
        origin = torch.where(begin[:, None], q, rec_state[:, 4:])

        action = self.raw_mean(rec_obs_hist, age, side)
        target = self.target_from_action(action, rec_obs_hist, age, side, origin)

        w_norm = torch.linalg.vector_norm(x[:, 44:47] * 4.0, dim=1)
        quiet_now = (-x[:, 47] > 0.84) & (w_norm < 1.45)
        quiet_cnt = torch.where(quiet_now, torch.where(begin, torch.zeros_like(rec_state[:, 3]), rec_state[:, 3]) + 1.0, 0.0)
        next_state = torch.cat((torch.stack((age + 1.0, age * 0.0 + 10.0, side, quiet_cnt), 1), origin), 1)
        return target, next_state


def eval_validation(env: Plant, model: SideRollNet, val_rows: np.ndarray, horizon: int, device: torch.device):
    n_val = len(val_rows)
    assert n_val <= env.n
    padded = np.tile(val_rows[0], (env.n, 1))
    padded[:n_val] = val_rows
    obs = env.reset(padded)
    histories = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
    side_np = (obs[:, 48] < 0).astype(np.float32)
    origin_np = (obs[:, :22] + STAND).astype(np.float32)
    age = torch.zeros(env.n, dtype=torch.float32, device=device)
    side = torch.from_numpy(side_np).to(device)
    origin = torch.from_numpy(origin_np).to(device)

    ep_peaks = np.zeros((n_val, 3), dtype=np.float64)
    quiet_cnt = np.zeros(n_val, dtype=np.int32)
    reached = np.zeros(n_val, dtype=bool)
    first_reach_peaks = np.zeros((n_val, 3), dtype=np.float64)

    with torch.no_grad():
        for tk in range(horizon):
            h = torch.from_numpy(histories).to(device)
            act = model.raw_mean(h, age, side)
            tgt = model.target_from_action(act, h, age, side, origin).cpu().numpy()
            obs, met = env.step(tgt)
            age = age + 1.0
            histories = np.concatenate((histories[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
            for j in range(n_val):
                if not reached[j]:
                    ep_peaks[j] = np.maximum(ep_peaks[j], met[j, [4, 5, 6]])
                    up_x = -obs[j, 47]
                    w_norm = float(np.linalg.norm(obs[j, 44:47] * 4.0))
                    if up_x > 0.84 and w_norm < 1.45:
                        quiet_cnt[j] += 1
                    else:
                        quiet_cnt[j] = 0
                    if quiet_cnt[j] >= 10:
                        reached[j] = True
                        first_reach_peaks[j] = ep_peaks[j]
            if reached.all():
                break

    soft_ok = reached & (first_reach_peaks[:, 0] <= 12.0) & (first_reach_peaks[:, 1] <= 10.0) & (first_reach_peaks[:, 2] <= 2.0)
    viol = np.maximum(ep_peaks / np.array([12.0, 10.0, 2.0]) - 1.0, 0.0)
    return dict(
        supine=int(reached.sum()),
        simultaneous_soft=int(soft_ok.sum()),
        total=n_val,
        max_qd=float(ep_peaks[:, 0].max()),
        max_ground_mm=float(ep_peaks[:, 1].max()),
        max_self_mm=float(ep_peaks[:, 2].max()),
        max_rel_violation=float(viol.max()),
        per_trial_self_gt2=[int(j) for j in np.flatnonzero(ep_peaks[:, 2] > 2.0)],
        per_trial_qd_gt12=[int(j) for j in np.flatnonzero(ep_peaks[:, 0] > 12.0)],
    )


def plot_curves(out: Path, train_rows: list[dict], val_rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    x = np.array([r["steps"] / 1e6 for r in train_rows])
    keys = [
        ("reward", "Sampled reward"),
        ("quiet_fraction", "Sampled supine quiet fraction"),
        ("peak_qd", "Batch max joint speed (rad/s)"),
        ("ground_mm", "Batch max ground pen (mm)"),
        ("self_mm", "Batch max arm-leg pen (mm)"),
        ("val_simultaneous", "Unseen val simultaneous soft / 32"),
    ]
    for ax, (k, title) in zip(axes.flat, keys):
        if k == "val_simultaneous" and val_rows:
            vx = np.array([r["steps"] / 1e6 for r in val_rows])
            ax.plot(vx, [r["supine"] for r in val_rows], "o--", label="supine/32")
            ax.plot(vx, [r["simultaneous_soft"] for r in val_rows], "s-", label="soft_supine/32")
            ax.axhline(30, color="green", ls="--", lw=1)
            ax.legend(fontsize=8)
        elif k in train_rows[0]:
            y = np.array([r[k] for r in train_rows])
            ax.plot(x, y, lw=0.7, alpha=0.6)
            if len(y) >= 25:
                sm = np.convolve(y, np.ones(25) / 25, mode="valid")
                ax.plot(x[24:], sm, lw=2.0)
        ax.set_title(title)
        ax.set_xlabel("Million 20 ms steps")
        ax.grid(alpha=0.25)
    for ax, v in ((axes[0, 2], 12.0), (axes[1, 0], 10.0), (axes[1, 1], 2.0)):
        ax.axhline(v, color="black", ls="--", lw=1)
    fig.suptitle("Q1 Stage A Side-to-Supine PPO | Closed-loop 2 ms plant")
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
    shutil.copy2(Path(__file__), out / "train_side_v2.py")
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    train_rows_pool = np.load(args.pools / "side_train.npz")["rows"]
    val_rows_pool = np.load(args.pools / "side_val.npz")["rows"]
    cmd0 = sample_record("q1_official_standup_faceup_20261120")["command_position"][0]

    env = Plant(args.envs, out / "scene", delay=11)
    val_env = Plant(max(32, len(val_rows_pool)), out / "val_scene", delay=11)

    model = SideRollNet(cmd0).to(device)
    critic = nn.Sequential(nn.Linear(271, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    logstd = nn.Parameter(torch.full((22,), -1.7, device=device))
    optimizer = torch.optim.Adam([*model.parameters(), *critic.parameters(), logstd], lr=args.lr)

    # Identify top-40 hardest side starts using zero-residual base over 20 ticks.
    base_diff = []
    with torch.no_grad():
        for s0 in range(0, len(train_rows_pool), val_env.n):
            b = train_rows_pool[s0 : s0 + val_env.n]
            m = len(b)
            pad = np.tile(b[0], (val_env.n, 1))
            pad[:m] = b
            obs_b = val_env.reset(pad)
            sd_b = torch.from_numpy((obs_b[:, 48] < 0).astype(np.float32)).to(device)
            org_b = torch.from_numpy((obs_b[:, :22] + STAND).astype(np.float32)).to(device)
            h_zero = torch.from_numpy(np.repeat(obs_b[:, None, :].astype(np.float32), 10, axis=1)).to(device)
            pks_b = np.zeros((m, 3), dtype=np.float64)
            z_act = torch.zeros((val_env.n, 22), dtype=torch.float32, device=device)
            for tk in range(20):
                ag_b = torch.full((val_env.n,), float(tk), dtype=torch.float32, device=device)
                tgt_b = model.target_from_action(z_act, h_zero, ag_b, sd_b, org_b).cpu().numpy()
                _, met_b = val_env.step(tgt_b)
                pks_b = np.maximum(pks_b, met_b[:m, [4, 5, 6]])
            base_diff.append(pks_b[:, 2] + 0.5 * np.maximum(0.0, pks_b[:, 0] - 11.0))
    difficulty = np.concatenate(base_diff, axis=0)
    hard_pool = np.argsort(difficulty)[-40:]
    print(f"Hard pool (top 40/200) difficulty range: [{difficulty[hard_pool[0]]:.2f}, {difficulty[hard_pool[-1]]:.2f}]", flush=True)

    histories = np.zeros((args.envs, 10, 74), np.float32)
    age = torch.zeros(args.envs, dtype=torch.float32, device=device)
    side = torch.zeros(args.envs, dtype=torch.float32, device=device)
    origin = torch.zeros((args.envs, 22), dtype=torch.float32, device=device)
    ages = np.zeros(args.envs, dtype=np.int32)
    quiet_streak = np.zeros(args.envs, dtype=np.int32)
    ep_peaks = np.zeros((args.envs, 3), dtype=np.float64)
    prev_up_x = np.zeros(args.envs, dtype=np.float64)

    def reset_envs(mask: np.ndarray):
        use_hard = rng.random(args.envs) < 0.50
        idxs = np.where(
            use_hard,
            rng.choice(hard_pool, size=args.envs, replace=True),
            rng.integers(len(train_rows_pool), size=args.envs),
        )
        batch = train_rows_pool[idxs].copy()
        # Inject mild initial joint/velocity perturbation on 35% of resets for generalization.
        perturb = rng.random(args.envs) < 0.35
        if perturb.any():
            batch[perturb, 7:29] = np.clip(batch[perturb, 7:29] + rng.uniform(-0.03, 0.03, (perturb.sum(), 22)), LO, HI)
            batch[perturb, 57:] = batch[perturb, 7:29]
        frames = env.reset(batch, mask.astype(np.int32))
        m_t = torch.from_numpy(mask).to(device)
        histories[mask] = frames[mask, None, :].astype(np.float32)
        age[m_t] = 0.0
        side[m_t] = torch.from_numpy((frames[mask, 48] < 0).astype(np.float32)).to(device)
        origin[m_t] = torch.from_numpy((frames[mask, :22] + STAND).astype(np.float32)).to(device)
        ages[mask] = 0
        quiet_streak[mask] = 0
        ep_peaks[mask] = 0.0
        prev_up_x[mask] = -frames[mask, 47]

    reset_envs(np.ones(args.envs, dtype=bool))
    metric_scale = torch.tensor([1, 1, 0.01, 1, 0.1, 0.1, 0.1, 1, 0.2, 1, 1, 1, 1], dtype=torch.float32, device=device)

    def critic_features(h, ag, sd, met_np):
        af = model.actor_features(h, ag, sd)
        priv = torch.from_numpy(met_np.astype(np.float32)).to(device) * metric_scale
        return torch.cat((af, priv), 1)

    train_log, val_log = [], []
    best_score = (-1, -1, -1e9)
    start_t = time.monotonic()

    for update in range(1, args.updates + 1):
        hs, ags, sds, orgs, cfs, acts, oldlps, vals, rews, dones = [], [], [], [], [], [], [], [], [], []
        batch_stats = []
        with torch.no_grad():
            for _ in range(args.steps):
                h = torch.from_numpy(histories.copy()).to(device)
                cf = critic_features(h, age, side, env.metrics)
                mean = model.raw_mean(h, age, side)
                dist = Normal(mean, logstd.clamp(-3.0, -0.4).exp())
                act = dist.sample()
                tgt = model.target_from_action(act, h, age, side, origin)

                hs.append(h)
                ags.append(age.clone())
                sds.append(side.clone())
                orgs.append(origin.clone())
                cfs.append(cf)
                acts.append(act)
                oldlps.append(dist.log_prob(act).sum(1))
                vals.append(critic(cf).squeeze(1))

                last_sent = STAND + 4.0 * SCALE * histories[:, -1, 50:72]
                obs, met = env.step(tgt.cpu().numpy())
                ages += 1
                age += 1.0
                ep_peaks = np.maximum(ep_peaks, met[:, [4, 5, 6]])

                up_x = -obs[:, 47]
                w_norm = np.linalg.norm(obs[:, 44:47] * 4.0, axis=1)
                qd_max_obs = np.abs(obs[:, 22:44] * 10.0).max(axis=1)
                quiet_now = (up_x > 0.84) & (w_norm < 1.45) & (qd_max_obs < 4.5)
                quiet_streak = np.where(quiet_now, quiet_streak + 1, 0)
                supine_goal = (up_x > 0.5).astype(np.float64) * np.exp(-2.0 * np.mean((env.state[:, 7:29] - model.goal_cmd.cpu().numpy()) ** 2, axis=1))

                r = (
                    3.0 * up_x
                    + 10.0 * np.clip(up_x - prev_up_x, -0.3, 0.3)
                    - 3.0 * np.maximum(0.0, -up_x - 0.25)
                    + 1.5 * supine_goal
                    + 3.0 * quiet_now
                )
                r -= 0.35 * np.maximum(met[:, 4] - 9.5, 0.0) ** 2 + 3.0 * np.maximum(met[:, 4] - 11.2, 0.0) + 10.0 * (met[:, 4] > 11.8)
                r -= 0.50 * np.maximum(met[:, 5] - 7.5, 0.0) ** 2 + 3.0 * np.maximum(met[:, 5] - 9.2, 0.0) + 10.0 * (met[:, 5] > 9.8)
                r -= 4.00 * np.maximum(met[:, 6] - 0.5, 0.0) ** 2 + 8.0 * np.maximum(met[:, 6] - 1.5, 0.0) + 15.0 * (met[:, 6] > 1.9)
                r -= 0.04 * np.mean((tgt.cpu().numpy() - last_sent) ** 2, axis=1)

                success_done = quiet_streak >= 10
                soft_ep = (ep_peaks[:, 0] <= 11.8) & (ep_peaks[:, 1] <= 9.6) & (ep_peaks[:, 2] <= 1.85)
                r += np.where(success_done, np.where(soft_ep, 20.0, 2.0), 0.0)

                blowup = (met[:, 4] > 45.0) | (met[:, 6] > 30.0)
                done = success_done | blowup | (ages >= args.horizon)
                r = np.clip(r - 15.0 * blowup, -100.0, 30.0)

                prev_up_x = up_x.copy()
                histories = np.concatenate((histories[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                rews.append(torch.tensor(r * 0.1, dtype=torch.float32, device=device))
                dones.append(torch.tensor(done, device=device))
                batch_stats.append([float(r.mean()), float(quiet_now.mean()), float(met[:, 4].max()), float(met[:, 5].max()), float(met[:, 6].max())])
                if done.any():
                    reset_envs(done)

            next_cf = critic_features(torch.from_numpy(histories.copy()).to(device), age, side, env.metrics)
            following = critic(next_cf).squeeze(1)
            gae = torch.zeros_like(following)
            advs = []
            for t in reversed(range(args.steps)):
                alive = (~dones[t]).float()
                gae = rews[t] + 0.99 * alive * following - vals[t] + 0.99 * 0.95 * alive * gae
                advs.append(gae)
                following = vals[t]
            advantage = torch.cat(advs[::-1])
            returns = advantage + torch.cat(vals)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        h, ag, sd, cf_b, act_b, lp_b = map(torch.cat, (hs, ags, sds, cfs, acts, oldlps))
        lr_now = args.lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * update / args.updates)))
        for pg_opt in optimizer.param_groups:
            pg_opt["lr"] = lr_now
        for _ in range(4):
            for ix in torch.randperm(len(act_b), device=device).split(512):
                dist = Normal(model.raw_mean(h[ix], ag[ix], sd[ix]), logstd.clamp(-3.0, -0.4).exp())
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

        if update == 1 or update % 50 == 0 or update == args.updates:
            v = eval_validation(val_env, model, val_rows_pool, args.horizon, device)
            v["update"] = update
            v["steps"] = row["steps"]
            val_log.append(v)
            with (out / "val_metrics.jsonl").open("a") as fp:
                fp.write(json.dumps(v) + "\n")
            key = (v["simultaneous_soft"], v["supine"], -v["max_rel_violation"])
            ckpt = dict(actor=model.state_dict(), critic=critic.state_dict(), logstd=logstd.detach(), update=update, val=v)
            torch.save(ckpt, out / f"checkpoint_{update:05d}.pt")
            if key > best_score:
                best_score = key
                torch.save(ckpt, out / "best.pt")
            print(f"[SideV2 upd={update:3d}/{args.updates}] r={row['reward']:+.2f} batch_peaks=({row['peak_qd']:.1f},{row['ground_mm']:.1f},{row['self_mm']:.1f}) | VAL soft={v['simultaneous_soft']}/{v['total']} sup={v['supine']}/{v['total']} peaks=({v['max_qd']:.2f},{v['max_ground_mm']:.2f},{v['max_self_mm']:.2f}) bad_s={v['per_trial_self_gt2']} bad_qd={v['per_trial_qd_gt12']}", flush=True)

    # Export best checkpoint ONNX and generate side_to_supine pools + training plots.
    best_ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    export_model = SideRollNet(cmd0).eval()
    export_model.load_state_dict(best_ckpt["actor"])
    sample = torch.zeros((1, 10, 74))
    sample[:, :, 49] = -1.0
    torch.onnx.export(
        export_model,
        (sample, torch.zeros(1, 26)),
        str(out / "side_expert.onnx"),
        input_names=["rec_obs_hist", "rec_state"],
        output_names=["joint_target", "next_state"],
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "side_expert.onnx"), providers=["CPUExecutionProvider"])
    rng_t = np.random.default_rng(1)
    th = torch.tensor(rng_t.normal(0, 0.2, (1, 10, 74)), dtype=torch.float32)
    ts = torch.zeros((1, 26), dtype=torch.float32)
    with torch.no_grad():
        pt_y, pt_s = export_model(th, ts)
    ox_y, ox_s = sess.run(["joint_target", "next_state"], {"rec_obs_hist": th.numpy(), "rec_state": ts.numpy()})
    ort_diff = max(float(np.max(np.abs(pt_y.numpy() - ox_y))), float(np.max(np.abs(pt_s.numpy() - ox_s))))
    assert ort_diff < 1e-5, f"Stage A ONNX parity failed: {ort_diff}"
    plot_curves(out, train_log, val_log)

    # Roll out best SideRollNet on side_train & side_val to capture side_to_supine transition pools.
    best_gpu = SideRollNet(cmd0).to(device).eval()
    best_gpu.load_state_dict(best_ckpt["actor"])
    for split, src_rows, dst_name in [
        ("train", train_rows_pool, "side_to_supine_train.npz"),
        ("val", val_rows_pool, "side_to_supine_val.npz"),
    ]:
        captured_rows = []
        reached = 0
        peaks_all = []
        for s0 in range(0, len(src_rows), val_env.n):
            batch = src_rows[s0 : s0 + val_env.n]
            m = len(batch)
            pad = np.tile(batch[0], (val_env.n, 1))
            pad[:m] = batch
            obs = val_env.reset(pad)
            hist = np.repeat(obs[:, None, :].astype(np.float32), 10, axis=1)
            ag = torch.zeros(val_env.n, dtype=torch.float32, device=device)
            sd = torch.from_numpy((obs[:, 48] < 0).astype(np.float32)).to(device)
            org = torch.from_numpy((obs[:, :22] + STAND).astype(np.float32)).to(device)
            sent = pad[:, 57:].astype(np.float32).copy()
            qcnt = np.zeros(m, dtype=np.int32)
            pks = np.zeros((m, 3), dtype=np.float64)
            got = [None] * m
            kp32, kd32, eff32, stand32, lo32, hi32 = [x.astype(np.float32) for x in (KP, KD, EFF, STAND, LO, HI)]
            margin = 0.5 * eff32 / kp32
            with torch.no_grad():
                for tk in range(args.horizon):
                    h_t = torch.from_numpy(hist).to(device)
                    tgt = best_gpu.target_from_action(best_gpu.raw_mean(h_t, ag, sd), h_t, ag, sd, org).cpu().numpy()
                    for j in range(m):
                        if got[j] is None:
                            q_o = obs[j, :22].astype(np.float32) + stand32
                            qd_o = obs[j, 22:44].astype(np.float32) * 10.0
                            raw = np.clip(tgt[j].astype(np.float32), lo32 - margin, hi32 + margin)
                            tau = np.clip(kp32 * (raw - q_o) - kd32 * qd_o, -eff32, eff32)
                            des = np.clip(q_o + (tau + kd32 * qd_o) / kp32, stand32 - 2.0, stand32 + 2.0)
                            for _ in range(10):
                                sent[j] = sent[j] + np.clip(des - sent[j], -0.2, 0.2)
                    obs, met = val_env.step(tgt)
                    ag += 1.0
                    hist = np.concatenate((hist[:, 1:], obs[:, None, :].astype(np.float32)), axis=1)
                    for j in range(m):
                        if got[j] is None:
                            pks[j] = np.maximum(pks[j], met[j, [4, 5, 6]])
                            if -obs[j, 47] > 0.84 and np.linalg.norm(obs[j, 44:47] * 4.0) < 1.45:
                                qcnt[j] += 1
                            else:
                                qcnt[j] = 0
                            if qcnt[j] >= 8:
                                got[j] = np.r_[val_env.state[j], sent[j].astype(np.float64)].copy()
                    if all(g is not None for g in got):
                        break
            for j in range(m):
                peaks_all.append(pks[j])
                if got[j] is not None:
                    reached += 1
                    captured_rows.append(got[j])
        if captured_rows:
            np.savez(out / dst_name, rows=np.stack(captured_rows))
        p_arr = np.array(peaks_all)
        print(f"Side->Supine {split}: reached={reached}/{len(src_rows)} peaks_max=({p_arr[:,0].max():.2f},{p_arr[:,1].max():.2f},{p_arr[:,2].max():.2f})", flush=True)

    summary = dict(
        best_val=best_ckpt["val"],
        best_update=int(best_ckpt["update"]),
        onnx_sha256=hashlib.sha256((out / "side_expert.onnx").read_bytes()).hexdigest(),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("STAGE A DONE:", json.dumps(summary), flush=True)
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
        "--out",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/side_v2_01"),
    )
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=94202)
    run(p.parse_args())
