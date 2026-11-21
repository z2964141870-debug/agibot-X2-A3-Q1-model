"""Fit a proprioceptive correction during the supine rise's foot-loading phase."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import Plant, mj
from command_support import sample_record
from policy import export
from roll_to_rise import RollToRise


START, END = 250, 360
JOINTS = (0, 3, 4, 14, 17)


def basis_matrix():
    matrix = np.zeros((5, 22))
    for row, left in enumerate(JOINTS):
        matrix[row, [left, left + (6 if left < 12 else 4)]] = 1
    return matrix


def snapshots(run, model):
    events = json.loads((run / "results.json").read_text())
    poses, references, ids = [], [], []
    with np.load(run / "commands.npz") as command:
        for row in events:
            key = "trial_%02d" % row["trial"]
            records = command[key]
            # Recorded expert state is post-action. At next tick its age
            # indexes the command that is about to be generated.
            active = (records[:, 3] == 6) & (records[:, 4] == 1)
            hit = np.flatnonzero(active & (records[:, 5] == START))
            if not len(hit):
                continue
            j = hit[0]
            with np.load(run / (key + ".npz")) as trial:
                qpos = trial["qpos"]
                v = np.zeros(model.nv)
                mj.mj_differentiatePos(model, v, .02, qpos[j-1], qpos[j])
                poses.append(np.r_[qpos[j], v, records[j, 50:72]])
                ids.append(row["trial"])
                if row["success"] and row["trial"] % 3 != 0:
                    sequence = qpos[j:j + END - START + 1]
                    references.append(1 - 2 * (sequence[:, 4]**2 + sequence[:, 5]**2))
    return np.stack(poses), np.median(references, axis=0), np.array(ids)


def fit(run, out, generations):
    out.mkdir(parents=True)
    n = 48
    env = Plant(n, out / "scene")
    poses, reference, ids = snapshots(run, env.model)
    np.savez(out / "snapshots.npz", initial=poses, ids=ids, reference=reference)
    cmd = sample_record("q1_official_standup_faceup_20261120")["command_position"]
    basis = basis_matrix()
    rng = np.random.default_rng(95631)
    def score(parameters, starts):
        obs = env.reset(poses[starts])
        peak = np.zeros((n, 3))
        best = np.full(n, -10.)
        for tick in range(START, END + 10):
            idx = min(tick - START, len(reference)-1)
            error = np.clip(reference[idx] + obs[:, 49], -.6, .6)
            window = np.sin(np.pi * idx/(END-START))**2
            raw = window * (parameters[:, :5] + parameters[:, 5:]*error[:, None])
            target = cmd[int((tick-25)*.75)] + raw @ basis
            obs, metrics = env.step(target)
            peak = np.maximum(peak, metrics[:, [4, 5, 6]])
            if tick >= 330:
                feet = (metrics[:, 2] > 10) & (metrics[:, 3] == 0)
                capture = (obs[:, 73] > .5) & (obs[:, 72] > .19) & (-obs[:, 49] > .42)
                value = 4*metrics[:, 1] + 4*metrics[:, 0] + 2*feet + 4*capture
                best = np.maximum(best, value)
        return best - .45*np.maximum(peak[:, 0]-11, 0) - .25*np.maximum(peak[:, 1]-8, 0) - .5*np.maximum(peak[:, 2]-1, 0)

    train = np.flatnonzero(ids % 3 != 0)
    valid = np.flatnonzero(ids % 3 == 0)
    # Evaluate every validation snapshot for every finalist, padding only to fit the plant.
    if not 1 <= len(valid) <= 16:
        raise ValueError("unexpected validation batch size")
    spread = np.array([.14]*5 + [.7]*5)
    mean = np.zeros(10)
    sigma = spread.copy()
    saved, best = mean.copy(), -np.inf
    for gen in range(generations):
        bank = rng.normal(mean, sigma, (16, 10))
        bank = np.clip(bank, -3*spread, 3*spread)
        bank[:2] = [saved, mean]
        batch = rng.choice(train, 3, replace=False)
        values = score(np.repeat(bank, 3, axis=0), np.tile(batch, 16)).reshape(16, 3)
        ranking = np.argsort(.5*values.mean(1)+.5*values.min(1))
        elite = bank[ranking[-4:]]
        mean = .3*mean + .7*elite.mean(0)
        sigma = np.maximum(.5*sigma + .5*elite.std(0), .12*spread)
        finalists = np.array([saved, mean, elite[-1]])
        p = np.repeat(finalists, len(valid), axis=0)
        s = np.tile(valid, 3)
        v = score(np.resize(p,(n,10)), np.resize(s,n))[:len(s)].reshape(3,-1)
        scores = .5*v.mean(1) + .5*v.min(1)
        i = int(np.argmax(scores))
        if scores[i] > best:
            best = float(scores[i])
            saved = finalists[i].copy()
            np.savez(out/"best.npz", parameters=saved, reference=reference, basis=basis)
        with (out/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(generation=gen, training=float(values[ranking[-1]].mean()),
                                    validation=best, baseline=float(scores[0])))+"\n")
    env.close()


class Transfer(torch.nn.Module):
    def __init__(self, roll, fit, strength=1., deviation=False):
        super().__init__()
        self.motion = RollToRise(roll, 10)
        self.strength = strength
        self.deviation = deviation
        for name in ("parameters", "reference", "basis"):
            self.register_buffer("fit_" + name, torch.tensor(fit[name], dtype=torch.float32))

    def forward(self, rec_obs_hist, rec_state):
        target, state = self.motion(rec_obs_hist, rec_state)
        age = rec_state[:, 3]
        i = (age - START).clamp(0, END-START).long()
        error = (self.fit_reference[i] + rec_obs_hist[:, -1, 49]).clamp(-.6, .6)
        active = (rec_state[:, 1] == 6) & (rec_state[:, 2] == 1) & (age > START) & (age < END)
        weight = torch.sin(torch.pi*i/(END-START)).square() * active
        if self.deviation:
            weight = weight * ((error-.012)/.075).clamp(0.,1.)
        correction = self.fit_parameters[:5] + error[:, None]*self.fit_parameters[5:]
        adjusted = target + self.strength*weight[:, None]*(correction @ self.fit_basis)
        return adjusted.clamp(self.motion.rise.lo, self.motion.rise.hi), state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("roll", type=Path)
    parser.add_argument("--generations", type=int, default=36)
    a = parser.parse_args()
    fit(a.run, a.out, a.generations)
    export(Transfer(np.load(a.roll), np.load(a.out/"best.npz")), a.out/"expert.onnx")
