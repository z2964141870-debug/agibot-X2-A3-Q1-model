"""Fit separate smooth left- and right-side approaches to a supine start."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from command_support import sample_record
from common import Plant, LO, HI, STAND, mj
from policy import export
from roll_to_rise import RollToRise


def load_sides(env, path):
    rows = [[], []]
    for item in json.loads(path.read_text()):
        if item["pose"] != "side":
            continue
        d = mj.MjData(env.model)
        d.qpos[:] = item["qpos"]
        d.ctrl[:] = d.qpos[7:]
        mj.mj_forward(env.model, d)
        mj.mj_step(env.model, d, nstep=50)
        side = int(d.xmat[env.model.body("pelvis").id, 7] > 0)
        rows[side].append(np.r_[d.qpos, d.qvel, d.qpos[7:]])
    return [np.stack(group) for group in rows]


def fit(out, train, valid, generations):
    out.mkdir(parents=True)
    env = Plant(48, out/"scene")
    training = load_sides(env, train)
    validation = load_sides(env, valid)
    goal = sample_record("q1_official_standup_faceup_20261120")["command_position"][0]
    low, high = np.maximum(LO, STAND-2), np.minimum(HI, STAND+2)
    rng = np.random.default_rng(95873)
    ticks = 24
    def rollout(keys, starts):
        env.reset(starts)
        best = np.full(len(keys), -20.)
        calm = np.zeros(len(keys))
        longest = calm.copy()
        peak = np.zeros((len(keys), 3))
        for tick in range(124):
            phase = min((tick+1)/ticks, 4.)
            k = min(int(phase), 3)
            alpha = .5 - .5*np.cos(np.pi*(phase-k))
            first = starts[:, 7:29] if k == 0 else keys[:, k-1]
            obs, met = env.step(first + alpha*(keys[:, k]-first))
            good = (-obs[:, 47] > .84) & (met[:, 8] < 1.45)
            calm = np.where(good, calm+1, 0.)
            longest = np.maximum(calm, longest)
            peak = np.maximum(peak, met[:, [4,5,6]])
            merit = -6*obs[:, 47] - .7*np.mean((env.state[:,7:29]-goal)**2,axis=1)
            best = np.maximum(best, merit + 2*np.minimum(calm/10, 1))
        violation = np.maximum(peak-[10., 8., 1.], 0) @ [.55,.22,.65]
        return best-violation, longest, peak
    learned = []
    for side, (tr, va) in enumerate(zip(training,validation,strict=True)):
        assert len(va) == 4 and len(tr) == 4
        origin = tr[:,7:29].mean(0)
        mean = np.stack([origin+(goal-origin)*k/4 for k in range(1,5)])
        sd = np.full_like(mean,.58)
        saved, best = mean.copy(), -np.inf
        for g in range(generations):
            bank = np.clip(rng.normal(mean,sd,(24,4,22)),low,high)
            bank[:2] = [saved,mean]
            ids = rng.choice(len(tr),2,replace=False)
            values,_,_ = rollout(np.repeat(bank,2,0),np.tile(tr[ids],(24,1)))
            ranking = np.argsort(values.reshape(24,2).min(1))
            elite = bank[ranking[-5:]]
            mean = .3*mean + .7*elite.mean(0)
            sd = np.maximum(.5*sd + .5*elite.std(0),.08)
            if g % 3 == 0 or g == generations-1:
                finals = np.concatenate((bank[ranking[-10:]],mean[None],saved[None]))
                vals, calm, peaks = rollout(np.repeat(finals,4,0),np.tile(va,(12,1)))
                scores = vals.reshape(12,4).min(1)
                pick = int(scores.argmax())
                if scores[pick] > best:
                    saved, best = finals[pick].copy(), float(scores[pick])
                with (out/"metrics.jsonl").open("a") as f:
                    f.write(json.dumps(dict(side=side,generation=g,training=float(values.max()),
                        validation=best, supine=int((calm.reshape(12,4)[pick]>=10).sum()),
                        peaks=peaks.reshape(12,4,3)[pick].max(0).tolist()))+"\n")
        learned.append(saved)
        np.savez(out/f"side_{side}.npz",keys=saved)
    env.close()
    return np.stack(learned)


class SideEntry(torch.nn.Module):
    def __init__(self, roll, keys):
        super().__init__()
        self.inner = RollToRise(roll, 10)
        self.register_buffer("keys", torch.tensor(keys,dtype=torch.float32))

    def forward(self, rec_obs_hist, rec_state):
        x = rec_obs_hist[:, -1]
        begin = (rec_state[:, 0] == 0) & (x[:, 48].abs() > .65) & (x[:, 48].abs() > x[:, 47].abs())
        active = begin | (rec_state[:, 1] == 10)
        age = torch.where(begin, 0., rec_state[:, 0])
        side = torch.where(begin, (x[:,48] < 0).float(), rec_state[:,2]).clamp(0,1)
        origin = torch.where(begin[:,None], x[:,:22]+self.inner.rise.crouch.stand, rec_state[:,4:])
        phase = ((age+1)/24).clamp(0,4)
        k = phase.long().clamp(0,3)
        a = .5-.5*torch.cos(torch.pi*(phase-k))
        first = torch.where((k==0)[:,None],origin,self.keys[side.long(),(k-1).clamp(0,3)])
        target = first+a[:,None]*(self.keys[side.long(),k]-first)
        quiet = (-x[:,47]>.84) & (torch.linalg.vector_norm(x[:,44:47]*4,dim=1)<1.45)
        count = torch.where(quiet,torch.where(begin,0.,rec_state[:,3])+1,0.)
        rolling = active & (count < 10)
        state = torch.cat((torch.stack((age+1,age*0+10,side,count),1),origin),1)
        after, following = self.inner(rec_obs_hist,torch.where(active[:,None],0.,rec_state))
        return torch.where(rolling[:,None],target,after), torch.where(rolling[:,None],state,following)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("train",type=Path)
    p.add_argument("validation",type=Path)
    p.add_argument("roll",type=Path)
    p.add_argument("--generations",type=int,default=36)
    args = p.parse_args()
    keys = fit(args.out,args.train,args.validation,args.generations)
    export(SideEntry(np.load(args.roll), keys), args.out/"expert.onnx")
