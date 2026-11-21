"""Search a gentle prone-to-supine maneuver before the command-guided rise."""
import argparse
import json
from pathlib import Path

import numpy as np
from command_support import sample_record
from common import HERE, LO, HI, Plant
from fit_command import prepare_starts


def rollout(plant, starts, keys, duration, goal):
    n=len(keys)
    population=np.arange(n)
    obs=plant.reset(starts)
    origin=starts[:,7:29]
    peak=np.zeros((n,3))
    best=np.full(n,-np.inf)
    time_in_pose=np.zeros(n)
    most_time=np.zeros(n)
    history=[]
    for tick in range(235):
        phase=np.minimum((tick+1)/duration,5.)
        segment=np.minimum(phase.astype(int),4)
        weight=.5-.5*np.cos(np.pi*(phase-segment))
        first=np.where((segment==0)[:,None],origin,keys[population,np.maximum(0,segment-1)])
        target=first+weight[:,None]*(keys[population,segment]-first)
        obs,metrics=plant.step(target)
        up=-obs[:,47:50]
        q=plant.state[:,7:29]
        peak=np.maximum(peak,metrics[:,[4,5,6]])
        valid=(up[:,0]>.80)&(metrics[:,8]<1.7)
        time_in_pose=np.where(valid,time_in_pose+1,0)
        most_time=np.maximum(most_time,time_in_pose)
        quality=6*up[:,0]+1.6*np.maximum(np.abs(up[:,1]),0)-1.2*np.mean((q-goal)**2,axis=1)
        quality+=1.6*np.minimum(time_in_pose/15,1)-.08*np.minimum(metrics[:,8],8)
        best=np.maximum(best,quality)
        history.append(up[:,0].copy())
    penalty=.32*np.maximum(peak[:,0]-10,0)+.16*np.maximum(peak[:,1]-8,0)+.4*np.maximum(peak[:,2]-1,0)
    return best-penalty,most_time,peak,np.max(history,axis=0)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    plant=Plant(72,a.out/"scene")
    prone=sample_record("q1_official_standup_from_mcap")["q"]
    supine=sample_record("q1_official_standup_faceup_20261120")["q"]
    starts=prepare_starts(plant,"prone",a.seed,prone,32)
    np.save(a.out/"starts.npy",starts)
    source=HERE.parent/"h003_recovery_expert/rolls_01/keys_0.npy"
    original=np.load(source)
    assert original.shape==(5,22)
    rng=np.random.default_rng(a.seed+1)
    mean=original.copy()
    sigma=np.full_like(mean,.60)
    lo=np.maximum(LO, np.array(plant.model.key_qpos[plant.model.key("stand").id,7:])-2.)
    hi=np.minimum(HI, np.array(plant.model.key_qpos[plant.model.key("stand").id,7:])+2.)
    hold_center=26.
    saved=mean.copy()
    best=-np.inf
    for generation in range(a.generations):
        candidates=np.clip(rng.normal(mean,sigma,(36,5,22)),lo,hi)
        candidates[0]=saved
        candidates[1]=mean
        durations=np.clip(rng.normal(hold_center,5.,36),16,40)
        ids=rng.choice(24,2,replace=False)
        values,_,_,_=rollout(plant,np.tile(starts[ids],(36,1)),np.repeat(candidates,2,0),np.repeat(durations,2),supine[0])
        scores=values.reshape(36,2).min(1)
        elite=np.argsort(scores)[-7:]
        mean=.25*mean+.75*candidates[elite].mean(0)
        sigma=np.maximum(.4*sigma+.6*candidates[elite].std(0),.075)
        hold_center=.25*hold_center+.75*durations[elite].mean()
        record=dict(generation=generation,train_score=float(scores.max()))
        if generation%4==0 or generation==a.generations-1:
            bank=np.concatenate((candidates[elite],mean[None],saved[None]),0)
            times=np.r_[durations[elite],hold_center,saved_time if np.isfinite(best) else 26.]
            vals,calm,peaks,up=rollout(plant,np.tile(starts[24:32],(9,1)),np.repeat(bank,8,0),np.repeat(times,8),supine[0])
            vals=vals.reshape(9,8);aggregate=.55*vals.mean(1)+.45*vals.min(1);i=int(np.argmax(aggregate))
            if aggregate[i]>best:
                best=float(aggregate[i]);saved=bank[i].copy();saved_time=float(times[i])
                np.savez(a.out/"best.npz",keys=saved,duration=saved_time,score=best)
            record.update(valid_best=best,steady_supine=int((calm.reshape(9,8)[i]>=10).sum()),
                          min_best_up=float(up.reshape(9,8)[i].min()),
                          peaks=peaks.reshape(9,8,3)[i].max(0).tolist())
        with (a.out/"metrics.jsonl").open("a") as fp:fp.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
    plant.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--generations",type=int,default=80)
    p.add_argument("--seed",type=int,default=95501);run(p.parse_args())
