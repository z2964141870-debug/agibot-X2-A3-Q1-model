"""Fit smooth symmetric corrections to a recorded approach under the actual sender."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import HERE, HI, LO, Plant, mj
from command_support import sample_record


def prepare_starts(plant, kind, seed, reference, count):
    rng = np.random.default_rng(seed)
    floor = plant.model.geom("floor").id
    rows = []
    for _ in range(count):
        for _ in range(300):
            d = mj.MjData(plant.model)
            pitch = (1 if kind == "prone" else -1)*np.pi/2 + rng.uniform(-.06,.06)
            roll = rng.uniform(-.04,.04)
            yaw = rng.uniform(-np.pi,np.pi)
            q = np.empty(4)
            mj.mju_euler2Quat(q, np.array([roll,pitch,yaw]), "XYZ")
            d.qpos[:] = np.r_[0,0,-.4,q,np.clip(reference[rng.integers(20)]+rng.uniform(-.08,.08,22),LO,HI)]
            mj.mj_forward(plant.model,d)
            if any(c.dist < -.0005 for c in d.contact if floor not in c.geom):
                continue
            d.qpos[2] -= min(c.dist for c in d.contact if floor in c.geom)-.0002
            d.ctrl[:] = d.qpos[7:]
            mj.mj_forward(plant.model,d)
            mj.mj_step(plant.model,d,nstep=50)
            rows.append(np.r_[d.qpos,d.qvel,d.qpos[7:]])
            break
        else:
            raise ValueError("no legal starting pose")
    return np.stack(rows)


def run(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    track=sample_record("q1_official_standup_from_mcap" if args.pose=="prone" else "q1_official_standup_faceup_20261120")
    ref=track["command_position"]
    n=64
    env=Plant(n,out/"scene")
    starts=prepare_starts(env,args.pose,args.seed+1,track["q"],24)
    np.save(out/"starts.npy",starts)
    rng=np.random.default_rng(args.seed)
    # Four knots modify hip pitch, knee, ankle pitch, shoulder pitch, elbow and shoulder abduction.
    mean=np.zeros(25)
    spread=np.tile([.33,.28,.22,.35,.30,.25],4).tolist()+[.12]
    std=np.array(spread)
    basis=np.zeros((6,22))
    for k,idx in enumerate(([0,6],[3,9],[4,10],[14,18],[17,21],[15,19])):
        basis[k,idx]=[1,-1] if k==5 else [1,1]
    saved=None
    def rollout(params,initial):
        if len(params)!=n:raise ValueError("population mismatch")
        env.reset(initial)
        rate=(.75+params[:,-1]).clip(.50,1.1)
        knots=params[:,:24].reshape(n,4,6).clip(-.9,.9)@basis
        knots=np.concatenate((np.zeros((n,1,22)),knots),axis=1)
        score_ring=np.zeros((25,n));peak=np.zeros((n,3));best=np.full(n,-1e4)
        idx=np.arange(n)
        for t in range(470):
            frame=((max(0,t-25)*rate)).clip(0,len(ref)-1)
            frac=frame/(len(ref)-1)*4
            left=np.floor(frac).astype(int).clip(0,3)
            mix=.5-.5*np.cos(np.pi*(frac-left))
            target=ref[frame.astype(int)]+(1-mix[:,None])*knots[idx,left]+mix[:,None]*knots[idx,left+1]
            blend=.5-.5*np.cos(np.pi*min(1,(t+1)/25))
            _,m=env.step(initial[:,7:29]+blend*(target-initial[:,7:29]))
            peak=np.maximum(peak,m[:,[4,5,6]])
            feet=(m[:,2]>12)&(m[:,3]==0)
            value=4*m[:,1]+8*m[:,0]+3*feet+7*((m[:,0]>.32)&(m[:,1]>.80)&feet)
            value-=.65*np.minimum(m[:,8],8)+2*np.minimum(m[:,7],2)
            score_ring[t%25]=value
            if t>=55:best=np.maximum(best,score_ring.mean(0))
        penalty=.23*np.maximum(peak[:,0]-11,0)+.14*np.maximum(peak[:,1]-7,0)+.45*np.maximum(peak[:,2]-1,0)
        return best-penalty,peak
    validation=np.tile(starts[16:24],(8,1))
    best_val=-1e4
    for gen in range(args.generations):
        unique=rng.normal(mean,std,(32,25));unique[0]=mean
        params=np.repeat(unique,2,axis=0)
        poses=rng.choice(16,2,replace=False)
        score,peak=rollout(params,np.tile(starts[poses],(32,1)))
        fitness=score.reshape(32,2).min(1)
        elite=unique[np.argsort(fitness)[-6:]]
        mean=.25*mean+.75*elite.mean(0)
        std=np.maximum(.4*std+.6*elite.std(0),np.array(spread)*.15)
        record=dict(generation=gen,train_pair=poses.tolist(),max_score=float(fitness.max()))
        if gen%4==0 or gen==args.generations-1:
            champs=np.vstack([unique[np.argsort(fitness)[-6:]],mean,np.zeros(25) if saved is None else saved])
            val,peaks=rollout(np.repeat(champs,8,0),validation)
            values=val.reshape(8,8);aggregate=values.mean(1)*.65+values.min(1)*.35
            i=int(np.argmax(aggregate))
            if aggregate[i]>best_val:
                best_val=float(aggregate[i]);saved=champs[i].copy()
                np.savez(out/"best.npz",parameters=saved,basis=basis,score=best_val,pose=args.pose)
            record.update(validation_score=best_val,validation_current=float(aggregate[i]),validation_peak=peaks.reshape(8,8,3)[i].max(0).tolist())
        with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
    env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path)
    p.add_argument("--pose",choices=["prone","supine"],required=True)
    p.add_argument("--seed",type=int,default=95101);p.add_argument("--generations",type=int,default=32)
    run(p.parse_args())
