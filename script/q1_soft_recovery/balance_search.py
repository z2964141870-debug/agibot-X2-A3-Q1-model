"""Search proprioceptive balance feedback along a mesh-grounded standing reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from common import Plant,HERE,LO,HI


def targets(curve,phase,obs,gains):
    low=np.floor(phase).astype(int);high=np.minimum(low+1,160);blend=phase-low
    base=(1-blend[:,None])*curve["command"][low]+blend[:,None]*curve["command"][high]
    up=(1-blend[:,None])*curve["up"][low]+blend[:,None]*curve["up"][high]
    error=obs[:,47:50]+up
    pitch=np.column_stack((error[:,0],obs[:,45]*4))
    roll=np.column_stack((error[:,1],obs[:,44]*4))
    for indices,features,g in (([4,10],pitch,gains[:,:2]),([0,6],pitch,gains[:,2:4]),
                               ([5,11],roll,gains[:,4:6]),([1,7],roll,gains[:,6:8])):
        base[:,indices]+=np.sum(features*g,axis=1)[:,None].clip(-.5,.5)
    return base.clip(LO,HI)


def evaluate(env,curve,gains,start,duration,seed,horizon,rows=None):
    rng=np.random.default_rng(seed);n=len(gains)
    if rows is None:
        rows=np.tile(np.r_[curve["qpos"][start],np.zeros(28),curve["q"][start]],(n,1))
        rows[:,7:29]+=rng.normal(0,.003,(n,22));rows[:,57:]=rows[:,7:29]
    obs=env.reset(rows);score=np.zeros(n);peak=np.zeros((n,3));calm=np.zeros(n,int);longest=calm.copy();traces=[];states=[]
    for tick in range(horizon):
        fraction=min(1.,tick/max(1,duration))
        phase=np.full(n,start+(160-start)*(.5-.5*np.cos(np.pi*fraction)))
        obs,met=env.step(targets(curve,phase,obs,gains))
        upright=(met[:,0]>.35)&(met[:,1]>.92)&(met[:,3]==0)
        quiet=upright&(met[:,2]>10)&(met[:,7]<.2)&(met[:,8]<.8)
        calm=np.where(quiet,calm+1,0);longest=np.maximum(longest,calm)
        score+=3*met[:,1]+4*met[:,0]+5*quiet-2*met[:,7]-met[:,8]
        score-=.03*np.maximum(met[:,4]-10,0)**2+.1*np.maximum(met[:,5]-8,0)+.2*np.maximum(met[:,6]-1,0)
        peak=np.maximum(peak,met[:,[4,5,6]])
        traces.append(met);states.append(env.state.copy())
    score+=200*(met[:,0]>.35)*(met[:,1]>.92)+longest
    return score,peak,longest,np.array(traces),np.array(states)


def run(a):
    a.out.mkdir(exist_ok=False);curve=np.load(a.reference);rng=np.random.default_rng(a.seed)
    env=Plant(a.population,a.out/"scene")
    mean=np.array([.5,.06,.0,.0,.5,.06,.0,.0]);std=np.array([1.,.3,.8,.2,1.,.3,.8,.2])
    if a.parent is not None:mean=np.load(a.parent)["gains"]
    best=-np.inf;started=time.monotonic()
    for generation in range(a.generations):
        gains=rng.normal(mean,std,(a.population,8));gains[0]=mean
        score,peak,calm,trace,state=evaluate(env,curve,gains,a.start,a.duration,81741,a.horizon)
        elite=np.argsort(score)[-max(6,a.population//8):]
        mean=.3*mean+.7*gains[elite].mean(0);std=np.maximum(.3*std+.7*gains[elite].std(0),[.03,.008,.03,.008,.03,.008,.03,.008])
        i=int(np.argmax(score))
        if score[i]>best:
            best=float(score[i]);np.savez(a.out/"best.npz",gains=gains[i],score=score[i],trace=trace[:,i],state=state[:,i],start=a.start,duration=a.duration)
        record=dict(generation=generation,best=best,current=float(score[i]),calm_s=float(calm[i]*.02),peak=peak[i].tolist(),gains=gains[i].tolist(),elapsed_s=time.monotonic()-started)
        with (a.out/"metrics.jsonl").open("a") as fp:fp.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
    env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--reference",type=Path,default=HERE/"balanced_reference_v2/curve.npz")
    p.add_argument("--parent",type=Path);p.add_argument("--population",type=int,default=64);p.add_argument("--generations",type=int,default=25)
    p.add_argument("--start",type=int,default=160);p.add_argument("--duration",type=int,default=250);p.add_argument("--horizon",type=int,default=250)
    p.add_argument("--seed",type=int,default=82931);run(p.parse_args())
