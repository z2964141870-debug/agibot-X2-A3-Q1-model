"""New-seed perturbation validation of a fixed balance controller, not a floor recovery test."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import Plant,HERE,mj,LO,HI
from balance_search import evaluate


def run(a):
    a.out.mkdir(exist_ok=False);curve=np.load(a.reference);g=np.load(a.gains)["gains"]
    env=Plant(24,a.out/"scene");rng=np.random.default_rng(a.seed);floor=env.model.geom("floor").id
    records=[]
    for start in (0,40,80,120,160):
        rows=[];rejections=0
        for i in range(env.n):
            for attempt in range(500):
                d=mj.MjData(env.model);d.qpos[:]=curve["qpos"][start]
                d.qpos[7:]=(d.qpos[7:]+rng.uniform(-.01,.01,22)).clip(LO,HI)
                delta=np.r_[1,rng.uniform(-.006,.006,2),0.];delta/=np.linalg.norm(delta)
                q=np.zeros(4);mj.mju_mulQuat(q,d.qpos[3:7],delta);d.qpos[3:7]=q
                d.qpos[2]=-.4;mj.mj_forward(env.model,d)
                if any(c.dist<-.0005 and floor not in c.geom for c in d.contact):rejections+=1;continue
                depth=min(c.dist for c in d.contact if floor in c.geom);d.qpos[2]-=depth-.0002
                rows.append(np.r_[d.qpos,np.zeros(28),d.qpos[7:]]);break
            else:break
        if len(rows)!=env.n:
            records.append(dict(start=start,initialized=len(rows),rejections=rejections,status="illegal_reference_neighborhood"));continue
        score,peak,calm,trace,state=evaluate(env,curve,np.tile(g,(env.n,1)),start,a.duration,a.seed,a.horizon,np.array(rows))
        np.savez_compressed(a.out/f"phase_{start:03d}.npz",metrics=trace,state=state,initial=rows)
        final=trace[-50:]
        sustained=((final[:,:,0]>.35)&(final[:,:,1]>.92)&(final[:,:,2]>10)&(final[:,:,3]==0)&(final[:,:,7]<.2)&(final[:,:,8]<.8)).all(0)
        record=dict(start=start,initialized=len(rows),rejections=rejections,calm_half_second=int((calm>=25).sum()),
                    final_calm_second=int(sustained.sum()),peak_qd=float(peak[:,0].max()),ground_mm=float(peak[:,1].max()),self_mm=float(peak[:,2].max()))
        records.append(record);print(json.dumps(record),flush=True)
    (a.out/"results.json").write_text(json.dumps(dict(seed=a.seed,duration=a.duration,horizon=a.horizon,phases=records),indent=2));env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("gains",type=Path)
    p.add_argument("--reference",type=Path,default=HERE/"balanced_reference_v2/curve.npz")
    p.add_argument("--seed",type=int,default=93011);p.add_argument("--duration",type=int,default=300);p.add_argument("--horizon",type=int,default=650)
    run(p.parse_args())
