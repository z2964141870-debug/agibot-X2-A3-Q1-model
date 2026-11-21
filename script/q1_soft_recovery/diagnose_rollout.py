"""Compare deterministic and stochastic execution on train and validation starts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import Plant,HERE
from balanced_policy import BalancedTeacher


def run(a):
    a.out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(94151);rng=np.random.default_rng(94151)
    ref=np.load(HERE/"prepared_balanced_01/references.npz");pool=np.load(HERE/"prepared_balanced_01/starts.npz")
    checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=BalancedTeacher(ref).eval();model.load_state_dict(checkpoint["actor"])
    env=Plant(128,a.out/"scene");reports=[]
    for name in ("train","valid"):
        fractions=pool[name+"_meta"][:,1]/160
        indices=rng.choice(np.flatnonzero(fractions>=.7),128,replace=True)
        rows=pool[name][indices];meta=pool[name+"_meta"][indices]
        for stochastic in (False,True):
            obs=env.reset(rows);hist=np.repeat(obs[:,None],10,axis=1).astype(np.float32)
            state=torch.zeros(128,26);state[:,0]=60;state[:,1]=4;state[:,2:4]=torch.tensor(meta,dtype=torch.float32);state[:,4:]=torch.tensor(rows[:,7:29],dtype=torch.float32)
            trace=[]
            with torch.no_grad():
                for tick in range(350):
                    h=torch.tensor(hist);mean=model.mean(h,state)
                    action=mean+torch.randn_like(mean)*checkpoint["logstd"].clamp(-2.8,-.3).exp() if stochastic else mean
                    target=model.target(action,h,state);state=model.advance(h,state)
                    obs,met=env.step(target.numpy());hist=np.concatenate((hist[:,1:],obs[:,None].astype(np.float32)),axis=1)
                    quiet=(met[:,0]>.35)&(met[:,1]>.92)&(met[:,2]>10)&(met[:,3]==0)&(met[:,7]<.2)&(met[:,8]<.8)
                    trace.append([quiet.mean(),met[:,0].mean(),(met[:,0]<.13).mean(),met[:,4].max()])
            record=dict(pool=name,stochastic=stochastic,trace=trace,logstd=checkpoint["logstd"].tolist(),update=checkpoint["update"])
            reports.append(record)
            print(json.dumps(dict(pool=name,stochastic=stochastic,at_1s=trace[49],at_3s=trace[149],at_5s=trace[249],at_7s=trace[349])),flush=True)
    (a.out/"results.json").write_text(json.dumps(reports,indent=2));env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("checkpoint",type=Path);run(p.parse_args())
