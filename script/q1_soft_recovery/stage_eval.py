"""Evaluate deterministic reference tracking from the same legal curriculum starts."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from common import Plant,HERE
from policy import Teacher,load_actor


def evaluate(checkpoint,out,data,balanced,horizon,network_only=False):
    torch.set_num_threads(2)
    reference=np.load(data/"references.npz")
    starts=np.load(data/"starts.npz")
    rows,meta=starts["valid"],starts["valid_meta"]
    if balanced:
        from balanced_policy import BalancedTeacher
        model=BalancedTeacher(reference).eval()
    else:model=Teacher(reference).eval()
    if checkpoint:
        load_actor(model,torch.load(checkpoint,map_location="cpu",weights_only=False)["actor"],network_only)
    sources=[Path(__file__),HERE/"common.py",HERE/"policy.py",HERE/"plant.cc",HERE/"plant.so",data/"references.npz",data/"starts.npz"]
    if balanced:sources.append(HERE/"balanced_policy.py")
    if checkpoint:sources.append(checkpoint)
    (out/"evaluation_manifest.json").write_text(json.dumps(dict(
        checkpoint=str(checkpoint) if checkpoint else None,data=str(data),balanced=balanced,horizon=horizon,network_only_transfer=network_only,
        control_s=.02,physics_s=.002,delay_ms=11,arm_leg_pairs=96,
        hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    env=Plant(len(rows),out/"scene")
    obs=env.reset(rows)
    hist=np.repeat(obs[:,None],10,axis=1).astype(np.float32)
    state=torch.zeros(len(rows),26)
    state[:,0]=60;state[:,1]=4;state[:,2:4]=torch.tensor(meta,dtype=torch.float32);state[:,4:]=torch.tensor(rows[:,7:29],dtype=torch.float32)
    peak=np.zeros((len(rows),3));longest=np.zeros(len(rows),int);run=longest.copy();trace=[];physics=[]
    with torch.no_grad():
        for tick in range(horizon):
            target,state=model(torch.tensor(hist),state)
            obs,met=env.step(target.numpy())
            hist=np.concatenate((hist[:,1:],obs[:,None].astype(np.float32)),axis=1)
            quiet=(met[:,0]>.35)&(met[:,1]>.92)&(met[:,2]>10)&(met[:,3]==0)&(met[:,7]<.2)&(met[:,8]<.8)
            run=np.where(quiet,run+1,0);longest=np.maximum(longest,run)
            peak=np.maximum(peak,met[:,[4,5,6]])
            trace.append(met)
            physics.append(env.state.copy())
    report=[]
    for i,(kind,phase) in enumerate(meta):
        report.append(dict(trial=i,kind=int(kind),phase=int(phase),fraction=float(phase/reference["length"][kind]),
                           calm_s=float(longest[i]*.02),final_calm_s=float(run[i]*.02),standing=bool(longest[i]>=25),final_calm_second=bool(run[i]>=50),
                           peak_qd=float(peak[i,0]),ground_mm=float(peak[i,1]),self_mm=float(peak[i,2])))
    (out/"results.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/"traces.npz",trace=trace,state=physics,initial=rows[:,:29])
    print(json.dumps(dict(standing=sum(r["standing"] for r in report),n=len(rows),
                         late_standing=sum(r["standing"] and r["fraction"]>.8 for r in report),
                         late_n=sum(r["fraction"]>.8 for r in report),final_calm_second=sum(r["final_calm_second"] for r in report))),flush=True)
    env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--checkpoint",type=Path)
    p.add_argument("--data",type=Path,default=HERE/"prepared_v3");p.add_argument("--balanced",action="store_true");p.add_argument("--horizon",type=int,default=300)
    p.add_argument("--transfer-network-only",action="store_true")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);evaluate(a.checkpoint,a.out,a.data,a.balanced,a.horizon,a.transfer_network_only)
