"""Generate legal perturbed crouch-to-standing starts for a separate curriculum."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import Plant,HERE,mj,LO,HI


def prepare(a):
    a.out.mkdir(exist_ok=False);c=np.load(a.reference);g=np.load(a.gains)["gains"]
    p=Plant(1,a.out/"scene");d=mj.MjData(p.model);floor=p.model.geom("floor").id
    for key in ("q","up","command"):
        if not np.isfinite(c[key]).all():raise ValueError("nonfinite reference")
    np.savez(a.out/"references.npz",q=np.tile(c["q"][None],(2,1,1)),up=np.tile(c["up"][None],(2,1,1)),
             command=np.tile(c["command"][None],(2,1,1)),length=[160,160],gains=g,phase_increment=a.phase_increment)
    result={};audit={}
    for name,size,seed in (("train",768,94001),("valid",144,94003)):
        rng=np.random.default_rng(seed);rows=[];meta=[];attempts=0
        for i in range(size):
            phase=int(rng.integers(161))
            for attempt in range(500):
                attempts+=1;d.qpos[:]=c["qpos"][phase];d.qvel[:]=0
                d.qpos[7:]=(d.qpos[7:]+rng.uniform(-.008,.008,22)).clip(LO,HI)
                delta=np.r_[1.,rng.uniform(-.004,.004,2),0.];delta/=np.linalg.norm(delta)
                quat=np.zeros(4);mj.mju_mulQuat(quat,d.qpos[3:7],delta);d.qpos[3:7]=quat
                d.qpos[2]=-.4;mj.mj_forward(p.model,d)
                if any(x.dist<-.0005 and floor not in x.geom for x in d.contact):continue
                d.qpos[2]-=min(x.dist for x in d.contact if floor in x.geom)-.0002
                mj.mj_forward(p.model,d)
                assert not any(x.dist<-.001 if floor in x.geom else x.dist<-.0005 for x in d.contact)
                rows.append(np.r_[d.qpos,d.qvel,d.qpos[7:]]);meta.append([i%2,phase]);break
            else:raise RuntimeError(f"no legal {name} start {i}/{phase}")
        result[name]=rows;result[name+"_meta"]=meta;audit[name]=dict(seed=seed,n=size,attempts=attempts)
    np.savez_compressed(a.out/"starts.npz",**result)
    audit["sources"]={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.reference,a.gains)}
    (a.out/"manifest.json").write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True);p.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--reference",type=Path,default=HERE/"balanced_reference_v3/curve.npz")
    p.add_argument("--gains",type=Path,default=HERE/"balance_stand_01/best.npz")
    p.add_argument("--phase-increment",type=float,default=.75);prepare(p.parse_args())
