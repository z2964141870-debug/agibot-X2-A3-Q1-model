"""Dynamic validation of the statically balanced curve in the exact sender plant."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import Plant,HERE,mj


def run(out,reference):
    out.mkdir(exist_ok=False)
    curve=np.load(reference)
    cases=[dict(start=s,duration=t,feedforward=ff) for s in (0,40,80,160)
           for t in (200,350,500) for ff in (False,True)]
    plant=Plant(len(cases),out/"scene")
    rows=np.array([np.r_[curve["qpos"][c["start"]],np.zeros(28),curve["q"][c["start"]]] for c in cases])
    floor=plant.model.geom("floor").id
    init=[]
    for row in rows:
        d=mj.MjData(plant.model);d.qpos[:]=row[:29];mj.mj_forward(plant.model,d)
        init.append(dict(ground_mm=1000*max([max(0.,-c.dist) for c in d.contact if floor in c.geom],default=0.),
                         self_mm=1000*max([max(0.,-c.dist) for c in d.contact if floor not in c.geom],default=0.)))
    plant.reset(rows)
    traces=[];states=[];targets=[];quiet=np.zeros(len(cases),int);longest=quiet.copy()
    for tick in range(750):
        commands=[]
        for c in cases:
            fraction=min(1.,tick/c["duration"])
            phase=c["start"]+(160-c["start"])*(.5-.5*np.cos(np.pi*fraction))
            low=int(phase);high=min(low+1,160);blend=phase-low
            key="command" if c["feedforward"] else "q"
            commands.append((1-blend)*curve[key][low]+blend*curve[key][high])
        _,met=plant.step(np.array(commands))
        calm=(met[:,0]>.35)&(met[:,1]>.92)&(met[:,2]>10)&(met[:,3]==0)&(met[:,7]<.2)&(met[:,8]<.8)
        quiet=np.where(calm,quiet+1,0);longest=np.maximum(longest,quiet)
        traces.append(met);states.append(plant.state.copy());targets.append(commands)
    traces=np.array(traces)
    for i,c in enumerate(cases):
        c.update(initial=init[i],calm_s=float(longest[i]*.02),final_z=float(traces[-1,i,0]),
                 final_up=float(traces[-1,i,1]),peak_qd=float(traces[:,i,4].max()),
                 ground_mm=float(traces[:,i,5].max()),self_mm=float(traces[:,i,6].max()))
    np.savez_compressed(out/"traces.npz",metrics=traces,state=states,target=targets,initial=rows)
    (out/"results.json").write_text(json.dumps(cases,indent=2))
    print(json.dumps(cases,indent=2),flush=True)
    plant.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path)
    p.add_argument("--reference",type=Path,default=HERE/"balanced_reference/curve.npz")
    a=p.parse_args();run(a.out,a.reference)
