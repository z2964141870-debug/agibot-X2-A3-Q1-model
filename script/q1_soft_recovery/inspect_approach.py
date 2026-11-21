"""Record commands and internal ownership around a matched full rise."""
import argparse
import json
from pathlib import Path

import numpy as np

import evaluate
import eval_core
from low_catch import MidCrouchCatch


def run(expert,out,seed):
    samples = []

    class Instrumented(MidCrouchCatch):
        def reset(self):
            super().reset()
            self.trial_samples = []
            samples.append(self.trial_samples)

        def step(self,R,w,q,qd,prev,base,anchor,*,sample_time_s,teleop_stamp_s):
            target,tau,owner = super().step(R,w,q,qd,prev,base,anchor,sample_time_s=sample_time_s,teleop_stamp_s=teleop_stamp_s)
            self.trial_samples.append(np.r_[sample_time_s, owner, self.rise.expert.state[0,:4], q, qd, target, tau])
            return target,tau,owner

    eval_core.Q1AutonomousRecovery=Instrumented
    args=argparse.Namespace(out=out,legacy=False,catch=False,group="static",seed=seed,expert=expert,delay=11)
    evaluate.main(args)
    trials=[np.array(r) for r in samples if len(r)]
    rows=json.loads((out/"results.json").read_text())
    assert len(trials)==len(rows)
    np.savez_compressed(out/"commands.npz",**{f"trial_{i:02d}":r for i,r in enumerate(trials)})
    summary=[]
    for i,(r,commands) in enumerate(zip(rows,trials,strict=True)):
        with np.load(out/f"trial_{i:02d}.npz") as rec:
            owned=np.flatnonzero(rec["trace"][:,4])
            t=owned[np.argmax(rec["peaks"][owned,0])]
            j=int(rec["peaks"][t,3])
            window=commands[max(0,t-2):t+2]
            summary.append(dict(trial=i,pose=r["pose"],success=r["success"],peak=r["peak_qd"],joint=r["peak_qd_joint"],tick=int(t),
                expert_mode=window[:,3].tolist(),expert_tick=window[:,5].tolist(),angle=window[:,6+j].tolist(),
                sensed_velocity=window[:,28+j].tolist(),target=window[:,50+j].tolist(),torque=window[:,72+j].tolist()))
    (out/"peaks_diagnosis.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("expert",type=Path);p.add_argument("out",type=Path);p.add_argument("--seed",type=int,default=92017)
    a=p.parse_args();run(a.expert,a.out,a.seed)
