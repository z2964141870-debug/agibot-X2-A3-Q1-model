"""CPU evaluation through the complete exclusive owner and frozen standing tracker."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import STAND,LO,HI,CONFIG,mj,replay
from prepare import pool
from recorded_rise import load_track
from limbs import add_limb_contacts
import eval_core


def main(args):
    out=args.out.resolve()
    original=eval_core.bench.scene
    holder={}
    def scene(directory):
        result=add_limb_contacts(directory)
        holder["model"]=result
        return result
    eval_core.bench.scene=scene
    if args.legacy:
        init="legacy"
    else:
        files=[Path.home()/"projects/Q1_deploy/real_logs"/n for n in ("q1_official_standup_from_mcap.npz","q1_official_standup_faceup_20261120.npz")]
        refs=[load_track(p,CONFIG["joint_names"]) for p in files]
        def poses(controller,seed):
            rng=np.random.default_rng(seed)
            m=holder["model"];d=mj.MjData(m);floor=m.geom("floor").id
            saved=[]
            rejected=[]
            settled=mj.MjData(m)
            for i in range(24):
                kind=i//8
                accepted=False
                for attempt in range(300):
                    phase=int(rng.integers(20))
                    base=refs[0 if kind==0 else 1][0][phase]
                    joints=np.clip(base+rng.uniform(-.08,.08,22),LO,HI)
                    pitch=(np.pi/2 if kind==0 else -np.pi/2 if kind==1 else 0)+rng.uniform(-.06,.06)
                    roll=(0 if kind<2 else (-1 if i%2 else 1)*np.pi/2)+rng.uniform(-.04,.04)
                    yaw=rng.uniform(-np.pi,np.pi)
                    cr,sr=np.cos(roll/2),np.sin(roll/2);cp,sp=np.cos(pitch/2),np.sin(pitch/2);cy,sy=np.cos(yaw/2),np.sin(yaw/2)
                    quat=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
                    d.qpos[:]=np.r_[0,0,-.4,quat,joints];d.qvel[:]=0;mj.mj_forward(m,d)
                    if any(c.dist<-.0005 and floor not in c.geom for c in d.contact):
                        rejected.append(dict(trial=i,attempt=attempt+1,reason="initial_self_overlap"))
                        continue
                    contacts=[c.dist for c in d.contact if floor in c.geom]
                    d.qpos[2]-=min(contacts)-.0002;mj.mj_forward(m,d)
                    mj.mj_resetData(m,settled)
                    settled.qpos[:]=d.qpos
                    settled.ctrl[:]=d.qpos[7:]
                    mj.mj_forward(m,settled)
                    mj.mj_step(m,settled,nstep=50)
                    ground=max((-1000*c.dist for c in settled.contact if floor in c.geom),default=0.)
                    arm_leg=max((-1000*c.dist for c in settled.contact if floor not in c.geom),default=0.)
                    if ground>10 or arm_leg>2:
                        rejected.append(dict(trial=i,attempt=attempt+1,reason="settled_overlap",
                                             ground_mm=ground,self_mm=arm_leg,qpos=d.qpos.tolist()))
                        continue
                    accepted=True;break
                if not accepted:raise RuntimeError(f"Cannot construct legal fall {seed}/{i}")
                saved.append(dict(pose=("prone","supine","side")[kind],attempts=attempt+1,root_z=float(d.qpos[2]),qpos=d.qpos.tolist()))
                yield saved[-1]["pose"],d.qpos.copy()
            (out/"initialization.json").write_text(json.dumps(saved,indent=2))
            (out/"rejected_initializations.json").write_text(json.dumps(rejected,indent=2))
        eval_core.bench.poses=poses
        init="hardware-shaped legal fallen starts"
    if args.catch:
        from low_catch import MidCrouchCatch
        eval_core.Q1AutonomousRecovery=MidCrouchCatch
    eval_core.simulate(args.group,args.seed,out,args.expert.resolve(),500,args.delay,True,2)
    manifest=json.loads((out/"manifest.json").read_text())
    manifest.update(initialization=init,settle_physics_ms=2,settle_steps=50,explicit_arm_leg_pairs=96,standing_limit_s=15)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    results=json.loads((out/"results.json").read_text())
    summary=dict(success=sum(r["success"] for r in results),n=len(results),fell=sum(r["phases"][0]["fell"] for r in results),
                 peak_qd=max(r.get("peak_qd",0) for r in results),ground_mm=max(r.get("ground_mm",0) for r in results),
                 self_mm=max(r.get("self_mm",0) for r in results),wrong_release=sum(r["wrong_release"] for r in results))
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("group",choices=["static","push","flat","offline","double"])
    p.add_argument("expert",type=Path);p.add_argument("out",type=Path);p.add_argument("--seed",type=int,default=92017)
    p.add_argument("--legacy",action="store_true");p.add_argument("--catch",action="store_true")
    p.add_argument("--delay",type=int,choices=[0,11],default=11)
    main(p.parse_args())
