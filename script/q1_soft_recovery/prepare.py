"""Build physics-valid hardware-derived starts and verify the exact sender contract."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from common import Plant,HERE,STAND,LO,HI,SCALE,KP,KD,EFF,mj,replay
from recorded_rise import load_track
from leg_height import LegHeight


def pool(plant,refs,seed,size):
    rng=np.random.default_rng(seed)
    d=mj.MjData(plant.model)
    floor=plant.model.geom("floor").id
    rows=[];meta=[]
    for i in range(size):
        kind=i%2
        qref,upref=refs[kind]
        # Interleave full trajectories with harder, fully fallen starts.
        phase=int(rng.integers(len(qref)-30)) if i%3 else int(rng.integers(35))
        accepted=False
        for attempt in range(100):
            d.qpos[:]=np.r_[0,0,-.4,1,0,0,0,STAND]
            up=upref[phase].copy();up/=np.linalg.norm(up)
            roll=np.arctan2(up[1],up[2]);pitch=np.arcsin(np.clip(-up[0],-1,1))
            yaw=rng.uniform(-np.pi,np.pi)
            cr,sr=np.cos(roll/2),np.sin(roll/2);cp,sp=np.cos(pitch/2),np.sin(pitch/2);cy,sy=np.cos(yaw/2),np.sin(yaw/2)
            d.qpos[3:7]=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
            d.qpos[7:]=np.clip(qref[phase]+rng.normal(0,.02,22),LO,HI)
            d.qvel[:]=0
            mj.mj_forward(plant.model,d)
            selfdepth=max([max(0.,-c.dist) for c in d.contact if floor not in c.geom],default=0.)
            if selfdepth>.0005:continue
            contacts=[c.dist for c in d.contact if floor in c.geom]
            if not contacts:raise RuntimeError("floor placement failed")
            d.qpos[2]-=min(contacts)-.0002
            mj.mj_forward(plant.model,d)
            ground=max([max(0.,-c.dist) for c in d.contact if floor in c.geom],default=0.)
            if ground<=.001:
                accepted=True;break
        if not accepted:raise RuntimeError(f"Cannot initialize legal pose {i}")
        rows.append(np.r_[d.qpos,d.qvel,d.qpos[7:]])
        meta.append([kind,phase])
    return np.array(rows),np.array(meta)


def parity(plant,rows):
    n=plant.n
    obs=plant.reset(rows[:n])
    states=rows[:n,:57].copy()
    ds=[];rings=[];sent=rows[:n,57:].astype(np.float32).copy()
    for state in states:
        d=mj.MjData(plant.model);d.qpos[:]=state[:29];d.qvel[:]=state[29:];mj.mj_forward(plant.model,d)
        ds.append(d);rings.append(deque([state.copy() for _ in range(7)],maxlen=7))
    legs=LegHeight(replay.BUNDLE/"robot_model/supermimic.xml")
    rng=np.random.default_rng(88313)
    maxobs=maxstate=maxmetric=0.
    diagnostics=[]
    for tick in range(45):
        targets=np.clip(STAND+rng.normal(0,.10,(n,22)),LO,HI)
        if tick<20:targets=rows[:n,7:29]+.06*np.sin(tick*.1)
        for i,d in enumerate(ds):
            sample=.5*(rings[i][0]+rings[i][1])
            a,b=rings[i][0][3:7],rings[i][1][3:7]
            sample[3:7]=.5*(a+(b if a@b>=0 else -b));mj.mj_normalizeQuat(plant.model,sample[:29])
            R=replay.bench.quat_wxyz_to_mat(sample[3:7])
            q,qd=sample[7:29].astype(np.float32),sample[35:].astype(np.float32)
            height,valid=legs.measure(R[2],q)
            frame=np.r_[q-STAND,qd*.1,sample[32:35]*.25,-R[2],.25*(sent[i]-STAND)/SCALE,height,valid]
            maxobs=max(maxobs,float(np.max(abs(frame-obs[i]))))
            kp,kd,eff,lo,hi,stand=[x.astype(np.float32) for x in (KP,KD,EFF,LO,HI,STAND)]
            margin=.5*eff/kp
            raw=targets[i].astype(np.float32).clip(lo-margin,hi+margin)
            tau=(kp*(raw-q)-kd*qd).clip(-eff,eff)
            desired=(q+(tau+kd*qd)/kp).clip(stand-2,stand+2)
            for sub in range(10):
                sent[i]+=np.clip(desired-sent[i],-.2,.2)
                d.ctrl[:]=sent[i];mj.mj_step(plant.model,d)
                rings[i].append(np.r_[d.qpos,d.qvel])
        obs,metrics=plant.step(targets)
        diagnostic=dict(tick=tick,obs_error=maxobs,state_error=maxstate)
        for i,d in enumerate(ds):
            maxstate=max(maxstate,float(abs(np.r_[d.qpos,d.qvel]-plant.state[i]).max()))
        diagnostic["new_state_error"]=max(float(abs(np.r_[d.qpos,d.qvel]-plant.state[i]).max()) for i,d in enumerate(ds))
        diagnostics.append(diagnostic)
    result=dict(control_steps=45,worlds=n,observation_max_abs=maxobs,state_max_abs=maxstate,
                command_step_max=float(metrics[:,11].max()),actual_torque_excess=float(metrics[:,10].max()))
    (HERE/"parity_diagnostic.json").write_text(json.dumps(diagnostics,indent=2))
    assert maxobs<2e-5 and maxstate<2e-5,result
    assert metrics[:,11].max()<.200001 and metrics[:,10].max()<1e-6,result
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
    from common import CONFIG
    records=Path.home()/"projects/Q1_deploy/real_logs"
    files=[records/"q1_official_standup_from_mcap.npz",records/"q1_official_standup_faceup_20261120.npz"]
    refs=[load_track(f,CONFIG["joint_names"]) for f in files]
    n=max(len(q) for q,_ in refs)+50
    qbank=np.tile(STAND,(2,n,1));ubank=np.tile([0.,0.,1.],(2,n,1))
    for k,(q,u) in enumerate(refs):
        qbank[k,:len(q)]=q;ubank[k,:len(u)]=u
        for j in range(50):
            blend=.5-.5*np.cos(np.pi*(j+1)/50)
            qbank[k,len(q)+j]=(1-blend)*q[-1]+blend*STAND
            ubank[k,len(u)+j]=(1-blend)*u[-1]+blend*np.array([0,0,1])
    np.savez(a.output/"references.npz",q=qbank,up=ubank,length=[len(q) for q,_ in refs])
    plant=Plant(8,a.output/"scene")
    train,tm=pool(plant,refs,91071,512);valid,vm=pool(plant,refs,91073,96)
    np.savez(a.output/"starts.npz",train=train,train_meta=tm,valid=valid,valid_meta=vm)
    result=parity(plant,train)
    result["sources"]={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    result["training_rows"]=len(train);result["validation_rows"]=len(valid)
    (a.output/"contract.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    plant.close()
