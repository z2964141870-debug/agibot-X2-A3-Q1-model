"""Refine the lifting motion against starts produced by the actual roll controller."""
import json
from pathlib import Path
import numpy as np
from bench import HERE,ROOT,BUNDLE,scene,mujoco,Q1V2bOnnxController,quat_wxyz_to_mat
from leg_height import LegHeight
from recovery import sensor_frame
from q1_v2b_runtime import apply_torque_safe_pd_np
from curriculum_teacher import Sequence
from search_teacher import lib,xml,ptr,c_char_p,c_int
lib.evaluate_balance.argtypes=[c_char_p,c_int,ptr,ptr,ptr,ptr,ptr]

def collect(path):
    pool=np.load(HERE/"pool.npy")
    c=Q1V2bOnnxController(BUNDLE,())
    policy=Sequence(c,HERE/"rolls_01",HERE/"teachers_01",settle=15)
    model=scene(path/"scene")
    rows=[]
    for i,row in enumerate(pool):
        if int(row[0])>3:continue
        d=mujoco.MjData(model)
        d.qpos[:]=row[1:30]
        d.qvel[:]=row[30:58]
        d.ctrl[:]=row[58:]
        # Randomized global heading with the correct world-frame linear velocity.
        yaw=np.random.default_rng(1500+i).uniform(-np.pi,np.pi)
        quat=np.empty(4)
        mujoco.mju_mulQuat(quat,[np.cos(yaw/2),0,0,np.sin(yaw/2)],d.qpos[3:7].copy())
        d.qpos[3:7]=quat
        vx,vy=d.qvel[:2]
        d.qvel[:2]=[np.cos(yaw)*vx-np.sin(yaw)*vy,np.sin(yaw)*vx+np.cos(yaw)*vy]
        mujoco.mj_forward(model,d)
        policy.reset()
        for t in range(180):
            R=quat_wxyz_to_mat(d.qpos[3:7])
            x=np.r_[d.qpos[7:]-c.default_q,d.qvel[6:]*.1,d.qvel[3:6]*.25,-R[2],
                    (d.ctrl-c.default_q)/c.action_scale*.25,0.,0.]
            raw=policy.run(None,{"rec_obs_hist":x.reshape(1,1,-1)})[0][0]
            if policy.mode=="rise":
                rows.append(np.r_[row[0],d.qpos,d.qvel])
                break
            d.ctrl[:]=apply_torque_safe_pd_np(raw,d.qpos[7:],d.qvel[6:],c.q_lo,c.q_hi,c.kp,c.kd,c.eff,engage_steps=0)
            mujoco.mj_step(model,d,nstep=4)
    np.save(path/"starts.npy",np.asarray(rows))

def fit():
    out=HERE/"rise_02"
    out.mkdir(exist_ok=True)
    if not (out/"starts.npy").exists(): collect(out)
    rows=np.load(out/"starts.npy")
    cfg=json.loads((BUNDLE/"config/q1_deploy_config.json").read_text())
    stand,lo,hi=(np.array(cfg[k]) for k in ("default_q","q_lo","q_hi"))
    base=np.load(HERE/"teachers_01/keys_1.npy")
    gains=np.load(HERE/"balance_01/feedback.npy")
    rng=np.random.default_rng(716568)
    # Shared adjustments preserve lateral symmetry of the parent trajectory.
    groups=([0,6],[3,9],[4,10],[14,18],[17,21])
    scales=np.r_[np.tile([.19,.14,.12,.30,.16],5),[.15,.4,.07,.12,.4,.07,.18,.18,.35,.4,.05,.3]]
    mean=np.zeros(len(scales))
    sigma=scales.copy()
    def decode(p):
        motion=np.tile(base,(len(p),1,1))
        for j,ix in enumerate(groups):motion[:,:,ix]+=p[:,:25].reshape(-1,5,5)[:,:,j,None]
        return motion.clip(lo,hi),p[:,25:]+gains
    def score(p,initial):
        motion,fb=decode(p)
        value=np.empty(len(p))
        lib.evaluate_balance(xml,len(p),np.ascontiguousarray(initial[:,1:]),np.ascontiguousarray(motion),
                             stand,value,np.ascontiguousarray(fb))
        return value
    valid=rows[::13][:32]
    best=-np.inf
    for gen in range(80):
        size=48
        par=mean+sigma*rng.normal(size=(size,len(mean)))
        par[0]=mean
        starts=rows[rng.choice(len(rows),5,replace=False)]
        score_by_start=score(np.repeat(par,5,axis=0),np.tile(starts,(size,1))).reshape(size,5)
        values=.75*score_by_start.mean(1)+.25*score_by_start.min(1)
        elite=par[np.argsort(values)[-8:]]
        win=par[values.argmax()]
        v=score(np.tile(win,(len(valid),1)),valid)
        key=.75*v.mean()+.25*np.percentile(v,15)
        if key>best:
            best=key
            m,g=decode(win[None])
            np.save(out/"keys_1.npy",m[0])
            np.save(out/"feedback.npy",g[0])
        print(gen,round(best,3),np.percentile(v,[10,50,90]).round(3).tolist(),flush=True)
        mean=.3*mean+.7*elite.mean(0)
        sigma=np.maximum(.14*scales,.3*sigma+.7*elite.std(0))
if __name__=="__main__":fit()

