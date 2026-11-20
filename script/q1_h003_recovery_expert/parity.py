"""Check source ordering and one-step mjwarp/CPU agreement before starting PPO."""
import numpy as np
import torch
import warp as wp
import mujoco
from train import World
from bench import HERE,BUNDLE,Q1KinematicsNP,ContactGatedFootAnchorNP,ROOT
from q1_kinematics_np import quat_wxyz_to_mat,Q1_JOINT_NAMES
import json

def check():
    wp.init()
    wp.set_device("cuda:0")
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream().cuda_stream,device="cuda:0"))
    torch.manual_seed(713)
    w=World(16)
    kin=Q1KinematicsNP(ROOT/"src/unisupermimic/assets/robots/q1/supermimic.xml")
    est=ContactGatedFootAnchorNP(kin)
    names=[w.model.actuator(j).name for j in range(22)]
    assert names==list(Q1_JOINT_NAMES)
    q,v,prev=(t.cpu().numpy().copy() for t in (w.qpos,w.qvel,w.prev))
    frames=w.frame().cpu().numpy()
    max_obs=0.
    targets=w.stand.expand(16,-1)
    rows=[]
    for i in range(16):
        d=mujoco.MjData(w.model)
        d.qpos[:]=q[i]
        d.qvel[:]=v[i]
        mujoco.mj_forward(w.model,d)
        R=quat_wxyz_to_mat(q[i,3:7])
        soles,feet,_,_=est.compute_sole_fk(R,v[i,3:6],q[i,7:],v[i,6:])
        least=soles[:,2].argmin()
        valid=float(feet[least,2,2]>.65)
        frame=np.r_[q[i,7:]-w.stand.cpu().numpy(),v[i,6:]*.1,v[i,3:6]*.25,-R[2],prev[i]*.25,
                    np.clip(-soles[least,2],0,.65)*valid,valid]
        max_obs=max(max_obs,float(abs(frames[i]-frame).max()))
        rows.append(d)
    w.step(targets)
    safe=w.ctrl.cpu().numpy()
    positions=w.qpos.cpu().numpy()
    errors=[]
    for i,d in enumerate(rows):
        d.ctrl[:]=safe[i]
        mujoco.mj_step(w.model,d,nstep=4)
        errors.append(float(abs(d.qpos-positions[i]).max()))
    report=dict(observation_max_abs=max_obs,qpos_step_max_abs=errors,
                sample_pd_max_excess=float(((w.kp*(w.ctrl-torch.tensor(q[:,7:],device="cuda"))-w.kd*torch.tensor(v[:,6:],device="cuda")).abs()-w.eff).amax()))
    print(json.dumps(report),flush=True)
    (HERE/"parity.json").write_text(json.dumps(report,indent=2))
    assert max_obs<1e-4 and np.isfinite(errors).all() and report["sample_pd_max_excess"]<1e-4
    assert np.median(errors)<.1 and max(errors)<.5
if __name__=="__main__": check()

