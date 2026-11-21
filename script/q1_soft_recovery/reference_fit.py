"""Retarget the crouch-to-stand reference to foot support in the Q1 model."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,nnls
from common import Plant,HERE,STAND,LO,HI,KP,mj


def fit(out):
    out.mkdir(exist_ok=False)
    plant=Plant(1,out/"scene")
    m=plant.model;d=mj.MjData(m)
    feet=[m.body(s+"_ankle_roll_link").id for s in ("left","right")]
    floor=m.geom("floor").id;pelvis=m.body("pelvis").id
    footgeom={m.geom(s+"_ankle_roll_link_collision_1").id for s in ("left","right")}
    sole=np.array([.02,0,-.022]);last=np.array([.6,-1.8,2.15,-.5])
    path=[];ups=[];commands=[];poses=[];diagnostics=[]
    def place(parameters):
        pitch,hip,knee,ankle=parameters
        d.qpos[:]=np.r_[0,0,0,np.cos(pitch/2),0,np.sin(pitch/2),0,STAND]
        d.qpos[[7,13]]=hip;d.qpos[[10,16]]=knee;d.qpos[[11,17]]=ankle
        d.qpos[[21,25]]=-.4*(1-progress);d.qpos[[24,28]]=1.25
        mj.mj_kinematics(m,d)
        soles=np.array([d.xpos[f]+d.xmat[f].reshape(3,3)@sole for f in feet])
        d.qpos[2]=-soles[:,2].min()
        mj.mj_forward(m,d)
        soles=np.array([d.xpos[f]+d.xmat[f].reshape(3,3)@sole for f in feet])
        return soles
    for index in range(161):
        progress=index/160
        desired_z=.20+.22*progress
        desired_pitch=.70*(1-progress)
        def objective(parameters):
            soles=place(parameters)
            com=d.subtree_com[pelvis]
            footnormal=d.xmat[feet[0]].reshape(3,3)[2,0]
            ground=max([max(0.,-c.dist) for c in d.contact if floor in c.geom and (c.geom1 if c.geom2==floor else c.geom2) not in footgeom],default=0.)
            selfdepth=max([max(0.,-c.dist) for c in d.contact if floor not in c.geom],default=0.)
            return np.r_[(d.qpos[2]-desired_z)*12,(com[0]-soles[:,0].mean())*20,
                         footnormal*3,(parameters[0]-desired_pitch)*.6,ground*30,selfdepth*40,
                         (parameters-last)*.015]
        result=least_squares(objective,last,bounds=([-1.2,LO[0],LO[3],LO[4]],[1.2,HI[0],HI[3],HI[4]]),
                             xtol=1e-9,ftol=1e-9,gtol=1e-9,max_nfev=100)
        last=result.x.copy();soles=place(last)
        generalized=[]
        for body,center in zip(feet,soles):
            for dx,dy in ((-.035,-.018),(-.035,.018),(.035,-.018),(.035,.018)):
                point=center+np.array([dx,dy,0.])
                jac=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,jacr,point,body)
                generalized.append(jac[2])
        matrix=np.stack(generalized,axis=1)
        forces,_=nnls(matrix[:6],d.qfrc_bias[:6])
        tau=d.qfrc_bias[6:]-matrix[6:]@forces
        command=(d.qpos[7:]+tau/KP).clip(LO,HI)
        path.append(d.qpos[7:].copy());ups.append(d.xmat[pelvis].reshape(3,3)[2].copy());commands.append(command)
        poses.append(d.qpos.copy())
        diagnostics.append(dict(index=index,z=float(d.qpos[2]),pitch=float(last[0]),com_margin_x=float(d.subtree_com[pelvis,0]-soles[:,0].mean()),
                                cost=float(np.sum(result.fun**2)),max_torque=float(abs(tau).max()),root_residual=float(np.linalg.norm(matrix[:6]@forces-d.qfrc_bias[:6]))))
    np.savez(out/"curve.npz",q=path,up=ups,command=commands,qpos=poses)
    (out/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(diagnostics[0]),flush=True);print(json.dumps(diagnostics[-1]),flush=True)
    plant.close()


if __name__=="__main__":
    import sys
    fit(Path(sys.argv[1]).resolve())
