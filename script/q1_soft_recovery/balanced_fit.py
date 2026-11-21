"""Fit foot-supported poses using the actual collision mesh sole, not the height sensor proxy."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,nnls,lsq_linear
from scipy.spatial import ConvexHull
from common import Plant,STAND,LO,HI,KP,mj


def fit(out,arm_raise,flat_feet=False,smooth_forces=False):
    out.mkdir(exist_ok=False)
    plant=Plant(1,out/"scene");m=plant.model;d=mj.MjData(m)
    feet=[m.body(s+"_ankle_roll_link").id for s in ("left","right")]
    fg=[m.geom(s+"_ankle_roll_link_collision_1").id for s in ("left","right")]
    floor=m.geom("floor").id;pelvis=m.body("pelvis").id
    meshes=[];soles=[]
    for geom in fg:
        mid=m.geom_dataid[geom];a=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid]
        rotation=np.zeros(9);mj.mju_quat2Mat(rotation,m.geom_quat[geom])
        points=m.mesh_vert[a:a+n]@rotation.reshape(3,3).T+m.geom_pos[geom]
        sole=points[points[:,2]<points[:,2].min()+.0005]
        sole=sole[ConvexHull(sole[:,:2]).vertices]
        meshes.append(points);soles.append(sole)
    center=np.array([p.mean(0) for p in soles])
    last=np.array([.3,-1.8,2.15,-.5]);path=[];ups=[];commands=[];poses=[];diagnostics=[]
    def place(parameters):
        pitch,hip,knee,ankle=parameters
        d.qpos[:]=np.r_[0,0,0,np.cos(pitch/2),0,np.sin(pitch/2),0,STAND]
        d.qpos[[7,13]]=hip;d.qpos[[10,16]]=knee;d.qpos[[11,17]]=ankle
        if flat_feet:d.qpos[[8,9,14,15]]=0.
        d.qpos[[21,25]]=-arm_raise*(1-progress);d.qpos[[24,28]]=1.25
        mj.mj_kinematics(m,d)
        lowest=min((points@d.xmat[f].reshape(3,3).T+d.xpos[f])[:,2].min() for points,f in zip(meshes,feet))
        d.qpos[2]=.0002-lowest
        mj.mj_forward(m,d)
        return np.array([d.xpos[f]+d.xmat[f].reshape(3,3)@c for f,c in zip(feet,center)])
    for index in range(161):
        progress=index/160;desired_z=.235+.185*progress;desired_pitch=.25*(1-progress)
        def objective(parameters):
            centers=place(parameters)
            ground=max([max(0.,-c.dist) for c in d.contact if floor in c.geom and (c.geom1 if c.geom2==floor else c.geom2) not in fg],default=0.)
            selfdepth=max([max(0.,-c.dist) for c in d.contact if floor not in c.geom],default=0.)
            return np.r_[(d.qpos[2]-desired_z)*12,(d.subtree_com[pelvis,0]-centers[:,0].mean())*20,
                         d.xmat[feet[0]].reshape(3,3)[2,0]*4,(parameters[0]-desired_pitch)*.4,
                         ground*50,selfdepth*50,(parameters-last)*.02]
        result=least_squares(objective,last,bounds=([-1.2,LO[0],LO[3],LO[4]],[1.2,HI[0],HI[3],HI[4]]),
                             xtol=1e-9,ftol=1e-9,gtol=1e-9,max_nfev=120)
        last=result.x.copy();centers=place(last)
        generalized=[]
        for body,points in zip(feet,soles):
            for point in points@d.xmat[body].reshape(3,3).T+d.xpos[body]:
                jac=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,jacr,point,body)
                generalized.append(jac[2])
        matrix=np.stack(generalized,axis=1)
        if smooth_forces:
            count=matrix.shape[1]
            system=np.vstack((matrix[:6],np.eye(count)*1e-4));rhs=np.r_[d.qfrc_bias[:6],np.zeros(count)]
            forces=lsq_linear(system,rhs,bounds=(0,np.inf),tol=1e-12).x
        else:forces,_=nnls(matrix[:6],d.qfrc_bias[:6])
        tau=d.qfrc_bias[6:]-matrix[6:]@forces
        path.append(d.qpos[7:].copy());ups.append(d.xmat[pelvis].reshape(3,3)[2].copy())
        commands.append((d.qpos[7:]+tau/KP).clip(LO,HI));poses.append(d.qpos.copy())
        diagnostics.append(dict(index=index,z=float(d.qpos[2]),pitch=float(last[0]),com_margin_x=float(d.subtree_com[pelvis,0]-centers[:,0].mean()),
                                cost=float(result.fun@result.fun),max_torque=float(abs(tau).max()),root_residual=float(np.linalg.norm(matrix[:6]@forces-d.qfrc_bias[:6])),
                                ground_mm=1000*max([max(0.,-c.dist) for c in d.contact if floor in c.geom],default=0.),
                                self_mm=1000*max([max(0.,-c.dist) for c in d.contact if floor not in c.geom],default=0.)))
    np.savez(out/"curve.npz",q=path,up=ups,command=commands,qpos=poses)
    (out/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps([diagnostics[0],diagnostics[-1]]),flush=True);plant.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--arm-raise",type=float,default=.4)
    p.add_argument("--flat-feet",action="store_true");p.add_argument("--smooth-forces",action="store_true")
    a=p.parse_args();fit(a.out,a.arm_raise,a.flat_feet,a.smooth_forces)
