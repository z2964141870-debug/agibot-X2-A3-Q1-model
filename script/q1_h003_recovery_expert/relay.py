"""Continuous walking perturbation, recovery and tracking handoff at 50 Hz."""
from bench import *
from q1_kinematics_np import quat_xyzw_to_mat,mat_to_quat_wxyz,yaw_angle_from_mat,rotz_np
from recovery import Recovery
from reference import HandoffReference

def run(path,out,public_runtime=False,data_dir=Path("/tmp/q1_v2b_narrow_tracks")):
    out.mkdir(exist_ok=False,parents=True)
    ctrl=Q1V2bOnnxController(BUNDLE,("robot_pose",))
    model=scene(out/"xml")
    kin=Q1KinematicsNP(ROOT/"src/unisupermimic/assets/robots/q1/supermimic.xml")
    report=[]
    for i,(track,kick) in enumerate([
            ("straight_walk_50hz.csv",(0,2.7)),("straight_walk_50hz.csv",(0,-2.7)),
            ("loop_figure8_walk.csv",(-2.7,0)),("loop_figure8_walk.csv",(2.7,0)),
            ("loop_figure8_walk.csv",(1.91,1.91))]):
        clip=np.loadtxt(data_dir/track,delimiter=",").astype(np.float32)
        pos=clip[:,:3]
        rot=quat_xyzw_to_mat(clip[:,3:7])
        joints=clip[:,7:]
        data=mujoco.MjData(model)
        data.qpos[:]=np.r_[pos[0],mat_to_quat_wxyz(rot[0]),joints[0]]
        data.ctrl[:]=joints[0]
        mujoco.mj_forward(model,data)
        if public_runtime:
            from q1_recovery_runtime import Q1RecoveryBranch
            c=Q1RecoveryBranch(ctrl,path,BUNDLE/"robot_model/supermimic.xml")
        else:
            c=Recovery(ctrl,path)
        est=ContactGatedFootAnchorNP(kin)
        ctrl.reset(joints[0])
        a_prev=np.zeros(22,np.float32)
        ref_window=None
        idx=0
        align=np.eye(3,dtype=np.float32)
        offset=np.zeros(3,np.float32)
        display_offset=np.zeros(3,np.float32)
        fallen=False
        handed=False
        walked=0
        released=False
        refall=False
        states=[]
        ghosts=[]
        trace=[]
        contact_after=False
        # Reserve 3 s of walking after a handoff within the 12 s recovery observation.
        for tick in range(850):
            if tick==100: data.qvel[:2]+=kick
            R=quat_wxyz_to_mat(data.qpos[3:7])
            q=data.qpos[7:].astype(np.float32)
            w=data.qvel[3:6].astype(np.float32)
            qd=data.qvel[6:].astype(np.float32)
            soles,feet,_,fw=est.compute_sole_fk(R,w,q,qd)
            def reanchor():
                nonlocal align,offset,ref_window
                align=rotz_np(yaw_angle_from_mat(R)-yaw_angle_from_mat(rot[idx]))
                offset[:2]=-(align@pos[idx])[:2]
                # Recorded world position is used only to place the visual reference.
                display_offset[:2]=data.qpos[:2]
                ref_window=HandoffReference(align@pos[idx]+offset,R,q)
                est.reset(R,w,q,qd,ref_window.origin[:2],R_ref=ref_window.rotation)
            goal_p,goal_R,goal_q=align@pos[idx]+offset,align@rot[idx],joints[idx]
            def observation(exy):
                return build_obs_80_np(q=q,qd=qd,default_q=ctrl.default_q,R_imu=R,w_body=w,
                         a_prev=a_prev,R_ref=goal_R,cgfa_exy=exy)
            def base():
                nonlocal goal_p,goal_R,goal_q
                if ref_window is not None:
                    goal_p,goal_R,goal_q,modal=ref_window.step(align@pos[idx]+offset,align@rot[idx],joints[idx])
                else:
                    modal=build_robot_pose_590_np(pos,rot,joints,idx)
                exy,_=est.step(R,w,q,qd,goal_p[:2],R_ref=goal_R)
                return ctrl.step(observation(exy),modal,q,qd)
            lowest=soles[:,2].argmin()
            valid=float(feet[lowest,2,2]>.65)
            height=float(np.clip(-soles[lowest,2],0,.65))*valid
            if public_runtime:
                target,tau,g=c.step(R,w,q,qd,a_prev,base,reanchor)
            else:
                target,tau,g=c.step_measured(observation(np.zeros(2,np.float32)),height,valid,fw*.01,base,reanchor)
            if tick>100:
                # Require a downed posture as well as body/ground contact outside the feet.
                for con in data.contact:
                    geom_names=[model.geom(int(j)).name or "" for j in (con.geom1,con.geom2)]
                    if "floor" in geom_names and any(x in n for x in ("pelvis","elbow","wrist","torso","shoulder") for n in geom_names):
                        contact_after=True
                fallen |= bool(contact_after and (R[2,2]<.65 or data.qpos[2]<.22))
                if fallen and g>0: handed=True
                if released and g==1: refall=True
                if handed and g==0: released=True
                if handed and g==0 and data.qpos[2]>=.35 and R[2,2]>.85: walked+=1
                else: walked=0
            action=(target-ctrl.default_q)/ctrl.action_scale
            change=abs(action-a_prev).mean()
            a_prev=action
            ghosts.append(np.r_[goal_p+display_offset,mat_to_quat_wxyz(goal_R),goal_q])
            if g<1. and (ref_window is None or ref_window.tick>=30): idx=min(idx+1,len(pos)-1)
            data.ctrl[:]=target
            mujoco.mj_step(model,data,nstep=4)
            depth=max(0.,-min((ct.dist for ct in data.contact),default=0.))
            trace.append([data.qpos[2],quat_wxyz_to_mat(data.qpos[3:7])[2,2],abs(tau).max(),
                       np.count_nonzero(abs(tau)>ctrl.eff+1e-4),change,g,depth,fw,idx])
            states.append(data.qpos.copy())
            if walked==150 or (tick>=700 and not released): break
        trace=np.asarray(trace)
        width=trace[-150:,7].mean() if walked==150 else None
        disp=float(np.linalg.norm(states[-1][:2]-states[-150][:2])) if walked==150 else 0.
        row=dict(trial=i,track=track,kick=kick,fallen=fallen,took_control=handed,walking_ticks=walked,
                 foot_width_cm=width,walk_disp_m=disp,pd_oob=int(trace[:,3].sum()),
                 refall=refall,success=bool(fallen and handed and not refall and walked==150 and width<=22 and disp>.25 and trace[:,3].sum()==0))
        report.append(row)
        np.savez_compressed(out/f"relay_{i}.npz",qpos=states,reference=ghosts,trace=trace)
        (out/"relay.json").write_text(json.dumps(report,indent=2))
        print(row,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("expert",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("--public-runtime",action="store_true")
    ap.add_argument("--tracks",type=Path,default=Path("/tmp/q1_v2b_narrow_tracks"))
    a=ap.parse_args()
    run(a.expert,a.out.resolve(),a.public_runtime,a.tracks)

