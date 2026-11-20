"""CPU Q1 closed-loop evaluator for strict fall recovery."""
import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = Path(os.environ.get("Q1_V2B_BUNDLE", ROOT / "runs/q1_v2b_gantry_deploy_bundle")).resolve()
H3 = ROOT / "runs/h003_causal_bwm"
sys.path[:0] = list(map(str, [HERE / "dependencies", BUNDLE / "code", H3, ROOT / "src"]))
from h003_terrains import build_fall_eval_xml
from q1_v2b_runtime import Q1V2bOnnxController
from q1_kinematics_np import Q1KinematicsNP, quat_wxyz_to_mat, yaw_mat_np
from leg_height import LegHeight
from q1_cgfa_obs_np import ContactGatedFootAnchorNP, build_obs_80_np, build_robot_pose_590_np

def scene(directory):
    tree = ET.parse(build_fall_eval_xml(directory))
    floor = tree.find(".//geom[@name='floor']")
    floor.attrib.update(priority="1", solref="0.01 1", solimp="0.99 0.999 0.001", friction="0.6 0.005 0.0001")
    path = directory / "fall_hard.xml"
    tree.write(path)
    return mujoco.MjModel.from_xml_path(str(path))

def poses(ctrl, seed):
    from math import cos, sin, radians, pi
    rng = np.random.default_rng(seed)
    angles = [(0, radians(88 + rng.uniform(-5,5))) for _ in range(8)]
    angles += [(0, radians(-88 + rng.uniform(-5,5))) for _ in range(8)]
    angles += [(radians((1 if i%2==0 else -1)*(88+rng.uniform(-5,5))),0) for i in range(8)]
    for i, (roll,pitch) in enumerate(angles):
        yaw = rng.uniform(-pi,pi)
        cr,sr = cos(roll/2),sin(roll/2)
        cp,sp = cos(pitch/2),sin(pitch/2)
        cy,sy = cos(yaw/2),sin(yaw/2)
        quat = [cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
        q = np.clip(ctrl.default_q+rng.uniform(-.15,.15,22),ctrl.q_lo,ctrl.q_hi)
        yield ("prone","supine","side")[i//8],np.r_[0,0,.16,quat,q]

def score(trace):
    upright = (trace[:,0]>=.39) & (trace[:,1]>=.85)
    count = 0
    first = None
    for step, ok in enumerate(upright, 1):
        count = count+1 if ok else 0
        if count==50 and first is None:
            first=step*.02
    return dict(success=bool(count>=50),first_stable_s=first,end_z=float(trace[-1,0]),
                end_cos=float(trace[-1,1]),terminal_stable_ticks=count,
                pd_oob=int(trace[:,3].sum()), act_rate=float(trace[:,4].mean()))

def evaluate(output,seed,expert_path=None,factory=None,public_runtime=False):
    directory=HERE/"scene"
    model=scene(directory)
    ctrl=Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    kin=Q1KinematicsNP(ROOT/"src/unisupermimic/assets/robots/q1/supermimic.xml")
    est=ContactGatedFootAnchorNP(kin)
    ref_R=np.eye(3,dtype=np.float32)
    modal=build_robot_pose_590_np(np.zeros((1,3),np.float32),ref_R[None],ctrl.default_q[None],0)
    xy=np.zeros(2,np.float32)
    legs=LegHeight(ROOT/"src/unisupermimic/assets/robots/q1/supermimic.xml")
    result=[]
    expert=None
    if expert_path:
        from recovery import Recovery
        if public_runtime:
            from q1_recovery_runtime import Q1RecoveryBranch
            expert=Q1RecoveryBranch(ctrl,expert_path,BUNDLE/"robot_model/supermimic.xml")
        else:
            expert=(factory or Recovery)(ctrl,expert_path)
    output.mkdir(parents=True,exist_ok=False)
    assets=[BUNDLE/"config/q1_deploy_config.json",*sorted((BUNDLE/"onnx").glob("*.onnx")),
            ROOT/"runs/h002_sonic_v2/checkpoints/h002_sonic_v2_bundle.pt",directory/"fall_hard.xml",directory/"supermimic_abs.xml"]
    (output/"sources.json").write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},indent=2))
    for i,(pose,qpos) in enumerate(poses(ctrl,seed)):
        ref_R[:]=np.eye(3,dtype=np.float32)
        data=mujoco.MjData(model)
        data.qpos[:]=qpos
        data.ctrl[:]=qpos[7:]
        mujoco.mj_forward(model,data)
        mujoco.mj_step(model,data,nstep=20)
        ctrl.reset(data.qpos[7:].astype(np.float32))
        if expert: expert.reset()
        prev=np.zeros(22,np.float32)
        trace=[]
        states=[]
        for tick in range(300):
            R=quat_wxyz_to_mat(data.qpos[3:7])
            q=data.qpos[7:].astype(np.float32)
            w=data.qvel[3:6].astype(np.float32)
            qd=data.qvel[6:].astype(np.float32)
            if tick==0: est.reset(R,w,q,qd,xy,R_ref=ref_R)
            def observation(exy):
                return build_obs_80_np(q=q,qd=qd,default_q=ctrl.default_q,R_imu=R,w_body=w,a_prev=prev,R_ref=ref_R,cgfa_exy=exy)
            def base():
                exy,_=est.step(R,w,q,qd,xy,R_ref=ref_R)
                return ctrl.step(observation(exy),modal,q,qd)
            def reanchor():
                ref_R[:]=yaw_mat_np(R)
                est.reset(R,w,q,qd,xy,R_ref=ref_R)
            if public_runtime:
                target,tau,gate=expert.step(R,w,q,qd,prev,base,reanchor)
            elif expert:
                height,valid=legs.measure(R[2],q)
                target,tau,gate=expert.step_measured(observation(xy),height,valid,legs.width(R[2],q),base,reanchor)
            else:
                target,tau,_=base()
                gate=0.
            data.ctrl[:]=target
            mujoco.mj_step(model,data,nstep=4)
            action=(target-ctrl.default_q)/ctrl.action_scale
            depth=max(0.,-min((c.dist for c in data.contact),default=0.))
            trace.append([data.qpos[2],quat_wxyz_to_mat(data.qpos[3:7])[2,2],abs(tau).max(),
                          np.count_nonzero(abs(tau)>ctrl.eff+1e-4),abs(action-prev).mean(),gate,depth])
            states.append(data.qpos.copy())
            prev=action
        trace=np.asarray(trace)
        np.savez_compressed(output/f"trial_{i:02}.npz",trace=trace,qpos=states,initial=qpos)
        result.append(dict(trial=i,pose=pose,**score(trace)))
        (output/"trials.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:sum(r["success"] for r in result if r["pose"]==k) for k in ("prone","supine","side")}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("output",type=Path)
    ap.add_argument("--seed",type=int,default=20261119)
    ap.add_argument("--expert",type=Path)
    ap.add_argument("--public-runtime",action="store_true")
    args=ap.parse_args()
    if args.public_runtime and not args.expert: ap.error("--public-runtime requires --expert")
    evaluate(args.output,args.seed,args.expert,public_runtime=args.public_runtime)

