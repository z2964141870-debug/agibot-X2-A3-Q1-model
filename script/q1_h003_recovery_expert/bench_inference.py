"""Time the complete gate, inference and PD call on a changing 50 Hz reference stream."""
import argparse
import hashlib
from check_base import verify
import json
import os
import platform
from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort

def run(args):
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(args.bundle/"code"),str(here/"dependencies")]
    from q1_v2b_runtime import Q1V2bOnnxController
    from q1_recovery_runtime import Q1RecoveryBranch
    from q1_cgfa_obs_np import (build_robot_pose_590_np,build_vr_3pt_360_np,build_human_smpl_430_np,
                                 build_planner_cmd_single_10d_np,build_obs_80_np)
    if hasattr(os,"sched_setaffinity"):
        os.sched_setaffinity(0,{args.cpu})
    base_hashes=verify(args.bundle)
    data=np.load(args.clip)
    p,R,q,bp,br=(data[k] for k in ("root_p","root_R","q","body_p","body_R"))
    qd=np.diff(q,axis=0,prepend=q[:1])/.02
    omega=np.zeros(3,np.float32)
    base=Q1V2bOnnxController(args.bundle)
    rec=Q1RecoveryBranch(base,args.expert,args.bundle/"robot_model/supermimic.xml")
    zero2=np.zeros(2,np.float32)
    report={"platform":platform.platform(),"cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            "ort_version":ort.__version__,"provider":base.head_sess.get_providers(),"intra_threads":1,"inter_threads":1,
            "load_before":os.getloadavg(),"samples_per_path":args.samples,"clip":args.clip.name,"base_sha256":base_hashes,"period_ms":args.period_ms,"active_wait":args.active_wait,"paths":{}}
    for mod in base.tok_sess:
        if mod=="robot_pose":features=[build_robot_pose_590_np(p,R,q,i) for i in range(len(q))]
        elif mod=="human_smpl":features=[build_human_smpl_430_np(p,R,bp,i) for i in range(len(q))]
        elif mod=="vr_3pt":features=[build_vr_3pt_360_np(bp,br,i) for i in range(len(q))]
        else:features=[np.tile(build_planner_cmd_single_10d_np(vx=.3+.05*np.sin(i*.02)),10) for i in range(len(q))]
        # Replay the same prepared, upright stream through two independent V2b histories.
        # This check is outside timing and never forces the gate mode.
        standalone=Q1V2bOnnxController(args.bundle,(mod,))
        standalone.reset(q[0])
        base.reset(q[0])
        rec.reset()
        previous=np.zeros(22,np.float32)
        digest=hashlib.sha256()
        for j in range(min(len(q),200)):
            obs=build_obs_80_np(q=q[j],qd=qd[j],default_q=base.default_q,R_imu=R[j],w_body=omega,
                                a_prev=previous,R_ref=R[j],cgfa_exy=zero2)
            def base_call():return base.step(obs,features[j],q[j],qd[j],modality=mod)
            target,tau,gate=rec.step(R[j],omega,q[j],qd[j],previous,base_call,lambda:None)
            ref,ref_tau,_=standalone.step(obs,features[j],q[j],qd[j],modality=mod)
            if gate!=0. or target.tobytes()!=ref.tobytes() or tau.tobytes()!=ref_tau.tobytes():
                raise AssertionError(f"Upright V2b mismatch at {mod} frame {j}")
            digest.update(target.tobytes()+tau.tobytes())
            previous=(target-base.default_q)/base.action_scale
        report.setdefault("flat_replay",{})[mod]={"ticks":min(len(q),200),"target_torque_sha256":digest.hexdigest()}
        del standalone
        for mode in ("track","recover","handoff"):
            if mode=="recover" and mod!="robot_pose":continue
            rec.reset();base.reset(q[0])
            previous=np.zeros(22,np.float32)
            times=[]
            deadline=time.perf_counter()
            for i in range(args.samples+150):
                if args.period_ms:
                    delay=deadline-time.perf_counter()
                    if args.active_wait:
                        while time.perf_counter()<deadline: pass
                    elif delay>0:
                        time.sleep(delay)
                    deadline+=args.period_ms*.001
                j=i%len(q)
                gravity_R=R[j] if mode!="recover" else np.array([[0,0,1],[0,1,0],[-1,0,0]],np.float32)
                obs=build_obs_80_np(q=q[j],qd=qd[j],default_q=base.default_q,R_imu=gravity_R,w_body=omega,
                                a_prev=previous,R_ref=R[j],cgfa_exy=zero2)
                rec.expert.gate.mode=mode
                rec.expert.gate.blend_ticks=5+i%20
                def base_call():return base.step(obs,features[j],q[j],qd[j],modality=mod)
                begin=time.perf_counter_ns()
                target,tau,g=rec.step(gravity_R,omega,q[j],qd[j],previous,base_call,lambda:None)
                elapsed=(time.perf_counter_ns()-begin)*1e-6
                if i>=150: times.append(elapsed)
                previous=(target-base.default_q)/base.action_scale
            values=dict(zip(("p50","p95","p99","max"),np.percentile(times,[50,95,99,100]).tolist()))
            values["over_2p5ms"]=int(np.count_nonzero(np.asarray(times)>=2.5))
            report["paths"][mod+"_"+mode]=values
            args.output.write_text(json.dumps(report,indent=2))
            print(mod,mode,round(values["p99"],4),flush=True)
    report["load_after"]=os.getloadavg()
    args.output.write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for key in ("bundle","clip","expert","output"):ap.add_argument(key,type=Path)
    ap.add_argument("--cpu",type=int,default=8)
    ap.add_argument("--samples",type=int,default=1000)
    ap.add_argument("--period-ms",type=float,default=0.)
    ap.add_argument("--active-wait",action="store_true",help="Poll the clock on the same single thread between scheduled calls.")
    run(ap.parse_args())

