"""Audit the original straight, figure-eight and box runners with identical frozen sessions."""
import argparse
import xml.etree.ElementTree as ET
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from bench import BUNDLE,HERE,ROOT,sys,np,mujoco,Q1V2bOnnxController,Q1KinematicsNP,ContactGatedFootAnchorNP
from recovery import Recovery,Gate
sys.path[:0]=[str(ROOT/"runs/h003_deploy_v2b_narrow"),str(ROOT/"runs/h002_sonic_v3")]
import render_v2b_narrow_compare_1x as walk
import eval_hard_bench_v3 as box

class Probe:
    def __init__(self,ctrl,expert,enabled):
        self.ctrl=ctrl
        self.session=ctrl.head_sess
        self.enabled=enabled
        self.digest=hashlib.sha256()
        self.observed=0
        self.base_calls=0
        self.expert=expert
        self.anchor=ContactGatedFootAnchorNP(Q1KinematicsNP(ROOT/"src/unisupermimic/assets/robots/q1/supermimic.xml"))

    def run(self,outs,inputs):
        self.obs=inputs["obs_hist"][0,-1]
        act=self.session.run(outs,inputs)
        self.digest.update(act[0].tobytes())
        return act

    def wrap_pd(self,original):
        def apply(raw,q,qd,lo,hi,kp,kd,eff,**kw):
            self.observed+=1
            def base():
                self.base_calls+=1
                target=original(raw,q,qd,lo,hi,kp,kd,eff,**kw)
                return target.numpy(),(kp*(target-q)-kd*qd).numpy(),{}
            if self.enabled:
                up=-self.obs[47:50]
                x=np.cross([0.,1.,0.],up)
                x=x/max(np.linalg.norm(x),1e-8)
                R=np.array([x,np.cross(up,x),up],dtype=np.float32)
                soles,feet,_,width_cm=self.anchor.compute_sole_fk(R,self.obs[44:47],q.numpy(),qd.numpy())
                lowest=soles[:,2].argmin()
                valid=float(feet[lowest,2,2]>.65)
                h=float(np.clip(-soles[lowest,2],0,.65))*valid
                target,tau,g=self.expert.step_measured(self.obs,h,valid,width_cm*.01,base,lambda:None)
                if g!=0: raise AssertionError("flat gate triggered at "+str(self.observed))
            else:
                target,tau,_=base()
            self.digest.update(target.tobytes())
            self.digest.update(tau.tobytes())
            return torch.from_numpy(target)
        return apply

def audit(output,path,ref_json=None,hard_box_trials=None):
    torch.set_num_threads(1)
    output.mkdir(exist_ok=False)
    tracks=output/"motions"
    tracks.mkdir()
    syn=box.synthesize_hard_bench_motions(tracks)
    stride=ROOT/"src/unisupermimic/assets/motions/q1/multidataset_2000/held_out/lafan_walk1_subject5_w480.csv"
    cyc,z=walk._extract_periodic_walk_cycle(stride)
    straight=walk._build_straight_walk_50hz_csv(cyc,z,tracks/"straight.csv",vx=.42,n_move=495)
    walk_model=mujoco.MjModel.from_xml_path(str(walk._build_hard_floor_sim_xml(output/"walk_xml")))
    (output/"box_xml").mkdir()
    box_xml=box.build_falling_box_xml(output/"box_xml",mass=.26)
    box_model=mujoco.MjModel.from_xml_path(str(box_xml))
    hard_tree=ET.parse(box_xml)
    hard_tree.find(".//geom[@name='floor']").attrib.update(priority="1",solref="0.01 1",solimp="0.99 0.999 0.001")
    hard_path=output/"box_xml/box_hard.xml"
    hard_tree.write(hard_path)
    hard_box=mujoco.MjModel.from_xml_path(str(hard_path))
    kin=walk.Q1Kinematics(str(ROOT/"src/unisupermimic/assets/robots/q1/supermimic.xml"),"cpu")
    motion=walk.MotionLib([str(syn["box_csv"])],kin,"cpu",src_fps=50.,ctrl_dt=.02)
    mdp=walk.MDP(kin,motion,torch.from_numpy(walk.Q_STAND),observation_mode="root_xy")
    legacy=json.loads((ROOT/"runs/h002_sonic_v3/hard_bench_v3_metrics.json").read_text())
    x=legacy["falling_box_calibrated_x_drop_m"]
    rng=np.random.default_rng(20261119)
    cases=[dict(x_drop=x+rng.uniform(-.01,.01),y_drop=rng.uniform(-.02,.02),z_drop=rng.uniform(1.2,1.3),box_mass=rng.uniform(.22,.34)) for _ in range(hard_box_trials or 20)]
    results={}
    ctrl=Q1V2bOnnxController(BUNDLE,("robot_pose",))
    # The pre-training test fails immediately if the flat path tries to run an expert.
    expert=Recovery(ctrl,path) if path else Recovery.__new__(Recovery)
    expert.ctrl=ctrl
    expert.gate=Gate()
    for enabled in (True,) if ref_json else (False,True):
        role="wrapped" if enabled else "reference"
        results[role]={}
        scenarios=(("straight",straight),("figure8",syn["fig8_csv"]),("box20",None),("box20_hard",None))
        if hard_box_trials:
            scenarios=((f"box{hard_box_trials}_hard",None),)
        for name,source in scenarios:
            expert.reset()
            probe=Probe(ctrl,expert,enabled)
            module=walk if source else box
            with patch.object(module,"_apply_torque_safe_pd_cpu",probe.wrap_pd(module._apply_torque_safe_pd_cpu)):
                if source:
                    res=walk.rollout_v2b_or_hybrid_1x(walk_model,source,src_fps=50.,pol_kind="v2b",label=name,sess_head=probe,sess_rob=ctrl.tok_sess["robot_pose"],is_loop=name=="figure8")
                    metrics=res["metrics"]
                else:
                    scene=hard_box if name.endswith("_hard") else box_model
                    catches=[]
                    for case in cases:
                        expert.reset()
                        r=box.rollout_falling_box_trial(scene,mujoco.MjData(scene),mdp,"v3_robot_0ms",sess_head=probe,sess_tok=ctrl.tok_sess["robot_pose"],**case)
                        catches.append(r["caught"])
                    metrics={"caught":catches,"successes":sum(catches)}
            assert probe.observed==probe.base_calls
            results[role][name]=dict(ticks=probe.observed,sha256=probe.digest.hexdigest(),metrics=metrics)
            (output/"results.json").write_text(json.dumps(results,indent=2))
            print(role,name,probe.observed,probe.digest.hexdigest(),flush=True)
    if ref_json:
        results["reference"]=json.loads(ref_json.read_text())["reference"]
        (output/"results.json").write_text(json.dumps(results,indent=2))
    assert results["reference"]==results["wrapped"]
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("output",type=Path)
    ap.add_argument("--expert",type=Path)
    ap.add_argument("--ref-json",type=Path)
    ap.add_argument("--hard-box-trials",type=int,choices=range(1,1001),metavar="N")
    a=ap.parse_args()
    audit(a.output.resolve(),a.expert,a.ref_json,a.hard_box_trials)

