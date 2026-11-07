"""Recover saved comparisons using joint names; never rerun failed inference."""

import argparse
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/"q1_five_stage"),str(HERE.parent/"primebot_q1")]
from common import ROOT, identity, require, save
from delivery import read
from student_same_state_math import CLIPS, VARIANTS, compare

STAGE="COVERAGE01_SAME_STATE_DIAGNOSTIC"
ERROR="ValueError: Require four aligned, finite canonical action prefixes and actual Q1 presence"


def canonical_names(text):
    values={}
    for node in ast.parse(text).body:
        targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AnnAssign) else []
        for target in targets:
            if isinstance(target,ast.Name) and target.id in ("_JOINT_STEMS","JOINT_NAMES"):
                require(target.id not in values,"Duplicate frozen joint declaration")
                values[target.id]=node.value
    require(set(values)=={"_JOINT_STEMS","JOINT_NAMES"},"Missing frozen canonical declarations")
    expected=ast.parse('tuple(f"{j}_joint" for j in _JOINT_STEMS)',mode="eval").body
    require(ast.dump(values["JOINT_NAMES"])==ast.dump(expected),"Canonical name derivation changed")
    stems=ast.literal_eval(values["_JOINT_STEMS"])
    require(len(stems)==29 and len(set(stems))==29 and all(type(s) is str for s in stems),"Invalid29 joint stems")
    return [s+"_joint" for s in stems]


def canonical_upper(names, present, case):
    upper=case["upper_joint_names"]
    require(len(names)==29 and len(set(names))==29 and len(upper)==8 and len(set(upper))==8
        and len(case["upper_joint_indices"])==8
        and all(n in names for n in upper),"Require exact physical upper names and unique canonical names")
    ids=[names.index(n) for n in upper]
    require(all(present[j] for j in ids),"A named physical upper joint has no canonical slot")
    return ids


def failed_gate(result):
    require(result.get("stage")==STAGE and result.get("status")=="FAILED" and result.get("error")==ERROR
        and len(result.get("phases",[]))==8 and result.get("new_physical_steps")==0
        and result.get("new_optimizer_updates")==0
        and result.get("cost_settlement",{}).get("state")=="SETTLED"
        and result["cost_settlement"]["cleanup"].get("owned_phase_workers_reaped")==8
        and not Path("/proc",str(result["controller"]["pid"])).exists(),
        "Only the fully reaped mapping-summary failure may use saved recovery")


def analyze(source,output,commit):
    started=time.monotonic()
    require(source==ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_SAME_STATE01_20261108"
        and output.is_absolute() and output.resolve()==output and not output.exists()
        and output.is_relative_to(ROOT/"data/supermimic_for_q1") and not output.is_relative_to(source)
        and len(commit)==40 and all(c in "0123456789abcdef" for c in commit),"Require independent published read-only recovery")
    result,auth=read(source/"result.json"),read(source/"authorization.json")
    failed_gate(result)
    require(auth.get("stage")==STAGE and len(auth.get("saved_prefixes",[]))==2
        and auth.get("maximum_policy_calls")==1024 and auth.get("new_optimizer_updates")==0,
        "Original bounded authorization changed")
    records=auth["protected_inputs"]+[identity(source/n) for n in ("result.json","authorization.json","checkpoint_audit.json")]
    records += [identity(Path(__file__).resolve()),identity(HERE/"test_student_same_state_summary.py")]
    plan=read(ROOT/"data/supermimic_for_q1/MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V04_20261104/e1024_train_plan.json")
    constants=Path(plan["source"])/"src/morphmimic/constants.py"
    require(str(constants) in {r["path"] for r in auth["protected_inputs"]},"Canonical names are not frozen inputs")
    names=canonical_names(constants.read_text())
    require(all(identity(r["path"])==r for r in records),"Original sources changed before saved recovery")
    guards=[]
    expected={f"{c}_{v}" for c in CLIPS for v in VARIANTS}
    require({p["name"] for p in result["phases"]}==expected,"Missing an actual same-state phase")
    for phase in result["phases"]:
        p=source/(phase["name"]+".guard.json")
        require(identity(p)==phase["guard"],"Original phase guard changed")
        g=read(p)
        require(g.get("status")=="WORKER_COMPLETED" and g.get("cleanup",{}).get("worker_reaped") is True
            and g["cleanup"].get("worker_return_code")==0,"Incomplete or unreaped inference phase")
        records.append(identity(p))
        guards.append(dict(name=phase["name"],guard=identity(p),status=g["status"],cleanup=g["cleanup"]))
    comparisons,software=[],{}
    total=0
    for clip in CLIPS:
        selected=next(s for s in auth["saved_prefixes"] if s["case"]["clip"]==clip)
        case=selected["case"]
        summaries={v:read(source/f"{clip}_{v}.summary.json") for v in VARIANTS}
        mask=np.asarray(summaries["coverage_f0"]["ready"]["canonical_present"],dtype=bool)
        require(mask.shape==(29,) and mask.sum()==22
            and mask.tolist()==summaries["original_f0"]["ready"]["canonical_present"],"Canonical presence changed")
        upper=canonical_upper(names,mask,case)
        actions={}
        for variant,s in summaries.items():
            require(s.get("status")=="SAME_STATE_CALLS_PASSED" and s.get("calls")==min(128,case["attempted_steps"])
                and s.get("source_trace")==selected["trace"],"Changed same-state action prefix")
            if variant=="coverage_f0":
                require(s.get("maximum_saved_action_difference")==0 and s.get("new_checkpoint_parity_checked") is True,
                    "Require actual saved selected-action parity")
            with np.load(source/f"{clip}_{variant}.actions.npz",allow_pickle=False) as a:
                actions[variant]=a["canonical_action"].copy()
            records += [identity(source/f"{clip}_{variant}.{suffix}") for suffix in ("summary.json","actions.npz","frames.jsonl")]
            total+=s["calls"]
        comparisons.append(dict(clip=clip,physical_upper_indices=case["upper_joint_indices"],
            canonical_upper_indices=upper,upper_joint_names=case["upper_joint_names"],
            **compare(actions,mask,upper,case)))
        software[clip]=summaries
    require(total==936,"Reuse the exact936 completed calls; do not rerun or add calls")
    unique={r["path"]:r for r in records}
    require(all(identity(r["path"])==r for r in unique.values()),"Final original source or saved output integrity failed")
    output.mkdir(exist_ok=False)
    save(output/"summary.json",dict(stage="SAME_STATE_SAVED_ANALYSIS",status="PASSED",evaluation_commit=commit,
        original_campaign_status="FAILED",original_error=result["error"],original_result=identity(source/"result.json"),
        source_integrity="PASSED",reused_policy_calls=total,new_policy_calls=0,new_physical_steps=0,new_optimizer_updates=0,
        comparisons=comparisons,software=software,phase_guards=guards,inputs=list(unique.values()),
        privileged_future_for_offline_reference_only=True,hardware_end_to_end_latency_ms=None,
        natural_teleoperation_qualified=False,user_accepted_checkpoint=None))
    (output/"source_result.json").write_bytes((source/"result.json").read_bytes())
    require(all(identity(r["path"])==r for r in unique.values()),"Source changed while writing summary")
    save(output/"audit.json",dict(status="SAME_STATE_SAVED_ANALYSIS_PASSED",seconds=time.monotonic()-started,
        source_files=len(unique),source_inputs_unchanged=True,reused_policy_calls=936,new_policy_calls=0,
        new_physical_steps=0,new_optimizer_updates=0,original_campaign_status="FAILED"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--evaluation-commit",required=True)
    a=p.parse_args();analyze(a.source,a.output,a.evaluation_commit)
