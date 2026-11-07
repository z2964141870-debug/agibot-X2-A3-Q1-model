"""Analyze seven saved H4 specialist replays with seventeen preserved controls."""

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE.parent/"supermimic_teleop"),
    str(HERE.parent/"q1_five_stage"),str(HERE.parent/"primebot_q1")]
from analyze import metrics
from common import identity,require,save
from delivery import read
from event_summary import capture_window
from stream_contract import DT_NS
from stream_events import EventSpec,EventTimeline
from student_causal_summary import video_contract
from student_h4_review import matched_baselines,review_cases
from student_h4_review_resume import STAGE
from student_review import validate_timing
from student_summary import differences,full_group,software_summary
from tracking_response import analyze_signal


def completed_gate(result,auth):
    require(result.get("stage") == auth.get("stage") == STAGE
        and result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
        and result.get("cost_settlement",{}).get("state") == "SETTLED"
        and result.get("physical_cases") == auth.get("physical_cases") == 7
        and result.get("new_physical_cases") == auth.get("new_physical_cases") == 6
        and result.get("reused_physical_cases") == auth.get("reused_physical_cases") == 1
        and result.get("reused_baseline_cases") == auth.get("reused_baseline_cases") == 17
        and result.get("total_comparison_cases") == 24
        and result.get("new_optimizer_updates") == auth.get("new_optimizer_updates") == 0
        and auth.get("human_real_future_frames") == 4 and auth.get("selected_update") == 1500
        and [tuple(c) for c in auth.get("cases",[])] == review_cases()
        and len(result.get("cases",[])) == len({r["path"] for r in result.get("cases",[])}) == 7,
        "Wait for seven real H4 cases, final source gate and clean cost settlement")
    expected = {f"{e}_student_h4_{c}_{p}" for i,(e,c) in enumerate(review_cases()) for p in ("replay","media")
        if not (i == 0 and p == "replay")}
    require(len(result.get("phases",[])) == 13 and {p["name"] for p in result["phases"]} == expected
        and not Path("/proc",str(result["controller"]["pid"])).exists(),
        "Wait for all thirteen H4 continuation guards and controller cleanup")


def join_comparisons(rows,controls):
    require(len(rows) == 7 and {(r["event"],r["clip"]) for r in rows} == set(review_cases())
        and all(r.get("variant") == "h4_specialist" and r.get("horizon_frames") == 4 for r in rows),
        "Require every selected-H4 case once")
    require(len(controls) == 17 and len({(r["event"],r["clip"],r["variant"],r["horizon_frames"]) for r in controls}) == 17,
        "Keep all seventeen distinct saved controls")
    lookup = {(r["event"],r["clip"],r["variant"],r["horizon_frames"]):r for r in controls}
    for row in rows:
        e,c = row["event"],row["clip"]
        teacher,previous = (lookup[(e,c,v,4)] for v in ("teacher","student"))
        require(row["derived_reference_sha256"] == teacher["derived_reference_sha256"] == previous["derived_reference_sha256"],
            "New H4 and saved comparison targets differ")
        row["change_from_r1_h4"] = differences(row,previous)
        row["change_from_teacher_h4"] = differences(row,teacher)
        row["completion_fraction_change_from_r1_h4"] = row["completion_fraction"]-previous["completion_fraction"]
        row["change_from_teacher_h10"] = differences(row,lookup[(e,c,"teacher",10)]) if e == "nominal" else None
    combined = copy.deepcopy(controls+rows)
    for row in combined[:17]:
        row["variant"] = "r1_h4" if row["variant"] == "student" else "teacher_h"+str(row["horizon_frames"])
    return combined


def analyze(source,previous_root,output,evaluation_commit):
    started = time.monotonic()
    require(all(p.is_absolute() and p.resolve() == p for p in (source,previous_root,output))
        and not output.exists() and not output.is_relative_to(source) and not output.is_relative_to(previous_root)
        and len(evaluation_commit) == 40 and set(evaluation_commit) <= set("0123456789abcdef"),
        "Use actual saved source and a new published H4 analysis root")
    result,auth,audit = (read(source/n) for n in ("result.json","authorization.json","checkpoint_audit.json"))
    completed_gate(result,auth)
    old,old_audit = (read(previous_root/n) for n in ("summary.json","audit.json"))
    require(old.get("stage") == "R1_MATCHED_SAVED_REVIEW_ANALYSIS" and old.get("total_cases") == 23
        and old_audit.get("status") == "PASSED" and old_audit.get("source_inputs_unchanged") is True,
        "Use the completed original R1 saved analysis")
    controls = matched_baselines(old)
    require(read(source/"baseline_cases.json") == controls,"H4 preserved controls changed")
    selected = read(source/"selected_encoder_audit.json")
    require(selected.get("status") == "CHANGED_SELECTED_H4_ENCODER" and selected.get("selected_update") == 1500
        and selected.get("cuda_initialized") is False,"Missing actual CPU selected-H4 audit")
    records = [identity(source/n) for n in ("result.json","authorization.json","checkpoint_audit.json",
        "baseline_cases.json","selected_encoder_audit.json")]
    records += [identity(previous_root/n) for n in ("summary.json","audit.json")]+old["inputs"]+auth["protected_inputs"]
    records += [identity(p) for p in (Path(__file__).resolve(),HERE/"student_summary.py",HERE/"student_causal_summary.py",
        HERE/"analyze.py",HERE/"event_summary.py",HERE.parent/"supermimic_teleop/student_h4_review.py",
        HERE.parent/"supermimic_teleop/tracking_response.py")]
    require(all(identity(r["path"]) == r for r in records),"Saved H4 or original protected sources changed")
    for phase in result["phases"]:
        p = source/(phase["name"]+".guard.json")
        require(phase["guard"] == identity(p),"Saved H4 guard changed")
        guard = read(p)
        require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup",{}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == 0,"An H4 phase was not cleanly reaped")
        records.append(identity(p))
    teacher = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    rows,upper_phase = [],[]
    for (event,clip),saved in zip(review_cases(),result["cases"]):
        directory = source/f"{event}_student_h4_{clip}"
        p = directory/"review/result.json"
        require(saved == identity(p),"Ordered H4 saved case changed")
        review = read(p)
        require(review.get("status") == "PASSED" and review.get("checkpoint") == teacher
            and review.get("policy_worker_reaped") is True and review.get("policy_worker_return_code") == 0
            and len(review.get("cases",[])) == 1,"Missing completed H4 physics/media case")
        case = review["cases"][0]
        require((case.get("event"),case.get("clip"),case.get("variant"),case.get("horizon_frames")) == (event,clip,"student",4)
            and case.get("student_checkpoint") == selected["files"][0]
            and case.get("seed") == 0 and case.get("interface") == "human"
            and case.get("policy_device") == "cuda" and case.get("measured_end_to_end_latency_ns") is None,
            "Selected H4 identity, clock or receiver differs")
        trace,timing = Path(case["trace"]),directory/"stream_timing.jsonl"
        require(trace == directory/"review"/(clip+"_s0")/"rollout.npz"
            and review["stream_telemetry"] == identity(timing),"Saved H4 trace or timing changed")
        times = [json.loads(line) for line in timing.read_text().splitlines()]
        require(len(times) == case["attempted_steps"],"Saved H4 calls are truncated")
        spec = EventSpec(event,50,10,0)
        timeline = EventTimeline(case["original_paired_reference"]["frames"],spec)
        for frame,t in enumerate(times):
            validate_timing(t,frame,4,"student",timeline)
        with np.load(trace,allow_pickle=False) as arrays:
            actual,reference,actions = (arrays[k] for k in ("actual","reference","canonical_action"))
            require(actual.shape == (len(actions)+1,79) and reference.shape == (case["reference_steps"]+1,79)
                and len(actions) == len(times) and all(np.isfinite(arrays[k]).all() for k in arrays.files),
                "Invalid saved H4 arrays")
            software = software_summary(times,actions,spec)
            clocked = [dict(t,simulated_control_ns=(t["frame"]+4)*DT_NS) for t in times]
            window = None if event == "nominal" else capture_window(clocked,actions,actual,reference,case["upper_joint_indices"],spec)
            if case["complete"]:
                upper_phase += [dict(event=event,clip=clip,variant="h4_specialist",joint=name,
                    **analyze_signal(reference[:,7+j],actual[:,7+j])) for j,name in zip(case["upper_joint_indices"],case["upper_joint_names"])]
        video = case["video"]
        video_contract(video,case)
        require(identity(video["path"]) == {k:video[k] for k in ("path","bytes","sha256")},"H4 video changed")
        rows.append(dict(event=event,clip=clip,variant="h4_specialist",horizon_frames=4,requested_preview_ms=80,
            **metrics(case),failure_frame=case["failure_frame"],attempted_steps=case["attempted_steps"],
            completion_fraction=case["valid_steps"]/case["reference_steps"],derived_reference_sha256=case["derived_reference_sha256"],
            software=software,capture_window=window,video=video,valid_prefix_metrics=case.get("valid_prefix_metrics"),
            attempted_prefix_diagnostics=None if case["complete"] else case["diagnostics"]))
        records += [identity(p) for p in (p,trace,timing,Path(video["path"]),
            trace.with_name("foot_posture.csv"),trace.with_name("upper_timing.csv"))]
    combined = join_comparisons(rows,controls)
    unique = {}
    for r in records:
        require(r["path"] not in unique or unique[r["path"]] == r,"Conflicting H4 saved identities")
        unique[r["path"]] = r
    require(all(identity(r["path"]) == r for r in unique.values()),"H4 source changed during saved analysis")
    groups = [dict(variant=v,**full_group([r for r in combined if r["event"] == "nominal" and r["variant"] == v]))
        for v in ("teacher_h10","teacher_h4","r1_h4","h4_specialist")]
    value = dict(stage="H4_SPECIALIST01_MATCHED_SAVED_ANALYSIS",evaluation_commit=evaluation_commit,
        selected_update=1500,new_saved_cases=7,reused_saved_controls=17,total_comparison_cases=24,
        complete_new_cases=sum(r["complete"] for r in rows),rows=combined,nominal_groups=groups,
        upper_phase=upper_phase,inputs=list(unique.values()),cost_settlement=result["cost_settlement"],
        cumulative_gpu_charged_seconds=result["cumulative_gpu_charged_seconds"],new_inference_calls=0,
        new_physical_transitions=0,new_training_updates=0,measured_hardware_end_to_end_latency_ms=None,
        natural_teleoperation_qualified=False,user_accepted_checkpoint=None,source_preprocessing="offline_unverified",
        scope="Previously seen clips and synthetic events; original R1 worker/physics/targets preserved",
        failure_note="Failed clips retain only prefix diagnostics, without whole-clip errors or incomplete group means",
        timing_note="Requested80ms preview and nested synchronized-target/worker/local-pipe timings are separate, not hardware latency")
    output.mkdir(parents=True,exist_ok=False)
    save(output/"summary.json",value)
    flat = [{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in combined]
    with (output/"per_case.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    require(all(identity(r["path"]) == r for r in unique.values()),"H4 analysis output changed a protected source")
    save(output/"audit.json",dict(status="PASSED",source_inputs_unchanged=True,source_files=len(unique),
        seconds=time.monotonic()-started,new_inference=0,new_physics=0,new_training=0))
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("source","previous-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--evaluation-commit",required=True)
    args = p.parse_args()
    value = analyze(args.source,args.previous_root,args.output,args.evaluation_commit)
    print(json.dumps(dict(total=value["total_comparison_cases"],complete_new=value["complete_new_cases"])))
