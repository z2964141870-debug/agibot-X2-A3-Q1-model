"""Join verified saved78 with the independently completed missing six."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from event_summary import (CLIPS, CONFIGS, EVENTS, EventSpec, EventTimeline, capture_window,
    folder, identity, metrics, read, require, telemetry_summary, validate_event_timing)
from event_partial_summary import groups, stopped_gate
from event_missing_six import MISSING, ORIGINAL_RESULT_SHA256, PARTIAL_SUMMARY_SHA256


def key(row):
    return row["event"], row["method"], row["horizon_frames"], row["clip"]


def merge(partial, added):
    old = partial.get("rows", [])
    expected = {(e,m,h,c) for e in EVENTS for m,h in CONFIGS for c in CLIPS}
    require(partial.get("source_campaign_status") == "FAILED" and partial.get("source_final_gate_passed") is False
            and len(old) == len({key(r) for r in old}) == 78
            and len(added) == len({key(r) for r in added}) == 6
            and {key(r) for r in added} == set(MISSING)
            and not {key(r) for r in old}&{key(r) for r in added}
            and {key(r) for r in old+added} == expected,
            "Require78 preserved and six disjoint missing cases, never relabel the old failure")
    lookup = {key(r):r for r in old+added}
    return [lookup[e,m,h,c] for e in EVENTS for m,h in CONFIGS for c in CLIPS]


def analyze(partial_root, followup, output, evaluation_commit):
    started = time.monotonic()
    require(len(evaluation_commit) == 40 and all(c in "0123456789abcdef" for c in evaluation_commit),
            "Publish the independent joined analyzer before execution")
    require(all(p.is_absolute() and p.resolve() == p for p in (partial_root,followup,output))
            and not output.exists() and not output.is_relative_to(partial_root)
            and not output.is_relative_to(followup), "Use original roots and a fresh independent analysis output")
    summary_path = partial_root/"summary.json"
    require(identity(summary_path)["sha256"] == PARTIAL_SUMMARY_SHA256, "The analyzed saved78 changed")
    partial, partial_audit = read(summary_path), read(partial_root/"audit.json")
    result, auth = read(followup/"result.json"), read(followup/"authorization.json")
    require(partial_audit.get("status") == "PARTIAL_ANALYSIS_PASSED"
            and result.get("status") == "COMPLETED" and result.get("stage") == "MISSING_SIX_EVENT_REPLAY"
            and result.get("source_integrity") == "PASSED" and result.get("new_physical_cases") == 6
            and result.get("reused_physical_cases") == 78 and result.get("new_training_updates") == 0
            and auth.get("stage") == "PAUSED32K_MISSING_SIX_EVENT_FOLLOWUP"
            and auth.get("maximum_wall_seconds") == auth.get("phase_wall_seconds") == 1800
            and auth.get("maximum_output_bytes") == 1024**3
            and len(auth.get("cases", [])) == 6 and {tuple(r) for r in auth["cases"]} == set(MISSING)
            and len(auth.get("reused_cases", [])) == 78
            and {tuple(r) for r in auth["reused_cases"]} == {key(r) for r in partial["rows"]}
            and auth.get("partial_summary_sha256") == PARTIAL_SUMMARY_SHA256
            and auth.get("original_result_sha256") == ORIGINAL_RESULT_SHA256,
            "Independent missing-six completion, budget or coverage gate differs")
    require(not Path("/proc",str(result["controller"]["pid"])).exists(), "Follow-up controller still present")
    original = Path(auth["original_root"])
    require(not output.is_relative_to(original) and identity(original/"result.json")["sha256"]
            == ORIGINAL_RESULT_SHA256, "Original failed campaign changed")
    original_result = read(original/"result.json")
    stopped_gate(original_result,read(original/(original_result["phase"]+".guard.json")),
                 lambda p:Path("/proc",str(p)).exists())
    expected_phases = {f"{e}_{m}_h{h}_{c}_{phase}" for e,m,h,c in MISSING for phase in ("replay","media")}
    require(len(result.get("phases", [])) == 12
            and {p["name"] for p in result["phases"]} == expected_phases,
            "All twelve independent guarded phases must be present")
    records = [identity(p) for p in (summary_path,partial_root/"audit.json",followup/"result.json",
        followup/"authorization.json",followup/"checkpoint_audit.json",followup/"equivalence.json",
        Path(__file__).resolve(),Path(__file__).with_name("event_summary.py"),
        Path(__file__).with_name("event_partial_summary.py"),Path(__file__).with_name("preview_summary.py"),
        Path(__file__).with_name("analyze.py"))]
    records += auth["reused_parent_artifacts"]+auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
    for phase in result["phases"]:
        require(identity(Path(phase["guard"]["path"])) == phase["guard"], "Follow-up guard record changed")
        guard = read(Path(phase["guard"]["path"]))
        require(guard.get("status") == "WORKER_COMPLETED" and guard["cleanup"].get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "An independent guarded phase did not complete")
        records.append(phase["guard"])
    unique = {}
    for value in records:
        require(value["path"] not in unique or unique[value["path"]] == value, "Conflicting saved input identities")
        unique[value["path"]] = value
    require(all(identity(Path(r["path"])) == r for r in unique.values()), "Saved parent or follow-up sources changed")
    audit, equality = read(followup/"checkpoint_audit.json"), read(followup/"equivalence.json")
    selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    require(audit.get("status") == equality.get("status") == "PASSED" and audit.get("iteration") == 32000
            and equality.get("checkpoint") == selected and len(equality.get("cases", [])) == 21,
            "Reused checkpoint/action audit differs")
    saved_cases = {r["path"]:r for r in result["case_records"]}
    added = []
    for event,method,horizon,clip in MISSING:
        require(time.monotonic()-started < 300, "Joined saved analysis exceeded300s")
        directory = folder(followup,event,method,horizon,clip)
        saved, telemetry = directory/"review/result.json", directory/"stream_timing.jsonl"
        require(str(saved) in saved_cases and identity(saved) == saved_cases[str(saved)], "New case changed after completion")
        case = read(saved)
        require(case.get("status") == "PASSED" and case.get("iteration") == 32000
                and case.get("checkpoint") == selected and len(case.get("cases", [])) == 1,
                "Missing completed same-policy physical/media case")
        row = case["cases"][0]
        require((row["event"],row["tail_method"],row["horizon_frames"],row["clip"])
                == (event,method,horizon,clip) and row["seed"] == 0 and row["interface"] == "human"
                and row["policy_device"] == "cuda" and row["reference_is_synthetic"] is True
                and row["measured_end_to_end_latency_ns"] is None, "New case identity or latency claims differ")
        trace = directory/"review"/(clip+"_s0")/"rollout.npz"
        video = trace.with_name("tracking.mp4")
        require(Path(row["trace"]) == trace and row["video"].get("status") == "PASSED"
                and row["video"].get("original_speed") is True and identity(video)
                == {k:row["video"][k] for k in ("path","bytes","sha256")}, "New original-speed video differs")
        spec = EventSpec(**row["event_mapping"]["event"])
        require(spec == EventSpec(event,first_frame=50,duration_frames=10,transport_frames=0), "New event changed")
        timings = [json.loads(line) for line in telemetry.read_text().splitlines()]
        require(len(timings) == row["attempted_steps"], "New timing and physical counts differ")
        timeline = EventTimeline(row["reference_steps"]+1,spec)
        for frame,value in enumerate(timings):
            validate_event_timing(value,frame,horizon,method,timeline)
        with np.load(trace,allow_pickle=False) as arrays:
            actual,reference,actions = (arrays[k] for k in ("actual","reference","canonical_action"))
            require(actual.shape == (len(actions)+1,79) and reference.shape == (row["reference_steps"]+1,79)
                    and all(np.isfinite(a).all() for a in (actual,reference,actions)), "Invalid new saved trace")
            timing = telemetry_summary(timings,actions,spec,horizon)
            window = capture_window(timings,actions,actual,reference,row["upper_joint_indices"],spec)
        added.append(dict(event=event,method=method,horizon_frames=horizon,preview_budget_ms=horizon*20,
            clip=clip,**metrics(row),failure_frame=row["failure_frame"],
            completion_fraction=row["valid_steps"]/row["reference_steps"],video=row["video"],
            derived_reference_sha256=row["derived_reference_sha256"],telemetry=timing,capture_window=window))
        for path in (saved,telemetry,trace,video):
            unique[str(path)] = identity(path)
    rows = merge(partial,added)
    lookup = {key(r):r for r in rows}
    for row in rows:
        control = lookup[row["event"],"masked",10,row["clip"]]
        require(row["derived_reference_sha256"] == control["derived_reference_sha256"], "Joined comparison targets differ")
        row["change_from_same_event_h10"] = ({k:row[k]-control[k] for k in
            ("upper_rmse_rad","slip_left_m_s","slip_right_m_s","tilt_rms_rad")
            if row.get(k) is not None and control.get(k) is not None}
            if row["complete"] and control["complete"] else None)
    require(all(identity(Path(r["path"])) == r for r in unique.values()), "Input changed during joined reading")
    value = dict(stage="JOINED84_SAVED_EVENT_ANALYSIS",status="PASSED",evaluation_commit=evaluation_commit,
        original_campaign_status="FAILED",original_full84_campaign_gate_passed=False,
        followup_campaign_status="COMPLETED",joined84_sources_verified=True,full84_comparison_performed=True,
        reused_physical_cases=78,additional_physical_cases=6,total_cases=84,
        complete_cases=sum(r["complete"] for r in rows),rows=rows,groups=groups(rows),inputs=list(unique.values()),
        new_physical_transitions=0,new_inference_calls=0,new_training_updates=0,
        measured_end_to_end_latency_ms=None,natural_teleoperation_qualified=False,user_accepted_checkpoint=None,
        note="Independent saved-artifact union; original wall-time failure remains unchanged. Synthetic clocks are not hardware delay.")
    output.mkdir(parents=True,exist_ok=False)
    (output/"summary.json").write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    flat = [{k:v for k,v in row.items() if k not in
             ("video","telemetry","capture_window","change_from_same_event_h10")} for row in rows]
    with (output/"per_case.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    require(all(identity(Path(r["path"])) == r for r in unique.values()), "Input changed while writing joined analysis")
    (output/"audit.json").write_text(json.dumps(dict(status="JOINED84_ANALYSIS_PASSED",seconds=time.monotonic()-started,
        sources_unchanged=True,original_campaign_status="FAILED",new_physics=0,new_inference=0,new_training=0),indent=2)+"\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("partial-root","followup","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--evaluation-commit",required=True)
    args = parser.parse_args()
    result = analyze(args.partial_root,args.followup,args.output,args.evaluation_commit)
    print(json.dumps(dict(status=result["status"],total=result["total_cases"],complete=result["complete_cases"])))
