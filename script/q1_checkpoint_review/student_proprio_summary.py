"""Read seven residual replays while preserving twenty-one actual controls."""

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"supermimic_teleop"),
    str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from analyze import metrics
from common import identity, require, save
from delivery import read
from event_summary import capture_window
from stream_contract import DT_NS
from stream_events import EventSpec, EventTimeline
from student_causal_review import review_cases, validate_causal_timing
from student_causal_summary import history_software, match_controls, video_contract
from student_proprio_review import STAGE, TRAINED
from student_summary import differences, full_group
from tracking_response import analyze_signal


def completed_gate(result, auth):
    require(result.get("stage") == auth.get("stage") == STAGE
        and result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and result.get("new_physical_cases") == auth.get("new_physical_cases") == 7
        and result.get("reused_baseline_cases") == auth.get("reused_baseline_cases") == 21
        and result.get("total_comparison_cases") == 28
        and result.get("new_optimizer_updates") == auth.get("new_optimizer_updates") == 0
        and auth.get("human_real_future_frames") == 0 and auth.get("selected_update") == 3000
        and auth.get("cpu_conversion_candidate_used") is False
        and auth.get("heldout_events_opened") is False
        and [tuple(c) for c in auth.get("cases", [])] == review_cases()
        and len(result.get("cases", [])) == len({r["path"] for r in result.get("cases", [])}) == 7,
        "Wait for all seven residual cases, original controls, source gate and settled cost")
    expected = {f"{e}_f0_student_h0_{c}_{p}" for e,c in review_cases() for p in ("replay", "media")}
    require(len(result.get("phases", [])) == 14 and {p["name"] for p in result["phases"]} == expected
        and not Path("/proc", str(result["controller"]["pid"])).exists(), "Wait for residual guards and controller cleanup")


def join_comparisons(rows, controls, previous):
    match_controls(rows, controls)
    expected = set(review_cases())
    require(len(previous) == 7 and {(r["event"], r["clip"]) for r in previous} == expected
        and all(r.get("variant") == "original_f0" and r.get("horizon_frames") == 0
            and r.get("human_real_future_frames") == 0 for r in previous),
        "Reuse the actual seven original f0 outcomes")
    lookup = {(r["event"], r["clip"]):r for r in previous}
    for row in rows:
        old = lookup[(row["event"], row["clip"])]
        require(row["derived_reference_sha256"] == old["derived_reference_sha256"], "Residual comparison targets differ")
        row["change_from_original_f0"] = differences(row, old)
        row["completion_fraction_change_from_original_f0"] = row["completion_fraction"]-old["completion_fraction"]
    combined = copy.deepcopy(controls+previous+rows)
    for row in combined[21:]:
        row["variant"] = "proprio_f0"
    return combined


def analyze(source, previous_root, output, evaluation_commit):
    started = time.monotonic()
    require(all(p.is_absolute() and p.resolve() == p for p in (source, previous_root, output))
        and not output.exists() and not output.is_relative_to(source) and not output.is_relative_to(previous_root)
        and len(evaluation_commit) == 40 and all(c in "0123456789abcdef" for c in evaluation_commit),
        "Use actual saved sources and one new published analysis root")
    result, auth, audit = (read(source/n) for n in ("result.json", "authorization.json", "checkpoint_audit.json"))
    completed_gate(result, auth)
    old, old_audit = (read(previous_root/n) for n in ("summary.json", "audit.json"))
    require(old.get("stage") == "COVERAGE01_MATCHED_SAVED_ANALYSIS" and old.get("total_comparison_cases") == 28
        and old_audit.get("status") == "PASSED" and old_audit.get("source_inputs_unchanged") is True
        and old.get("new_inference_calls") == old.get("new_physical_transitions") == old.get("new_training_updates") == 0,
        "Preserve the original passed saved comparison")
    controls = [r for r in old["rows"] if r["variant"] in ("teacher", "student")]
    previous = [r for r in old["rows"] if r["variant"] == "original_f0"]
    require(read(source/"baseline_cases.json") == controls and read(source/"original_f0_cases.json") == previous,
        "Reused residual controls differ from the original saved rows")
    selection = read(source/"selected_residual_audit.json")
    proof = selection.get("zero_history_reuse_proof", {})
    require(selection.get("status") == "SELECTED_NONZERO_CAUSAL_RESIDUAL"
        and selection.get("selected_update") == 3000 and selection.get("cuda_initialized") is False
        and proof.get("status") == "EXACT_ORIGINAL_F0_ZERO_RESIDUAL"
        and proof.get("base_tensors_exact") == 8 and proof.get("maximum_abs_difference") == 0,
        "Missing actual changed residual and zero-control identity proof")
    records = [identity(source/n) for n in ("result.json", "authorization.json", "checkpoint_audit.json",
        "baseline_cases.json", "original_f0_cases.json", "selected_residual_audit.json")]
    records += [identity(previous_root/n) for n in ("summary.json", "audit.json")]+old["inputs"]+auth["protected_inputs"]
    records += [identity(p) for p in (Path(__file__).resolve(), HERE/"student_causal_summary.py",
        HERE/"student_summary.py", HERE/"analyze.py", HERE/"event_summary.py")]
    require(all(identity(r["path"]) == r for r in records), "Saved residual or original sources changed")
    for phase in result["phases"]:
        p = source/(phase["name"]+".guard.json")
        require(phase["guard"] == identity(p), "Residual guard changed")
        guard = read(p)
        require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == 0, "Residual phase has unconfirmed cleanup")
        records.append(identity(p))
    teacher = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    residual_path = TRAINED/"causal_history/selected.pt"
    residual_record = identity(residual_path)
    require(residual_record in auth["protected_inputs"], "Selected residual identity is outside the original review admission")
    rows, upper_phase = [], []
    for (event, clip), saved in zip(review_cases(), result["cases"]):
        directory = source/f"{event}_f0_student_h0_{clip}"
        p = directory/"review/result.json"
        require(saved == identity(p), "Ordered residual case changed")
        review = read(p)
        require(review.get("status") == "PASSED" and review.get("checkpoint") == teacher
            and review.get("policy_worker_reaped") is True and review.get("policy_worker_return_code") == 0
            and len(review.get("cases", [])) == 1, "Missing clean saved residual physics/media case")
        case = review["cases"][0]
        require((case.get("event"), case.get("clip"), case.get("variant"), case.get("horizon_frames"))
            == (event, clip, "f0_student", 0) and case.get("residual_arm") == "causal_history"
            and case.get("human_real_future_frames") == 0 and case.get("seed") == 0 and case.get("interface") == "human"
            and case.get("policy_device") == "cuda" and case.get("measured_end_to_end_latency_ns") is None
            and case.get("student_checkpoint") == residual_record,
            "Changed residual task, input or timing scope")
        trace, timing = Path(case["trace"]), directory/"stream_timing.jsonl"
        require(trace == directory/"review"/(clip+"_s0")/"rollout.npz"
            and review["stream_telemetry"] == identity(timing), "Saved residual trace or timing path changed")
        times = [json.loads(line) for line in timing.read_text().splitlines()]
        require(len(times) == case["attempted_steps"], "Missing residual software calls")
        spec = EventSpec(event, 50, 10, 0)
        timeline = EventTimeline(case["original_paired_reference"]["frames"], spec)
        for frame,t in enumerate(times):
            validate_causal_timing(t, frame, timeline)
        with np.load(trace, allow_pickle=False) as arrays:
            actual, reference, actions = (arrays[k] for k in ("actual", "reference", "canonical_action"))
            require(actual.shape == (len(actions)+1, 79) and reference.shape == (case["reference_steps"]+1, 79)
                and len(actions) == len(times) and all(np.isfinite(arrays[k]).all() for k in arrays.files), "Invalid saved residual arrays")
            software = history_software(times, actions, spec)
            clocked = [dict(t, simulated_control_ns=t["frame"]*DT_NS) for t in times]
            window = None if event == "nominal" else capture_window(clocked, actions, actual, reference, case["upper_joint_indices"], spec)
            if case["complete"]:
                upper_phase += [dict(event=event, clip=clip, variant="proprio_f0", joint=name,
                    **analyze_signal(reference[:,7+j], actual[:,7+j])) for j,name in zip(case["upper_joint_indices"], case["upper_joint_names"])]
        video = case["video"]
        video_contract(video, case)
        require(identity(video["path"]) == {k:video[k] for k in ("path", "bytes", "sha256")}, "Residual video changed")
        rows.append(dict(event=event, clip=clip, variant="f0_student", horizon_frames=0, human_real_future_frames=0,
            requested_preview_ms=0, **metrics(case), failure_frame=case["failure_frame"], attempted_steps=case["attempted_steps"],
            completion_fraction=case["valid_steps"]/case["reference_steps"], derived_reference_sha256=case["derived_reference_sha256"],
            software=software, capture_window=window, video=video, valid_prefix_metrics=case.get("valid_prefix_metrics"),
            attempted_prefix_diagnostics=None if case["complete"] else case["diagnostics"]))
        records += [identity(f) for f in (p, trace, timing, Path(video["path"]),
            trace.with_name("foot_posture.csv"), trace.with_name("upper_timing.csv"))]
    combined = join_comparisons(rows, controls, previous)
    unique = {}
    for record in records:
        require(record["path"] not in unique or unique[record["path"]] == record, "Conflicting saved source identities")
        unique[record["path"]] = record
    require(all(identity(r["path"]) == r for r in unique.values()), "Residual analysis sources changed")
    groups = [dict(variant=v, **full_group([r for r in combined if r["event"] == "nominal" and r["variant"] == v]))
        for v in ("teacher", "student", "original_f0", "proprio_f0")]
    value = dict(stage="F0_PROPRIO_MATCHED_SAVED_ANALYSIS", evaluation_commit=evaluation_commit, new_saved_cases=7,
        reused_saved_controls=21, total_comparison_cases=28, complete_new_cases=sum(r["complete"] for r in rows),
        selected_update=3000, rows=combined, nominal_groups=groups, upper_phase=upper_phase, inputs=list(unique.values()),
        new_inference_calls=0, new_physical_transitions=0, new_training_updates=0, human_real_future_frames=0,
        measured_hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False, user_accepted_checkpoint=None,
        zero_history_reused_original_f0=True, heldout_events_opened=False,
        teacher_h10_nominal_context=old["teacher_h10_nominal_context"], failure_note="Failed clips have prefix diagnostics only",
        scope="Fixed clips and synthetic events, original CPU action path; upstream preprocessing remains offline unverified")
    output.mkdir(parents=True, exist_ok=False)
    save(output/"summary.json", value)
    flat = [{k:v for k,v in r.items() if not isinstance(v, (dict,list))} for r in combined]
    with (output/"per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    require(all(identity(r["path"]) == r for r in unique.values()), "Sources changed during saved analysis")
    save(output/"audit.json", dict(status="PASSED", source_inputs_unchanged=True, source_files=len(unique),
        seconds=time.monotonic()-started, new_inference=0, new_physics=0, new_training=0))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "previous-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    args = parser.parse_args()
    result = analyze(args.source, args.previous_root, args.output, args.evaluation_commit)
    print(json.dumps(dict(total=result["total_comparison_cases"], complete_new=result["complete_new_cases"])))
