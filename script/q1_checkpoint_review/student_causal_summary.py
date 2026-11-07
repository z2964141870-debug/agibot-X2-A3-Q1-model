"""Compare seven saved f0 replays with fourteen preserved R1 H4 controls."""

import argparse
from collections import Counter
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
from common import FIXED, identity, require, save
from delivery import read
from event_summary import call_statistics, capture_window
from stream_contract import DT_NS
from stream_events import EventSpec, EventTimeline
from student_causal_review import baseline_cases, review_cases, validate_causal_timing
from student_summary import differences, full_group
from tracking_response import analyze_signal


def completed_gate(result, auth):
    require(result.get("stage") == auth.get("stage") == "R2_F0_MATCHED_CLOSED_LOOP"
        and result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
        and result.get("new_physical_cases") == auth.get("new_physical_cases") == 7
        and result.get("reused_baseline_cases") == auth.get("reused_baseline_cases") == 14
        and result.get("total_comparison_cases") == 21
        and result.get("new_optimizer_updates") == auth.get("new_optimizer_updates") == 0
        and auth.get("human_real_future_frames") == 0
        and [tuple(case) for case in auth.get("cases", [])] == review_cases()
        and len(result.get("cases", [])) == len({r["path"] for r in result.get("cases", [])}) == 7,
        "Wait for all seven source-gated f0 cases and fourteen preserved controls")
    expected = {f"{e}_f0_student_h0_{c}_{p}" for e, c in review_cases() for p in ("replay", "media")}
    require(len(result.get("phases", [])) == 14
        and {p["name"] for p in result["phases"]} == expected,
        "Missing, duplicated or changed f0 phase list")
    require(not Path("/proc", str(result["controller"]["pid"])).exists(), "Wait for f0 controller cleanup")


def history_software(timings, actions, spec):
    require(actions.shape == (len(timings), 29) and len(timings) > 0 and np.isfinite(actions).all(),
        "Invalid saved f0 action trace")
    report = {}
    for field in ("gpu_policy_wall_compute_ns", "worker_action_wall_ns", "local_pipe_rpc_ns"):
        require(all(type(t.get(field)) is int and t[field] >= 0 for t in timings), "Missing f0 software timing")
        values = [t[field]/1e6 for t in timings]
        report[field.removesuffix("_ns")+"_ms"] = dict(all=call_statistics(values),
            without_first=call_statistics(values[1:]), without_first_two=call_statistics(values[2:]),
            first_two_ms=values[:2], input_ready_calls=call_statistics(
                [t[field]/1e6 for t in timings if t["input_ready"]]))
    jumps = np.max(np.abs(np.diff(actions, axis=0)), axis=1)
    rms = np.sqrt(np.mean(np.diff(actions, axis=0)**2, axis=1))
    intervals, start = [], None
    for frame in range(len(timings)+1):
        holding = frame < len(timings) and timings[frame]["fallback"] is not None
        if holding:
            expected = np.zeros(29, dtype=actions.dtype) if frame == 0 else actions[frame-1]
            require(np.array_equal(actions[frame], expected), "f0 fallback differs from saved previous-action hold")
            if start is None:
                start = frame
        if not holding and start is not None:
            recovery = frame if frame < len(timings) else None
            packet_end = ((spec.first_frame+spec.duration_frames+spec.transport_frames)*DT_NS
                if spec.kind in ("dropout", "late_burst") else None)
            intervals.append(dict(first_origin_frame=start, end_origin_frame_exclusive=frame,
                observed_fallback_ms=(frame-start)*20, recovered_origin_frame=recovery,
                recovered_synthetic_control_ms=None if recovery is None else recovery*20,
                recovery_after_packet_window_ms=None if recovery is None or packet_end is None
                    else (recovery*DT_NS-packet_end)/1e6,
                recovery_action_jump_max=None if recovery is None or recovery == 0 else float(jumps[recovery-1]),
                measured_hardware_recovery=False))
            start = None

    def ages(predicate):
        values = [t["current_capture_age_ns"]/1e6 for t in timings
                  if predicate(t) and t["current_capture_age_ns"] is not None]
        return dict(samples=len(values), minimum_ms=None if not values else min(values),
            p95_ms=None if not values else float(np.quantile(values, .95)),
            maximum_ms=None if not values else max(values))

    require(all(len(t["used_sequences"]) == 10 for t in timings), "Missing f0 history slots")
    report.update(total_calls=len(timings), ready_calls=sum(t["input_ready"] for t in timings),
        fallback_frames=sum(t["fallback"] is not None for t in timings), fallback_intervals=intervals,
        history_valid_slots_histogram=dict(sorted(Counter(
            sum(s is not None for s in t["used_sequences"]) for t in timings).items())),
        current_capture_age=dict(all=ages(lambda t: True), input_ready=ages(lambda t: t["input_ready"]),
            fallback=ages(lambda t: not t["input_ready"])),
        capture_age_note="Retained historical support may exceed200ms during current-slot fallback",
        action_delta_rms_p95=None if not len(rms) else float(np.quantile(rms, .95)),
        action_delta_max_joint_max=None if not len(jumps) else float(jumps.max()),
        timing_note="Nested scopes are not additive; local pipe is not human network or actuator latency")
    return report


def match_controls(rows, controls):
    expected = set(review_cases())
    require(len(rows) == 7 and {(r["event"], r["clip"]) for r in rows} == expected,
        "Keep all seven unique f0 tasks, including failures")
    require(len(controls) == 14 and {(r["event"], r["clip"], r["variant"]) for r in controls}
        == {(e, c, v) for e, c in expected for v in ("teacher", "student")}
        and all(r.get("horizon_frames") == 4 for r in controls), "Keep fourteen unique H4 controls")
    lookup = {(r["event"], r["clip"], r["variant"]):r for r in controls}
    for row in rows:
        require(row.get("variant") == "f0_student" and row.get("horizon_frames") == 0
            and row.get("human_real_future_frames") == 0, "Changed f0 comparison interface")
        for variant, key in (("teacher", "change_from_teacher_h4"), ("student", "change_from_r1_h4")):
            control = lookup[(row["event"], row["clip"], variant)]
            require(row["derived_reference_sha256"] == control["derived_reference_sha256"],
                "f0 and preserved H4 controls have different targets")
            row[key] = differences(row, control)
    return rows


def video_contract(video, row):
    require(video.get("status") == "PASSED" and video.get("original_speed") is True
        and video["probe"]["nb_frames"] == row["reference_steps"]+1
        and video.get("actual_frames_before_hold") == row["attempted_steps"]+1
        and video.get("failure_preserved") is True, "f0 video must retain its actual prefix and labelled failure hold")


def analyze(source, output, evaluation_commit):
    started = time.monotonic()
    require(source.is_absolute() and source.resolve() == source and output.is_absolute()
        and output.resolve() == output and not output.exists() and not output.is_relative_to(source),
        "Use actual saved f0 source and a new independent output")
    require(len(evaluation_commit) == 40 and all(c in "0123456789abcdef" for c in evaluation_commit),
        "Publish f0 saved analysis first")
    result, auth, audit = (read(source/n) for n in ("result.json", "authorization.json", "checkpoint_audit.json"))
    completed_gate(result, auth)
    records = [identity(source/n) for n in
        ("result.json", "authorization.json", "checkpoint_audit.json", "baseline_cases.json")]
    records += (auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
        +auth["preserved_research_inputs"])
    records += [identity(p) for p in (Path(__file__).resolve(), HERE/"student_summary.py",
        HERE/"analyze.py", HERE/"event_summary.py", HERE.parent/"supermimic_teleop/student_causal_review.py",
        HERE.parent/"supermimic_teleop/tracking_response.py")]
    require(all(identity(r["path"]) == r for r in records), "f0 review or preserved evidence changed")
    summaries = [Path(r["path"]) for r in auth["preserved_research_inputs"]
        if Path(r["path"]).name == "summary.json" and read(Path(r["path"])).get("stage")
        == "R1_MATCHED_SAVED_REVIEW_ANALYSIS"]
    require(len(set(summaries)) == 1, "Require one actual preserved R1 saved analysis")
    r1 = read(summaries[0])
    controls = baseline_cases(r1)
    require(read(source/"baseline_cases.json") == controls, "Saved H4 controls differ from actual R1 analysis")
    selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    for phase in result["phases"]:
        path = source/(phase["name"]+".guard.json")
        require(phase["guard"] == identity(path), "f0 guard identity changed")
        guard = read(path)
        require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == 0, "Saved f0 phase was not cleanly reaped")
        records.append(identity(path))
    rows, upper_phase = [], []
    for (event, clip), saved in zip(review_cases(), result["cases"]):
        directory = source/f"{event}_f0_student_h0_{clip}"
        path = directory/"review/result.json"
        require(saved == identity(path), "Ordered f0 case identity changed")
        review = read(path)
        require(review.get("status") == "PASSED" and review.get("checkpoint") == selected
            and len(review.get("cases", [])) == 1 and review.get("policy_worker_reaped") is True
            and review.get("policy_worker_return_code") == 0, "Missing saved f0 physical/media result")
        row = review["cases"][0]
        require((row.get("event"), row.get("clip"), row.get("variant"), row.get("horizon_frames"))
            == (event, clip, "f0_student", 0) and row.get("seed") == 0 and row.get("interface") == "human"
            and row.get("policy_device") == "cuda" and row.get("human_real_future_frames") == 0
            and row.get("measured_end_to_end_latency_ns") is None, "Changed saved f0 task or latency claims")
        trace, timing_path = Path(row["trace"]), directory/"stream_timing.jsonl"
        require(trace == directory/"review"/(clip+"_s0")/"rollout.npz", "f0 trace escaped saved directory")
        timings = [json.loads(line) for line in timing_path.read_text().splitlines()]
        require(len(timings) == row["attempted_steps"] and review["stream_telemetry"] == identity(timing_path),
            "f0 timing identity or length changed")
        spec = EventSpec(event, 50, 10, 0)
        timeline = EventTimeline(row["original_paired_reference"]["frames"], spec)
        for frame, timing in enumerate(timings):
            validate_causal_timing(timing, frame, timeline)
        with np.load(trace, allow_pickle=False) as arrays:
            actual, reference, actions = (arrays[k] for k in ("actual", "reference", "canonical_action"))
            require(actual.shape == (len(actions)+1, 79) and reference.shape == (row["reference_steps"]+1, 79)
                and len(actions) == row["attempted_steps"] and all(np.isfinite(arrays[k]).all() for k in arrays.files),
                "Saved f0 physical arrays changed")
            software = history_software(timings, actions, spec)
            clocked = [dict(t, simulated_control_ns=t["frame"]*DT_NS) for t in timings]
            window = None if event == "nominal" else capture_window(clocked, actions, actual, reference,
                row["upper_joint_indices"], spec)
            if row["complete"]:
                for joint, name in zip(row["upper_joint_indices"], row["upper_joint_names"]):
                    upper_phase.append(dict(event=event, clip=clip, variant="f0_student", horizon_frames=0,
                        joint=name, **analyze_signal(reference[:, 7+joint], actual[:, 7+joint])))
        video = row["video"]
        video_contract(video, row)
        require(identity(video["path"]) == {k:video[k] for k in ("path", "bytes", "sha256")},
            "f0 original-speed video identity changed")
        rows.append(dict(event=event, clip=clip, variant="f0_student", horizon_frames=0,
            human_real_future_frames=0, requested_preview_ms=0, **metrics(row),
            failure_frame=row["failure_frame"], attempted_steps=row["attempted_steps"],
            completion_fraction=row["valid_steps"]/row["reference_steps"],
            derived_reference_sha256=row["derived_reference_sha256"], software=software, capture_window=window,
            video=video, valid_prefix_metrics=row.get("valid_prefix_metrics"),
            attempted_prefix_diagnostics=None if row["complete"] else row["diagnostics"]))
        records.extend(identity(p) for p in (path, trace, timing_path, Path(video["path"]),
            trace.with_name("foot_posture.csv"), trace.with_name("upper_timing.csv")))
    match_controls(rows, controls)
    unique = {}
    for record in records:
        require(record["path"] not in unique or unique[record["path"]] == record, "Conflicting f0 source identities")
        unique[record["path"]] = record
    require(all(identity(r["path"]) == r for r in unique.values()), "f0 sources changed during saved analysis")
    combined = controls+rows
    groups = [dict(variant=v, horizon_frames=h, **full_group(
        [r for r in combined if r["event"] == "nominal" and r["variant"] == v]))
        for v, h in (("teacher", 4), ("student", 4), ("f0_student", 0))]
    value = dict(stage="R2_F0_MATCHED_SAVED_REVIEW_ANALYSIS", evaluation_commit=evaluation_commit,
        review_source_commit=auth["evaluation_commit"], new_saved_cases=7, reused_saved_controls=14,
        total_comparison_cases=21, complete_new_cases=sum(r["complete"] for r in rows), rows=combined,
        teacher_h10_nominal_context=[r for r in r1["rows"] if r["event"] == "nominal"
            and r["variant"] == "teacher" and r["horizon_frames"] == 10],
        context_included_in_21_case_count=False, nominal_groups=groups, upper_phase=upper_phase,
        inputs=list(unique.values()), new_inference_calls=0, new_physical_transitions=0, new_training_updates=0,
        human_real_future_frames=0, measured_hardware_end_to_end_latency_ms=None,
        natural_teleoperation_qualified=False, user_accepted_checkpoint=None,
        scope="Already-seen clips and synthetic receiver events; full-body precise imitation",
        timing_note="Zero requested future frames preserves packet age, preprocessing and nested software tails",
        source_preprocessing="offline_unverified", failure_note="No full-motion errors or group means for failed clips")
    output.mkdir(parents=True, exist_ok=False)
    save(output/"summary.json", value)
    omitted = {"software", "capture_window", "video", "valid_prefix_metrics", "attempted_prefix_diagnostics",
        "change_from_teacher_h4", "change_from_r1_h4", "change_from_matched_teacher", "change_from_nominal_teacher_h10"}
    flat = [{k:v for k, v in row.items() if k not in omitted} for row in combined]
    with (output/"per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    require(all(identity(r["path"]) == r for r in unique.values()), "f0 sources changed while writing analysis")
    save(output/"audit.json", dict(status="PASSED", source_inputs_unchanged=True, source_files=len(unique),
        seconds=time.monotonic()-started, new_inference=0, new_physics=0, new_training=0))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    args = parser.parse_args()
    result = analyze(args.source, args.output, args.evaluation_commit)
    print(json.dumps(dict(total=result["total_comparison_cases"], complete_new=result["complete_new_cases"], output=str(args.output))))
