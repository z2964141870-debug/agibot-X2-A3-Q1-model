"""Read matched R1 review artifacts without loading a policy or advancing physics."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
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
from student_review import review_cases, validate_timing
from student_review_resume import phases_for
from tracking_response import analyze_signal


KEYS = ("joint_rmse_rad", "upper_rmse_rad", "slip_left_m_s", "slip_right_m_s",
        "sole_left_m", "sole_right_m", "alternations", "tilt_rms_rad")


def full_group(rows):
    require(len(rows) == 3 and {r["clip"] for r in rows} == set(FIXED), "Keep all three fixed clips")
    count = sum(r["complete"] for r in rows)
    return dict(total_cases=3, complete_cases=count,
        full_fixed_set_means=({key: statistics.mean(r[key] for r in rows)
            if all(r.get(key) is not None for r in rows) else None for key in KEYS} if count == 3 else None))


def software_summary(timings, actions, spec):
    require(actions.shape == (len(timings), 29) and np.isfinite(actions).all(), "Invalid saved canonical actions")
    fields = ("gpu_policy_wall_compute_ns", "worker_action_wall_ns", "local_pipe_rpc_ns")
    report = {}
    for field in fields:
        require(all(type(t.get(field)) is int and t[field] >= 0 for t in timings), "Missing bounded software timing")
        values = [t[field]/1e6 for t in timings]
        report[field.removesuffix("_ns")+"_ms"] = dict(all=call_statistics(values),
            without_first=call_statistics(values[1:]), without_first_two=call_statistics(values[2:]),
            first_two_ms=values[:2], input_ready_calls=call_statistics(
                [t[field]/1e6 for t in timings if t["input_ready"]]))
    delta = np.diff(actions, axis=0)
    jumps = np.max(np.abs(delta), axis=1) if len(delta) else np.array([])
    rms = np.sqrt(np.mean(delta**2, axis=1)) if len(delta) else np.array([])
    intervals, start = [], None
    for frame in range(len(timings)+1):
        holding = frame < len(timings) and timings[frame]["fallback"] is not None
        if holding:
            expected = np.zeros(29, dtype=actions.dtype) if frame == 0 else actions[frame-1]
            require(np.array_equal(actions[frame], expected), "Saved missing-input action did not use declared hold")
            if start is None:
                start = frame
        if not holding and start is not None:
            recovery = frame if frame < len(timings) else None
            control = None if recovery is None else (recovery+timings[recovery]["horizon_frames"])*DT_NS
            packet_end = ((spec.first_frame+spec.duration_frames+spec.transport_frames)*DT_NS
                if spec.kind in ("dropout", "late_burst") else None)
            intervals.append(dict(first_origin_frame=start, end_origin_frame_exclusive=frame,
                observed_fallback_ms=(frame-start)*20, recovered_origin_frame=recovery,
                recovered_synthetic_control_ms=None if control is None else control/1e6,
                recovery_after_packet_window_ms=None if control is None or packet_end is None else (control-packet_end)/1e6,
                recovery_action_jump_max=None if recovery is None or recovery == 0 else float(jumps[recovery-1]),
                measured_hardware_recovery=False))
            start = None
    report.update(fallback_frames=sum(t["fallback"] is not None for t in timings), fallback_intervals=intervals,
        actual_prefix_histogram=dict(sorted(Counter(t["observed_prefix_frames"] for t in timings).items())),
        ready_calls=sum(t["input_ready"] for t in timings), total_calls=len(timings),
        action_delta_rms_p95=None if not len(rms) else float(np.quantile(rms, .95)),
        action_delta_max_joint_max=None if not len(jumps) else float(jumps.max()),
        timing_note="Nested scopes, not additive; RPC is a local pipe, not the human network or robot actuator")
    return report


def differences(value, control):
    if not value["complete"] or not control["complete"]:
        return None
    absolute = {k:value[k]-control[k] for k in KEYS if value.get(k) is not None and control.get(k) is not None}
    return dict(absolute=absolute, relative={k:v/control[k] for k, v in absolute.items() if control[k] != 0})


def analyze(source, output, evaluation_commit):
    started = time.monotonic()
    require(source.is_absolute() and source.resolve() == source and output.is_absolute()
        and output.resolve() == output and not output.exists() and not output.is_relative_to(source),
        "Use actual saved source and a new independent analysis root")
    result, auth, audit = (read(source/n) for n in ("result.json", "authorization.json", "checkpoint_audit.json"))
    require(result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
        and result.get("physical_cases") == 23 and result.get("new_physical_cases") == 22
        and result.get("new_optimizer_updates") == 0
        and result.get("stage") == auth.get("stage") == "R1_MATCHED_CLOSED_LOOP_CONTINUATION"
        and result.get("reused_physical_cases") == auth.get("reused_physical_cases") == 1
        and [tuple(r) for r in auth.get("cases", [])] == review_cases()
        and len(result.get("cases", [])) == 23 and len(set(r["path"] for r in result["cases"])) == 23
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "Wait for completed, reaped matched review and its source gate")
    require(len(evaluation_commit) == 40 and all(c in "0123456789abcdef" for c in evaluation_commit), "Publish saved analysis first")
    selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    records = [identity(source/n) for n in ("result.json", "authorization.json", "checkpoint_audit.json")]
    records += (auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
        +auth["trained_student_inputs"]+auth["preserved_failed_review_inputs"])
    records += [identity(p) for p in (Path(__file__).resolve(), HERE/"analyze.py", HERE/"event_summary.py",
        HERE/"preview_summary.py", HERE.parent/"supermimic_teleop/tracking_response.py")]
    require(all(identity(r["path"]) == r for r in records), "Published review or preserved sources changed")
    expected_phases = {f"{e}_{v}_h{h}_{c}_{p}" for i, (e, v, h, c) in enumerate(review_cases())
        for p in phases_for((e, v, h, c)) if not (i == 0 and p == "replay")}
    require({r["name"] for r in result["phases"]} == expected_phases and len(result["phases"]) == 29,
        "Completed phase list is incomplete or duplicated")
    for phase in result["phases"]:
        path = source/(phase["name"]+".guard.json")
        require(phase["guard"] == identity(path), "Saved phase guard changed")
        guard = read(path)
        require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == 0, "A saved evaluation phase did not finish cleanly")
        records.append(identity(path))
    rows, upper_phase = [], []
    for case, saved_record in zip(review_cases(), result["cases"]):
        event, variant, horizon, clip = case
        directory = source/f"{event}_{variant}_h{horizon}_{clip}"
        path = directory/"review/result.json"
        require(saved_record == identity(path), "Ordered case record changed")
        review = read(path)
        needs_video = "media" in phases_for(case)
        require(review.get("status") == ("PASSED" if needs_video else "PHYSICS_PASSED")
            and review.get("checkpoint") == selected and len(review.get("cases", [])) == 1
            and review.get("policy_worker_reaped") is True and review.get("policy_worker_return_code") == 0,
            "Missing clean saved physical/media case")
        row = review["cases"][0]
        require((row.get("event"), row.get("variant"), row.get("horizon_frames"), row.get("clip")) == case
            and row.get("seed") == 0 and row.get("interface") == "human" and row.get("policy_device") == "cuda"
            and row.get("measured_end_to_end_latency_ns") is None,
            "Saved case changed its policy, stream or latency claims")
        trace, timing_path = Path(row["trace"]), directory/"stream_timing.jsonl"
        require(trace == directory/"review"/(clip+"_s0")/"rollout.npz", "Trace escaped its saved case")
        timings = [json.loads(line) for line in timing_path.read_text().splitlines()]
        require(len(timings) == row["attempted_steps"] and review["stream_telemetry"] == identity(timing_path),
            "Saved timing is truncated or changed")
        spec = EventSpec(event, 50, 10, 0)
        timeline = EventTimeline(row["original_paired_reference"]["frames"], spec)
        for frame, timing in enumerate(timings):
            validate_timing(timing, frame, horizon, variant, timeline)
        with np.load(trace, allow_pickle=False) as arrays:
            actual, reference, actions = (arrays[k] for k in ("actual", "reference", "canonical_action"))
            require(actual.shape == (len(actions)+1, 79) and reference.shape == (row["reference_steps"]+1, 79)
                and len(actions) == row["attempted_steps"] and all(np.isfinite(arrays[k]).all() for k in arrays.files),
                "Saved physical arrays changed")
            software = software_summary(timings, actions, spec)
            clocked = [dict(t, simulated_control_ns=(t["frame"]+horizon)*DT_NS) for t in timings]
            window = None if event == "nominal" else capture_window(clocked, actions, actual, reference,
                row["upper_joint_indices"], spec)
            if row["complete"]:
                for joint, name in zip(row["upper_joint_indices"], row["upper_joint_names"]):
                    upper_phase.append(dict(event=event, variant=variant, horizon_frames=horizon, clip=clip,
                        joint=name, **analyze_signal(reference[:, 7+joint], actual[:, 7+joint])))
        video = row.get("video")
        if needs_video:
            require(video.get("status") == "PASSED" and video.get("original_speed") is True
                and video["probe"]["nb_frames"] == row["reference_steps"]+1
                and identity(video["path"]) == {k:video[k] for k in ("path", "bytes", "sha256")}, "Video identity changed")
            records.append(identity(video["path"]))
        value = dict(event=event, variant=variant, horizon_frames=horizon, clip=clip, **metrics(row),
            failure_frame=row["failure_frame"], attempted_steps=row["attempted_steps"],
            completion_fraction=row["valid_steps"]/row["reference_steps"], requested_preview_ms=horizon*20,
            derived_reference_sha256=row["derived_reference_sha256"], software=software, capture_window=window,
            video=video, valid_prefix_metrics=row.get("valid_prefix_metrics"),
            attempted_prefix_diagnostics=None if row["complete"] else row["diagnostics"])
        rows.append(value)
        records.extend(identity(p) for p in (path, trace, timing_path,
            trace.with_name("foot_posture.csv"), trace.with_name("upper_timing.csv")))
    lookup = {(r["event"], r["variant"], r["horizon_frames"], r["clip"]):r for r in rows}
    for row in rows:
        control = lookup[(row["event"], "teacher", row["horizon_frames"], row["clip"])]
        require(row["derived_reference_sha256"] == control["derived_reference_sha256"], "Matched targets differ")
        row["change_from_matched_teacher"] = differences(row, control) if row["variant"] == "student" else None
        row["change_from_nominal_teacher_h10"] = (differences(row, lookup[("nominal", "teacher", 10, row["clip"])])
            if row["event"] == "nominal" else None)
    groups = [dict(variant=v, horizon_frames=h, **full_group(
        [lookup[("nominal", v, h, c)] for c in FIXED])) for v, h in
        (("teacher", 10), ("teacher", 4), ("student", 4), ("teacher", 1), ("student", 1))]
    unique = {}
    for record in records:
        require(record["path"] not in unique or unique[record["path"]] == record, "Conflicting source identities")
        unique[record["path"]] = record
    require(all(identity(r["path"]) == r for r in unique.values()), "Source changed during saved analysis")
    value = dict(stage="R1_MATCHED_SAVED_REVIEW_ANALYSIS", evaluation_commit=evaluation_commit,
        review_source_commit=auth["evaluation_commit"], total_cases=23, complete_cases=sum(r["complete"] for r in rows),
        rows=rows, nominal_groups=groups, upper_phase=upper_phase, inputs=list(unique.values()),
        new_inference_calls=0, new_physical_transitions=0, new_training_updates=0,
        measured_hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        user_accepted_checkpoint=None, source_preprocessing="offline_unverified",
        scope="Matched synthetic receiver and already-seen clips; new H10 is support-aware and is not the old full-window H10",
        timing_note="Requested preview, nested software scopes and descriptive phase are separate; no measured human network or actuator delay",
        capture_window_note="Fixed capture-clock1000ms window; error uses the corresponding saved next origin-state reference, not a hardware response timestamp",
        failure_note="Incomplete clips have only saved-prefix diagnostics; full errors and full three-clip means remain absent")
    output.mkdir(parents=True, exist_ok=False)
    save(output/"summary.json", value)
    flat = [{k:v for k, v in row.items() if k not in ("software", "capture_window", "video", "valid_prefix_metrics",
        "attempted_prefix_diagnostics", "change_from_matched_teacher", "change_from_nominal_teacher_h10")} for row in rows]
    with (output/"per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    require(all(identity(r["path"]) == r for r in unique.values()), "Source changed while writing analysis")
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
    print(json.dumps(dict(total=result["total_cases"], complete=result["complete_cases"], output=str(args.output))))
