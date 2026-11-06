"""Read preserved event cases after a clean wall-time stop; never certify84."""

import argparse
import csv
import datetime
import json
from pathlib import Path
import time

import numpy as np

from event_summary import (CLIPS, CONFIGS, EVENTS, EventSpec, EventTimeline,
    capture_window, folder, full_group, identity, metrics, read, require,
    telemetry_summary, validate_event_timing)

WALL_ERROR = "RuntimeError: Paused campaign whole/phase wall-time bound reached"


def stopped_gate(result, guard, alive):
    require(result.get("status") == guard.get("status") == "FAILED"
            and result.get("error") == guard.get("error") == WALL_ERROR,
            "Only the preserved wall-time failure is eligible for partial reading")
    cleanup = guard.get("cleanup", {})
    require(cleanup.get("worker_reaped") is True and cleanup.get("worker_return_code") == -15,
            "Interrupted worker cleanup is not verified")
    require(not alive(result["controller"]["pid"]) and not alive(guard["worker_pid"]),
            "Campaign or worker is still present")
    duration = (datetime.datetime.fromisoformat(result["ended_utc"])
                - datetime.datetime.fromisoformat(result["started_utc"])).total_seconds()
    require(duration >= 7200, "The whole-campaign time bound is not evidenced")


def groups(rows):
    values = []
    for event in EVENTS:
        for method, horizon in CONFIGS:
            selected = [r for r in rows if (r["event"], r["method"], r["horizon_frames"])
                        == (event, method, horizon)]
            require(len({r["clip"] for r in selected}) == len(selected)
                    and all(r["clip"] in CLIPS for r in selected), "Duplicate or foreign clip")
            value = (full_group(selected) if len(selected) == 3 else
                     dict(total_cases=3, complete_cases=sum(r["complete"] for r in selected),
                          full_fixed_set_means=None))
            values.append(dict(event=event, method=method, horizon_frames=horizon,
                               available_cases=len(selected), **value))
    return values


def analyze_stopped(source, output):
    started = time.monotonic()
    require(source.is_absolute() and source.resolve() == source, "Require an actual absolute source")
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and not output.is_relative_to(source), "Use a fresh independent partial-analysis root")
    result = read(source / "result.json")
    guard_path = source / (result["phase"] + ".guard.json")
    stopped_gate(result, read(guard_path), lambda pid: Path("/proc", str(pid)).exists())
    auth, equality, audited = (read(source / name) for name in
        ("authorization.json", "equivalence.json", "checkpoint_audit.json"))
    expected = {(e, m, h, c) for e in EVENTS for m, h in CONFIGS for c in CLIPS}
    require(auth.get("stage") == "PAUSED32K_SYNTHETIC_EVENTS"
            and len(auth["cases"]) == 84 and {tuple(r) for r in auth["cases"]} == expected,
            "Changed event campaign or fixed case set")
    require(audited.get("status") == equality.get("status") == "PASSED"
            and audited.get("iteration") == 32000 and len(equality.get("cases", [])) == 21
            and {(r["method"], r["horizon"], r["clip"]) for r in equality["cases"]}
                == {(m, h, c) for m, h in CONFIGS for c in CLIPS}
            and equality.get("atol") == equality.get("rtol") == 2e-5,
            "Missing32k checkpoint or complete action gate")
    selected = next(r["checkpoint"] for r in audited["records"] if r["robot"] == "q1")
    require(equality.get("checkpoint") == selected, "Action gate used another policy")
    inputs = [identity(source / name) for name in
        ("result.json", "authorization.json", "equivalence.json", "checkpoint_audit.json")]
    inputs.append(identity(guard_path))
    inputs.extend(identity(Path(__file__).with_name(name)) for name in
        ("event_partial_summary.py", "event_summary.py", "analyze.py", "preview_summary.py"))
    source_counts = {}
    for name in ("evaluation_sources", "training_source_inputs", "controller_inputs", "reused_nominal_evidence"):
        records = auth[name]
        require(records and all(identity(Path(r["path"])) == r for r in records),
                "Published or frozen source changed: " + name)
        source_counts[name] = len(records)
        inputs.extend(records)
    rows, missing = [], []
    for event, method, horizon, clip in auth["cases"]:
        require(time.monotonic()-started < 300, "Partial analysis exceeded its300s bound")
        directory = folder(source, event, method, horizon, clip)
        saved, telemetry = directory / "review/result.json", directory / "stream_timing.jsonl"
        if not saved.exists():
            missing.append(dict(event=event, method=method, horizon_frames=horizon, clip=clip,
                                reason="NO_COMPLETED_RECORD"))
            continue
        case = read(saved)
        if case.get("status") != "PASSED":
            missing.append(dict(event=event, method=method, horizon_frames=horizon, clip=clip,
                                reason="RECORD_NOT_MEDIA_COMPLETE", saved_status=case.get("status")))
            inputs.append(identity(saved))
            continue
        require(case.get("iteration") == 32000 and case.get("checkpoint") == selected
                and len(case.get("cases", [])) == 1, "Saved policy identity differs")
        row = case["cases"][0]
        require((row.get("event"), row.get("tail_method"), row.get("horizon_frames"), row.get("clip"))
                == (event, method, horizon, clip) and row.get("seed") == 0
                and row.get("policy_device") == "cuda" and row.get("reference_is_synthetic") is True
                and row.get("measured_end_to_end_latency_ns") is None
                and row.get("video", {}).get("status") == "PASSED"
                and row["video"].get("original_speed") is True, "Saved case identity or media differs")
        trace = directory / "review" / (clip+"_s0") / "rollout.npz"
        require(Path(row["trace"]) == trace, "Trace escaped the saved case")
        video = trace.with_name("tracking.mp4")
        require(Path(row["video"]["path"]) == video and identity(video)
                == {k:row["video"][k] for k in ("path", "bytes", "sha256")},
                "Original-speed video changed after its full decode")
        before = [identity(saved), identity(trace), identity(telemetry), identity(video)]
        timings = [json.loads(line) for line in telemetry.read_text().splitlines()]
        spec = EventSpec(**row["event_mapping"]["event"])
        require(spec == EventSpec(event, first_frame=50, duration_frames=10, transport_frames=0)
                and len(timings) == row["attempted_steps"], "Changed event or transition count")
        timeline = EventTimeline(row["reference_steps"]+1, spec)
        for frame, value in enumerate(timings):
            validate_event_timing(value, frame, horizon, method, timeline)
        with np.load(trace, allow_pickle=False) as arrays:
            actual, reference, actions = (arrays[k] for k in ("actual", "reference", "canonical_action"))
            require(actual.shape == (len(actions)+1, 79)
                    and reference.shape == (row["reference_steps"]+1, 79)
                    and all(np.isfinite(a).all() for a in (actual, reference, actions)), "Invalid trace")
            timing = telemetry_summary(timings, actions, spec, horizon)
            window = capture_window(timings, actions, actual, reference, row["upper_joint_indices"], spec)
        rows.append(dict(event=event, method=method, horizon_frames=horizon, preview_budget_ms=horizon*20,
            clip=clip, **metrics(row), failure_frame=row["failure_frame"],
            completion_fraction=row["valid_steps"]/row["reference_steps"], video=row["video"],
            derived_reference_sha256=row["derived_reference_sha256"], telemetry=timing, capture_window=window))
        inputs.extend(before)
    require(len(rows)+len(missing) == 84 and 0 < len(missing) < 84,
            "This entry only reports an explicitly incomplete campaign")
    lookup = {(r["event"], r["method"], r["horizon_frames"], r["clip"]): r for r in rows}
    for row in rows:
        control = lookup.get((row["event"], "masked", 10, row["clip"]))
        require(control is not None and row["derived_reference_sha256"] == control["derived_reference_sha256"],
                "Missing same-event H10 or changed comparison target")
        row["change_from_same_event_h10"] = ({k:row[k]-control[k] for k in
            ("upper_rmse_rad", "slip_left_m_s", "slip_right_m_s", "tilt_rms_rad")
            if row.get(k) is not None and control.get(k) is not None}
            if row["complete"] and control["complete"] else None)
    value = dict(stage="PARTIAL_SAVED_EVENT_ANALYSIS", status="PARTIAL", source_campaign_status="FAILED",
        source_final_gate_passed=False, partial_source_reaudit="PASSED", source_counts=source_counts,
        evaluation_commit=auth["evaluation_commit"], total_expected_cases=84, total_cases=len(rows),
        complete_cases=sum(r["complete"] for r in rows), missing=missing, rows=rows, groups=groups(rows),
        inputs=inputs, new_physical_transitions=0, new_inference_calls=0, new_training_updates=0,
        measured_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        user_accepted_checkpoint=None, full84_comparison_performed=False,
        note="Stopped wall-time campaign; incomplete groups have no full-set mean. Synthetic clocks are not hardware latency.")
    require(all(identity(Path(r["path"])) == r for r in inputs), "Saved inputs changed during reading")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    flat = [{k:v for k,v in row.items() if k not in
             ("video", "telemetry", "capture_window", "change_from_same_event_h10")} for row in rows]
    with (output / "per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    require(all(identity(Path(r["path"])) == r for r in inputs), "Saved inputs changed while writing")
    (output / "audit.json").write_text(json.dumps(dict(status="PARTIAL_ANALYSIS_PASSED",
        seconds=time.monotonic()-started, source_inputs_unchanged=True, new_physics=0,
        new_inference=0, new_training=0, original_campaign_status="FAILED"), indent=2)+"\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_stopped(args.source, args.output)
    print(json.dumps(dict(status=result["status"], available=result["total_cases"],
                          missing=len(result["missing"]), output=str(args.output))))
