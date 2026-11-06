"""Read completed synthetic-event artifacts; never load a policy or run physics."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_teleop"))
from analyze import metrics
from event_inputs import validate_event_timing
from preview_summary import identity, plot
from stream_contract import DT_NS
from stream_events import EventSpec, EventTimeline

EVENTS = ("stop", "reverse", "dropout", "late_burst")
CONFIGS = (("masked", 10), ("masked", 5), ("masked", 4), ("hold", 5),
           ("hold", 4), ("damped_cv", 5), ("damped_cv", 4))
CLIPS = ("walk_forward_loop_001__A022", "walk_forward_start_002__A023", "walk_forward_stop_002__A024")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def folder(source, event, method, horizon, clip):
    root = source / event
    if method != "masked":
        root = root / ("tail_" + method)
    return root / f"h{horizon}" / clip


def call_statistics(values):
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    return dict(samples=len(values), p50_ms=float(np.quantile(values, .5)),
                p95_ms=float(np.quantile(values, .95)), maximum_ms=float(values.max()),
                calls_over_20ms=int(np.count_nonzero(values > 20.)))


def telemetry_summary(timings, actions, spec, horizon):
    actions = np.asarray(actions)
    require(actions.shape == (len(timings), 29) and np.isfinite(actions).all(),
            "Require one finite canonical action29 per timing row")
    for frame, value in enumerate(timings):
        require(value.get("frame") == frame and value.get("simulated_control_ns") == (frame+horizon)*DT_NS
                and type(value.get("gpu_policy_wall_compute_ns")) is int
                and value["gpu_policy_wall_compute_ns"] >= 0,
                "Invalid saved software or synthetic control timestamp")
    compute = [value["gpu_policy_wall_compute_ns"]/1e6 for value in timings]
    jumps = np.max(np.abs(np.diff(actions, axis=0)), axis=1) if len(actions) > 1 else np.array([])
    rms = np.sqrt(np.mean(np.diff(actions, axis=0)**2, axis=1)) if len(actions) > 1 else np.array([])
    intervals, start = [], None
    for frame in range(len(timings)+1):
        holding = frame < len(timings) and timings[frame].get("fallback") is not None
        if holding and start is None:
            start = frame
        if not holding and start is not None:
            recovered = frame if frame < len(timings) else None
            control = timings[recovered]["simulated_control_ns"] if recovered is not None else None
            transport_end = ((spec.first_frame+spec.duration_frames+spec.transport_frames)*DT_NS
                             if spec.kind in ("dropout", "late_burst") else None)
            intervals.append(dict(first_origin_frame=start, end_origin_frame_exclusive=frame,
                observed_fallback_ms=(frame-start)*DT_NS/1e6, recovered_origin_frame=recovered,
                recovered_control_ms=None if control is None else control/1e6,
                recovery_after_packet_window_ms=(None if control is None or transport_end is None
                                                  else (control-transport_end)/1e6),
                recovery_max_action_jump=(float(jumps[recovered-1])
                                          if recovered is not None and recovered > 0 else None)))
            start = None
    return dict(gpu_goal_call_all=call_statistics(compute),
        gpu_goal_call_without_first=call_statistics(compute[1:]),
        first_goal_call_ms=compute[0] if compute else None,
        action_delta_rms_p95=float(np.quantile(rms, .95)) if len(rms) else None,
        action_delta_max_joint_max=float(jumps.max()) if len(jumps) else None,
        fallback_frames=sum(t.get("fallback") is not None for t in timings),
        fallback_intervals=intervals,
        maximum_used_pose_age_ms=max((t["used_pose_age_ns"]/1e6 for t in timings), default=None))


def capture_window(timings, actions, actual, reference, upper_indices, spec):
    # Compare a fixed capture-clock second; robot origin clocks differ by preview.
    start = spec.first_frame*DT_NS
    selected = [i for i, value in enumerate(timings)
                if start <= value["simulated_control_ns"] < start+1_000_000_000]
    complete = len(selected) == 50
    result = dict(capture_window_start_ns=start, capture_window_duration_ns=1_000_000_000,
        expected_samples=50, observed_samples=len(selected), complete=complete,
        upper_rmse_rad=None, action_delta_rms_p95=None, action_delta_max_joint_max=None,
        endpoint="one saved simulator state after the action at each synthetic control timestamp",
        measured_hardware_delay=False)
    if not complete:
        return result
    require(len(actual) == len(actions)+1 and len(reference) >= len(actual), "Trace and action lengths differ")
    upper = np.asarray(upper_indices)
    require(upper.ndim == 1 and len(upper) > 0 and np.issubdtype(upper.dtype, np.integer)
            and np.all((0 <= upper) & (upper < 22)), "Invalid Q1 upper joint indices")
    states = np.asarray(selected)+1
    error = actual[states, 7:29][:, upper]-reference[states, 7:29][:, upper]
    transitions = np.asarray([i for i in selected if i > 0])
    delta = actions[transitions]-actions[transitions-1]
    result.update(upper_rmse_rad=float(np.sqrt(np.mean(error**2))),
        action_delta_rms_p95=(float(np.quantile(np.sqrt(np.mean(delta**2, axis=1)), .95))
                              if len(delta) else None),
        action_delta_max_joint_max=float(np.abs(delta).max()) if len(delta) else None)
    return result


def full_group(rows):
    require(len(rows) == 3 and {r["clip"] for r in rows} == set(CLIPS), "Group must retain all three fixed clips")
    complete = sum(row["complete"] for row in rows)
    value = dict(total_cases=3, complete_cases=complete, full_fixed_set_means=None)
    if complete == 3:
        keys = ("upper_rmse_rad", "joint_rmse_rad", "slip_left_m_s", "slip_right_m_s",
                "sole_left_m", "sole_right_m", "alternations", "tilt_rms_rad")
        value["full_fixed_set_means"] = {key: (statistics.mean(r[key] for r in rows)
            if all(r.get(key) is not None for r in rows) else None) for key in keys}
    return value


def analyze(source, output, *, make_plots=False):
    started = time.monotonic()
    require(source.is_absolute() and source.resolve() == source, "Require the actual absolute source root")
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and not output.is_relative_to(source), "Use a new independent analysis root")
    result = read(source / "result.json")
    require(result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
            and result.get("new_training_updates") == 0, "Event campaign is not formally complete")
    auth, equality, comparison, audited = (read(source / name) for name in
        ("authorization.json", "equivalence.json", "comparison.json", "checkpoint_audit.json"))
    expected = {(event, method, h, clip) for event in EVENTS for method, h in CONFIGS for clip in CLIPS}
    require(auth.get("stage") == "PAUSED32K_SYNTHETIC_EVENTS" and auth.get("iteration") == 32000
            and len(auth.get("cases", [])) == 84 and {tuple(r) for r in auth["cases"]} == expected,
            "Changed event campaign or fixed case set")
    gate_cases = equality.get("cases", [])
    require(equality.get("status") == "PASSED" and len(gate_cases) == 21
            and {(r["method"], r["horizon"], r["clip"]) for r in gate_cases}
                == {(method, h, clip) for method, h in CONFIGS for clip in CLIPS}
            and equality.get("atol") == equality.get("rtol") == 2e-5
            and equality.get("physical_transitions") == equality.get("optimizer_updates") == 0
            and comparison.get("status") == "COMPLETED" and comparison.get("new_physical_cases") == 84
            and comparison.get("natural_teleoperation_qualified") is False,
            "Missing completed event action gate or comparison")
    require(audited.get("status") == "PASSED" and audited.get("iteration") == 32000,
            "Missing reused32k checkpoint audit")
    selected = next((r["checkpoint"] for r in audited.get("records", []) if r["robot"] == "q1"), None)
    require(selected is not None and equality.get("checkpoint") == selected, "Event action gate used another checkpoint")
    paths = [source / name for name in
             ("result.json", "authorization.json", "comparison.json", "equivalence.json", "checkpoint_audit.json")]
    paths.extend((Path(__file__).resolve(), HERE / "analyze.py", HERE / "preview_summary.py",
                  HERE.parent / "supermimic_teleop/event_inputs.py",
                  HERE.parent / "supermimic_teleop/stream_events.py"))
    input_records = [identity(path) for path in paths]
    published = {row["path"]: row for row in auth["evaluation_sources"]}
    for path in (HERE.parent / "supermimic_teleop/event_inputs.py", HERE.parent / "supermimic_teleop/stream_events.py"):
        require(str(path) in published and identity(path) == published[str(path)],
                "Event validation code differs from the published replay")
    rows = []
    for event, method, horizon, clip in auth["cases"]:
        directory = folder(source, event, method, horizon, clip)
        saved, telemetry = directory / "review/result.json", directory / "stream_timing.jsonl"
        saved_identity = identity(saved)
        case = read(saved)
        require(case.get("status") == "PASSED" and case.get("iteration") == 32000
                and case.get("checkpoint") == selected and len(case.get("cases", [])) == 1,
                "Missing completed physical/media record")
        row = case["cases"][0]
        require((row.get("event"), row.get("tail_method"), row.get("horizon_frames"), row.get("clip"))
                == (event, method, horizon, clip) and row.get("seed") == 0
                and row.get("policy_device") == "cuda" and row.get("reference_is_synthetic") is True
                and row.get("measured_end_to_end_latency_ns") is None
                and row.get("video", {}).get("status") == "PASSED"
                and row["video"].get("original_speed") is True,
                "Saved case identity or synthetic/media claims differ")
        trace = Path(row["trace"])
        require(trace == directory / "review" / (clip+"_s0") / "rollout.npz", "Trace escaped its saved case")
        before = [saved_identity, identity(trace), identity(telemetry)]
        timings = [json.loads(line) for line in telemetry.read_text().splitlines()]
        require(len(timings) == row["attempted_steps"], "Timing and transition counts differ")
        spec = EventSpec(**row["event_mapping"]["event"])
        require(spec == EventSpec(event, first_frame=50, duration_frames=10, transport_frames=0),
                "Changed event or transport configuration")
        timeline = EventTimeline(row["reference_steps"]+1, spec)
        for frame, value in enumerate(timings):
            validate_event_timing(value, frame, horizon, method, timeline)
        with np.load(trace, allow_pickle=False) as arrays:
            actual, reference, actions = (arrays[key] for key in ("actual", "reference", "canonical_action"))
            require(actual.shape == (len(actions)+1, 79) and reference.shape == (row["reference_steps"]+1, 79)
                    and all(np.isfinite(a).all() for a in (actual, reference, actions)), "Invalid saved event trace")
            timing = telemetry_summary(timings, actions, spec, horizon)
            window = capture_window(timings, actions, actual, reference, row["upper_joint_indices"], spec)
        value = dict(event=event, method=method, horizon_frames=horizon, preview_budget_ms=horizon*20,
            clip=clip, **metrics(row), failure_frame=row["failure_frame"],
            completion_fraction=row["valid_steps"]/row["reference_steps"], video=row["video"],
            derived_reference_sha256=row["derived_reference_sha256"], telemetry=timing, capture_window=window)
        require(all(identity(Path(record["path"])) == record for record in before), "Input changed during event analysis")
        input_records.extend(before)
        rows.append(value)
    lookup = {(r["event"], r["method"], r["horizon_frames"], r["clip"]): r for r in rows}
    for row in rows:
        control = lookup[(row["event"], "masked", 10, row["clip"])]
        require(row["derived_reference_sha256"] == control["derived_reference_sha256"], "Comparison targets changed")
        keys = ("upper_rmse_rad", "slip_left_m_s", "slip_right_m_s", "tilt_rms_rad")
        row["change_from_same_event_h10"] = ({key: row[key]-control[key] for key in keys
            if row.get(key) is not None and control.get(key) is not None}
            if row["complete"] and control["complete"] else None)
    groups = [dict(event=event, method=method, horizon_frames=h,
        **full_group([lookup[(event, method, h, clip)] for clip in CLIPS]))
        for event in EVENTS for method, h in CONFIGS]
    value = dict(stage="SAVED_SYNTHETIC_EVENT_ANALYSIS", rows=rows, groups=groups, inputs=input_records,
        evaluation_commit=auth["evaluation_commit"], total_cases=len(rows),
        complete_cases=sum(row["complete"] for row in rows), new_physical_transitions=0,
        new_inference_calls=0, new_training_updates=0, measured_end_to_end_latency_ms=None,
        user_accepted_checkpoint=None, natural_teleoperation_qualified=False,
        failed_case_policy="Full-motion metrics and full-set means are absent when required cases fail",
        timing_note="Synthetic capture/control clocks only; software goal-call time excludes RPC and execution",
        window_note="Fixed first1000ms after event capture; action-associated next simulator state, not hardware response latency",
        action_note="Canonical29 action changes, including absent zero joints; not physical acceleration",
        source_preprocessing="offline_unverified", input_constraints_validated=False)
    require(all(identity(Path(record["path"])) == record for record in input_records), "Saved sources changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    flat = []
    for row in rows:
        record = {k:v for k,v in row.items() if k not in ("video", "telemetry", "capture_window", "change_from_same_event_h10")}
        for name in ("fallback_frames", "first_goal_call_ms", "action_delta_rms_p95", "action_delta_max_joint_max"):
            record[name] = row["telemetry"][name]
        record.update(event_window_complete=row["capture_window"]["complete"],
                      event_window_upper_rmse_rad=row["capture_window"]["upper_rmse_rad"],
                      event_window_action_jump_max=row["capture_window"]["action_delta_max_joint_max"])
        flat.append(record)
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (output / "per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    if make_plots:
        for event in EVENTS:
            directory = output / event
            directory.mkdir()
            plot([row for row in rows if row["event"] == event], directory)
    require(all(identity(Path(record["path"])) == record for record in input_records), "Sources changed while writing analysis")
    (output / "audit.json").write_text(json.dumps(dict(status="PASSED", source_files=len(input_records),
        source_inputs_unchanged=True, seconds=time.monotonic()-started, new_physics=0,
        new_inference=0, new_training=0, plots_requested=make_plots), indent=2)+"\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = analyze(args.source, args.output, make_plots=args.plot)
    print(json.dumps(dict(total=result["total_cases"], complete=result["complete_cases"], output=str(args.output))))
