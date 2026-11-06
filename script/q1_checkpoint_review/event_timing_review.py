"""Locate saved event-call outliers without new inference or causal claims."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from preview_summary import identity
from event_summary import folder, read, require


def statistics_for(rows):
    values = sorted(row["compute_ms"] for row in rows)
    if not values:
        return dict(calls=0, over20ms=0, over100ms=0, maximum_ms=None, p95_ms=None)
    position = (len(values)-1)*.95
    lo = int(position)
    hi = min(lo+1, len(values)-1)
    return dict(calls=len(values), over20ms=sum(v > 20 for v in values),
        over100ms=sum(v > 100 for v in values), maximum_ms=values[-1],
        median_ms=statistics.median(values), p95_ms=values[lo]+(values[hi]-values[lo])*(position-lo))


def locate(timings, row):
    recoveries = [i for i in range(1, len(timings))
                  if timings[i-1]["fallback"] is not None and timings[i]["fallback"] is None]
    points = []
    for frame, value in enumerate(timings):
        require(value["frame"] == frame and value["simulated_control_ns"]
                == (frame+row["horizon_frames"])*20_000_000,
                "Saved call frame/control clock differs")
        elapsed = value["simulated_control_ns"]-value["event"]["first_frame"]*20_000_000
        duration = value["event"]["duration_frames"]*20_000_000
        compute = value["gpu_policy_wall_compute_ns"]
        require(type(compute) is int and compute >= 0, "Invalid saved compute time")
        points.append(dict(event=row["event"], method=row["method"], horizon_frames=row["horizon_frames"],
            clip=row["clip"], frame=frame, compute_ms=compute/1e6,
            capture_clock_since_event_ms=elapsed/1e6,
            region="before_event" if elapsed < 0 else ("event_window" if elapsed < duration else "after_event"),
            first_capture_second=0 <= elapsed < 1_000_000_000,
            fallback=value["fallback"] is not None,
            recovered_this_frame=frame in recoveries,
            within_two_frames_of_recovery=any(abs(frame-r) <= 2 for r in recoveries),
            nearest_recovery_distance_frames=min((abs(frame-r) for r in recoveries), default=None)))
    return points


def analyze(summary_path, audit_path, output):
    started = time.monotonic()
    require(summary_path.is_absolute() and summary_path.resolve() == summary_path
            and output.is_absolute() and output.resolve() == output and not output.exists(),
            "Use actual saved summary and a new absolute output")
    before = [identity(summary_path), identity(audit_path), identity(Path(__file__).resolve())]
    summary, audit = read(summary_path), read(audit_path)
    require(summary.get("stage") == "PARTIAL_SAVED_EVENT_ANALYSIS" and summary.get("total_cases") == 78
            and summary.get("complete_cases") == 78 and len(summary.get("missing", [])) == 6
            and audit.get("status") == "PARTIAL_ANALYSIS_PASSED" and audit.get("source_inputs_unchanged") is True,
            "Require the already-completed78-case partial analysis")
    sources = {r["path"]: r for r in summary["inputs"]}
    original = next(Path(r["path"]).parent for r in summary["inputs"]
                    if r["path"].endswith("/result.json") and r.get("sha256")
                    == "81f7f90c39b68a25fe5c15760e75902e6a9839f9a5ef3a44bd7a7ac3e38a7191")
    require(not output.is_relative_to(original), "Do not write into the failed campaign")
    all_points, case_rows = [], []
    for row in summary["rows"]:
        require(time.monotonic()-started < 300, "Saved timing review exceeded300s")
        path = folder(original, row["event"], row["method"], row["horizon_frames"], row["clip"])/"stream_timing.jsonl"
        require(str(path) in sources and identity(path) == sources[str(path)], "Saved timing SHA changed")
        values = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(values) == row["reference_steps"], "Complete case timing count differs")
        points = locate(values, row)
        kept = points[1:]
        expected = row["telemetry"]["gpu_goal_call_without_first"]
        measured = statistics_for(kept)
        require(measured["calls"] == expected["samples"] and measured["over20ms"] == expected["calls_over_20ms"]
                and measured["maximum_ms"] == expected["maximum_ms"], "Saved timing aggregate changed")
        case_rows.append(dict(event=row["event"], method=row["method"], horizon_frames=row["horizon_frames"],
            clip=row["clip"], first_call_ms=points[0]["compute_ms"], **measured))
        all_points.extend(kept)
        before.append(sources[str(path)])
    strata = []
    for event in ("ALL", "stop", "reverse", "dropout", "late_burst"):
        selected = [p for p in all_points if event == "ALL" or p["event"] == event]
        for region in ("ALL", "before_event", "event_window", "after_event", "first_capture_second",
                       "fallback", "within_two_frames_of_recovery"):
            subset = [p for p in selected if region == "ALL" or p.get("region") == region or p.get(region) is True]
            strata.append(dict(event=event, subset=region, **statistics_for(subset)))
    require(all(identity(Path(r["path"])) == r for r in before), "Input changed during timing review")
    spikes = sorted((p for p in all_points if p["compute_ms"] > 20), key=lambda p:p["compute_ms"], reverse=True)
    value = dict(stage="SAVED_EVENT_TIMING_LOCALIZATION", status="PASSED", summary=before[0],
        overall_without_first=statistics_for(all_points), strata=strata, per_case=case_rows,
        largest_calls=spikes[:30], inputs=before, source_inputs_unchanged=True,
        new_inference_calls=0, new_physical_transitions=0, new_training_updates=0,
        overlap_note="Named boolean subsets overlap; each includes its call denominator. The three regions partition all calls.",
        clock_note="Capture/event clocks are synthetic. compute_ms is synchronized software goal-call wall time.",
        causal_attribution="NOT_IDENTIFIED; saved timing cannot separate Python scheduling, goal construction, transfers and GPU forward",
        measured_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        seconds=time.monotonic()-started)
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    with (output/"over20ms.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_points[0]))
        writer.writeheader()
        writer.writerows(spikes)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("summary", "audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.summary, args.audit, args.output)
    print(json.dumps(dict(status=value["status"], **value["overall_without_first"])))
