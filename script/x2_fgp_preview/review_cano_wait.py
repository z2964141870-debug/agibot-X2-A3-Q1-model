"""Compare CUDA event and host timing inside saved FGP slow frames."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from .review_cano_solver_tail import matcher, read_jsonl
from .review_fgp_gap_pairing import review as review_gaps


def evidence(path):
    raw = read_jsonl(path / "profile.jsonl")
    find = {name: matcher([p for p in raw if p["stage"] == name and (mode is None or p["direction"] == mode)])
            for name, mode in (("cano_session", None), ("imu_calibration", None), ("cuda_sync_host", None),
                               ("cuda_graph_enqueue_host", None), ("cuda_memcpy_host", 2))}
    solve = matcher([p for p in read_jsonl(path / "cpu_child/profile.jsonl") if p["stage"] == "native_solver"])
    cycles = []
    for frame in read_jsonl(path / "fgp.jsonl"):
        if not frame["output"]:
            continue
        got = {k: f(frame["started_ns"], frame["denoised_ns"], k) for k, f in find.items()}
        cano, calibration = got["cano_session"], got["imu_calibration"]
        sync, enqueue, copy = (got[key] for key in ("cuda_sync_host", "cuda_graph_enqueue_host", "cuda_memcpy_host"))
        if any(not cano["began_ns"] <= row["began_ns"] < row["ended_ns"] <= cano["ended_ns"]
               for row in (sync, enqueue, copy)) or copy["bytes"] != 360:
            raise ValueError("unexpected CANO profiler nesting")
        graph = cano["graph_stream_span_ms"]
        h2d = cano["h2d_stream_span_ms"]
        d2h = cano["d2h_stream_span_ms"]
        cuda_span = graph + h2d + d2h
        ns = {k: frame[k] for k in ("capture_ns", "started_ns", "denoised_ns", "cpu_started_ns", "inferred_ns")}
        native_solver = solve(ns["cpu_started_ns"], ns["inferred_ns"], "native solver")
        detail = dict(start=(ns["started_ns"]-ns["capture_ns"])/1e6,
                      calibration=calibration["wall_ms"], cano=cano["wall_ms"],
                      rest=(ns["denoised_ns"]-ns["started_ns"])/1e6-calibration["wall_ms"]-cano["wall_ms"],
                      cpu_queue=(ns["cpu_started_ns"]-ns["denoised_ns"])/1e6,
                      native=native_solver["wall_ms"],
                      cpu_other=(ns["inferred_ns"]-ns["cpu_started_ns"])/1e6-native_solver["wall_ms"])
        timing = dict(cano=detail["cano"], cano_thread=cano["thread_cpu_ms"],
                      graph_span=graph, h2d_span=h2d, d2h_span=d2h,
                      cano_less_stream_span=cano["wall_ms"]-cuda_span,
                      d2h_host=copy["wall_ms"], d2h_host_thread=copy["thread_cpu_ms"],
                      launch_host=enqueue["wall_ms"], explicit_sync=sync["wall_ms"],
                      calibration=detail["calibration"], calibration_cpu=calibration["thread_cpu_ms"],
                      calibration_wall_less_cpu=detail["calibration"]-calibration["thread_cpu_ms"])
        cycles.append((frame["frame"], ns["inferred_ns"], detail, timing))
    cano_slow, calibration_slow, examples = [], [], []
    for before, after in zip(cycles, cycles[1:]):
        gap = (after[1] - before[1]) / 1e6
        if gap < 40:
            continue
        changes = {k: v - before[2][k] for k, v in after[2].items()}
        diff = {k: v - before[3][k] for k, v in after[3].items()}
        dominant = max(changes, key=changes.get)
        row = dict(frame=after[0], gap_ms=gap, delta_ms=diff, before_ms=before[3], after_ms=after[3])
        if dominant == "cano":
            cano_slow.append(row)
        elif dominant == "calibration":
            calibration_slow.append(row)
        examples.append(row)
    exact = review_gaps(path)["slow_largest_detailed_change"]
    if (len(cano_slow), len(calibration_slow)) != (exact.get("cano_host", 0), exact.get("imu_calibration", 0)):
        raise ValueError("dominant stage populations differ from gap attribution")
    return dict(cycles=len(cycles), exact_slow_dominant_counts=exact,
                cano_dominant=summary(cano_slow), calibration_dominant=summary(calibration_slow),
                frame129=next((r for r in examples if r["frame"] == 129), None))


def summary(rows):
    keys = ("graph_span", "h2d_span", "d2h_span", "cano_less_stream_span")
    dominant = Counter(max(keys, key=lambda k: row["delta_ms"][k]) for row in rows)
    return dict(count=len(rows), stream_increase_largest=dict(dominant),
                median_delta_ms={k: median(r["delta_ms"][k] for r in rows) for k in rows[0]["delta_ms"]}
                if rows else {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounds", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({p.name: evidence(p) for p in args.rounds}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
