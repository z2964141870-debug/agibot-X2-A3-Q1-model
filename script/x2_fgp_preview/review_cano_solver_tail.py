"""Attribute saved FGP frame time to matched GPU and CPU spans."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def distribution(values):
    values = sorted(values)
    return dict(n=len(values), p50=median(values), p95=values[math.ceil(.95 * len(values)) - 1],
                p99=values[math.ceil(.99 * len(values)) - 1], max=values[-1])


def matcher(rows):
    ordered = sorted(rows, key=lambda row: row["began_ns"])
    starts = [row["began_ns"] for row in ordered]

    def one(start, stop, stage):
        idx = bisect_left(starts, start)
        if idx >= len(ordered) or ordered[idx]["ended_ns"] > stop:
            raise ValueError(f"{stage} did not match one full frame ({start}, {stop})")
        if idx + 1 < len(starts) and starts[idx + 1] < stop:
            raise ValueError(f"ambiguous {stage} for frame")
        return ordered[idx]

    return one


def review(path):
    spans = read_jsonl(path / "profile.jsonl")
    find_cano = matcher([p for p in spans if p["stage"] == "cano_session"])
    find_adapter = matcher([p for p in spans if p["stage"] == "cano_adapter"])
    find_calibration = matcher([p for p in spans if p["stage"] == "imu_calibration"])
    find_solver = matcher([p for p in spans if p["stage"] == "native_solver"])
    source = [r for r in read_jsonl(path / "fgp.jsonl") if r["output"]]
    if not source:
        raise ValueError("no FGP outputs")
    frames = []
    for row in source:
        calibration = find_calibration(row["started_ns"], row["denoised_ns"], "calibration")
        adapter = find_adapter(row["started_ns"], row["denoised_ns"], "CANO adapter")
        gpu = find_cano(adapter["began_ns"], adapter["ended_ns"], "CANO TensorRT session")
        native = find_solver(row["cpu_started_ns"], row["inferred_ns"], "solver")
        total = row["fgp_ms"]
        spans_ms = dict(gpu_stage=row["gpu_stage_ms"], cpu_queue=row["cpu_queue_ms"],
                        native_solver=native["wall_ms"], cpu_other=row["cpu_stage_ms"] - native["wall_ms"])
        if min(spans_ms.values()) < 0 or not math.isclose(sum(spans_ms.values()), total, abs_tol=1e-4):
            raise ValueError(f"non-additive per-frame intervals at frame {row['frame']}")
        gpu_spans = dict(calibration=calibration["wall_ms"],
                         cano_adapter_outside_session=adapter["wall_ms"]-gpu["wall_ms"],
                         gpu_host_outside_monitored_calls=(row["gpu_stage_ms"] - calibration["wall_ms"]
                                                           - adapter["wall_ms"]))
        if min(gpu_spans.values()) < 0:
            raise ValueError(f"negative GPU-host residual at frame {row['frame']}")
        frames.append(dict(frame=row["frame"], full=total, gpu_graph=gpu["graph_stream_span_ms"],
                           cano_host=gpu["wall_ms"], solver_thread_cpu=native["thread_cpu_ms"],
                           calibration_thread_cpu=calibration["thread_cpu_ms"],
                           **spans_ms, **gpu_spans))
    keys = ("full", "gpu_stage", "gpu_graph", "cano_host", "cpu_queue", "native_solver",
            "solver_thread_cpu", "cpu_other", "calibration", "calibration_thread_cpu",
            "cano_adapter_outside_session",
            "gpu_host_outside_monitored_calls")
    baseline = {key: distribution([r[key] for r in frames]) for key in keys}
    top = sorted(frames, key=lambda row: row["full"], reverse=True)[:math.ceil(.05 * len(frames))]
    tail = {key: distribution([r[key] for r in top]) for key in keys}
    components = ("gpu_stage", "cpu_queue", "native_solver", "cpu_other")
    largest = Counter(max(components, key=lambda key: row[key] - baseline[key]["p50"]) for row in top)
    detailed = ("calibration", "cano_host", "cano_adapter_outside_session",
                "gpu_host_outside_monitored_calls", "cpu_queue", "native_solver", "cpu_other")
    largest_detail = Counter(max(detailed, key=lambda key: row[key] - baseline[key]["p50"]) for row in top)
    return dict(frames=len(frames), top_frames=len(top), baseline_ms=baseline, slowest_five_percent_ms=tail,
                largest_increase_from_own_median=dict(largest),
                largest_detailed_increase=dict(largest_detail), slowest_examples=top[:4])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({run.name: review(run) for run in args.runs}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
