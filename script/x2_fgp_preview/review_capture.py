"""Summarize a saved FGP terminal capture without loading the models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return dict(n=len(values), p50=median(values),
                p95=values[math.ceil(.95 * len(values)) - 1],
                p99=values[math.ceil(.99 * len(values)) - 1], max=values[-1])


def summarize(rows):
    frames = [row for row in rows if row["stage"] == "output"]
    intervals, jumps = [], []
    for left, right in zip(frames, frames[1:]):
        gap = (right["output_ns"] - left["output_ns"]) / 1e6
        if gap <= 0 or right["sequence"] <= left["sequence"]:
            raise ValueError("FGP output timestamps and frame sequences must increase")
        intervals.append(gap)
        jumps.append(dict(before=left["sequence"], after=right["sequence"], gap_ms=gap))
    profile_columns = {}
    for row in frames:
        for name, value in row.get("inference_stages", {}).items():
            if name.endswith("_ms") and value is not None:
                profile_columns.setdefault(name, []).append(value)
    infer = [row["inference_ms"] for row in frames]
    return dict(
        stages=dict(Counter(row["stage"] for row in rows)),
        output_hz=(1000 * len(intervals) / sum(intervals) if intervals else None),
        inference_ms=stats(infer),
        inference_over_30hz_budget=sum(ms > 1000/30 for ms in infer),
        output_interval_ms=stats(intervals),
        pair_to_output_ms=stats((row["output_ns"] - row["source_ns"]) / 1e6 for row in frames),
        scheduled_lateness_ms=stats((row["tick_ns"] - row["scheduled_ns"]) / 1e6
                                    for row in rows if "scheduled_ns" in row),
        stage_ms={name: stats(values) for name, values in profile_columns.items()},
        slowest_frames=sorted(frames, key=lambda row: row["inference_ms"], reverse=True)[:5],
        widest_output_intervals=sorted(jumps, key=lambda row: row["gap_ms"], reverse=True)[:5],
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    with args.capture.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    print(json.dumps(summarize(rows), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
