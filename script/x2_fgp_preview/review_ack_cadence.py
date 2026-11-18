"""Decompose acknowledgement cadence in saved X2 joint replay logs."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
from statistics import median

from .review_cano_solver_tail import read_jsonl

TICKS = ("capture_ns", "started_ns", "denoised_ns", "cpu_started_ns",
         "inferred_ns", "acknowledged_ns")
STAGES = ("release_to_host", "gpu_host_with_calibration", "cpu_queue", "cpu_stage",
          "inference_to_ack")
POST = ("sender_queue_ms", "bridge_ms", "human_rpc_ms")


def review(run, threshold):
    inputs = [row for row in read_jsonl(run / "fgp.jsonl") if row["output"]]
    service = read_jsonl(run / "service.jsonl")
    if len(inputs) < 2 or not service:
        raise ValueError("expected FGP outputs and service timing")
    intervals = [(row["received_ns"], row["sent_ns"]) for row in service]
    if (any(start > end for start, end in intervals)
            or any(before[1] > after[0] for before, after in zip(intervals, intervals[1:]))):
        raise ValueError("state service intervals are not serial and chronological")
    service_ends = [end for _, end in intervals]

    def sample(row):
        times = [row[key] for key in TICKS]
        durations = [end - start for start, end in zip(times, times[1:])]
        post = [round(row[key] * 1e6) for key in POST]
        if min(durations + post) < 0 or sum(post) != durations[-1]:
            raise ValueError(f"frame {row['frame']} has inconsistent stage durations")
        rpc_start = times[-1] - post[-1]
        end = times[-1]
        overlap = 0
        for index in range(bisect_right(service_ends, rpc_start), len(intervals)):
            occupied_start, occupied_end = intervals[index]
            if occupied_start >= end:
                break
            overlap += min(end, occupied_end) - max(rpc_start, occupied_start)
        return dict(frame=row["frame"], capture=times[0], ack=end,
                    durations=durations, post=post, overlap=overlap)

    pairs = []
    samples = list(map(sample, inputs))
    for a, b in zip(samples, samples[1:]):
        gap = b["ack"] - a["ack"]
        source_gap = b["capture"] - a["capture"]
        if gap <= 0 or source_gap <= 0 or b["frame"] <= a["frame"]:
            raise ValueError("FGP output order regressed")
        changes = [after - before for before, after in zip(a["durations"], b["durations"])]
        post_changes = [after - before for before, after in zip(a["post"], b["post"])]
        if gap != source_gap + sum(changes) or changes[-1] != sum(post_changes):
            raise ValueError("gap decomposition does not close")
        rpc = b["post"][-1]
        pairs.append(dict(
            before=a["frame"], after=b["frame"], ack_gap_ms=gap / 1e6,
            source_gap_ms=source_gap / 1e6,
            change_ms={key: value / 1e6 for key, value in zip(STAGES, changes)},
            post_change_ms={key: value / 1e6 for key, value in zip(POST, post_changes)},
            human_rpc_ms=rpc / 1e6, overlapped_state_service_ms=b["overlap"] / 1e6,
            remaining_rpc_ms=(rpc - b["overlap"]) / 1e6,
        ))

    slow = [row for row in pairs if row["ack_gap_ms"] > threshold]
    largest = Counter(max(row["change_ms"], key=row["change_ms"].get) for row in slow)
    post_largest = Counter(max(row["post_change_ms"], key=row["post_change_ms"].get) for row in slow)
    return dict(
        outputs=len(inputs), pairs=len(pairs), state_service_spans=len(service),
        slow_ack_gaps=len(slow),
        slow_gap_largest_stage_increase=dict(largest),
        slow_gap_largest_post_increase=dict(post_largest),
        slow_gap_median_ms={key: median(row[key] for row in slow) if slow else None
                            for key in ("human_rpc_ms", "overlapped_state_service_ms", "remaining_rpc_ms")},
        slow_gap_rpc_half_or_more_overlaps_state_service=sum(
            row["overlapped_state_service_ms"] >= .5 * row["human_rpc_ms"] for row in slow),
        largest_gaps=sorted(pairs, key=lambda row: row["ack_gap_ms"], reverse=True)[:3],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--threshold-ms", type=float, default=45.0)
    args = parser.parse_args()
    print(json.dumps(dict(threshold_ms=args.threshold_ms,
                          runs={str(run): review(run, args.threshold_ms) for run in args.runs}),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
