"""Audit SONIC mode requests against actual policy heads in saved, same-clock logs."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from .review_sonic_cycles import read_cycles
from .sonic_reference_provenance import unpack_origin


HEAD_MODE = {"pose": "planner", "dual_pose": "planner", "token": "tracking", "dual_smpl": "tracking"}
POSE_SLOTS = 10


def pose_source(sample):
    """Classify tagged pose-reference slots; upstream identifies no producer."""
    kinds = Counter()
    for i in range(POSE_SLOTS):
        stamp = sample.get(f"watchdog_provenance_{i}", "")
        try:
            origin = unpack_origin(int(stamp))
        except ValueError:
            origin = None
        kinds[origin["source"] if origin else "unknown"] += 1
    if len(kinds) == 1:
        resolved = next(iter(kinds))
    else:
        resolved = "partial" if "unknown" in kinds else "mixed"
    return dict(source_kind=resolved, slot_counts=dict(kinds))


def read_requests(path):
    requests = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line, parse_float=Decimal)
            stamp = Decimal(str(row["t"]))
            mode = row["mode"]
            if not stamp.is_finite() or mode not in ("planner", "tracking"):
                raise ValueError("each request needs a finite timestamp and a planner/tracking mode")
            if requests and stamp <= requests[-1][0]:
                raise ValueError("request timestamps must increase")
            requests.append((stamp, mode))
    return requests


def audit(samples, requests, *, max_gap_ms, max_confirmation_ms):
    if not all(v.is_finite() and v > 0 for v in (max_gap_ms, max_confirmation_ms)):
        raise ValueError("timing limits must be finite and positive")
    if not samples or not requests:
        raise ValueError("actual head samples and operator requests are required")
    max_gap, max_delay = max_gap_ms / 1000, max_confirmation_ms / 1000
    times = sorted(samples)
    modes = [HEAD_MODE.get(samples[t]["policy_head"]) for t in times]
    confirmed = set()
    results = []
    for request_idx, (stamp, wanted) in enumerate(requests):
        next_stamp = requests[request_idx + 1][0] if request_idx + 1 < len(requests) else None
        index = bisect_left(times, stamp)
        detail = dict(request_t=str(stamp), requested_mode=wanted, outcome="pre_request_coverage_missing")
        results.append(detail)
        if index == 0 or stamp - times[index - 1] > max_gap or modes[index - 1] is None:
            continue
        before = modes[index - 1]
        preceding = samples[times[index - 1]]
        detail["prior_head"] = preceding["policy_head"]
        if before == "planner":
            detail["prior_pose_reference"] = pose_source(preceding)
        if before == wanted:
            detail["outcome"] = "already_active"
            continue
        deadline = stamp + max_delay
        previous = times[index - 1]
        for i in range(index, len(times)):
            observed = times[i]
            if previous >= deadline:
                detail["outcome"] = "confirmation_timeout"
                break
            if next_stamp is not None and next_stamp <= min(observed, deadline):
                detail["outcome"] = "superseded"
                break
            if observed - previous > max_gap or modes[i] is None:
                # Even when outside the deadline, this pair may hide a
                # transition during the confirmation interval.
                detail["outcome"] = "head_coverage_gap"
                break
            if observed > deadline:
                detail["outcome"] = "confirmation_timeout"
                break
            if modes[i] == wanted:
                if observed == stamp:
                    detail["outcome"] = "timestamp_tie"
                else:
                    detail.update(outcome="confirmed", actual_head=samples[observed]["policy_head"],
                                  confirmed_t=str(observed), confirmation_ms=float((observed - stamp) * 1000))
                    if wanted == "planner":
                        detail["confirmed_pose_reference"] = pose_source(samples[observed])
                    confirmed.add(observed)
                break
            previous = observed
        else:
            detail["outcome"] = ("confirmation_timeout" if previous >= deadline else "insufficient_tail")

    unpaired = []
    for i, (before, after) in enumerate(zip(modes, modes[1:]), 1):
        stamp = times[i]
        if (before is not None and after is not None and before != after
                and stamp - times[i - 1] <= max_gap and stamp not in confirmed):
            unpaired.append(dict(t=str(stamp), prior_mode=before, actual_mode=after))
    return dict(
        cycles=len(times), requests=len(requests), unknown_head_cycles=modes.count(None),
        outcomes=dict(Counter(r["outcome"] for r in results)), events=results,
        confirmed_planner_side_source_kinds=dict(Counter(
            r["confirmed_pose_reference"]["source_kind"] for r in results if "confirmed_pose_reference" in r
        )),
        unmatched_observed_head_changes=unpaired,
        clock_scope="one saved monotonic epoch; requests and samples must already share a clock",
        evidence_scope=("actual model head and optional pose-slot provenance; upstream does not identify "
                        "kplanner; command publication and stand-pose quality not assessed"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cycle_trace", type=Path, help="native cycle_trace.csv with actual policy_head")
    parser.add_argument("requests", type=Path, help="same-clock JSONL with monotonic seconds t and requested mode")
    parser.add_argument("--max-gap-ms", type=Decimal, required=True)
    parser.add_argument("--max-confirmation-ms", type=Decimal, required=True)
    args = parser.parse_args()
    samples, excluded = read_cycles(args.cycle_trace)
    report = audit(samples, read_requests(args.requests), max_gap_ms=args.max_gap_ms,
                   max_confirmation_ms=args.max_confirmation_ms)
    report["excluded_csv_rows"] = excluded
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
