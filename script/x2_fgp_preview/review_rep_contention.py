"""Compare saved HumanFrame RPCs by whether state handling was already in progress."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from .review_cano_solver_tail import read_jsonl


def summarize(values):
    return dict(count=len(values), p50=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(max(values)), above_10_ms=sum(value > 10 for value in values))


def review(directory):
    spans = [(r["received_ns"], r["sent_ns"]) for r in read_jsonl(directory / "service.jsonl")]
    if not spans or any(b <= a for a, b in spans) or any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
        raise ValueError("expected serial ordered state handling spans")
    starts = [begin for begin, _ in spans]
    buckets = {"state_already_running": [], "no_state_running_at_send": []}
    state_finish_overlaps = []
    for row in read_jsonl(directory / "fgp.jsonl"):
        if not row["output"]:
            continue
        end = row["acknowledged_ns"]
        start = end - round(row["human_rpc_ms"] * 1e6)
        i = bisect_right(starts, start) - 1
        used = i >= 0 and start < spans[i][1]
        key = "state_already_running" if used else "no_state_running_at_send"
        buckets[key].append((end - start) / 1e6)
        if used:
            state_finish_overlaps.append((min(end, spans[i][1])-start) / 1e6)
    return dict(cases={name: summarize(items) for name, items in buckets.items()},
                overlap_with_already_running_state_ms=summarize(state_finish_overlaps))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    print(json.dumps(dict(clock_scope="same board monotonic clock per archived run",
                          runs={str(path): review(path) for path in args.runs}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
