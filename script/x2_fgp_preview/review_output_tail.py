"""Compare FGP output and downstream confirmation gaps in archived joint replays."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .review_cano_solver_tail import read_jsonl


def summary(values):
    samples = np.asarray(values)
    flags = samples > 1000 / 30
    runs = np.diff(np.flatnonzero(np.diff(np.r_[False, flags, False])))
    return dict(n=len(samples), p50=float(np.median(samples)), p95=float(np.percentile(samples, 95)),
                p99=float(np.percentile(samples, 99)), max=float(max(samples)),
                above_33_33_ms=int(flags.sum()), above_50_ms=int(np.count_nonzero(samples > 50)),
                above_100_ms=int(np.count_nonzero(samples > 100)),
                longest_consecutive_over_33_33_ms=int(max(runs[::2], default=0)))


def review(path):
    data = [row for row in read_jsonl(path / "fgp.jsonl") if row["output"]]
    ids = np.array([row["frame"] for row in data])
    source = np.array([row["capture_ns"] for row in data])
    output = np.array([row["inferred_ns"] for row in data])
    ack = np.array([row["acknowledged_ns"] for row in data])
    all_stamps = (ids, source, output, ack)
    if len(ids) < 8 or any((np.diff(column) <= 0).any() for column in all_stamps):
        raise ValueError(f"nonmonotonic recorded FGP output timestamps in {path}")
    if (output < source).any() or (ack < output).any():
        raise ValueError(f"recorded FGP output violates causal clock ordering in {path}")
    output_gaps, ack_gaps = (np.diff(column) / 1e6 for column in (output, ack))
    cases = Counter()
    for output_gap, ack_gap in zip(output_gaps, ack_gaps, strict=True):
        cases[("slow" if output_gap > 50 else "within_50") + "_fgp_"
              + ("slow" if ack_gap > 50 else "within_50") + "_ack"] += 1
    age = (ack - source) / 1e6
    n = len(age) // 4
    return dict(outputs=len(ids), skipped_source_ids=int(np.sum(np.diff(ids) - 1)),
                measured_output_rate_hz=float((len(ids) - 1) * 1e9 / (output[-1] - output[0])),
                output_gap_ms=summary(output_gaps), acknowledgement_gap_ms=summary(ack_gaps),
                frame_source_to_ack_ms=summary(age), source_to_ack_first_last_quarter_p50_ms=[
                    float(np.median(age[:n])), float(np.median(age[-n:]))],
                paired_gap_counts_over_50_ms=dict(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(dict(scope="archived runs of the same recorded IMU; no pooling across runs",
                          runs={str(path): review(path) for path in args.runs}),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
