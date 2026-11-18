"""Join saved SONIC control CSVs by their original monotonic tick."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
import math
from pathlib import Path

from .sonic_reference_provenance import unpack_origin


def read_cycles(path):
    records = {}
    bad = 0
    with path.open(encoding="utf-8") as source:
        def intact_lines():
            nonlocal bad
            for line in source:
                if "\0" in line:
                    bad += 1
                else:
                    yield line

        rows = csv.reader(intact_lines(), strict=True)
        fields = next(rows)
        if not fields or fields[0] != "t" or len(fields) != len(set(fields)):
            raise ValueError(f"invalid SONIC CSV header: {path}")
        for row in rows:
            if len(row) != len(fields):
                bad += 1
                continue
            stamp = Decimal(row[0])
            if not stamp.is_finite() or stamp in records:
                raise ValueError(f"non-finite or duplicate cycle timestamp in {path}: {row[0]}")
            records[stamp] = dict(zip(fields[1:], row[1:]))
    return records, bad


def degrees(text):
    result = math.degrees(float(text))
    if not math.isfinite(result):
        raise ValueError("non-finite SONIC angle")
    return result


def review(root, joint):
    names = ("tick", "joint_pos", "target_pos", "action_il")
    sources = {name: read_cycles(root / f"{name}.csv") for name in names}
    optional = root / "cycle_trace.csv"
    if optional.is_file():
        sources["cycle_trace"] = read_cycles(optional)
    cycles = sorted(set.intersection(*(set(sources[name][0]) for name in names)))
    tick, measured, target, action = (sources[name][0] for name in names)
    trace = sources.get("cycle_trace", ({}, 0))[0]

    def record(stamp):
        row = dict(t=str(stamp), ramp_alpha=float(tick[stamp]["ramp_alpha"]),
                   reason=tick[stamp]["reason"],
                   target_deg=degrees(target[stamp][f"target_{joint}"]),
                   measured_deg=degrees(measured[stamp][f"q_{joint}"]))
        if stamp in trace:
            details = dict(trace[stamp])
            slots = [unpack_origin(int(details[name])) for name in details
                     if name.startswith("watchdog_provenance_")]
            if slots:
                details["watchdog_slots"] = slots
            row["trace"] = details
        if joint == "waist_roll_joint":
            row["action_il_8_post_clip"] = float(action[stamp]["a_il_8"])
        return row

    rows = [record(stamp) for stamp in cycles]
    reset, steady, other = [], [], []
    for prev, row in zip(rows, rows[1:]):
        span = Decimal(row["t"]) - Decimal(prev["t"])
        if span > Decimal(".035"):
            continue
        delta = row["target_deg"] - prev["target_deg"]
        item = dict(before=prev, after=row, step_deg=delta, interval_ms=float(span * 1000))
        if prev["ramp_alpha"] > .9 and row["ramp_alpha"] < .05:
            reset.append(item)
        elif prev["ramp_alpha"] == row["ramp_alpha"] == 1 and prev["reason"] == row["reason"] == "ok":
            steady.append(item)
        else:
            other.append(item)

    def peak(items):
        return dict(n=len(items), largest=max(items, key=lambda item: abs(item["step_deg"]), default=None))

    return dict(
        joint=joint, matched_cycles=len(rows),
        sources={name: dict(rows=len(items), excluded=bad) for name, (items, bad) in sources.items()},
        reasons=dict(Counter(row["reason"] for row in rows)),
        adjacent_target_steps=dict(ramp_reset=peak(reset), steady=peak(steady), other=peak(other)),
        largest_target_error=max((abs(row["target_deg"] - row["measured_deg"]) for row in rows), default=None),
        observed_policy_heads=dict(Counter(row["policy_head"] for row in trace.values())),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--joint", default="waist_roll_joint")
    args = parser.parse_args()
    print(json.dumps(review(args.folder, args.joint), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
