"""Summarize one isolated SONIC shadow capture: handoff audit, phase joint stats,
pose-wire provenance and the wearer's wrist motion. Read-only, offline."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from script.x2_fgp_preview.audit_sonic_handoff import audit, read_requests
from script.x2_fgp_preview.review_sonic_cycles import read_cycles
from script.x2_fgp_preview.sonic_reference_provenance import unpack_origin

JOINTS = (
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_time(path):
    with path.open(encoding="utf-8") as source:
        return {row["t"]: row for row in csv.DictReader(line for line in source if "\0" not in line)}


def summarize(capture, max_gap_ms, max_confirmation_ms):
    deploy = capture / "deploy"
    trace_path = deploy / "cycle_trace.csv"
    samples, excluded = read_cycles(trace_path)
    report = dict(audit=audit(samples, read_requests(capture / "requests.jsonl"),
                              max_gap_ms=Decimal(max_gap_ms), max_confirmation_ms=Decimal(max_confirmation_ms)))
    report["audit"]["excluded_csv_rows"] = excluded
    rows = [dict(t=str(t), **samples[t]) for t in sorted(samples)]
    heads = [row["policy_head"] for row in rows]
    phases = []
    for head in heads:
        tracking = head in ("token", "dual_smpl")
        if not phases or phases[-1][0] != tracking:
            phases.append([tracking, 0])
        phases[-1][1] += 1
    labels, cursor = [], 0
    for index, (tracking, count) in enumerate(phases):
        name = ("tracking" if tracking else "planner") + f"_{index}"
        labels += [name] * count
        cursor += count
    report["phases"] = [dict(name=("tracking" if t else "planner") + f"_{i}", ticks=c)
                        for i, (t, c) in enumerate(phases)]

    provenance = {}
    for row, label in zip(rows, labels):
        if not label.startswith("planner"):
            continue
        kinds = provenance.setdefault(label, {})
        for k in range(10):
            origin = unpack_origin(int(row.get(f"watchdog_provenance_{k}", "0") or 0))
            kind = origin["source"] if origin else "unknown"
            kinds[kind] = kinds.get(kind, 0) + 1
    report["planner_slot_provenance"] = provenance
    report["pose_received_total"] = dict(first=int(rows[0]["pose_received_total"]),
                                         last=int(rows[-1]["pose_received_total"]))
    report["token_received_total"] = dict(first=int(rows[0]["token_received_total"]),
                                          last=int(rows[-1]["token_received_total"]))

    targets, measured = by_time(deploy / "target_pos.csv"), by_time(deploy / "joint_pos.csv")
    joints = {}
    for joint in JOINTS:
        per = {}
        previous = None
        for row, label in zip(rows, labels):
            t = row["t"]
            if t not in targets or t not in measured:
                continue
            target = math.degrees(float(targets[t][f"target_{joint}"]))
            error = abs(target - math.degrees(float(measured[t][f"q_{joint}"])))
            stats = per.setdefault(label, dict(min=target, max=target, max_step=0.0, max_error=0.0))
            stats["min"], stats["max"] = min(stats["min"], target), max(stats["max"], target)
            stats["max_error"] = max(stats["max_error"], error)
            if previous is not None:
                step = abs(target - previous[1])
                key = "max_step" if previous[0] == label else "boundary_step_into"
                stats[key] = max(stats.get(key, 0.0), step)
            previous = (label, target)
        joints[joint] = {k: {n: round(v, 3) for n, v in s.items()} for k, s in per.items()}
    report["joint_targets_deg"] = joints

    spans = sorted((capture / "spans").glob("span_*.npz")) if (capture / "spans").is_dir() else []
    wrists = []
    for span in spans:
        data = np.load(span)
        joints24 = data["smpl_joints"].reshape(-1, 24, 3)
        if len(joints24) < 2:
            continue
        wrists.append(dict(file=span.name, frames=int(len(joints24)),
                           duration_s=round(float(data["t"][-1] - data["t"][0]), 3),
                           left_wrist_z_m=[round(float(joints24[:, 20, 2].min()), 3), round(float(joints24[:, 20, 2].max()), 3)],
                           right_wrist_z_m=[round(float(joints24[:, 21, 2].min()), 3), round(float(joints24[:, 21, 2].max()), 3)]))
    report["wearer_spans"] = wrists
    report["files"] = {str(p.relative_to(capture)): sha256(p) for p in
                       [trace_path, capture / "requests.jsonl", deploy / "target_pos.csv", deploy / "joint_pos.csv"]}
    report["scope"] = ("read-only shadow: --no-publish, robot never moved; target-vs-measured error grows "
                       "open-loop and is not a closed-loop tracking error")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--max-gap-ms", default="60")
    parser.add_argument("--max-confirmation-ms", default="6000")
    args = parser.parse_args()
    print(json.dumps(summarize(args.capture, args.max_gap_ms, args.max_confirmation_ms),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
