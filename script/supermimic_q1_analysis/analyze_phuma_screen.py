#!/usr/bin/env python3
"""Summarize fixed PHUMA screening evidence without changing reference gates."""

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path


def identity(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def analyze(manifest_path, checks_path):
    manifest = json.loads(manifest_path.read_text())
    checks = json.loads(checks_path.read_text())
    observed = identity(manifest_path)
    if any(observed[key] != checks["input_manifest"][key] for key in ("bytes", "sha256")):
        raise ValueError("Screen receipt belongs to a different conversion manifest")
    if (checks["status"] != "INDEPENDENT_REFERENCE_CHECK_COMPLETED"
            or checks["reference_protocol"] != manifest["reference_protocol"]):
        raise ValueError("Incomplete or mismatched PHUMA screening evidence")
    converted = {row["clip"]: row for row in manifest["motions"]}
    screened = {row["clip"]: row for row in checks["clips"]}
    if (len(converted) != len(manifest["motions"]) or len(screened) != len(checks["clips"])
            or converted.keys() != screened.keys()):
        raise ValueError("Duplicate, missing or unexpected screened clips")
    groups, failures, patterns = {}, collections.Counter(), collections.Counter()
    qualified, sources, training_examples = [], set(), []
    expected_gates = set(next(iter(screened.values()))["gates"])
    for name, row in screened.items():
        source = converted[name]
        if (row["split"] != source["split"] or row["bucket"] != source["bucket"]
                or row["source_group"] != source["phuma_source"]["source_group"]):
            raise ValueError("Screen and conversion split/source metadata differ")
        if row["source_group"] in sources:
            raise ValueError("A source group is repeated across fixed candidates")
        sources.add(row["source_group"])
        gates = row["gates"]
        if set(gates) != expected_gates or not all(type(value) is bool for value in gates.values()):
            raise ValueError("Inconsistent or non-boolean screening gates")
        accepted = all(gates.values())
        if accepted != row["qualified_for_training_pilot"]:
            raise ValueError("Qualification disagrees with the recorded gates")
        failed = tuple(sorted(key for key, passed in gates.items() if not passed))
        failures.update(failed)
        patterns[failed] += 1
        key = row["split"] + "/" + row["bucket"]
        group = groups.setdefault(key, {"candidates": 0, "qualified": 0,
                                       "failure_counts": collections.Counter()})
        group["candidates"] += 1
        group["qualified"] += int(accepted)
        group["failure_counts"].update(failed)
        if accepted:
            qualified.append(name)
        if row["split"] == "train" and row["bucket"] == "locomotion" and len(failed) == 1:
            training_examples.append({
                "clip": name, "failed_gate": failed[0], "duration_s": row["duration_s"],
                "support_fraction": row["support_within_15mm_fraction"],
                "support_gap_max_m": row["foot_support_gap_max_m"],
                "self_penetration_max_m": row["self_penetration_max_m"],
                "IK_position_p95_m": row["IK_position_p95_m"],
                "horizontal_displacement_m": row["reference_horizontal_displacement_m"]})
    if sorted(qualified) != sorted(checks["qualified_clips"]):
        raise ValueError("Qualified list differs from independently counted rows")
    split_counts = collections.Counter(screened[name]["split"] for name in qualified)
    if dict(split_counts) != checks["qualified_split_counts"]:
        raise ValueError("Qualified split totals disagree")
    for group in groups.values():
        group["failure_counts"] = dict(sorted(group["failure_counts"].items()))
    return {
        "status": "PHUMA_FIXED_SCREEN_SUMMARY_RECONCILED",
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs": {"manifest": observed, "screen_receipt": identity(checks_path)},
        "candidates": len(screened), "qualified": len(qualified),
        "rejected": len(screened) - len(qualified), "independent_gates": len(expected_gates),
        "unique_source_groups": len(sources), "qualified_split_counts": dict(split_counts),
        "failure_counts": dict(failures.most_common()), "groups": dict(sorted(groups.items())),
        "failure_patterns": [{"failed_gates": list(keys), "clips": count}
                             for keys, count in patterns.most_common()],
        "training_locomotion_single_failure_examples": training_examples,
        "new_physics_steps": 0, "new_policy_calls": 0, "new_optimizer_updates": 0,
        "reference_or_gate_changes": False, "storage_status": "LOCAL_ONLY",
        "limits": "Failure counts overlap. Single-failure examples are not qualified. "
                  "Summary of existing geometry evidence; no new FK, dynamics or tracking evaluation.",
        "source": identity(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.checks)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "candidates", "qualified", "rejected",
                                                  "failure_counts", "qualified_split_counts")}, allow_nan=False))


if __name__ == "__main__":
    main()
