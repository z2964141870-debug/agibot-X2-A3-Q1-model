#!/usr/bin/env python3
"""Validate all packaged G1 chunks and preselect grouped PHUMA-only Q1 inputs."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

from acquire_source import COMMIT, file_id
from motion_io import decode_g1, describe, member_key, source_group


ARCHIVE_SHA = "7bd38634f1ec075af907a2bce10616b083a5414885674ce6630664124e22e495"
SEED = 2026102405
QUOTAS = {
    "train": {"locomotion": 32, "squat": 16, "upper": 16, "other": 16},
    "development": {"locomotion": 6, "squat": 3, "upper": 3, "other": 4},
    "test": {"locomotion": 6, "squat": 3, "upper": 3, "other": 4},
}


def resource_check():
    base = Path("/sys/fs/cgroup/memory")
    current = int((base / "memory.usage_in_bytes").read_text())
    events = dict(line.split() for line in (base / "memory.oom_control").read_text().splitlines())
    if current + 256 * 1024 ** 2 >= 30 * 1024 ** 3:
        raise RuntimeError("PHUMA scan memory protection")
    return int(events["oom_kill"])


def source_split(group, official):
    if "test" in official:
        return "test"
    if "train" not in official:
        return "unlisted"
    digest = hashlib.sha256((str(SEED) + ":" + group).encode()).hexdigest()
    return "development" if int(digest[:8], 16) % 10 == 0 else "train"


def bucket(key, description):
    if re.search(r"squat|crouch", key, re.I):
        return "squat"
    if description["max_horizontal_displacement_m"] >= .25:
        return "locomotion"
    if re.search(r"punch|reach|wave|clap|raise|arm|hand|throw|box", key, re.I):
        return "upper"
    return "other"


def scan(archive, vendor, output):
    if output.exists():
        raise FileExistsError(output)
    baseline_oom = resource_check()
    archive_identity = file_id(archive)
    if archive_identity["sha256"] != ARCHIVE_SHA:
        raise ValueError("PHUMA archive differs from the independently inspected package")
    source = json.loads((vendor / "source_receipt.json").read_text())
    if source["commit"] != COMMIT or any(file_id(x["path"]) != x for x in source["files"]):
        raise ValueError("Pinned PHUMA source assets changed")
    split_paths = {name: vendor / "data/split" / f"phuma_{name}.txt"
                   for name in ("train", "test")}
    official_sets = {name: set(path.read_text().splitlines()) for name, path in split_paths.items()}
    if official_sets["train"] & official_sets["test"]:
        raise ValueError("Official raw split files intersect")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    candidates, counts, errors = [], collections.Counter(), []
    with zipfile.ZipFile(archive) as package:
        members = [x for x in package.infolist()
                   if x.filename.startswith("data/g1/") and x.filename.endswith(".npy")]
        if len({x.filename for x in members}) != len(members):
            raise ValueError("Duplicate archive members")
        groups = collections.defaultdict(set)
        for member in members:
            key = member_key(member.filename)
            for split, keys in official_sets.items():
                if key in keys:
                    groups[source_group(key)].add(split)
        records = output / "all_chunks.jsonl"
        with records.open("x") as stream:
            for index, member in enumerate(sorted(members, key=lambda x: x.filename)):
                key = member_key(member.filename)
                group = source_group(key)
                split = source_split(group, groups[group])
                row = {"member": member.filename, "source_key": key,
                       "source_group": group, "split": split,
                       "official_group_splits": sorted(groups[group])}
                try:
                    if member.file_size > 4 * 1024 ** 2:
                        raise ValueError("Unexpected source member size")
                    raw = package.read(member)
                    motion = decode_g1(raw)
                    stats = describe(motion)
                    eligible = (stats["source_duration_s"] >= 1.98
                                and stats["max_tilt_rad"] <= .6
                                and stats["source_joint_speed_p95_rad_s"] <= 4
                                and stats["source_joint_speed_max_rad_s"] <= 12)
                    row.update({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                "source_statistics": stats, "bucket": bucket(key, stats),
                                "source_schema_valid": True,
                                "moderate_motion_candidate": eligible})
                    if eligible and split != "unlisted":
                        candidates.append(row)
                    counts["schema_valid"] += 1
                    counts[f"category/{Path(key).parts[0]}"] += 1
                    counts[f"split/{split}"] += 1
                except (ValueError, TypeError, KeyError, EOFError) as exc:
                    row.update({"source_schema_valid": False, "error": f"{type(exc).__name__}: {exc}"})
                    errors.append({"member": member.filename, "error": row["error"]})
                    counts["schema_invalid"] += 1
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if index % 1000 == 0:
                    if resource_check() != baseline_oom:
                        raise RuntimeError("OOM count changed during PHUMA scan")
                    print(json.dumps({"scanned": index + 1, "total": len(members),
                                      "schema_valid": counts["schema_valid"]}), flush=True)
    selected, used_groups, available = [], set(), {}
    for split, quotas in QUOTAS.items():
        for kind, quota in quotas.items():
            pool = [x for x in candidates if x["split"] == split and x["bucket"] == kind]
            pool.sort(key=lambda x: hashlib.sha256(
                (str(SEED) + ":" + x["member"]).encode()).hexdigest())
            available[f"{split}/{kind}"] = len({x["source_group"] for x in pool})
            chosen = 0
            for row in pool:
                if chosen >= quota:
                    break
                if row["source_group"] in used_groups:
                    continue
                selected.append(row)
                used_groups.add(row["source_group"])
                chosen += 1
    result = {
        "status": "PHUMA_G1_CORPUS_SCAN_AND_GROUPED_SELECTION_COMPLETED",
        "archive": archive_identity, "upstream_commit": COMMIT,
        "source_receipt": file_id(vendor / "source_receipt.json"),
        "official_splits": {key: file_id(value) for key, value in split_paths.items()},
        "raw_g1_members": len(members), "counts": dict(counts), "errors": errors,
        "source_groups": len(groups),
        "official_train_test_overlapping_source_groups": sum(len(x) > 1 for x in groups.values()),
        "group_rule": "Strip chunk and clip suffixes; reserve every official-test group from training.",
        "development_rule": "Fixed seed hash reserves 10 percent of remaining official-train groups.",
        "candidate_rule": {"minimum_duration_s": 1.98, "max_root_tilt_rad": .6,
                           "joint_speed_p95_cap_rad_s": 4, "joint_speed_max_cap_rad_s": 12},
        "seed": SEED, "requested_quotas": QUOTAS, "available_source_groups": available,
        "selected_counts": dict(collections.Counter(x["split"] + "/" + x["bucket"] for x in selected)),
        "selected": selected, "all_records": file_id(records),
        "elapsed_s": time.monotonic() - started, "oom_kill": resource_check(),
        "sources": [file_id(Path(__file__)), file_id(Path(__file__).with_name("motion_io.py"))],
        "qualified_for_training_pilot": False, "new_physics_steps": 0,
        "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
        "limits": "Source schema and moderate-motion selection only; Q1 retargeting and physics remain unverified.",
    }
    if result["oom_kill"] != baseline_oom:
        raise RuntimeError("OOM count changed at scan completion")
    with (output / "selection.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "selected_counts": result["selected_counts"],
                      "selection": file_id(output / "selection.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan(args.archive.resolve(), args.vendor.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
