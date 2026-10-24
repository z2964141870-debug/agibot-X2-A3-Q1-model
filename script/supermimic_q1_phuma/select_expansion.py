#!/usr/bin/env python3
"""Select new, grouped PHUMA candidates from the already verified full index."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

from acquire_source import file_id
from scan_corpus import ARCHIVE_SHA, SEED


EXPANSION_SEED = 2026102407
SPLITS = ("train", "development", "test")
PRIORITY = {"unlisted": 0, "train": 1, "development": 2, "test": 3}
QUOTAS = {
    "train": {"locomotion": 1024, "squat": 256, "upper": 256, "other": 512},
    "development": {"locomotion": 16, "squat": 4, "upper": 4, "other": 8},
    "test": {"locomotion": 16, "squat": 4, "upper": 4, "other": 8},
}
INTERACTION = re.compile(r"(?:^|[/_ -])(stairs?|ladder|chair|climb|sitting|seated|crawl)(?:$|[/_ -])", re.I)


def choose(rows, previous, quotas=QUOTAS):
    blocked = {row["source_group"] for row in previous}
    hash_splits = collections.defaultdict(set)
    for row in rows:
        if row.get("source_schema_valid"):
            hash_splits[row["sha256"]].add(row["split"])
    pools = collections.defaultdict(list)
    excluded = collections.Counter()
    for row in rows:
        if not row.get("source_schema_valid") or not row.get("moderate_motion_candidate"):
            excluded["source_schema_or_moderate_filter"] += 1
        elif row["split"] not in SPLITS:
            excluded["unlisted"] += 1
        elif row["source_group"] in blocked:
            excluded["previous_candidate_group"] += 1
        elif PRIORITY[row["split"]] != max(PRIORITY[x] for x in hash_splits[row["sha256"]]):
            excluded["exact_content_reserved_for_higher_priority_split"] += 1
        elif INTERACTION.search(row["source_key"]):
            excluded["named_environment_interaction"] += 1
        else:
            pools[row["split"], row["bucket"]].append(row)
    selected, groups, hashes, available = [], set(), set(), {}
    for split in SPLITS:
        for bucket, quota in quotas[split].items():
            pool = pools[split, bucket]
            pool.sort(key=lambda row: hashlib.sha256(
                f"{EXPANSION_SEED}:{row['member']}".encode()).hexdigest())
            available[f"{split}/{bucket}"] = len({row["source_group"] for row in pool})
            used = 0
            for row in pool:
                if used >= quota:
                    break
                if row["source_group"] in groups or row["sha256"] in hashes:
                    continue
                selected.append(row)
                groups.add(row["source_group"])
                hashes.add(row["sha256"])
                used += 1
    # Reserve the evaluation candidates first; interleave training categories in each batch.
    training = {kind: [x for x in selected if x["split"] == "train" and x["bucket"] == kind]
                for kind in quotas["train"]}
    selected = [x for x in selected if x["split"] != "train"]
    for index in range(max([0] + [len(x) for x in training.values()])):
        for pool in training.values():
            if index < len(pool):
                selected.append(pool[index])
    proof = {
        "excluded_counts": dict(excluded),
        "available_source_groups": available,
        "previous_source_groups_excluded": len(blocked),
        "exact_content_hashes_crossing_source_splits": sum(len(x) > 1 for x in hash_splits.values()),
        "selected_unique_source_groups": len(groups),
        "selected_unique_content_hashes": len(hashes),
        "selected_counts": dict(collections.Counter(x["split"] + "/" + x["bucket"] for x in selected)),
        "nominal_selected_seconds": {
            split: sum(x["source_statistics"]["source_duration_s"] for x in selected if x["split"] == split)
            for split in SPLITS
        },
    }
    return selected, proof


def select(base_path, output):
    if output.exists():
        raise FileExistsError(output)
    base = json.loads(base_path.read_text())
    if base["archive"]["sha256"] != ARCHIVE_SHA or base["seed"] != SEED:
        raise ValueError("Unexpected original PHUMA scan")
    if file_id(base["all_records"]["path"]) != base["all_records"]:
        raise ValueError("PHUMA full index changed")
    with Path(base["all_records"]["path"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != base["raw_g1_members"]:
        raise ValueError("Full index record count changed")
    selected, proof = choose(rows, base["selected"])
    result = dict(base)
    result.update({
        "status": "PHUMA_P04_GROUPED_EXPANSION_SELECTION_COMPLETED",
        "selected": selected, "selected_counts": proof["selected_counts"],
        "requested_quotas": QUOTAS, "available_source_groups": proof["available_source_groups"],
        "expansion": {
            "base_selection": file_id(base_path), "selection_seed": EXPANSION_SEED,
            "deduplication": "One chunk per named source group and exact source-file hash; reserve identical bytes for test, then development, then train.",
            "previous_groups": "All 112 previous candidate groups excluded from new conversion; existing R01 inputs remain unchanged.",
            "interaction_name_pattern": INTERACTION.pattern, "proof": proof,
        },
        "sources": [file_id(__file__)], "qualified_for_training_pilot": False,
        "new_physics_steps": 0, "new_optimizer_updates": 0,
        "limits": "Source candidates only. Nominal duration is not Q1-qualified duration; name/exact-byte deduplication does not prove semantic or subject independence.",
    })
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"selection": file_id(output), **proof}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select(args.base_selection.resolve(), args.output.resolve())
