#!/usr/bin/env python3
"""Independently reconcile the PHUMA scan, group splits and selected ZIP bytes."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

from acquire_source import file_id
from motion_io import decode_g1, describe, member_key, source_group
from scan_corpus import ARCHIVE_SHA, SEED, source_split


def audit(selection_path, output):
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(selection_path.read_text())
    if file_id(selection["archive"]["path"]) != selection["archive"]:
        raise ValueError("Archive changed after scanning")
    if selection["archive"]["sha256"] != ARCHIVE_SHA or selection["seed"] != SEED:
        raise ValueError("Unexpected corpus identity or selection seed")
    if file_id(selection["all_records"]["path"]) != selection["all_records"]:
        raise ValueError("Full scan index changed")
    official = {}
    for split, record in selection["official_splits"].items():
        if file_id(record["path"]) != record:
            raise ValueError("Official split file changed")
        official[split] = set(Path(record["path"]).read_text().splitlines())
    expected_groups = collections.defaultdict(set)
    rows = {}
    counts = collections.Counter()
    with Path(selection["all_records"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = member_key(row["member"])
            if row["member"] in rows or row["source_group"] != source_group(key):
                raise ValueError("Duplicate member or incorrect source grouping")
            rows[row["member"]] = row
            for split, keys in official.items():
                if key in keys:
                    expected_groups[source_group(key)].add(split)
            counts["schema_valid" if row["source_schema_valid"] else "schema_invalid"] += 1
            counts["split/" + row["split"]] += 1
            counts["category/" + key.split("/")[0]] += 1
    for row in rows.values():
        group = row["source_group"]
        if (row["split"] != source_split(group, expected_groups[group])
                or row["official_group_splits"] != sorted(expected_groups[group])):
            raise ValueError("A source group crossed its reserved split")
    if dict(counts) != selection["counts"] or len(rows) != selection["raw_g1_members"]:
        raise ValueError("Full scan counts do not reconcile")
    groups = {split: set() for split in ("train", "development", "test")}
    with zipfile.ZipFile(selection["archive"]["path"]) as package:
        for row in selection["selected"]:
            if rows[row["member"]] != row or not row["moderate_motion_candidate"]:
                raise ValueError("Selected row was not a valid scan candidate")
            split, group = row["split"], row["source_group"]
            if any(group in used for used in groups.values()):
                raise ValueError("Repeated source group in candidate selection")
            groups[split].add(group)
            raw = package.read(row["member"])
            if (len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]
                    or describe(decode_g1(raw)) != row["source_statistics"]):
                raise ValueError("Candidate source bytes or recomputed statistics changed")
    report = {"status": "PHUMA_SCAN_INDEPENDENT_RECONCILIATION_PASSED",
              "selection": file_id(selection_path), "all_records": selection["all_records"],
              "recomputed_counts": dict(counts), "candidate_source_groups": {k: len(v) for k, v in groups.items()},
              "candidate_pairwise_group_intersections": 0,
              "candidate_bytes_redecoded": len(selection["selected"]),
              "official_group_overlap_reserved_for_test": sum(len(x) > 1 for x in expected_groups.values()),
              "source": file_id(Path(__file__)), "new_physics_steps": 0,
              "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
              "limits": "Grouping is by named source stem, not a subject-disjoint or semantic deduplication claim."}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.selection.resolve(), args.output.resolve())
