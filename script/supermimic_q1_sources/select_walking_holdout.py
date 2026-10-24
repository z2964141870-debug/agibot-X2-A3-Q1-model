#!/usr/bin/env python3
"""Reserve unused PHUMA walking sources for a supplementary Q1 evaluation."""

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from audit_humanml_aliases import COLLECTION_CANDIDATES, HUMANML_INDEX_SHA, alias_key, original_key
from prepare_amass_body import identity


SEED = 2026102510
WALK = re.compile(r"walk", re.I)
EXCLUDE = re.compile(r"backward|treadmill|stair|run|jump|hop|crouch|climb|sit|kneel|push|pull|carry|box", re.I)


def canonical_event(key):
    path = PurePosixPath(key)
    return str(PurePosixPath(COLLECTION_CANDIDATES.get(path.parts[0], path.parts[0]), *path.parts[1:]))


def lineage(row, aliases):
    if not row["source_group"].startswith("humanml/"):
        return None
    name, mirrored = alias_key(row["source_group"])
    alias = aliases.get(name)
    if alias is None:
        return None
    key = original_key(alias["source_path"])
    candidate = canonical_event(key)
    return {"humanml_original_id": name, "mirrored_variant": mirrored,
            "humanml_index_row": alias, "original_motion_key": key,
            "exclusion_event_key": candidate,
            "directory_subject_candidate": str(PurePosixPath(candidate).parent)}


def choose(rows, previous, aliases, amass_rows):
    groups = {row["source_group"] for row in previous}
    hashes = {row["sha256"] for row in previous}
    events, directories, unresolved = set(), set(), collections.Counter()
    for row in previous:
        info = lineage(row, aliases)
        if info is None:
            unresolved[row["source_group"].split("/", 1)[0]] += 1
        else:
            events.add(info["exclusion_event_key"])
            directories.add(info["directory_subject_candidate"])
    for row in amass_rows:
        path = row["relative_path"]
        if not path.endswith("_stageii.npz"):
            raise ValueError("Unexpected previous AMASS source filename")
        key = canonical_event(path[:-len("_stageii.npz")])
        events.add(key)
        directories.add(str(PurePosixPath(key).parent))
    pool, exclusions = [], collections.Counter()
    for row in rows:
        if (not row.get("source_schema_valid") or not row.get("moderate_motion_candidate")
                or row["bucket"] != "locomotion" or row["split"] not in ("train", "development", "test")):
            continue
        info = lineage(row, aliases)
        if info is None:
            continue
        name = PurePosixPath(info["original_motion_key"]).name
        if not WALK.search(name) or EXCLUDE.search(name):
            continue
        if row["source_group"] in groups or row["sha256"] in hashes:
            exclusions["previous_group_or_exact_bytes"] += 1
        elif info["exclusion_event_key"] in events:
            exclusions["previous_original_event_candidate"] += 1
        elif info["directory_subject_candidate"] in directories:
            exclusions["previous_directory_subject_candidate"] += 1
        else:
            pool.append((row, info))
    ordered = sorted(pool, key=lambda pair: hashlib.sha256(
        (str(SEED) + "/" + pair[0]["member"]).encode()).hexdigest())
    selected, used_directories, used_hashes = [], set(), set()
    for row, info in ordered:
        directory = info["directory_subject_candidate"]
        if directory not in used_directories and row["sha256"] not in used_hashes:
            selected.append((row, info))
            used_directories.add(directory)
            used_hashes.add(row["sha256"])
    proof = {"eligible_unused_chunks": len(pool),
             "eligible_original_event_candidates": len({info["exclusion_event_key"] for _, info in pool}),
             "reserved_directory_candidates": sorted({info["directory_subject_candidate"] for _, info in pool}),
             "selected_count": len(selected), "exclusion_counts": dict(exclusions),
             "previous_unresolved_namespace_counts": dict(unresolved)}
    return selected, proof


def main(args):
    root = args.root.resolve(strict=True)
    base = root / "data/supermimic_for_q1"
    previous_paths = [base / name for name in (
        "PHUMA_P02_scan_20261024/selection.json", "PHUMA_P04_selection_20261024.json",
        "PHUMA_P04_additional_selection_20261024.json")]
    amass_path = base / "AMASS_P02_moving_candidates_20261025/selection.json"
    index = root / "data/references/humanml3d_metadata_9176e8fb_B_20261025/index.csv"
    previous_selections = [json.loads(path.read_text()) for path in previous_paths]
    full = previous_selections[0]
    scan_path = Path(full["all_records"]["path"])
    inputs = [identity(path) for path in [*previous_paths, amass_path, index, scan_path]]
    if inputs[-2]["sha256"] != HUMANML_INDEX_SHA or inputs[-1] != full["all_records"]:
        raise ValueError("Use the fixed full scan and HumanML metadata")
    with index.open(newline="") as stream:
        aliases = {}
        for row in csv.DictReader(stream):
            if row["new_name"] in aliases:
                raise ValueError("Repeated HumanML metadata identity")
            aliases[row["new_name"]] = row
    with scan_path.open() as stream:
        rows = [json.loads(line) for line in stream]
    previous = [row for selection in previous_selections for row in selection["selected"]]
    selected, proof = choose(rows, previous, aliases, json.loads(amass_path.read_text())["selected"])
    if not 1 <= len(selected) <= 32:
        raise ValueError("Expected a finite pool of one to thirty-two unused directory candidates")
    if any(identity(record["path"]) != record for record in inputs):
        raise ValueError("Selection inputs changed")
    output = args.output.resolve()
    if output.exists() or not output.parent.is_relative_to(base):
        raise ValueError("Use a new selection inside the Q1 experiment area")
    result = dict(full)
    result.update(status="PHUMA_P10_SUPPLEMENTARY_WALKING_HOLDOUT_SELECTED",
                  captured_utc=datetime.now(timezone.utc).isoformat(),
                  selected=[row for row, _ in selected],
                  selected_counts=dict(collections.Counter(row["split"] + "/" + row["bucket"] for row, _ in selected)),
                  walking_holdout={"selection_seed": SEED, "role": "SUPPLEMENTARY_TEST_ONLY_NEVER_TRAIN",
                      "source_split_preserved": True, "inputs": inputs, "proof": proof,
                      "selected_lineage": [{"member": row["member"], **info} for row, info in selected],
                      "method": "Name-filtered walking, excluding treadmill and listed interactions; seeded one chunk per directory candidate, excluding every previous selected group, byte hash, known original event and directory candidate. No Q1 geometry or policy score used for selection.",
                      "sources": [identity(__file__), identity(Path(__file__).with_name("audit_humanml_aliases.py"))]},
                  qualified_for_training_pilot=False, source_isolation_certified=False,
                  new_physics_steps=0, new_optimizer_updates=0, storage_status="LOCAL_ONLY",
                  limits="Walking names are semantic candidates only. Original PHUMA splits are immutable provenance, not the supplementary evaluation role. Collection renames only support conservative exclusion; unresolved namespaces and cross-collection identities prevent a full source-independence claim. Geometry and actual dynamics remain unverified.")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"selection": identity(output), **proof}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
