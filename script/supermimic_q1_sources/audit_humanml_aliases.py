#!/usr/bin/env python3
"""Audit accepted PHUMA HumanML aliases against the fixed AMASS header index."""

import argparse
import collections
import csv
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re

from prepare_amass_body import identity


COLLECTION_CANDIDATES = {"BioMotionLab_NTroje": "BMLrub", "DFaust_67": "DFaust",
                         "MPI_HDM05": "HDM05", "MPI_Limits": "PosePrior", "MPI_mosh": "MoSh",
                         "SSM_synced": "SSM", "Transitions_mocap": "Transitions"}
HUMANML_INDEX_SHA = "80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065"
AMASS_INDEX_SHA = "501e723765a71167e7186f90df0791b506191cb83fbe98a103810d86caf2a233"


def alias_key(source_group):
    match = re.fullmatch(r"humanml/(M?)([0-9]{6})", source_group)
    if match is None:
        raise ValueError("Require an explicit six-digit HumanML identity with optional M mirror prefix")
    return match.group(2) + ".npy", bool(match.group(1))


def original_key(source_path):
    path = PurePosixPath(source_path)
    if (path.is_absolute() or ".." in path.parts or len(path.parts) < 4
            or path.parts[0] != "pose_data" or not path.name.endswith("_poses.npy")):
        raise ValueError("Unexpected HumanML original source path")
    return str(PurePosixPath(*path.parts[1:]).with_name(path.name[:-len("_poses.npy")]))


def indexed_match(key, inventory):
    expected = key + "_stageii.npz"
    if expected in inventory:
        return expected, "EXACT_RELATIVE_EVENT_NAME_NOT_BYTE_IDENTITY"
    path = PurePosixPath(expected)
    renamed = COLLECTION_CANDIDATES.get(path.parts[0])
    if renamed:
        candidate = str(PurePosixPath(renamed, *path.parts[1:]))
        if candidate in inventory:
            return candidate, "COLLECTION_RENAME_CANDIDATE_EXCLUSION_ONLY"
    return None, "UNRESOLVED"


def overlap_groups(rows, field):
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[row[field]].append(row)
    return [{"key": key, "splits": sorted({row["split"] for row in members}),
             "clips": [row["clip"] for row in members]}
            for key, members in sorted(groups.items()) if len({row["split"] for row in members}) > 1]


def run(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, save

    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new source-audit directory")
    budget = MemoryBoundBudget(StorageBudget(output, args.library.resolve(strict=True),
                                            100 * 1024 ** 3, 16 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    before = budget.memory()
    manifest_path, checked_path = args.library / "references/manifest.json", args.library / "reference_check.json"
    inputs = [identity(path) for path in (manifest_path, checked_path, args.humanml_index, args.amass_index)]
    if inputs[2]["sha256"] != HUMANML_INDEX_SHA or inputs[3]["sha256"] != AMASS_INDEX_SHA:
        raise ValueError("Use the received fixed HumanML index and completed AMASS header inventory")
    manifest, checked = (json.loads(path.read_text()) for path in (manifest_path, checked_path))
    if checked["input_manifest"] != inputs[0]:
        raise ValueError("Independent geometry receipt does not bind this PHUMA library")
    accepted = {row["clip"]: row for row in checked["clips"] if row["qualified_for_training_pilot"]}
    if len(accepted) != len(checked["qualified_clips"]) or set(accepted) != set(checked["qualified_clips"]):
        raise ValueError("Accepted references disagree with the complete independent receipt")
    with args.humanml_index.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["source_path", "start_frame", "end_frame", "new_name"]:
            raise ValueError("HumanML metadata schema differs from the fixed index")
        aliases = {}
        for row in reader:
            if row["new_name"] in aliases:
                raise ValueError("Repeated HumanML index identity")
            aliases[row["new_name"]] = row
    with args.amass_index.open() as stream:
        inventory = {row["relative_path"]: row for line in stream if (row := json.loads(line)).get("schema_valid")
                     and row.get("kind") == "smplx_motion"}
    budget.memory()
    rows, other_sources = [], collections.Counter()
    for motion in manifest["motions"]:
        if motion["clip"] not in accepted:
            continue
        source = motion["phuma_source"]
        if (source["source_group"] != accepted[motion["clip"]]["source_group"]
                or motion["split"] != accepted[motion["clip"]]["split"]):
            raise ValueError("Accepted clip source or split differs from its conversion")
        if not source["source_group"].startswith("humanml/"):
            other_sources[source["source_group"].split("/", 1)[0]] += 1
            continue
        name, mirrored = alias_key(source["source_group"])
        alias = aliases.get(name)
        row = {"clip": motion["clip"], "split": motion["split"], "bucket": motion["bucket"],
               "source_group": source["source_group"], "humanml_original_id": name,
               "mirrored_variant": mirrored, "source_member": source["member"],
               "original_motion_key": None, "directory_subject_candidate": None}
        if alias is not None:
            key = original_key(alias["source_path"])
            match, status = indexed_match(key, inventory)
            candidate = match[:-len("_stageii.npz")] if match else key
            row.update(humanml_index_row=alias, original_motion_key=key,
                       amass_candidate=match, amass_match_status=status,
                       exclusion_event_key=candidate,
                       directory_subject_candidate=str(PurePosixPath(candidate).parent))
            if match:
                row["amass_header"] = {key: inventory[match][key] for key in
                                       ("source_hz", "source_frames", "gender", "bytes", "mtime_ns")}
        else:
            row["amass_match_status"] = "HUMANML_ID_MISSING"
        rows.append(row)
    if len(rows) + sum(other_sources.values()) != len(accepted):
        raise ValueError("Accepted clip join is not complete")
    if any(identity(row["path"]) != row for row in inputs):
        raise ValueError("A source index or library receipt changed during audit")
    after = budget.memory()
    report = {"status": "PHUMA_ACCEPTED_HUMANML_ALIAS_AUDIT_COMPLETED_REVIEW_REQUIRED",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
              "accepted_reference_count": len(accepted), "humanml_reference_count": len(rows),
              "humanml_split_counts": dict(collections.Counter(row["split"] for row in rows)),
              "other_source_namespace_counts_unresolved": dict(other_sources),
              "match_status_counts": dict(collections.Counter(row["amass_match_status"] for row in rows)),
              "original_motion_cross_split": overlap_groups(rows, "original_motion_key"),
              "directory_subject_candidate_cross_split": overlap_groups(rows, "directory_subject_candidate"),
              "rows": rows, "source": identity(Path(__file__)), "health_before": before, "health_after": after,
              "new_q1_references": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "source_isolation_certified": False, "storage_status": "LOCAL_ONLY",
              "limits": "Source-path event-name reconciliation for accepted HumanML references only. Stage-I/II fits may have different bytes; candidate collection renames are exclusion evidence only. CSV frame ranges are opaque metadata, not asserted to be native-120Hz frames. Directory subject candidates do not verify biological identity or cross-collection identity. Other PHUMA namespaces, semantic duplicates and final held-out split assignment remain unresolved."}
    output.mkdir()
    save(output / "receipt.json", report)
    budget.check()
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--humanml-index", type=Path, required=True)
    parser.add_argument("--amass-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    run(parser.parse_args())
