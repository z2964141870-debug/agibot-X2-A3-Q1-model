#!/usr/bin/env python3
"""Prepare a separate PHUMA view excluding held-out GRAB directory candidates."""

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

from audit_humanml_aliases import COLLECTION_CANDIDATES
from phuma_source_view import identity, validated_view


HERE = Path(__file__).resolve().parent
HOLDOUT_SHA = "1b23939b731dcdfedc8b27732e4f22f92a4faf07412dff0dbea8529e6d969ba9"
VIEW_STATUS = "PHUMA_DIRECTORY_EXCLUSION_VIEW_AND_TRAINER_ADMISSION_PASSED"


def grab_directory(group):
    path = PurePosixPath(group)
    if (not path.is_absolute() and ".." not in path.parts and len(path.parts) == 3
            and path.parts[0] == "GRAB" and re.fullmatch(r"s[0-9]+", path.parts[1])):
        return str(path.parent)
    return None


def exclusions(manifest, checked, splits):
    expected = {name: split for split, names in splits.items() for name in names}
    motions = {row["clip"]: row for row in manifest["motions"]}
    checks = {row["clip"]: row for row in checked["clips"]}
    qualified = checked["qualified_clips"]
    if (set(splits) != {"train", "development", "test"}
            or len(expected) != sum(map(len, splits.values()))
            or len(motions) != len(manifest["motions"]) or len(checks) != len(checked["clips"])
            or set(motions) != set(checks) or len(qualified) != len(set(qualified))
            or set(qualified) != set(expected) or manifest["qualified_clips"] != qualified):
        raise ValueError("Require complete unique parent conversion, checks and registered splits")
    groups, hashes = set(), set()
    for name, check in checks.items():
        motion = motions[name]
        if check["qualified_for_training_pilot"] != (name in expected):
            raise ValueError("Parent qualification disagrees with its accepted list")
        if name not in expected:
            continue
        source = motion["phuma_source"]
        if (source["split"] != motion["split"] or motion["split"] != check["split"]
                or check["split"] != expected[name] or source["source_group"] != check["source_group"]
                or not motion["qualified_for_training_pilot"]
                or check["reference"] != motion["output"]
                or len(check["gates"]) != 19 or not all(v is True for v in check["gates"].values())
                or not math.isfinite(check["duration_s"]) or check["duration_s"] <= 0):
            raise ValueError("Parent qualification, source, reference or inherited geometry changed")
        if source["source_group"] in groups or source["sha256"] in hashes:
            raise ValueError("Repeated parent source group or exact source bytes")
        groups.add(source["source_group"])
        hashes.add(source["sha256"])
    heldout = {directory for name, split in expected.items() if split != "train"
               and (directory := grab_directory(motions[name]["phuma_source"]["source_group"]))}
    return {name: {"grab_named_directory_candidate": directory}
            for name, split in expected.items() if split == "train"
            and (directory := grab_directory(motions[name]["phuma_source"]["source_group"])) in heldout}


def canonical_event(key):
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError("Unexpected original-event candidate")
    return str(PurePosixPath(COLLECTION_CANDIDATES.get(path.parts[0], path.parts[0]), *path.parts[1:]))


def check_reserved_sources(manifest, splits, aliases, holdout):
    contract = holdout["walking_holdout"]
    if (holdout["status"] != "PHUMA_P10_SUPPLEMENTARY_WALKING_HOLDOUT_SELECTED"
            or contract["role"] != "SUPPLEMENTARY_TEST_ONLY_NEVER_TRAIN"
            or contract["source_split_preserved"] is not True):
        raise ValueError("Require the preserved supplementary walking-test selection")
    accepted = {name for names in splits.values() for name in names}
    source_rows = {row["clip"]: row["phuma_source"] for row in manifest["motions"] if row["clip"] in accepted}
    rows = {row["clip"]: row for row in aliases["rows"]}
    humanml = {name for name, row in source_rows.items() if row["source_group"].startswith("humanml/")}
    if set(rows) != humanml or len(rows) != len(aliases["rows"]):
        raise ValueError("Require complete parent HumanML alias coverage")
    reserved_groups = {row["source_group"] for row in holdout["selected"]}
    reserved_hashes = {row["sha256"] for row in holdout["selected"]}
    lineage = contract["selected_lineage"]
    if ({row["member"] for row in lineage} != {row["member"] for row in holdout["selected"]}
            or len(lineage) != len(holdout["selected"])):
        raise ValueError("Reserved walking lineage is incomplete")
    reserved_events = {canonical_event(row["original_motion_key"]) for row in lineage}
    reserved_directories = {str(PurePosixPath(key).parent) for key in reserved_events}
    for name, source in source_rows.items():
        if source["source_group"] in reserved_groups or source["sha256"] in reserved_hashes:
            raise ValueError("A reserved walking source group or exact byte identity entered the library")
        if name in rows:
            alias = rows[name]
            if alias["source_group"] != source["source_group"]:
                raise ValueError("Parent HumanML alias source changed")
            if alias.get("original_motion_key") is not None:
                key = canonical_event(alias["original_motion_key"])
                if key in reserved_events or str(PurePosixPath(key).parent) in reserved_directories:
                    raise ValueError("A reserved walking event or directory candidate entered the library")
    return {"status": "P10_RESERVED_SOURCES_EXCLUDED_FROM_ENTIRE_PARENT_VIEW",
            "parent_references_checked": len(accepted), "humanml_aliases_checked": len(rows),
            "reserved_clips": len(lineage), "reserved_directory_candidates": len(reserved_directories)}


def documents(manifest, checked, parent, provenance):
    if (provenance["mode"] != "TRAIN_ONLY_GRAB_NAMED_DIRECTORY_CANDIDATE_EXCLUSION"
            or provenance["source_isolation_certified"] is not False
            or provenance["geometry_checks_rerun"] is not False):
        raise ValueError("Keep the named-directory scope and unresolved inherited-geometry status")
    removed = exclusions(manifest, checked, parent["split_clips"])
    if not removed or removed != provenance["excluded_train_clips"]:
        raise ValueError("Require exactly the independently derived training-directory exclusions")
    view, checks = copy.deepcopy((manifest, checked))
    for records in (view["motions"], checks["clips"]):
        for row in records:
            if row["clip"] in removed:
                row.update(qualification_before_directory_exclusion=row["qualified_for_training_pilot"],
                           qualified_for_training_pilot=False,
                           directory_source_exclusion_reasons=removed[row["clip"]])
    accepted = [name for name in checked["qualified_clips"] if name not in removed]
    accepted_set = set(accepted)
    by_name = {row["clip"]: row for row in checked["clips"]}
    splits = {split: sorted(name for name in accepted if by_name[name]["split"] == split)
              for split in ("train", "development", "test")}
    counts = {split: len(names) for split, names in splits.items()}
    view.update(status="PHUMA_DIRECTORY_EXCLUSION_VIEW_WITH_INHERITED_GEOMETRY", qualified_clips=accepted,
                qualified_split_counts=counts, directory_exclusion=provenance,
                sources=manifest["sources"] + [provenance["source"]])
    checks.update(qualified_clips=accepted, qualified_split_counts=counts, directory_exclusion=provenance,
                  rejected_clips=[row["clip"] for row in checks["clips"] if row["clip"] not in accepted_set])
    if (any(splits[s] != parent["split_clips"][s] for s in ("development", "test"))
            or exclusions(view, checks, splits)):
        raise ValueError("Held-out coverage changed or a named directory conflict remains")
    return view, checks, splits


def run(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, save

    root, library = args.root.resolve(strict=True), args.library.resolve(strict=True)
    if not library.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use the existing Q1 source-exclusion library")
    budget = MemoryBoundBudget(StorageBudget(library, library, 100 * 1024 ** 3,
                                            512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    before = budget.memory() if args.preview else budget.check()
    evidence = validated_view(root, library, args.source_aliases)
    paths = [library / "references/manifest.json", library / "reference_check.json",
             library / "completion.json", args.source_aliases.resolve(strict=True),
             args.walking_holdout.resolve(strict=True)]
    inputs = [identity(path) for path in paths]
    if inputs[-1]["sha256"] != HOLDOUT_SHA:
        raise ValueError("Use the fixed P10 supplementary walking selection")
    manifest, checked, parent, aliases, holdout = (json.loads(path.read_text()) for path in paths)
    if (evidence["library_completion"] != inputs[2]
            or parent["manifest"] != inputs[0] or parent["reference_check"] != inputs[1]):
        raise ValueError("Parent changed after source-view validation")
    reserved = check_reserved_sources(manifest, parent["split_clips"], aliases, holdout)
    removed = exclusions(manifest, checked, parent["split_clips"])
    if not removed:
        raise ValueError("No named-directory conflicts; retain the existing library")
    by_name = {row["clip"]: row for row in checked["clips"]}
    preview = {"status": "PHUMA_DIRECTORY_EXCLUSION_PREVIEW_ONLY_NOT_A_LIBRARY",
               "captured_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
               "parent_validation": evidence["status"], "parent_split_counts": {s: len(v) for s, v in parent["split_clips"].items()},
               "excluded_train_clips": removed, "excluded_duration_s_from_check": sum(by_name[n]["duration_s"] for n in removed),
               "proposed_train_count": len(parent["split_clips"]["train"]) - len(removed),
               "proposed_train_duration_s_from_check": sum(row["duration_s"] for row in checked["clips"]
                   if row["qualified_for_training_pilot"] and row["split"] == "train" and row["clip"] not in removed),
               "walking_reservation": reserved, "source_isolation_certified": False,
               "new_library_created": False, "new_physics_steps": 0, "new_optimizer_updates": 0}
    if any(identity(path) != record for path, record in zip(paths, inputs)):
        raise ValueError("Parent documents changed during exclusion derivation")
    if args.preview:
        preview.update(health_before=before, health_after=budget.memory(), new_server_files=0)
        print(json.dumps(preview, allow_nan=False), flush=True)
        return preview
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent Q1 directory view")
    producer = identity(Path(__file__).resolve())
    provenance = {"mode": "TRAIN_ONLY_GRAB_NAMED_DIRECTORY_CANDIDATE_EXCLUSION",
                  "parent_manifest": inputs[0], "parent_reference_check": inputs[1], "parent_completion": inputs[2],
                  "parent_alias_audit": inputs[3], "walking_holdout": inputs[4], "source": producer,
                  "excluded_train_clips": removed, "walking_reservation": reserved,
                  "source_isolation_certified": False, "geometry_checks_rerun": False,
                  "limits": "Additional conservative GRAB directory exclusions on the independently validated HumanML view. Named directories are not biological identities. Original geometry and held-out lists are inherited; full cross-source independence and actual tracking remain unverified."}
    view, checks, splits = documents(manifest, checked, parent, provenance)
    output.mkdir()
    budget.storage = StorageBudget(output, library, 100 * 1024 ** 3, 64 * 1024 ** 2, 512 * 1024 ** 2)
    budget.check()
    references = output / "references"
    references.mkdir()
    import numpy as np
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    from run_training import admitted_clips
    motions = {row["clip"]: row for row in view["motions"]}
    durations = {s: 0. for s in splits}
    moving = dict(durations)
    for index, name in enumerate(view["qualified_clips"]):
        row = motions[name]
        target = Path(row["output"]["path"]).resolve(strict=True)
        if identity(target) != row["output"]:
            raise ValueError("A retained Q1 reference changed")
        with np.load(target, allow_pickle=False) as saved:
            times = saved["time_s"]
            duration = (len(times) - 1) / 50.
            if (float(saved["fps"]) != 50. or saved["qpos"].shape != (len(times), 29)
                    or not np.allclose(times, np.arange(len(times)) / 50., rtol=0, atol=1e-10)
                    or abs(duration - by_name[name]["duration_s"]) > 1e-10):
                raise ValueError("A retained reference clock changed")
        durations[row["split"]] += duration
        if row["bucket"] == "locomotion":
            moving[row["split"]] += duration
        (references / (name + ".npz")).symlink_to(target)
        if index % 32 == 0:
            budget.check()
    save(references / "manifest.json", view)
    checks["input_manifest"] = identity(references / "manifest.json")
    save(output / "reference_check.json", checks)
    (output / "scan_check.json").symlink_to(Path(parent["scan_check"]["path"]))
    if admitted_clips(references, output / "reference_check.json", output / "scan_check.json") != splits:
        raise ValueError("The preserved trainer disagrees with the exclusion view")
    for record in [*inputs, producer, *evidence["evidence_files"]]:
        if identity(record["path"]) != record:
            raise ValueError("An original input changed during view construction")
    report = {"status": VIEW_STATUS, "captured_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": inputs, "manifest": identity(references / "manifest.json"),
              "reference_check": identity(output / "reference_check.json"), "scan_check": identity(output / "scan_check.json"),
              "directory_exclusion": provenance, "parent_validation": evidence, "split_clips": splits,
              "qualified_count": sum(map(len, splits.values())), "qualified_duration_s": durations,
              "qualified_locomotion_duration_s": moving, "health_before": before, "health_after": budget.check(),
              "source_isolation_certified": False, "new_ik_conversions": 0, "new_physics_steps": 0,
              "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    save(output / "completion.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("parent_validation", "split_clips")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source-aliases", type=Path, required=True)
    parser.add_argument("--walking-holdout", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--preview", action="store_true", help="Read-only validation and exclusion arithmetic; creates no library")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preview == (args.output is not None):
        parser.error("Choose --preview or a new --output directory")
    run(args)
