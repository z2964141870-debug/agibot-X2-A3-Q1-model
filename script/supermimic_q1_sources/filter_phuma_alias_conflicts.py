#!/usr/bin/env python3
"""Create a separate PHUMA training view excluding known held-out alias conflicts."""

import argparse
import collections
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def exclusions(motions, qualified, splits, aliases):
    expected = {name: split for split, names in splits.items() for name in names}
    if (set(splits) != {"train", "development", "test"}
            or len(expected) != sum(map(len, splits.values())) or set(expected) != set(qualified)):
        raise ValueError("Require complete disjoint registered split coverage")
    source_rows = {row["clip"]: row for row in motions}
    if len(source_rows) != len(motions):
        raise ValueError("Repeated conversion identity")
    humanml = {name for name in expected
               if source_rows[name]["phuma_source"]["source_group"].startswith("humanml/")}
    rows = aliases["rows"]
    if len(rows) != len(humanml) or {row["clip"] for row in rows} != humanml:
        raise ValueError("Alias evidence must cover every admitted HumanML reference once")
    for row in rows:
        if (row["split"] != expected[row["clip"]]
                or row["source_group"] != source_rows[row["clip"]]["phuma_source"]["source_group"]):
            raise ValueError("Alias split or source differs from the admitted conversion")
    heldout = {field: {row[field] for row in rows if row["split"] != "train" and row.get(field)}
               for field in ("original_motion_key", "directory_subject_candidate")}
    removed = {}
    for row in rows:
        reasons = {field: row[field] for field, keys in heldout.items() if row.get(field) in keys}
        if row["split"] == "train" and reasons:
            removed[row["clip"]] = reasons
    return removed


def run(args):
    import numpy as np
    from guard_phuma_batch import MemoryBoundBudget
    from prepare_amass_body import identity
    from runtime_guard import StorageBudget, save

    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    from run_training import admitted_clips

    root, library, output = args.root.resolve(strict=True), args.library.resolve(strict=True), args.output.resolve()
    if (output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1")
            or not library.is_relative_to(root / "data/supermimic_for_q1")):
        raise ValueError("Require an existing Q1 library and a new independent output")
    budget = MemoryBoundBudget(StorageBudget(output, library, 100 * 1024 ** 3,
                                            64 * 1024 ** 2, 64 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    before = budget.check()
    os.nice(10)
    paths = [library / "references/manifest.json", library / "reference_check.json",
             library / "completion.json", library / "scan_check.json", args.aliases.resolve(strict=True),
             Path(__file__).resolve()]
    inputs = [identity(path) for path in paths]
    manifest, checked, completed, scan, aliases = (json.loads(path.read_text()) for path in paths[:5])
    if (completed["manifest"] != inputs[0] or completed["reference_check"] != inputs[1]
            or completed["scan_check"] != inputs[3] or checked["input_manifest"] != inputs[0]
            or aliases["inputs"][:2] != inputs[:2]
            or aliases["source"]["sha256"] != "c7c6b6d79dd88af7f3d66beba9957f97a67be8b2996737dda04f8b3b8d8e359c"
            or aliases["status"] != "PHUMA_ACCEPTED_HUMANML_ALIAS_AUDIT_COMPLETED_REVIEW_REQUIRED"):
        raise ValueError("Require original assembled library and its fixed independent alias audit")
    for record in [*aliases["inputs"], aliases["source"]]:
        if identity(record["path"]) != record:
            raise ValueError("An alias-audit input or its source code changed")
    original = admitted_clips(library / "references", paths[1], paths[3])
    if original != completed["split_clips"] or aliases["accepted_reference_count"] != sum(map(len, original.values())):
        raise ValueError("Original trainer admission differs from the assembly or alias audit")
    removed = exclusions(manifest["motions"], checked["qualified_clips"], original, aliases)
    if not removed:
        raise ValueError("No known held-out conflicts to exclude; preserve the original library")
    provenance = {"mode": "TRAIN_ONLY_KNOWN_HELDOUT_ALIAS_EXCLUSION_VIEW",
                  "parent_manifest": inputs[0], "parent_reference_check": inputs[1],
                  "parent_completion": inputs[2], "alias_audit": inputs[4], "source": inputs[5],
                  "excluded_train_clips": removed, "geometry_checks_rerun": False,
                  "source_isolation_certified": False,
                  "limits": "Original 19 geometry gates are inherited unchanged. Training eligibility additionally excludes known held-out event or directory-subject candidates. Unresolved identities remain unresolved."}
    accepted = [name for name in checked["qualified_clips"] if name not in removed]
    for rows in (manifest["motions"], checked["clips"]):
        for row in rows:
            if row["clip"] in removed:
                row["qualification_before_source_exclusion"] = row["qualified_for_training_pilot"]
                row["qualified_for_training_pilot"] = False
                row["source_exclusion_reasons"] = removed[row["clip"]]
    by_name = {row["clip"]: row for row in checked["clips"]}
    counts = dict(collections.Counter(by_name[name]["split"] for name in accepted))
    manifest.update(status="PHUMA_SOURCE_EXCLUSION_VIEW_WITH_INHERITED_GEOMETRY", qualified_clips=accepted,
                    qualified_split_counts=counts, source_exclusion=provenance,
                    sources=manifest["sources"] + [inputs[5]])
    checked.update(qualified_clips=accepted, qualified_split_counts=counts, source_exclusion=provenance,
                   rejected_clips=[row["clip"] for row in checked["clips"] if row["clip"] not in set(accepted)])
    budget.memory()
    output.mkdir()
    references = output / "references"
    references.mkdir()
    durations = {split: 0. for split in original}
    moving = dict(durations)
    manifest_rows = {row["clip"]: row for row in manifest["motions"]}
    for index, name in enumerate(accepted):
        row, check = manifest_rows[name], by_name[name]
        target = Path(row["output"]["path"]).resolve(strict=True)
        if identity(target) != row["output"] or len(check["gates"]) != 19 or not all(check["gates"].values()):
            raise ValueError("Retained reference identity or original geometry gates changed")
        with np.load(target, allow_pickle=False) as saved:
            times = saved["time_s"]
            duration = (len(times) - 1) / 50.
            if (float(saved["fps"]) != 50. or saved["qpos"].shape != (len(times), 29)
                    or not np.allclose(times, np.arange(len(times)) / 50., rtol=0, atol=1e-10)
                    or abs(duration - check["duration_s"]) > 1e-10):
                raise ValueError("Retained reference native clock or checked duration changed")
        durations[row["split"]] += duration
        if row["bucket"] == "locomotion":
            moving[row["split"]] += duration
        (references / (name + ".npz")).symlink_to(target)
        if index % 32 == 0:
            budget.check()
    save(references / "manifest.json", manifest)
    checked["input_manifest"] = identity(references / "manifest.json")
    save(output / "reference_check.json", checked)
    (output / "scan_check.json").symlink_to(paths[3])
    current = admitted_clips(references, output / "reference_check.json", output / "scan_check.json")
    if (any(current[split] != original[split] for split in ("development", "test"))
            or set(original["train"]) - set(current["train"]) != set(removed)):
        raise ValueError("The exclusion view changed held-out references or removed other training clips")
    for record in inputs:
        if identity(record["path"]) != record:
            raise ValueError("An original library input changed during view construction")
    report = {"status": "PHUMA_SOURCE_EXCLUSION_VIEW_AND_TRAINER_ADMISSION_PASSED",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
              "manifest": identity(references / "manifest.json"),
              "reference_check": identity(output / "reference_check.json"), "scan_check": inputs[3],
              "split_clips": current, "qualified_count": len(accepted),
              "qualified_duration_s": durations, "qualified_locomotion_duration_s": moving,
              "source_exclusion": provenance, "health_before": before, "health_after": budget.check(),
              "new_q1_references": 0, "new_ik_conversions": 0, "new_physics_steps": 0,
              "new_optimizer_updates": 0, "original_files_modified": False,
              "source_isolation_certified": False, "storage_status": "LOCAL_ONLY",
              "limits": provenance["limits"]}
    save(output / "completion.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("split_clips", "inputs")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    run(parser.parse_args())
