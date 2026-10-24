"""Validate an exclusion view without changing its inherited geometry contract."""

import collections
import copy
import json
import math
from pathlib import Path
import sys

from audit_humanml_aliases import overlap_groups
from filter_phuma_alias_conflicts import exclusions
from prepare_amass_body import identity


HERE = Path(__file__).resolve().parent
VIEW_STATUS = "PHUMA_SOURCE_EXCLUSION_VIEW_AND_TRAINER_ADMISSION_PASSED"
PARENT_STATUS = "PHUMA_EXPANDED_LIBRARY_ASSEMBLED_AND_TRAINER_ADMISSION_PASSED"
ALIAS_STATUS = "PHUMA_ACCEPTED_HUMANML_ALIAS_AUDIT_COMPLETED_REVIEW_REQUIRED"


def validate_documents(completed, manifest, checked, parent, parent_manifest,
                       parent_checked, old_aliases, aliases):
    provenance = completed["source_exclusion"]
    if (completed["status"] != VIEW_STATUS or parent["status"] != PARENT_STATUS
            or provenance["mode"] != "TRAIN_ONLY_KNOWN_HELDOUT_ALIAS_EXCLUSION_VIEW"
            or provenance["geometry_checks_rerun"] is not False
            or any(record["source_isolation_certified"] is not False
                   for record in (completed, provenance, old_aliases, aliases))):
        raise ValueError("An exclusion view must retain its actual status and unresolved source isolation")
    if (any(parent[key] != provenance["parent_" + key]
            for key in ("manifest", "reference_check"))
            or parent["scan_check"] != completed["scan_check"]
            or parent_checked["input_manifest"] != parent["manifest"]
            or old_aliases["inputs"][:2] != [parent["manifest"], parent["reference_check"]]
            or aliases["inputs"][:2] != [completed["manifest"], completed["reference_check"]]
            or aliases["inputs"][2:] != old_aliases["inputs"][2:]
            or aliases["source"] != old_aliases["source"]
            or old_aliases["status"] != ALIAS_STATUS or aliases["status"] != ALIAS_STATUS):
        raise ValueError("Parent geometry and both source audits must bind the same inputs")
    removed = exclusions(parent_manifest["motions"], parent_checked["qualified_clips"],
                         parent["split_clips"], old_aliases)
    if not removed or removed != provenance["excluded_train_clips"]:
        raise ValueError("Excluded training clips must exactly match known held-out alias conflicts")

    accepted = [name for name in parent_checked["qualified_clips"] if name not in removed]
    accepted_set = set(accepted)
    expected_manifest, expected_checked = copy.deepcopy((parent_manifest, parent_checked))
    for records in (expected_manifest["motions"], expected_checked["clips"]):
        for row in records:
            if row["clip"] in removed:
                row.update(qualification_before_source_exclusion=row["qualified_for_training_pilot"],
                           qualified_for_training_pilot=False,
                           source_exclusion_reasons=removed[row["clip"]])
    checks = {row["clip"]: row for row in expected_checked["clips"]}
    counts = dict(collections.Counter(checks[name]["split"] for name in accepted))
    expected_manifest.update(status="PHUMA_SOURCE_EXCLUSION_VIEW_WITH_INHERITED_GEOMETRY",
                             qualified_clips=accepted, qualified_split_counts=counts,
                             source_exclusion=provenance,
                             sources=parent_manifest["sources"] + [provenance["source"]])
    expected_checked.update(input_manifest=completed["manifest"], qualified_clips=accepted,
                            qualified_split_counts=counts, source_exclusion=provenance,
                            rejected_clips=[row["clip"] for row in expected_checked["clips"]
                                            if row["clip"] not in accepted_set])
    if manifest != expected_manifest or checked != expected_checked:
        raise ValueError("Only declared training eligibility may change; inherited geometry and references must match")
    splits = {split: sorted(name for name in accepted if checks[name]["split"] == split)
              for split in ("train", "development", "test")}
    if (completed["split_clips"] != splits or completed["qualified_count"] != len(accepted)
            or any(splits[split] != parent["split_clips"][split] for split in ("development", "test"))):
        raise ValueError("Retained split coverage and held-out lists must agree exactly")
    for name in accepted:
        gates = checks[name]["gates"]
        if len(gates) != 19 or not all(value is True for value in gates.values()):
            raise ValueError("Every retained reference must keep all original 19 gates")
    for split, names in splits.items():
        for field, selected in (("qualified_duration_s", names),
                                ("qualified_locomotion_duration_s",
                                 [name for name in names if checks[name]["bucket"] == "locomotion"])):
            duration = sum(checks[name]["duration_s"] for name in selected)
            if not math.isclose(completed[field][split], duration, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Recorded view durations differ from inherited checked references")
    expected_aliases = [row for row in old_aliases["rows"] if row["clip"] not in removed]
    if (aliases["rows"] != expected_aliases or aliases["accepted_reference_count"] != len(accepted)
            or aliases["humanml_reference_count"] != len(expected_aliases)
            or aliases["other_source_namespace_counts_unresolved"]
            != old_aliases["other_source_namespace_counts_unresolved"]):
        raise ValueError("The independent re-audit must cover exactly the retained references")
    for field, key in (("original_motion_key", "original_motion_cross_split"),
                       ("directory_subject_candidate", "directory_subject_candidate_cross_split")):
        actual = overlap_groups(aliases["rows"], field)
        if actual or aliases[key] != actual:
            raise ValueError("Known cross-split source conflicts remain in the view")
    return splits


def validated_view(root, library, aliases_path):
    root, library = root.resolve(strict=True), library.resolve(strict=True)
    if not library.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use an existing Q1 source-exclusion library")
    completion_path = library / "completion.json"
    completed = json.loads(completion_path.read_text())
    provenance = completed["source_exclusion"]
    records = [identity(completion_path), *completed["inputs"], completed["manifest"],
               completed["reference_check"], completed["scan_check"], identity(aliases_path)]
    records += [identity(HERE / name) for name in (
        "phuma_source_view.py", "prepare_amass_body.py", "audit_humanml_aliases.py")]
    if (identity(library / "references/manifest.json") != completed["manifest"]
            or identity(library / "reference_check.json") != completed["reference_check"]
            or identity(library / "scan_check.json") != completed["scan_check"]
            or provenance["source"] != identity(HERE / "filter_phuma_alias_conflicts.py")):
        raise ValueError("The view paths or its producer differ from the recorded evidence")

    def load(record):
        path = Path(record["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or identity(path) != record:
            raise ValueError("Source-view evidence changed or lies outside Q1: " + str(path))
        records.append(record)
        return json.loads(path.read_text())

    parent = load(provenance["parent_completion"])
    parent_manifest = load(provenance["parent_manifest"])
    parent_checked = load(provenance["parent_reference_check"])
    old_aliases = load(provenance["alias_audit"])
    aliases = load(identity(aliases_path))
    manifest, checked = load(completed["manifest"]), load(completed["reference_check"])
    records += [*old_aliases["inputs"], old_aliases["source"], *aliases["inputs"], aliases["source"]]
    records += [parent[key] for key in ("selection", "reconciliation", "source")]
    if aliases["source"] != identity(HERE / "audit_humanml_aliases.py"):
        raise ValueError("The independent source auditor changed")
    splits = validate_documents(completed, manifest, checked, parent, parent_manifest,
                                parent_checked, old_aliases, aliases)
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    from run_training import admitted_clips

    if admitted_clips(library / "references", library / "reference_check.json",
                      library / "scan_check.json") != splits:
        raise ValueError("Preserved trainer admission differs from the validated source view")
    unique = {}
    for record in records:
        path = Path(record["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or identity(path) != record:
            raise ValueError("Source-view input identity changed: " + str(path))
        if str(path) in unique and unique[str(path)] != record:
            raise ValueError("Inconsistent evidence identities")
        unique[str(path)] = record
    return {"status": "PHUMA_SOURCE_EXCLUSION_REGISTRATION_PREFLIGHT_PASSED",
            "library_completion": identity(completion_path), "source_aliases": identity(aliases_path),
            "split_clips": splits, "qualified_duration_s": completed["qualified_duration_s"],
            "excluded_train_clips": provenance["excluded_train_clips"],
            "source_isolation_certified": False, "geometry_checks_rerun": False,
            "evidence_files": [unique[key] for key in sorted(unique)],
            "limits": provenance["limits"]}
