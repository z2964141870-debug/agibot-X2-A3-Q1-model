#!/usr/bin/env python3
"""Fix another finite PHUMA training tranche while preserving held-out selections."""

import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))

from acquire_source import file_id
from scan_corpus import ARCHIVE_SHA, SEED, resource_check
from select_expansion import EXPANSION_SEED, INTERACTION, choose


def select(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed before additional selection")
    base = json.loads(args.base_selection.read_text())
    previous = json.loads(args.exclude_selection.read_text())
    if (base["archive"]["sha256"] != ARCHIVE_SHA or base["seed"] != SEED
            or any(previous[key] != base[key] for key in ("archive", "all_records", "official_splits", "seed"))):
        raise ValueError("Additional candidates must use the same fixed corpus and split protocol")
    if file_id(base["all_records"]["path"]) != base["all_records"]:
        raise ValueError("Full source index changed")
    with Path(base["all_records"]["path"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed while reading the source index")
    excluded = base["selected"] + previous["selected"]
    quotas = {"train": {"locomotion": 512, "other": 512}, "development": {}, "test": {}}
    selected, proof = choose(rows, excluded, quotas)
    previous_hashes = {row["sha256"] for row in excluded}
    removed = sum(row["sha256"] in previous_hashes for row in selected)
    selected = [row for row in selected if row["sha256"] not in previous_hashes]
    proof.update({"exact_previous_bytes_removed": removed,
                  "selected_unique_source_groups": len({row["source_group"] for row in selected}),
                  "selected_unique_content_hashes": len({row["sha256"] for row in selected}),
                  "selected_counts": dict(collections.Counter(row["split"] + "/" + row["bucket"] for row in selected)),
                  "nominal_selected_seconds": {split: sum(row["source_statistics"]["source_duration_s"]
                        for row in selected if row["split"] == split) for split in ("train", "development", "test")}})
    if (not selected or any(row["split"] != "train" for row in selected)
            or len(selected) != proof["selected_unique_source_groups"]
            or len(selected) != proof["selected_unique_content_hashes"]):
        raise ValueError("Additional tranche is empty, duplicated or changes held-out inputs")
    result = dict(base)
    result.update({"status": "PHUMA_P04_GROUPED_EXPANSION_SELECTION_COMPLETED",
                   "selected": selected, "selected_counts": proof["selected_counts"],
                   "requested_quotas": quotas, "available_source_groups": proof["available_source_groups"],
                   "expansion": {"base_selection": file_id(args.base_selection),
                       "excluded_expansion_selection": file_id(args.exclude_selection),
                       "selection_seed": EXPANSION_SEED, "proof": proof,
                       "interaction_name_pattern": INTERACTION.pattern,
                       "deduplication": "Prior 112 and full P04 named groups and exact bytes excluded; held-out candidates unchanged."},
                   "sources": [file_id(__file__), file_id(HERE.parent / "supermimic_q1_phuma/select_expansion.py")],
                   "qualified_for_training_pilot": False, "new_physics_steps": 0, "new_optimizer_updates": 0,
                   "oom_kill": resource_check(),
                   "limits": "New fixed training candidates only; nominal duration is not Q1-qualified duration. Existing development/test selections are unchanged."})
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"selection": file_id(args.output), **proof}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    select(parser.parse_args())
