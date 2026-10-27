"""Prepare an independent, bounded SOMA expansion without reconverting AP02."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

from resources import identity, save, stamp

QUOTAS = {"walking": 250, "jogging": 150, "turning": 100,
          "gesture": 300, "dancing": 150, "jumping": 50}


def select(args):
    import pandas as pd

    sys.path.insert(0, str(args.runtime))
    from tools.prepare_soma_pilot import EXCLUDE

    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "source_bvh"
    destination.mkdir()
    previous = json.loads((args.previous / "pilot/selection.json").read_text())
    old_library = json.loads((args.previous / "q1_candidates/pilot_manifest.json").read_text())
    old_names = {r["name"] for rows in old_library["splits"].values() for r in rows}
    old_records = [r for r in previous["clips"] if r["name"] in old_names]
    frame = pd.read_parquet(args.metadata)
    eligible = frame[(~frame.is_mirror) & (frame.is_neutral == 1) & (frame.content_props == "0")
                     & frame.move_duration_frames.between(240, 1200)
                     & frame.content_type_of_movement.isin(QUOTAS)
                     & frame.content_body_position.isin(["standing", "crouching"])
                     & ~frame.filename.str.contains(EXCLUDE, regex=True)]
    lookup = {r["move_soma_uniform_path"]: r for r in eligible.to_dict("records")}
    counts = Counter(r["group"] for r in old_records if r["split"] == "train")
    actors = Counter(r["actor_uid"] for r in old_records if r["split"] == "train")
    validation_actors = {r["actor_uid"] for r in old_records if r["split"] == "validation"}
    seen_names = set(old_names)
    seen_source_sha = {r["sha256"] for r in old_records}
    added = []
    last_report = time.monotonic()
    with args.archive.open("rb") as archive_file, tarfile.open(fileobj=archive_file, mode="r|gz") as archive:
        for entry_count, entry in enumerate(archive, 1):
            if time.monotonic() - last_report > 20:
                print(json.dumps({"phase": "streaming_SOMA", "entries": entry_count,
                                  "counts": dict(counts), "added": len(added)}), flush=True)
                last_report = time.monotonic()
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(archive_file.fileno(), 0, archive_file.tell(), os.POSIX_FADV_DONTNEED)
            if not entry.isfile() or entry.name not in lookup:
                continue
            row = lookup[entry.name]
            name, actor, group = row["filename"], str(row["actor_uid"]), row["content_type_of_movement"]
            actor_train = int(hashlib.sha256(actor.encode()).hexdigest()[:8], 16) % 5 != 0
            if (not actor_train or actor in validation_actors or name in seen_names
                    or counts[group] >= QUOTAS[group] or actors[actor] >= 4):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("Unexpected source identity")
            payload = archive.extractfile(entry).read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen_source_sha:
                continue
            (destination / (name + ".bvh")).write_bytes(payload)
            added.append({"name": name, "archive_path": entry.name, "sha256": digest,
                          "split": "train", "actor_uid": actor, "group": group,
                          "frames_120hz": int(row["move_duration_frames"]),
                          "description": row["content_natural_desc_4"]})
            counts[group] += 1
            actors[actor] += 1
            seen_names.add(name)
            seen_source_sha.add(digest)
            if all(counts[g] >= n for g, n in QUOTAS.items()):
                break
    save(args.output / "selection.json", {
        "status": "AP03_ADDITIONS_SELECTED", "captured_utc": stamp(),
        "previous_library": identity(args.previous / "q1_candidates/pilot_manifest.json"),
        "archive": str(args.archive), "archive_sha256_previously_verified": previous.get("archive_sha256_previously_verified", previous.get("archive_sha256")),
        "metadata": identity(args.metadata), "quotas": QUOTAS, "counts_including_original300": dict(counts),
        "deficits": {g: n - counts[g] for g, n in QUOTAS.items() if counts[g] < n},
        "clips": added, "retained_original_names": sorted(old_names),
        "selection": "AP02 unchanged; train actor-hash split; <=4 clips/actor across AP02+AP03; HP metadata filters, unique names and source hashes. No physical performance admission gate."})
    print(json.dumps({"added": len(added), "counts": dict(counts)}), flush=True)


def convert(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supermimic_q1_hp"))
    from retarget_soma import process

    selection = json.loads((args.output / "selection.json").read_text())
    destination = args.output / "q1_candidates"
    destination.mkdir(exist_ok=False)
    jobs = [(args.runtime, args.output, args.canonical, args.gmr, destination, 4, True, r)
            for r in selection["clips"]]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for record in pool.map(process, jobs):
            results.append(record)
            save(destination / "retarget_report.json", results)
    finalize(args)


def finalize(args):
    selection = json.loads((args.output / "selection.json").read_text())
    destination = args.output / "q1_candidates"
    results = json.loads((destination / "retarget_report.json").read_text())
    previous = json.loads((args.previous / "q1_candidates/pilot_manifest.json").read_text())
    additions = [r for r in results if r["status"] == "READABLE_FINITE_Q1"]
    old_train, validation = previous["splits"]["train"], previous["splits"]["validation"]
    train = old_train + additions
    if len(train) > 1000 or {r["actor_uid"] for r in train} & {r["actor_uid"] for r in validation}:
        raise ValueError("AP03 count or actor separation contract failed")
    if len({r["sha256"] for r in train + validation}) != len(train + validation):
        raise ValueError("Duplicate converted content in AP03")
    save(destination / "pilot_manifest.json", {
        **previous, "status": "Q1_AP03_LIBRARY_READY", "captured_utc": stamp(),
        "requested_quotas": {"train": QUOTAS, "validation": previous["requested_quotas"]["validation"]},
        "source_selection_sha256": identity(args.output / "selection.json")["sha256"],
        "previous_library": identity(args.previous / "q1_candidates/pilot_manifest.json"),
        "splits": {"train": train, "validation": validation}, "counts": {"train": len(train), "validation": len(validation)},
        "original_train_names": [r["name"] for r in old_train], "added_train_names": [r["name"] for r in additions],
        "actors": {"train": sorted({r["actor_uid"] for r in train}), "validation": previous["actors"]["validation"]},
        "processing_errors": [r for r in results if r["status"] != "READABLE_FINITE_Q1"],
        "category_counts": dict(Counter(r["group"] for r in train)),
        "duration_seconds": {"train": sum(r["duration_seconds"] for r in train), "validation": sum(r["duration_seconds"] for r in validation)},
        "deficits": selection["deficits"], "retarget_adapter": identity(Path(__file__).resolve().parents[1] / "supermimic_q1_hp/retarget_soma.py"),
        "retarget_settings": {"workers": 4, "iterations": 4, "root_relative_rest": True},
        "limits": "Original 300/30 arrays retained by reference; only additions converted. Ready does not imply physical tracking quality. Expansion used only if RF12 gate passes."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "convert", "finalize"))
    for name in ("runtime", "previous", "output", "archive", "metadata", "canonical", "gmr"):
        parser.add_argument("--" + name, type=Path, required=name in ("runtime", "previous", "output"))
    args = parser.parse_args()
    {"select": select, "convert": convert, "finalize": finalize}[args.mode](args)


if __name__ == "__main__":
    main()
