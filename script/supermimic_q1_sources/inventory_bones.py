#!/usr/bin/env python3
"""Audit supplied BONES metadata and freeze source groups before Q1 conversion."""

import argparse
import collections
import csv
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import time

from inventory_amass import health, identity


SEED = "Q1_BONES_G00_20261024_v1"
REQUIRED = {
    "filename", "move_name", "move_g1_path", "move_duration_frames", "is_mirror",
    "take_name", "take_org_name", "take_actor", "take_date", "actor_uid",
    "category", "is_neutral", "content_uniform_style", "content_type_of_movement",
    "content_body_position", "content_horizontal_move", "content_props",
}
INDEX_COLUMNS = ["filename", "move_g1_path", "move_duration_frames", "source_group",
                 "split", "is_mirror", "candidate_kind", "quarantine"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def csv_value(raw, field):
    import pyarrow as pa

    if pa.types.is_string(field.type):
        return raw
    if pa.types.is_boolean(field.type):
        if raw not in {"True", "False", "true", "false"}:
            raise ValueError(f"Invalid CSV boolean in {field.name}: {raw}")
        return raw.lower() == "true"
    if pa.types.is_integer(field.type):
        return int(raw)
    if pa.types.is_floating(field.type):
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite CSV value in {field.name}")
        return number
    raise ValueError(f"Unsupported metadata field: {field}")


def read_reconciled(parquet, csv_path):
    import pyarrow.parquet as pq

    source = pq.ParquetFile(parquet)
    if not REQUIRED.issubset(source.schema_arrow.names):
        raise ValueError("Missing required BONES metadata columns")
    rows = []
    names, paths = set(), set()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != source.schema_arrow.names:
            raise ValueError("CSV and parquet columns differ")
        for batch in source.iter_batches(batch_size=2048):
            for expected in batch.to_pylist():
                actual = next(reader, None)
                if actual is None or None in actual:
                    raise ValueError("CSV row count or field count differs")
                for field in source.schema_arrow:
                    value = expected[field.name]
                    if value is None or csv_value(actual[field.name], field) != value:
                        raise ValueError(f"CSV/parquet mismatch at row {len(rows)}, {field.name}")
                name = expected["filename"]
                member = PurePosixPath(expected["move_g1_path"])
                if (not name or member.is_absolute() or ".." in member.parts
                        or "\\" in str(member) or member.parts[:2] != ("g1", "csv")
                        or member.name != name + ".csv"):
                    raise ValueError(f"Unsafe or inconsistent G1 source path: {member}")
                if name in names or str(member) in paths:
                    raise ValueError(f"Duplicate metadata identity: {name}")
                if (expected["take_actor"] != expected["actor_uid"]
                        or not expected["take_actor"] or not expected["take_name"]
                        or not expected["take_org_name"] or expected["move_duration_frames"] < 2):
                    raise ValueError(f"Invalid actor, take or frame count: {name}")
                names.add(name)
                paths.add(str(member))
                rows.append(expected)
        if next(reader, None) is not None:
            raise ValueError("CSV has extra rows")
    return rows


def base_name(row):
    name = row["filename"]
    if row["is_mirror"]:
        if not name.endswith("_M"):
            raise ValueError(f"Mirror metadata lacks documented suffix: {name}")
        name = name[:-2]
    return name


def group_rows(rows):
    parents, aliases = list(range(len(rows))), {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Both corrected take names and original capture names are aliases. This also
    # keeps any later crop with its source; filenames alone are not the split key.
    for index, row in enumerate(rows):
        prefix = (str(row["take_date"]), row["take_actor"])
        keys = [("file", base_name(row)),
                ("take", *prefix, row["take_name"]),
                ("take", *prefix, row["take_org_name"])]
        for key in keys:
            if key in aliases:
                parents[find(index)] = find(aliases[key])
            else:
                aliases[key] = index
    groups = collections.defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    name_index = {row["filename"]: row for row in rows}
    for indices in groups.values():
        canonical = min((str(rows[i]["take_date"]), rows[i]["take_actor"],
                         rows[i]["take_org_name"], rows[i]["take_name"],
                         base_name(rows[i])) for i in indices)
        group = digest(canonical)
        bucket = int(digest([SEED, "split", group])[:12], 16) % 10000
        split = "train" if bucket < 8000 else "development" if bucket < 9000 else "test"
        reasons = set()
        for i in indices:
            row = rows[i]
            if row["is_mirror"]:
                original = name_index.get(base_name(row))
                if original is None or original["is_mirror"]:
                    reasons.add("missing_original")
                elif any(original[key] != row[key] for key in
                         ("move_duration_frames", "take_actor", "take_date")):
                    reasons.add("mirror_identity_disagreement")
        for i in indices:
            rows[i].update(source_group=group, split=split,
                           quarantine=",".join(sorted(reasons)))
    return rows


def candidate_kind(row):
    if (row["quarantine"] or row["is_mirror"] or row["category"] != "Basic Locomotion Neutral"
            or row["is_neutral"] != 1 or row["content_uniform_style"].strip() != "neutral"
            or row["content_body_position"].strip() != "standing"
            or row["content_props"].strip() != "0"
            or not 101 <= row["move_duration_frames"] <= 3001):
        return ""
    tokens = {item.strip() for item in row["content_type_of_movement"].split(",")}
    if "walking" in tokens and tokens <= {"walking", "turning", "standing"}:
        return "locomotion_candidate"
    if tokens <= {"standing", "standing idle", "idle"} and row["content_horizontal_move"] == 0:
        return "standing_candidate"
    return ""


def selected_candidates(rows, train_limit=1024, heldout_limit=128):
    selected = []
    for split, kind, limit in [("train", "locomotion_candidate", train_limit),
                               ("train", "standing_candidate", 128),
                               ("development", "locomotion_candidate", heldout_limit),
                               ("test", "locomotion_candidate", heldout_limit)]:
        candidates = [row for row in rows if row["split"] == split and row["candidate_kind"] == kind]
        candidates.sort(key=lambda row: digest([SEED, "rank", row["source_group"], row["filename"]]))
        used = set()
        for row in candidates:
            if row["source_group"] in used:
                continue
            used.add(row["source_group"])
            selected.append(row)
            if len(used) >= limit:
                break
    return selected


def temporal_audit(path, names):
    seen, anomalies, examples = set(), collections.Counter(), []
    event_count, line_number = 0, 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            name = row["filename"]
            reasons = []
            if name in seen:
                reasons.append("duplicate_filename")
            seen.add(name)
            events = row["events"]
            if row["num_events"] != len(events):
                reasons.append("event_count_disagreement")
            previous_end = -math.inf
            for event in events:
                event_count += 1
                start, end = event["start_time"], event["end_time"]
                if (isinstance(start, bool) or isinstance(end, bool)
                        or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                        or not math.isfinite(start) or not math.isfinite(end)
                        or start < 0 or end <= start):
                    reasons.append("invalid_interval")
                    continue
                if start < previous_end - 1e-6:
                    reasons.append("overlapping_or_unordered")
                previous_end = end
            for reason in set(reasons):
                anomalies[reason] += 1
            if reasons and len(examples) < 10:
                examples.append({"line": line_number, "filename": name, "reasons": reasons})
    return {"rows": line_number, "unique_filenames": len(seen), "events": event_count,
            "covered_metadata_filenames": len(names & seen),
            "metadata_filenames_without_v002_labels": len(names - seen),
            "v002_filenames_outside_v004_metadata": len(seen - names),
            "anomaly_rows": dict(anomalies), "examples": examples,
            "used_for_training_or_segmentation": False,
            "duration_comparison": "Pending actual CSV native-time evidence; no frame-rate default"}


def run(args):
    root = args.metadata_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must not modify the supplied metadata tree")
    baseline = health(args.expected_oom_kill)
    receipt = json.loads(args.intake_receipt.read_text())
    files, stats = {}, {}
    started = time.monotonic()
    for entry in receipt["files"]:
        path = root / entry["name"]
        if path.is_symlink() or path.parent != root:
            raise ValueError("Metadata must be a direct regular file")
        info = path.stat()
        stats[path.name] = (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
        actual = identity(path)
        if (actual["bytes"], actual["sha256"]) != (entry["bytes"], entry["sha256"]):
            raise ValueError(f"Intake identity mismatch: {path.name}")
        files[path.name] = actual
    rows = group_rows(read_reconciled(root / "seed_metadata_v004.parquet",
                                      root / "seed_metadata_v004.csv"))
    for row in rows:
        row["candidate_kind"] = candidate_kind(row)
    candidates = selected_candidates(rows)
    temporal = temporal_audit(root / "seed_metadata_v002_temporal_labels.jsonl",
                              {row["filename"] for row in rows})
    health(baseline.get("oom_kill"))
    for name, expected in stats.items():
        info = (root / name).stat()
        if (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino) != expected:
            raise ValueError(f"Source changed during audit: {name}")
    output.mkdir(parents=True, exist_ok=False)
    index_path = output / "all_source_splits.csv"
    with index_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    selected_path = output / "selected_candidates.jsonl"
    with selected_path.open("x") as stream:
        for row in candidates:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    counts = collections.Counter((row["split"], row["candidate_kind"]) for row in rows)
    groups = {split: {row["source_group"] for row in rows if row["split"] == split}
              for split in ("train", "development", "test")}
    if any(groups[left] & groups[right] for left, right in itertools.combinations(groups, 2)):
        raise AssertionError("Source group crossed a split")
    summary = {
        "status": "METADATA_AUDIT_AND_CANDIDATE_SPLITS_COMPLETED",
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_files": files, "intake_receipt": identity(args.intake_receipt),
        "code": identity(Path(__file__).resolve()), "seed": SEED,
        "csv_parquet_all_fields_equal": True, "rows": len(rows),
        "source_groups": {key: len(value) for key, value in groups.items()},
        "mirror_rows": sum(row["is_mirror"] for row in rows),
        "quarantined_rows": sum(bool(row["quarantine"]) for row in rows),
        "candidate_rows": {f"{split}/{kind}": value for (split, kind), value in counts.items() if kind},
        "selected_candidate_groups": dict(collections.Counter(row["split"] for row in candidates)),
        "selected_kinds": dict(collections.Counter(row["candidate_kind"] for row in candidates)),
        "selected_source_frames": dict((split, sum(row["move_duration_frames"] for row in candidates
                                                  if row["split"] == split)) for split in groups),
        "temporal_v002": temporal, "all_splits": identity(index_path),
        "selected_candidates": identity(selected_path),
        "health_before": baseline, "health_after": health(baseline.get("oom_kill")),
        "elapsed_s": time.monotonic() - started, "storage_status": "LOCAL_ONLY",
        "limits": "Metadata grouping and preconversion candidates only. Corrected and original capture aliases are unioned within actor/date, and mirror filename identities are unioned. Future crops must inherit the group. Numeric motion duplicates/cross-source overlap, actual rate/schema/model, Q1 conversion and qualified seconds are not yet verified. v002 event labels do not select or segment references.",
        "new_q1_references": 0, "new_optimizer_updates": 0,
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--intake-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int)
    run(parser.parse_args())
