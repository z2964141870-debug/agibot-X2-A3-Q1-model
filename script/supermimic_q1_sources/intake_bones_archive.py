#!/usr/bin/env python3
"""Verify a completed BONES G1 archive and stream only frozen candidate CSVs."""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

from inventory_amass import health, identity
from runtime_guard import StorageBudget, save


CHUNK = 1024 ** 2


def canonical_member(name, prefix=""):
    if not name or len(name) > 1024 or "\\" in name or "\x00" in name:
        raise ValueError("Invalid archive member name")
    for item in (name, prefix):
        path = PurePosixPath(item)
        if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
            raise ValueError(f"Archive member escapes its relative namespace: {item}")
    return (PurePosixPath(prefix) / PurePosixPath(name)).as_posix()


def stream_extract(archive_path, wanted, output, *, prefix="", catalog=None,
                   maximum_member_bytes=64 * 1024 ** 2,
                   maximum_selected_bytes=8 * 1024 ** 3,
                   maximum_expanded_bytes=250 * 1024 ** 3,
                   maximum_members=500000, progress=None):
    wanted = set(wanted)
    if not wanted or any(canonical_member(name) != name or not name.endswith(".csv") for name in wanted):
        raise ValueError("A nonempty canonical frozen CSV selection is required")
    canonical_member(".", prefix)
    files = output / "files"
    files.mkdir(exist_ok=False)
    index_path = output / "archive_members.jsonl"
    seen, regular, selected = set(), set(), {}
    total_size, selected_size, count = 0, 0, 0
    last_progress = time.monotonic()
    with archive_path.open("rb") as raw, gzip.GzipFile(fileobj=raw) as decoded, \
            tarfile.open(fileobj=decoded, mode="r|") as package, index_path.open("x") as index:
        for member in package:
            count += 1
            name = canonical_member(member.name, prefix)
            if count > maximum_members or name in seen:
                raise ValueError("Archive member count exceeded or canonical member repeated")
            seen.add(name)
            if not member.isdir() and (not member.isreg() or member.issparse()):
                raise ValueError(f"Only regular files and directories are admitted: {member.name}")
            if member.size < 0:
                raise ValueError("Negative archive member size")
            item = {"archive_name": member.name, "source_path": name,
                    "kind": "directory" if member.isdir() else "file",
                    "bytes": member.size, "selected": name in wanted}
            if member.isreg():
                regular.add(name)
                total_size += member.size
                if total_size > maximum_expanded_bytes:
                    raise ValueError("Declared expanded archive exceeds the admitted bound")
            if name in wanted:
                if (not member.isreg() or member.size > maximum_member_bytes
                        or selected_size + member.size > maximum_selected_bytes):
                    raise ValueError("Selected archive member exceeds type or storage bounds")
                target = files / name
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, length = hashlib.sha256(), 0
                with package.extractfile(member) as source, target.open("xb") as destination:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        length += len(block)
                        if length > member.size:
                            raise ValueError("Extracted size exceeds the declared member size")
                        destination.write(block)
                        digest.update(block)
                if length != member.size:
                    raise EOFError("Extracted member is shorter than its header")
                selected_size += length
                selected[name] = {"path": str(target.resolve()), "bytes": length,
                                  "sha256": digest.hexdigest()}
                item["sha256"] = digest.hexdigest()
            index.write(json.dumps(item, allow_nan=False) + "\n")
            if progress and (count % 1000 == 0 or time.monotonic() - last_progress >= 5):
                index.flush()
                progress({"indexed_members": count, "selected_files": len(selected),
                          "selected_bytes": selected_size, "declared_expanded_bytes": total_size})
                last_progress = time.monotonic()
        # Drain through the gzip footer: finding tar's zero blocks alone does not
        # prove that the compressed stream is complete or its CRC is valid.
        for block in iter(lambda: package.fileobj.read(CHUNK), b""):
            if any(block):
                raise ValueError("Nonzero content after the end of the tar archive")
    missing = sorted(wanted - selected.keys())
    if missing:
        raise ValueError(f"Frozen selection is missing {len(missing)} members; first: {missing[:5]}")
    summary = {"indexed_members": count, "regular_files": len(regular),
               "declared_expanded_bytes": total_size, "selected_files": len(selected),
               "selected_bytes": selected_size, "selected": selected,
               "gzip_read_to_eof_and_crc_checked": True, "index": identity(index_path)}
    if catalog is not None:
        summary.update(catalog_paths=len(catalog), catalog_paths_present=len(regular & catalog),
                       catalog_paths_missing=len(catalog - regular),
                       extra_csv_paths=len({name for name in regular if name.endswith(".csv")} - catalog))
    if progress:
        progress({key: value for key, value in summary.items() if key != "selected"})
    return summary


def frozen_selection(summary_path):
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "METADATA_AUDIT_AND_CANDIDATE_SPLITS_COMPLETED":
        raise ValueError("Expected the completed BONES metadata audit")
    for key in ("selected_candidates", "all_splits"):
        item = summary[key]
        if identity(Path(item["path"])) != item:
            raise ValueError(f"Frozen metadata artifact changed: {key}")
    with Path(summary["selected_candidates"]["path"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    if (len({row["source_group"] for row in rows}) != len(rows)
            or len({row["move_g1_path"] for row in rows}) != len(rows)
            or any(row["is_mirror"] or row["quarantine"] for row in rows)
            or dict(collections.Counter(row["split"] for row in rows)) != summary["selected_candidate_groups"]):
        raise ValueError("Candidate identities or split counts differ from the frozen audit")
    import csv
    with Path(summary["all_splits"]["path"]).open(newline="") as stream:
        catalog = {row["move_g1_path"] for row in csv.DictReader(stream)}
    return summary, rows, catalog


def run(args):
    archive, output, source_root = args.archive.absolute(), args.output_dir.absolute(), args.source_root.resolve(strict=True)
    if (archive.is_symlink() or not archive.is_file() or not archive.name.endswith(".tar.gz")
            or not archive.resolve().is_relative_to(source_root)
            or output.resolve().is_relative_to(source_root)):
        raise ValueError("Use a completed regular tar.gz inside the source tree and an independent output")
    if args.expected_bytes <= 0 or not re.fullmatch("[0-9a-f]{64}", args.expected_sha256):
        raise ValueError("Explicit complete source size and lowercase SHA-256 are required")
    if archive.stat().st_size != args.expected_bytes:
        raise ValueError("Archive size differs from the admitted source")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAILED", "new_q1_references": 0, "new_optimizer_updates": 0,
               "storage_status": "LOCAL_ONLY", "error": None}
    started = time.monotonic()
    try:
        storage = json.loads(args.goal_config.read_text())["storage"]
        budget = StorageBudget(output, source_root, storage["minimum_free_bytes"],
                               storage["maximum_experiment_unique_allocated_bytes"],
                               storage["maximum_new_source_unique_allocated_bytes"])
        receipt["storage_before"] = budget.check()
        baseline = health(args.expected_oom_kill)
        info = archive.stat()
        stable = (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
        actual = identity(archive)
        if actual["sha256"] != args.expected_sha256:
            raise ValueError("Archive full SHA-256 differs from admitted source")
        summary, rows, catalog = frozen_selection(args.metadata_summary)
        frozen = {"metadata_summary": identity(args.metadata_summary),
                  "goal_config": identity(args.goal_config),
                  "selected_candidates": summary["selected_candidates"], "all_splits": summary["all_splits"]}

        def progress(state):
            budget.check()
            health(baseline.get("oom_kill"))
            save(output / "progress.json", {"status": "INDEXING_ARCHIVE", **state,
                                           "elapsed_s": time.monotonic() - started})

        extracted = stream_extract(archive, {row["move_g1_path"] for row in rows}, output,
                                   prefix=args.member_prefix, catalog=catalog, progress=progress)
        final = archive.stat()
        if (final.st_size, final.st_mtime_ns, final.st_dev, final.st_ino) != stable:
            raise ValueError("Archive changed during acceptance")
        if any(identity(Path(item["path"])) != item for item in frozen.values()):
            raise ValueError("Frozen metadata or goal changed during intake")
        receipt.update(status="ARCHIVE_ACCEPTED_FROZEN_SELECTION_EXTRACTED", archive=actual,
                       member_prefix=args.member_prefix, frozen=frozen, extraction=extracted,
                       health_before=baseline, health_after=health(baseline.get("oom_kill")),
                       storage_after=budget.check(), code=identity(Path(__file__).resolve()),
                       selected_split_counts=summary["selected_candidate_groups"],
                       limits="Complete archive transport/hash, gzip CRC, member inventory and selected-file extraction only. Actual CSV schema/time, source model, numeric duplicates and Q1 conversion/qualification remain pending. No archive or previous input was deleted.")
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_s"] = time.monotonic() - started
        save(output / "receipt.json", receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--metadata-summary", type=Path, required=True)
    parser.add_argument("--goal-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--member-prefix", default="")
    parser.add_argument("--expected-oom-kill", type=int)
    run(parser.parse_args())
