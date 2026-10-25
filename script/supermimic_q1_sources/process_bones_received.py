#!/usr/bin/env python3
"""Accept the received real BONES archive, then convert a bounded frozen batch."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
from types import SimpleNamespace

from bounded_archive_io import ArchivePath, bounded_identity
from guard_phuma_batch import MemoryBoundBudget
import relocated_io
from relocated_io import JsonView
from runtime_guard import StorageBudget, guarded_run, save


HERE = Path(__file__).resolve().parent


def budget(args):
    return MemoryBoundBudget(StorageBudget(args.output, args.archive.parent,
        100 * 1024 ** 3, 12 * 1024 ** 3, 250 * 1024 ** 3),
        int(28.75 * 1024 ** 3), args.expected_oom_kill)


def release_archive(path):
    with ArchivePath(path).open("rb"):
        pass


def worker(args):
    os.chdir(args.root)
    # Hashing immutable migrated inputs must not populate tens of GB of page cache.
    relocated_io.identity = bounded_identity
    view = relocated_io.migration_view(args.root)
    save(args.output / (args.phase + "_relocation.json"), view.evidence())

    def heartbeat(value):
        budget(args).check()
        save(args.output / "heartbeat.json", {"phase": args.phase,
            "utc": datetime.now(timezone.utc).isoformat(), **value})

    if args.phase == "prepare":
        archive = bounded_identity(args.archive,
            lambda done, total: heartbeat({"hashed_bytes": done, "archive_bytes": total}))
        prefix, names, first_csv = None, [], None
        with args.archive.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as package:
            for member in package:
                names.append(member.name)
                if member.isreg() and member.name.endswith(".csv"):
                    name = member.name.removeprefix("./")
                    if name.startswith("g1/csv/"):
                        prefix = ""
                    elif name.startswith("csv/"):
                        prefix = "g1"
                    else:
                        raise ValueError("Unexpected BONES CSV namespace; no implicit renaming")
                    first_csv = package.extractfile(member).read(2048).decode("utf-8-sig").splitlines()[0]
                    break
                if len(names) > 32:
                    raise ValueError("No real CSV within the bounded archive preview")
        if prefix is None:
            raise ValueError("The received archive has no previewable CSV")
        save(args.output / "prepared_archive.json", {"status": "RECEIVED_ARCHIVE_HASHED_CRC_PENDING",
            "archive": archive, "member_prefix": prefix, "first_members": names,
            "first_csv_header": first_csv, "source_side_sha256_available": False,
            "code": bounded_identity(__file__), "storage_status": "LOCAL_ONLY",
            "limits": "Complete local SHA and bounded namespace preview; full gzip CRC and frozen selection pending."})
        print(json.dumps({"status": "RECEIVED_ARCHIVE_HASHED_CRC_PENDING", "archive": archive,
                          "member_prefix": prefix, "first_csv_header": first_csv}), flush=True)
        return

    prepared = json.loads((args.output / "prepared_archive.json").read_text())
    if (prepared["status"] != "RECEIVED_ARCHIVE_HASHED_CRC_PENDING"
            or prepared["archive"]["path"] != str(args.archive)
            or prepared["archive"]["bytes"] != args.archive.stat().st_size):
        raise ValueError("The completed archive preparation is required")
    import intake_bones_archive as intake

    intake.json, intake.identity = JsonView(view), bounded_identity
    original_save = intake.save

    def save_progress(path, value):
        original_save(path, value)
        if path.name == "progress.json":
            heartbeat({key: value[key] for key in ("indexed_members", "selected_files", "selected_bytes") if key in value})

    intake.save = save_progress
    intake_folder = args.output / "intake"
    if args.phase == "intake":
        intake.run(SimpleNamespace(archive=args.archive, output_dir=intake_folder,
            source_root=args.archive.parent, expected_bytes=prepared["archive"]["bytes"],
            expected_sha256=prepared["archive"]["sha256"], member_prefix=prepared["member_prefix"],
            metadata_summary=args.root / "data/supermimic_for_q1/BONES_P01_metadata_20261024/summary.json",
            goal_config=HERE / "bones_goal_20261024.json", expected_oom_kill=args.expected_oom_kill))
        save(args.output / "intake_relocation_after.json", view.evidence())
        return

    import bones_retarget as converter
    import bones_verify_reference as verifier

    converter.json, verifier.json = JsonView(view), JsonView(view)
    motions = args.output / "retarget"
    converter.convert(SimpleNamespace(root=args.root, intake=intake_folder / "receipt.json",
        model_comparison=args.root / "data/supermimic_for_q1/BONES_P02_model_20261024/official_comparison.json",
        output=motions, offset=0, count=args.count, expected_oom_kill=args.expected_oom_kill))
    heartbeat({"converted_batch": args.count})
    verifier.verify(SimpleNamespace(root=args.root, motions=motions,
        output=args.output / "reference_check.json", expected_oom_kill=args.expected_oom_kill))
    checks = json.loads((args.output / "reference_check.json").read_text())
    save(args.output / "batch_summary.json", {"status": "RECEIVED_BONES_REAL_BATCH_CHECKED",
        "archive": prepared["archive"], "intake": bounded_identity(intake_folder / "receipt.json"),
        "manifest": bounded_identity(motions / "manifest.json"),
        "independent_check": bounded_identity(args.output / "reference_check.json"),
        "requested_clips": args.count, "qualified_split_counts": checks["qualified_split_counts"],
        "qualified_clips": checks["qualified_clips"], "corpus_ready_for_training": False,
        "physics_steps": 0, "optimizer_updates": 0, "storage_status": "LOCAL_ONLY"})
    print(json.dumps({"status": "RECEIVED_BONES_REAL_BATCH_CHECKED",
        "qualified_split_counts": checks["qualified_split_counts"], "requested_clips": args.count}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare", "intake", "batch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=ArchivePath, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if (args.archive.is_symlink() or not args.archive.is_file()
            or not args.archive.resolve().is_relative_to(args.root / "data/references/bones_seed/archives")
            or not args.output.is_relative_to(args.root / "data/supermimic_for_q1")
            or not 1 <= args.count <= 16):
        raise ValueError("Use the received archive and a bounded batch inside the unified root")
    args.archive = ArchivePath(args.archive.resolve(strict=True))
    if args.worker:
        worker(args)
        return 0
    release_archive(args.archive)
    budget(args).memory()
    if args.phase == "prepare":
        args.output.mkdir()
    elif not args.output.is_dir():
        raise ValueError("Prepare the archive first")
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), args.phase, "--worker",
        "--root", str(args.root), "--archive", str(args.archive), "--output", str(args.output),
        "--expected-oom-kill", str(args.expected_oom_kill), "--count", str(args.count)]
    result = guarded_run(command, args.root, budget(args), args.output / ("guard_" + args.phase + ".json"),
        args.root / "logs" / (args.output.name + "_" + args.phase + ".log"),
        check_interval_s=1., grace_s=5., stop_requested=lambda: requested,
        progress_path=args.output / "heartbeat.json", maximum_idle_s=900.)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
