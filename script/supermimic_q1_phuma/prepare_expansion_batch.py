#!/usr/bin/env python3
"""Convert and independently check a new, finite PHUMA expansion batch."""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from acquire_source import file_id


def uncached_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        offset = dropped = 0
        for block in iter(lambda: stream.read(2 * 1024 ** 2), b""):
            digest.update(block)
            offset += len(block)
            if offset - dropped >= 32 * 1024 ** 2:
                os.posix_fadvise(stream.fileno(), dropped, offset - dropped, os.POSIX_FADV_DONTNEED)
                dropped = offset
        os.posix_fadvise(stream.fileno(), dropped, offset - dropped, os.POSIX_FADV_DONTNEED)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare(args):
    import audit_scan
    import retarget
    from scan_corpus import resource_check

    verifier_path = Path(__file__).with_name("verify_reference.py")
    spec = importlib.util.spec_from_file_location("phuma_expansion_reference_verifier", verifier_path)
    verify_reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_reference)
    output = args.output.resolve()
    if output.exists() and not args.verify_existing:
        raise FileExistsError(output)
    full = json.loads(args.selection.read_text())
    if full["status"] != "PHUMA_P04_GROUPED_EXPANSION_SELECTION_COMPLETED":
        raise ValueError("Expected the independent P04 expansion selection")
    if args.offset < 0 or args.count < 1 or args.offset >= len(full["selected"]):
        raise ValueError("Invalid finite batch slice")
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed before this batch")
    archive = Path(full["archive"]["path"]).resolve()
    original_identity = retarget.identity

    def checked_identity(path):
        return uncached_identity(path) if Path(path).resolve() == archive else original_identity(path)

    # Keep the existing byte checks, but release pages of this project's large ZIP as they are hashed.
    retarget.identity = checked_identity
    audit_scan.file_id = checked_identity
    if checked_identity(archive) != full["archive"]:
        raise ValueError("PHUMA archive changed")
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed during archive verification")
    started = time.monotonic()
    batch = dict(full)
    batch["selected"] = full["selected"][args.offset:args.offset + args.count]
    batch["selected_counts"] = dict(collections.Counter(
        row["split"] + "/" + row["bucket"] for row in batch["selected"]))
    batch["batch"] = {"full_selection": file_id(args.selection), "offset": args.offset,
                      "requested_count": args.count, "count": len(batch["selected"])}
    selection_path = output / "selection.json"
    if args.verify_existing:
        if (output / "completion.json").exists() or (output / "reference_check.json").exists():
            raise FileExistsError("This batch already has a reference check or completion")
        if json.loads(selection_path.read_text()) != batch:
            raise ValueError("Existing batch selection differs from the requested immutable slice")
        existing = json.loads((output / "references/manifest.json").read_text())
        scan = json.loads((output / "scan_check.json").read_text())
        if (existing["selection"] != file_id(selection_path)
                or scan["selection"] != file_id(selection_path)
                or scan["status"] != "PHUMA_SCAN_INDEPENDENT_RECONCILIATION_PASSED"
                or [row["phuma_source"] for row in existing["motions"]] != batch["selected"]):
            raise ValueError("Existing complete conversion or input reconciliation does not match")
        print(json.dumps({"event": "verify_existing_conversion_only", **batch["batch"]}), flush=True)
    else:
        output.mkdir(parents=True)
        with selection_path.open("x") as stream:
            json.dump(batch, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"event": "batch_started", **batch["batch"]}), flush=True)
        audit_scan.audit(selection_path, output / "scan_check.json")
        if resource_check() != args.expected_oom_kill:
            raise RuntimeError("Resource/OOM state changed during input reconciliation")
        retarget.convert(args.root.resolve(), selection_path, args.vendor.resolve(), output / "references", False)
    verify_reference.verify(args.root.resolve(), output / "references", output / "reference_check.json")
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed during conversion/checking")
    checked = json.loads((output / "reference_check.json").read_text())
    admitted = [row for row in checked["clips"] if row["qualified_for_training_pilot"]]
    rejection = collections.Counter(
        key for row in checked["clips"] for key, passed in row.get("gates", {}).items() if not passed)
    summary = {
        "status": ("PHUMA_P04_EXISTING_CONVERSION_INDEPENDENT_CHECK_COMPLETED" if args.verify_existing else
                   "PHUMA_P04_BATCH_CONVERSION_AND_INDEPENDENT_CHECK_COMPLETED"),
        "selection": file_id(selection_path), "scan_check": file_id(output / "scan_check.json"),
        "retarget_manifest": file_id(output / "references/manifest.json"),
        "reference_check": file_id(output / "reference_check.json"),
        "candidate_count": len(batch["selected"]), "qualified_count": len(admitted),
        "qualified_counts": dict(collections.Counter(row["split"] + "/" + row["bucket"] for row in admitted)),
        "qualified_duration_s": {split: sum(row["duration_s"] for row in admitted if row["split"] == split)
                                 for split in ("train", "development", "test")},
        "qualified_locomotion_duration_s": {split: sum(row["duration_s"] for row in admitted
                                            if row["split"] == split and row["bucket"] == "locomotion")
                                            for split in ("train", "development", "test")},
        "conversion_errors": sum("conversion_error" in row for row in checked["clips"]),
        "rejection_gate_counts": dict(rejection), "sources": [file_id(__file__)],
        "elapsed_s": time.monotonic() - started, "verification_only": args.verify_existing,
        "new_ik_conversion_in_this_invocation": not args.verify_existing, "oom_kill": resource_check(),
        "new_physics_steps": 0, "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
        "limits": "Existing 19 geometry/export checks preserved. No dynamic tracking or new policy training; reference counts are not policy success.",
    }
    with (output / "completion.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--verify-existing", action="store_true",
                        help="Only check an existing complete conversion whose original wrapper failed before verification")
    prepare(parser.parse_args())
