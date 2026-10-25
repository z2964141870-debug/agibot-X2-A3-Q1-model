#!/usr/bin/env python3
"""Combine native-time Q1 reference batches through links to existing arrays."""

import argparse
import json
from pathlib import Path

from reference_admission import POLICY, identity, validate_arrays


def merge_libraries(libraries, output):
    rows, checks, provenance = [], [], []
    seen = set()
    contract = None
    for library in libraries:
        library = Path(library).resolve(strict=True)
        receipt_path = library / "admission.json"
        receipt = json.loads(receipt_path.read_text())
        manifest_path = library / "references/manifest.json"
        if (receipt["status"] != "Q1_REFERENCE_IO_ADMISSION_COMPLETED"
                or receipt["admission_policy"] != POLICY
                or identity(manifest_path) != receipt["input_manifest"]):
            raise ValueError("Input batch differs from its IO receipt")
        manifest = json.loads(manifest_path.read_text())
        current = {key: manifest[key] for key in ("q1_assets", "reference_protocol")}
        if (manifest["admission_policy"] != POLICY
                or (contract is not None and current != contract)):
            raise ValueError("Reference batches use different Q1 or time contracts")
        contract = current
        admitted = set(receipt["admitted_clips"])
        by_clip = {row["clip"]: row for row in manifest["motions"]}
        if len(by_clip) != len(manifest["motions"]) or set(by_clip) != admitted:
            raise ValueError("Input batch clip coverage differs from its receipt")
        for name in receipt["admitted_clips"]:
            row = by_clip[name]
            if name in seen:
                raise ValueError("Duplicate clip in the requested batches: " + name)
            if not row["admitted_for_training"] or identity(row["output"]["path"]) != row["output"]:
                raise ValueError("Reference content differs from the prepared batch")
            checked = validate_arrays(row["output"]["path"])
            seen.add(name)
            rows.append(row)
            checks.append({"clip": name, "split": row["split"], **checked})
        provenance.append({"admission": identity(receipt_path), "manifest": identity(manifest_path)})
    if not rows:
        raise ValueError("No readable references in the requested batches")
    output = Path(output)
    output.mkdir()
    references = output / "references"
    references.mkdir()
    for row in rows:
        (references / (row["clip"] + ".npz")).symlink_to(row["output"]["path"])
    manifest = {"status": "Q1_REPRODUCTION_LIBRARY_ASSEMBLED", "admission_policy": POLICY,
                **contract, "original_libraries": provenance, "motions": rows,
                "local_geometry_gates_used_for_admission": False,
                "legacy_diagnostics_preserved": True, "storage_status": "LOCAL_ONLY"}
    target = references / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    result = {"status": "Q1_REFERENCE_IO_ADMISSION_COMPLETED", "admission_policy": POLICY,
              "input_manifest": identity(target), "clips": checks,
              "admitted_clips": [row["clip"] for row in rows],
              "admitted_duration_s": sum(row["duration_s"] for row in checks),
              "original_libraries": provenance, "local_geometry_gates_used_for_admission": False,
              "validation_scope": "Readable finite arrays, Q1 dimensions, body order and native 50Hz time only.",
              "source": identity(__file__), "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    (output / "admission.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Keep the merged reference library in the unified project")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = merge_libraries(args.libraries, output)
    print(json.dumps({"status": result["status"], "clips": len(result["admitted_clips"]),
                      "duration_s": result["admitted_duration_s"]}), flush=True)


if __name__ == "__main__":
    main()
