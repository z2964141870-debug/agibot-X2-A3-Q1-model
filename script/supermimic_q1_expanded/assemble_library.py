#!/usr/bin/env python3
"""Assemble checked PHUMA batches without converting or reclassifying references."""

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PHUMA = HERE.parent / "supermimic_q1_phuma"
sys.path.insert(0, str(PHUMA))

from acquire_source import file_id


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def combine(selection, bundles):
    """Compose previously checked records, requiring coverage of the full fixed list."""
    bundles = sorted(bundles, key=lambda item: item[0]["batch"]["offset"])
    prototype, checked_prototype = bundles[0][1:]
    comparable = ("reference_protocol", "source_receipt", "archive", "source_model",
                  "q1_assets", "scale", "pilot", "sources")
    motions, checks, inherited = [], [], []
    for batch, manifest, checked in bundles:
        if (any(manifest[key] != prototype[key] for key in comparable) or manifest["pilot"]
                or checked["status"] != "INDEPENDENT_REFERENCE_CHECK_COMPLETED"
                or checked["reference_protocol"] != prototype["reference_protocol"]
                or checked["source"] != checked_prototype["source"]):
            raise ValueError("Incompatible batch assets, protocol or reference verifier")
        if ([row["phuma_source"] for row in manifest["motions"]] != batch["selected"]
                or [row["clip"] for row in checked["clips"]]
                != [row["clip"] for row in manifest["motions"]]):
            raise ValueError("Batch conversion/check records do not cover the same candidates")
        motions.extend(manifest["motions"])
        checks.extend(checked["clips"])
        inherited.append({"selection": manifest["selection"], "reference_manifest": checked["input_manifest"]})
    if [row["phuma_source"] for row in motions] != selection["selected"]:
        raise ValueError("Complete fixed selection must be covered once in original order")
    groups = [row["phuma_source"]["source_group"] for row in motions]
    hashes = [row["phuma_source"]["sha256"] for row in motions]
    names = [row["clip"] for row in motions]
    if len(set(groups)) != len(groups) or len(set(hashes)) != len(hashes) or len(set(names)) != len(names):
        raise ValueError("Repeated source group, source bytes or clip name")
    admitted = []
    for row, check in zip(motions, checks):
        if (check["split"] != row["split"] or check["source_group"] != row["phuma_source"]["source_group"]):
            raise ValueError("Check split or source group differs from conversion")
        if check["qualified_for_training_pilot"]:
            if (not row["qualified_for_training_pilot"] or len(check["gates"]) != 19
                    or not all(check["gates"].values()) or check["reference"] != row["output"]
                    or Path(row["output"]["path"]).name != row["clip"] + ".npz"):
                raise ValueError("An accepted record does not preserve its complete 19 checks")
            admitted.append(row["clip"])
    if not admitted:
        raise ValueError("No independently accepted references")
    manifest = dict(prototype)
    manifest.pop("seconds", None)
    manifest.update({"status": "PHUMA_REFERENCE_LIBRARY_ASSEMBLED_FROM_CHECKED_BATCHES",
                     "motions": motions, "qualified_clips": admitted,
                     "qualified_split_counts": dict(collections.Counter(
                         row["split"] for row in checks if row["clip"] in set(admitted)))})
    receipt = dict(checked_prototype)
    receipt.update({"clips": checks, "qualified_clips": admitted,
                    "qualified_split_counts": manifest["qualified_split_counts"],
                    "rejected_clips": [row["clip"] for row in checks if row["clip"] not in set(admitted)]})
    return manifest, receipt, inherited


def assemble(args):
    import audit_scan
    from prepare_expansion_batch import uncached_identity
    from scan_corpus import resource_check
    from summarize_expansion import summarize

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed before library assembly")
    started = time.monotonic()
    output.mkdir(parents=True)
    selection_path = args.selection.resolve()
    progress_path = output / "batch_reconciliation.json"
    summarize(selection_path, [p.resolve() for p in args.batches], progress_path, args.expected_oom_kill)
    progress = json.loads(progress_path.read_text())
    selection = json.loads(selection_path.read_text())
    bundles, receipts = [], []
    for entry in progress["batches"]:
        done = json.loads(Path(entry["completion"]["path"]).read_text())
        for key in ("selection", "retarget_manifest", "reference_check"):
            if file_id(done[key]["path"]) != done[key]:
                raise ValueError("Batch input changed after reconciliation")
        bundles.append(tuple(json.loads(Path(done[key]["path"]).read_text())
                             for key in ("selection", "retarget_manifest", "reference_check")))
        receipts.append({key: done[key] for key in ("selection", "scan_check", "retarget_manifest", "reference_check")})
    manifest, checked, inherited = combine(selection, bundles)
    archive = Path(selection["archive"]["path"]).resolve()
    original_file_id = audit_scan.file_id
    audit_scan.file_id = lambda p: uncached_identity(p) if Path(p).resolve() == archive else original_file_id(p)
    audit_scan.audit(selection_path, output / "scan_check.json")
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed during source reconciliation")
    references = output / "references"
    references.mkdir()
    accepted = set(checked["qualified_clips"])
    for row in manifest["motions"]:
        if row["clip"] in accepted:
            target = Path(row["output"]["path"]).resolve()
            if file_id(target) != row["output"]:
                raise ValueError("Accepted reference changed before linking")
            (references / (row["clip"] + ".npz")).symlink_to(target)
    provenance = {"mode": "COMPOSITION_OF_PRIOR_INDEPENDENT_19_GATE_RECORDS",
                  "geometric_checks_rerun": False, "new_ik_conversions": 0,
                  "batch_checks": receipts, "inherited_records": inherited,
                  "saved_time_reconciliation": file_id(progress_path), "assembler": file_id(__file__)}
    manifest.update({"selection": file_id(selection_path), "assembly": provenance,
                     "new_ik_conversions": 0,
                     "sources": manifest["sources"] + [file_id(__file__), file_id(PHUMA / "summarize_expansion.py")],
                     "limits": "Existing conversion records composed; only references accepted by prior independent 19-gate checks are linked. No new retargeting or dynamic tracking."})
    write_new(references / "manifest.json", manifest)
    checked.update({"input_manifest": file_id(references / "manifest.json"), "assembly": provenance,
                    "limits": "Independent geometric check records inherited unchanged from the listed batches; not a new geometric verification run or a policy result."})
    write_new(output / "reference_check.json", checked)
    from run_training import admitted_clips
    admitted = admitted_clips(references, output / "reference_check.json", output / "scan_check.json")
    if set(sum(admitted.values(), [])) != accepted:
        raise ValueError("Preserved training admission did not reproduce the composed check set")
    if resource_check() != args.expected_oom_kill:
        raise RuntimeError("Resource/OOM state changed during library assembly")
    result = {"status": "PHUMA_EXPANDED_LIBRARY_ASSEMBLED_AND_TRAINER_ADMISSION_PASSED",
              "captured_utc": datetime.now(timezone.utc).isoformat(),
              "selection": file_id(selection_path), "reconciliation": file_id(progress_path),
              "manifest": file_id(references / "manifest.json"),
              "reference_check": file_id(output / "reference_check.json"),
              "scan_check": file_id(output / "scan_check.json"), "source": file_id(__file__),
              "candidate_count": progress["candidate_count"], "qualified_count": len(accepted),
              "split_clips": admitted, "qualified_duration_s": progress["qualified_duration_s"],
              "qualified_locomotion_duration_s": progress["qualified_locomotion_duration_s"],
              "elapsed_s": time.monotonic() - started, "oom_kill": resource_check(),
              "new_ik_conversions": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY", "geometric_checks_rerun": False,
              "limits": "Library and preserved trainer admission only. Existing checked NPZs remain in their original locations; no new policy has been trained."}
    write_new(output / "completion.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "split_clips"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    assemble(parser.parse_args())
