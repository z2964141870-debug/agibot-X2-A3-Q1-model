#!/usr/bin/env python3
"""Admit PHUMA-only references to the preserved Q1 SuperMimic trainer."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))
from supermimic.q1_assets import identity


PROTOCOL = "PHUMA_G1_30HZ_TO_Q1_50HZ_NATIVE_TIME_V2"


def admitted_clips(motions, reference_receipt, scan_check):
    manifest_path = motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = json.loads(reference_receipt.read_text())
    scan = json.loads(scan_check.read_text())
    if (manifest["reference_protocol"] != PROTOCOL or manifest["pilot"]
            or checks["status"] != "INDEPENDENT_REFERENCE_CHECK_COMPLETED"
            or checks["reference_protocol"] != PROTOCOL
            or checks["input_manifest"] != identity(manifest_path)):
        raise ValueError("PHUMA full candidate conversion and independent checks are required")
    if (scan["status"] != "PHUMA_SCAN_INDEPENDENT_RECONCILIATION_PASSED"
            or scan["selection"] != manifest["selection"]
            or scan["candidate_pairwise_group_intersections"] != 0):
        raise ValueError("PHUMA scan/group reconciliation is missing or belongs to other inputs")
    for record in manifest["sources"]:
        if identity(record["path"]) != record:
            raise ValueError("PHUMA retargeting sources changed")
    if identity(checks["source"]["path"]) != checks["source"]:
        raise ValueError("PHUMA independent reference verifier changed")
    by_name = {row["clip"]: row for row in manifest["motions"]}
    admitted = {split: [] for split in ("train", "development", "test")}
    used_groups = set()
    for row in checks["clips"]:
        if not row["qualified_for_training_pilot"]:
            continue
        source = by_name[row["clip"]]
        if not source["qualified_for_training_pilot"] or source["split"] != row["split"]:
            raise ValueError("PHUMA qualification or split mismatch")
        group = source["phuma_source"]["source_group"]
        if group in used_groups:
            raise ValueError("PHUMA source group repeated across admitted clips")
        used_groups.add(group)
        if identity(source["output"]["path"]) != source["output"]:
            raise ValueError("An admitted PHUMA Q1 reference changed")
        admitted[row["split"]].append(row["clip"])
    if (len(admitted["train"]) < 2 or not admitted["development"] or not admitted["test"]
            or not any(by_name[name]["bucket"] == "locomotion" for name in admitted["train"])):
        raise ValueError("Need separate train/development/test groups and a moving training reference")
    return {key: sorted(value) for key, value in admitted.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scan-check", type=Path, required=True)
    parser.add_argument("--binding-output", type=Path, required=True)
    wrapper, forwarded = parser.parse_known_args()
    contract = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    contract.add_argument("--motions", type=Path, required=True)
    contract.add_argument("--reference-receipt", type=Path, required=True)
    contract.add_argument("--code-commit", required=True)
    known, _ = contract.parse_known_args(forwarded)
    if any(argument == "--clips" or argument.startswith("--clips=") for argument in forwarded):
        raise ValueError("Training clips are derived only from the PHUMA split/quality receipts")
    if wrapper.binding_output.exists():
        raise FileExistsError(wrapper.binding_output)
    admitted = admitted_clips(known.motions, known.reference_receipt, wrapper.scan_check)
    binding = {"status": "PHUMA_ONLY_RANDOM_INITIALIZATION_ADMITTED",
               "reference_protocol": PROTOCOL, "initialization": "RANDOM_NO_PARENT_CHECKPOINT",
               "code_commit": known.code_commit, "wrapper": identity(__file__),
               "scan_check": identity(wrapper.scan_check), "reference_check": identity(known.reference_receipt),
               "motion_manifest": identity(known.motions / "manifest.json"), "split_clips": admitted,
               "a3_source_motions": 0, "parent_checkpoint": None,
               "training_method": "Online MuJoCo transitions; learned differentiable world model; imagined actor BPTT and critic.",
               "storage_status": "LOCAL_ONLY"}
    with wrapper.binding_output.open("x") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write("\n")
    import train
    import run_contact_corrected
    train.HELDOUT = tuple(admitted["development"] + admitted["test"])
    sys.argv = [str(HERE / "run_training.py"), "train", *forwarded, "--clips", *admitted["train"]]
    print(json.dumps({"status": binding["status"], "split_counts": {k: len(v) for k, v in admitted.items()}}), flush=True)
    run_contact_corrected.main()


if __name__ == "__main__":
    main()
