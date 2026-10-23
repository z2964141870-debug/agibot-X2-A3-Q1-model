#!/usr/bin/env python3
"""Bind the versioned contact adapter to the preserved R01 train/evaluate drivers."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

from contact_protocol import ContactCorrectedAssets
from supermimic.q1_assets import Q1Assets, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    wrapper, forwarded = parser.parse_known_args()
    inputs = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    inputs.add_argument("--root", type=Path, required=True)
    inputs.add_argument("--backend-receipt", type=Path)
    inputs.add_argument("--checkpoint", type=Path)
    args, _ = inputs.parse_known_args(forwarded)
    if wrapper.check_only and wrapper.preflight_output is None:
        parser.error("--check-only requires a new --preflight-output path")
    if wrapper.preflight_output is not None and wrapper.preflight_output.exists():
        raise FileExistsError(wrapper.preflight_output)
    assets = ContactCorrectedAssets(args.root)
    checkpoint = None
    if wrapper.phase == "train":
        if args.backend_receipt is None:
            parser.error("train requires --backend-receipt from the corrected protocol")
        backend_path = args.backend_receipt
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint from the corrected protocol")
        checkpoint = identity(args.checkpoint)
        ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        if ck["manifest"]["assets"] != assets.manifest():
            raise ValueError("Checkpoint is not from this corrected contact protocol")
        backend_record = ck["manifest"]["backend_receipt"]
        backend_path = Path(backend_record["path"])
        if identity(backend_path) != backend_record:
            raise ValueError("Checkpoint backend receipt changed")
        del ck
    receipt = json.loads(backend_path.read_text())
    if (receipt["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
            or receipt["assets"] != assets.manifest()):
        raise ValueError("Backend receipt is not for this corrected contact protocol")
    for item in receipt["source_files"]:
        if identity(item["path"]) != item:
            raise ValueError("Verified backend source changed")
    driver = importlib.import_module(wrapper.phase)
    if driver.Q1Assets is not Q1Assets:
        raise ValueError("Preserved driver's asset binding is unexpected")
    # The preserved driver already takes an asset object; replace its constructor only.
    driver.Q1Assets = ContactCorrectedAssets
    preflight = {"status": "CONTACT_PROTOCOL_PREFLIGHT_PASSED", "phase": wrapper.phase,
                 "wrapper": identity(__file__), "preserved_driver": identity(driver.__file__),
                 "backend_receipt": identity(backend_path), "checkpoint": checkpoint,
                 "assets": assets.manifest(), "forwarded_args": forwarded,
                 "new_physics_steps": 0, "new_policy_calls": 0, "optimizer_updates": 0,
                 "limits": "Asset/source binding preflight; training arguments, collection and optimizer lifecycle are validated by the underlying driver during an actual run.",
                 "storage_status": "LOCAL_ONLY"}
    if wrapper.preflight_output is not None:
        with wrapper.preflight_output.open("x") as stream:
            json.dump(preflight, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": preflight["status"], "phase": wrapper.phase,
                      "check_only": wrapper.check_only,
                      "contact_protocol": assets.manifest()["contact_protocol"]["version"]}), flush=True)
    if wrapper.check_only:
        return
    sys.argv = [str(driver.__file__), *forwarded]
    driver.main()


if __name__ == "__main__":
    main()
