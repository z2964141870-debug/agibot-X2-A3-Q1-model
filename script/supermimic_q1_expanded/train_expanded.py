#!/usr/bin/env python3
"""Bind the verified model-slot backend to the unchanged PHUMA SuperMimic trainer."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))

from run_training import main as phuma_main, identity
from supermimic.q1_sim import Q1Sim
from slotted_sim import SlottedQ1Sim


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-slots", type=int, required=True)
    parser.add_argument("--slot-binding-output", type=Path, required=True)
    wrapper, forwarded = parser.parse_known_args()
    inputs = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    inputs.add_argument("--backend-receipt", type=Path, required=True)
    inputs.add_argument("--num-envs", type=int, required=True)
    args, _ = inputs.parse_known_args(forwarded)
    receipt = json.loads(args.backend_receipt.read_text())
    if (receipt["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
            or receipt["model_slots"] != wrapper.model_slots
            or receipt["resource_probe64"]["environments"] != 64
            or args.num_envs != 64):
        raise ValueError("Expanded training requires the verified model-slot configuration and 64 environments")
    for record in receipt["source_files"]:
        if identity(record["path"]) != record:
            raise ValueError("Verified model-slot backend source changed")
    if wrapper.slot_binding_output.exists():
        raise FileExistsError(wrapper.slot_binding_output)
    import train
    if train.Q1Sim is not Q1Sim:
        raise ValueError("Unexpected preserved simulator binding")
    train.Q1Sim = partial(SlottedQ1Sim, slots=wrapper.model_slots)
    binding = {"status": "VERIFIED_SLOTTED_Q1_COLLECTION_BOUND", "entry": identity(__file__),
               "backend_receipt": identity(args.backend_receipt), "simulator": identity(HERE / "slotted_sim.py"),
               "preserved_trainer": identity(train.__file__), "num_envs": args.num_envs,
               "model_slots": wrapper.model_slots, "training_algorithm_changed": False,
               "forwarded_arguments": forwarded, "storage_status": "LOCAL_ONLY"}
    with wrapper.slot_binding_output.open("x") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sys.argv = [str(Path(__file__)), *forwarded]
    phuma_main()


if __name__ == "__main__":
    main()
