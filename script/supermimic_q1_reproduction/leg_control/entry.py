#!/usr/bin/env python3
"""Use the RF06 leg contract consistently with the preserved training/eval code."""

import argparse
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1]
for directory in (SCRIPT / "project_tools", HERE.parent, SCRIPT / "supermimic_q1_analysis",
                  SCRIPT / "supermimic_for_q1"):
    sys.path.insert(0, str(directory))


def activate():
    import contact_protocol
    from control import Leg50Assets
    contact_protocol.ContactCorrectedAssets = Leg50Assets
    return Leg50Assets


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("task", choices=("train", "evaluate", "audit", "render", "fullbody"))
    args, forwarded = parser.parse_known_args()
    if args.task == "render":
        os.environ.setdefault("MUJOCO_GL", "egl")
    assets_class = activate()
    if args.task == "train":
        import train_reproduction
        from control import augment_training_manifest
        train_reproduction.Q1Assets = assets_class
        original_write = train_reproduction.atomic_json

        def record_manifest(path, value):
            if path.name == "admission.json":
                augment_training_manifest(value, __file__)
                value["main_variable"] = "LEG_ACTUATOR_FORCE_CAPS_AND_USER_SHORTENED_TIME_BUDGET"
                value["user_training_budget"] = {"maximum_training_seconds": 3600.,
                    "soft_target_seconds": 3000., "maximum_training_phase_seconds": 3500.,
                    "cleanup_grace_seconds": 55., "automatic_extension": False,
                    "comparison": "RF05 has the earlier two-hour budget; RF06 is not duration-matched"}
            return original_write(path, value)

        train_reproduction.atomic_json = record_manifest
        module_name = "train_q1_with_dashboard"
    else:
        module_name = {"evaluate": "evaluate_reproduction", "audit": "audit_reproduction_evaluation",
                       "render": "render_reproduction", "fullbody": "diagnose_fullbody"}[args.task]
    module = importlib.import_module(module_name)
    sys.argv = [module.__file__, *forwarded]
    module.main()


if __name__ == "__main__":
    main()
