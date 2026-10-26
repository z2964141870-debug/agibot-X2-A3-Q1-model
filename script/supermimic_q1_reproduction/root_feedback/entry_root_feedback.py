#!/usr/bin/env python3
"""RF07 dispatch: the same root feedback in collection, imagination and evaluation."""

import argparse
from functools import partial
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1]
for directory in (HERE, HERE.parent / "leg_control", HERE.parent,
                  SCRIPT / "project_tools", SCRIPT / "supermimic_q1_analysis",
                  SCRIPT / "supermimic_q1_sources", SCRIPT / "supermimic_for_q1"):
    sys.path.insert(0, str(directory))


def activate():
    import contact_protocol
    from control import Leg50Assets
    from feedback import FeedbackQ1Sim, configure_imagination
    from supermimic import q1_sim
    contact_protocol.ContactCorrectedAssets = Leg50Assets
    q1_sim.Q1Sim = FeedbackQ1Sim
    configure_imagination()
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
        from feedback import FeedbackSlottedQ1Sim, augment_manifest
        from supermimic.policy.nets import PolicyCfg
        from supermimic.imagination.policy_trainer import PolicyTrainerCfg
        from supermimic.robot_spec import GOAL_DIM
        from supermimic.world_model.heads import ANCHOR_OBS_DIM

        train_reproduction.Q1Assets = assets_class
        train_reproduction.Q1Sim = partial(FeedbackSlottedQ1Sim, slots=4)
        train_reproduction.PolicyCfg = partial(PolicyCfg, goal_dim=GOAL_DIM + ANCHOR_OBS_DIM)
        train_reproduction.PolicyTrainerCfg = partial(PolicyTrainerCfg, anchor_obs=True)
        original_write = train_reproduction.atomic_json

        def record(path, value):
            if path.name == "admission.json":
                augment_manifest(value, __file__)
            return original_write(path, value)

        train_reproduction.atomic_json = record
        module_name = "train_q1_with_dashboard"
    else:
        module_name = {"evaluate": "evaluate_reproduction", "audit": "audit_reproduction_evaluation",
                       "render": "render_reproduction", "fullbody": "diagnose_fullbody"}[args.task]
    module = importlib.import_module(module_name)
    sys.argv = [module.__file__, *forwarded]
    module.main()


if __name__ == "__main__":
    main()
