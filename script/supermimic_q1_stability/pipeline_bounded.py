#!/usr/bin/env python3
"""Register and run one fixed-budget action-domain trial after S02 verification."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from run_contract import CONFIG, EXPERIMENT, HERE, parent, correction_declaration, verified_prerequisites
import pipeline_partial


def check(args):
    root = args.root.resolve(strict=True)
    if (args.experiment.resolve() != root / "data/supermimic_for_q1" / EXPERIMENT
            or args.experiment.exists() or args.seed != 20261025
            or len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit)):
        raise ValueError("Use the new fixed R02D directory, matched seed, and published full commit")
    if (root / "logs" / EXPERIMENT).exists():
        raise FileExistsError("The new trial's log directory already exists")
    proof = verified_prerequisites(root)
    original = root / "data/supermimic_for_q1/PHUMA_R02_expanded64_H32_12000_20261024"
    development = root / "data/supermimic_for_q1/PHUMA_R02_DEV_A_step5000_20261025"
    evidence = parent.validated_parent(development, original / "run")
    baseline_path = root / "data/supermimic_for_q1/PHUMA_R02C_partial5000_extend7000_20261025/spec.json"
    baseline = json.loads(baseline_path.read_text())
    if (baseline["partial_parent"] != evidence or baseline["seed"] != args.seed
            or baseline["steps"] != 7000 or baseline["num_envs"] != 64
            or baseline["model_slots"] != 4 or baseline["horizon"] != 32
            or {key: len(value) for key, value in baseline["split_clips"].items()}
               != {"train": 340, "development": 6, "test": 7}):
        raise ValueError("Keep the R02C parent, input, seed and fixed remaining budget")
    pipeline_partial.pipeline.verify_bindings(baseline)
    config = json.loads(CONFIG.read_text())
    if config["experiment"] != EXPERIMENT or config["training"] != {
            "parent_step": 5000, "additional_steps": 7000, "additional_transitions": 448000,
            "environments": 64, "model_slots": 4, "horizon": 32, "seed": 20261025,
            "fresh_replay": True, "retain_networks_optimizers_and_sampler": True,
            "automatic_restart": False}:
        raise ValueError("R02D resource/training configuration differs")
    sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))
    from run_batch_pair import health

    observed_health = health(root, 14)
    if observed_health["disk_free_bytes"] < config["storage"]["minimum_free_bytes"]:
        raise RuntimeError("Disk reserve preflight failed")
    return {"status": "BOUNDED_R02D_REGISTRATION_CHECK_PASSED",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "parent": evidence["checkpoint"], "baseline_spec": parent.identity(baseline_path),
            "prerequisites": proof, "action_domain_correction": correction_declaration(),
            "config": parent.identity(CONFIG), "source": parent.identity(__file__),
            "health": observed_health, "split_counts": {key: len(value) for key, value in baseline["split_clips"].items()},
            "new_physics_steps": 0, "new_optimizer_updates": 0}


def register(args):
    preflight = check(args)
    pipeline_partial.register(args)
    path = args.experiment / "spec.json"
    spec = json.loads(path.read_text())
    files = {row["path"]: row for row in spec["frozen_files"]}
    for row in preflight["prerequisites"]["frozen_evidence"]:
        files[row["path"]] = row
    for source in [*HERE.glob("*.py"), CONFIG]:
        row = parent.identity(source)
        files[row["path"]] = row
    spec.update(registration_entry=parent.identity(__file__),
                training_entry=parent.identity(HERE / "train_bounded.py"),
                action_domain_correction=correction_declaration(),
                stability_prerequisites={key: value for key, value in preflight["prerequisites"].items()
                                         if key != "frozen_evidence"},
                matched_failed_trial=preflight["baseline_spec"],
                selection_reason="Matched R02C action-domain correction; same parent5000, 340 train references, seed, H32 and 7000 new rounds. Pre-Adam nonfinite-gradient rejection is an additional stop guard.",
                frozen_files=[files[key] for key in sorted(files)])
    pipeline_partial.pipeline.verify_bindings(spec)
    pipeline_partial.pipeline.save(path, spec)
    print(json.dumps({"status": "BOUNDED_R02D_REGISTERED_NOT_STARTED", "spec": parent.identity(path),
                      "frozen_files": len(files), "prerequisites": spec["stability_prerequisites"]}), flush=True)


def tasks_for(spec, experiment):
    tasks = pipeline_partial.tasks_for(spec, experiment)
    tasks[0][1][0] = str(HERE / "train_bounded.py")
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("check", "register", "run"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.phase != "run":
        if args.root is None or args.code_commit is None:
            parser.error("check/register needs root and code-commit")
        if args.phase == "check":
            print(json.dumps(check(args), allow_nan=False), flush=True)
        else:
            register(args)
    else:
        spec = json.loads((args.experiment / "spec.json").read_text())
        if (spec.get("action_domain_correction") != correction_declaration()
                or spec["training_entry"] != parent.identity(HERE / "train_bounded.py")
                or args.experiment.name != EXPERIMENT):
            raise ValueError("The independent bounded-action trial must be fully registered")
        pipeline_partial.pipeline.tasks_for = tasks_for
        pipeline_partial.pipeline.run(args.experiment.resolve(strict=True))


if __name__ == "__main__":
    main()
