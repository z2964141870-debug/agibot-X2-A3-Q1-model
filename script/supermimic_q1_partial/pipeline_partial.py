#!/usr/bin/env python3
"""Register a new 7000-round run from the failed R02's audited step5000."""

import argparse
import json
from pathlib import Path
import sys

import parent


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_expanded"))
import expanded_pipeline as pipeline


def tasks_for(spec, experiment):
    core = HERE.parent / "supermimic_for_q1"
    training = [str(HERE / "train_partial.py"), "--root", spec["root"],
                "--pair", spec["partial_development_directory"], "--parent-run", spec["parent_run"],
                "--output", str(experiment / "run"), "--steps", "7000", "--seed", str(spec["seed"]),
                "--code-commit", spec["code_commit"]]
    tasks = [("train", training), ("train_audit", [str(HERE / "audit_partial.py"),
              "--run", str(experiment / "run"), "--output", str(experiment / "training_audit.json")])]
    for label, checkpoint in (("initial", "step_000000.pt"), ("final", "final.pt")):
        destination = experiment / ("evaluation_" + label)
        tasks.append(("evaluate_" + label, [str(core / "run_contact_corrected.py"), "evaluate",
                      "--root", spec["root"], "--motions", spec["motions"],
                      "--checkpoint", str(experiment / "run" / checkpoint), "--output", str(destination),
                      "--clips", *spec["evaluation_clips"]]))
        tasks.append(("audit_" + label, [str(core / "audit_evaluation.py"), "--root", spec["root"],
                      "--evaluation", str(destination), "--output", str(experiment / ("evaluation_audit_" + label + ".json"))]))
    tasks.append(("compare", [str(HERE.parent / "supermimic_q1_phuma/compare.py"), "--experiment", str(experiment)]))
    return tasks


def register(args):
    root, output = args.root.resolve(strict=True), args.experiment.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("A new independent partial continuation directory is required")
    if len(args.code_commit) != 40:
        raise ValueError("Use a published full Q1 commit")
    original = root / "data/supermimic_for_q1/PHUMA_R02_expanded64_H32_12000_20261024"
    development = root / "data/supermimic_for_q1/PHUMA_R02_DEV_A_step5000_20261025"
    evidence = parent.validated_parent(development, original / "run")
    spec = json.loads((original / "spec.json").read_text())
    admission = json.loads((original / "run/admission.json").read_text())
    if (spec["steps"] != 12000 or evidence["parent_counts"]["step"] + 7000 != spec["steps"]
            or spec["num_envs"] != 64 or spec["model_slots"] != 4 or admission["sim"]["num_envs"] != 64
            or args.seed == admission["args"]["seed"]):
        raise ValueError("Retain the original 64-env/4-slot contract, explicit fresh seed, and remaining 7000 budget")
    logs = root / "logs" / output.name
    if logs.exists():
        raise FileExistsError(logs)
    files = {row["path"]: row for row in spec["frozen_files"]}
    for path in [*HERE.glob("*.py"), *parent.EXTENSION.glob("*.py"),
                 HERE.parent / "supermimic_q1_sources/runtime_guard.py",
                 HERE / "partial_R02_20261025.json"]:
        record = parent.identity(path)
        files[record["path"]] = record
    for record in evidence.values():
        if isinstance(record, dict) and {"path", "bytes", "sha256"} <= record.keys():
            files[record["path"]] = record
    for path in (Path(evidence["save_audit"]["path"]), development / "plan.json"):
        record = json.loads(path.read_text())["source"]
        parent.check_identity(record)
        files[record["path"]] = record
    spec.update(status="REGISTERED_NOT_STARTED", log_directory=str(logs), code_commit=args.code_commit,
                steps=7000, seed=args.seed, initialization="AUDITED_PARTIAL_PARENT_FRESH_REPLAY",
                parent_run=str(original / "run"), partial_development_directory=str(development),
                partial_parent=evidence, cumulative_collection_rounds_at_completion=12000,
                parent_failure_unchanged=True, automatic_restart=False,
                registration_entry=parent.identity(__file__), training_entry=parent.identity(HERE / "train_partial.py"),
                selection_reason="Finish the interrupted fixed interaction budget after verified resource recovery; development shows stability gains but no locomotion success. No claim that more training is sufficient.",
                replay_and_rng="Fresh replay, episodes and seed; networks, Adam, return bins and LR offset5000 retained.",
                frozen_files=[files[key] for key in sorted(files)])
    pipeline.verify_bindings(spec)
    output.mkdir()
    logs.mkdir()
    pipeline.save(output / "spec.json", spec)
    print(json.dumps({"status": spec["status"], "spec": parent.identity(output / "spec.json"),
                      "parent_step": 5000, "additional_steps": 7000, "additional_transitions": 448000,
                      "frozen_files": len(files), "parent_terminal": "FAILED_UNCHANGED"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.phase == "register":
        if args.root is None or args.code_commit is None:
            parser.error("register needs --root and --code-commit")
        register(args)
    else:
        pipeline.tasks_for = tasks_for
        pipeline.run(args.experiment.resolve(strict=True))


if __name__ == "__main__":
    main()
