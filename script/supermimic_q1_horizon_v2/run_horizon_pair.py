#!/usr/bin/env python3
"""One matched H32/H8 Q1 continuation using preserved training and audit loops."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import horizon_contract as contract

continuation = contract.continuation
runner = continuation.runner
PACKAGE = Path("script/supermimic_q1_horizon_v2")
STEPS = 3000
SEED = 2026102404


def sources(root):
    return sorted((root / PACKAGE).glob("*.py"))


def verify_plan(plan):
    continuation.verify_plan(plan)
    if {str(p.resolve()) for p in sources(Path(plan["root"]))} != {
            row["path"] for row in plan["horizon_sources"]}:
        raise ValueError("Horizon source set changed")
    if (plan["choice"]["steps_per_arm"] != STEPS or plan["choice"]["seed"] != SEED
            or [(a["name"], a["horizon"]) for a in plan["arms"]] != [("A_H32", 32), ("B_H8", 8)]):
        raise ValueError("Registered horizon budget or arms changed")


def release(root, output):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip() != "Q1":
        raise ValueError("Expected Q1 release")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = [*runner.source_paths(root), *continuation.extension_paths(root), *sources(root),
             root / "script/supermimic_q1_analysis/diagnose_completed_run.py",
             root / "script/supermimic_q1_analysis/render_audited_trace.py",
             root / "script/supermimic_q1_analysis/audit_horizon_initial.py"]
    records = []
    for path in paths:
        relative = str(path.relative_to(root))
        published = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=root)
        observed = contract.identity(path)
        if hashlib.sha256(published).hexdigest() != observed["sha256"]:
            raise ValueError(f"Unpublished source {relative}")
        records.append({**observed, "path": relative})
    runner.write_json(output, {"code_commit": head, "branch": "Q1", "files": records}, exclusive=True)


def stages(root, output, parent, clips):
    plan_path = output / "plan.json"
    result, arms = [], []

    def add(name, arguments, target, expected, cpu=False):
        result.append({"name": name, "argv": [sys.executable, "-u", *map(str, arguments)],
                       "result": str(target), "expected_status": expected, "cpu": cpu})

    for name, horizon in (("A_H32", 32), ("B_H8", 8)):
        arm = {"name": name, "horizon": horizon, "run": str(output / name / "run"), "evaluations": {}}
        base = output / name
        add(name + "_train", [root / PACKAGE / "run_horizon_pair.py", "train", "--plan", plan_path,
            "--arm", name], base / "run/receipt.json", "TRAINING_FINISHED_PENDING_EVALUATION")
        add(name + "_save_audit", [root / PACKAGE / "run_horizon_pair.py", "audit", "--plan", plan_path,
            "--arm", name], base / "training_audit.json", "CONTINUATION_SAVE_AUDIT_PASSED", True)
        for phase, checkpoint in (("initial", "step_000000.pt"), ("final", "final.pt")):
            evaluation = base / ("eval_" + phase)
            add(name + "_eval_" + phase, [root / "script/supermimic_for_q1/run_contact_corrected.py",
                "evaluate", "--root", root, "--motions", Path(parent["motion_manifest"]["path"]).parent,
                "--checkpoint", base / "run" / checkpoint, "--output", evaluation,
                "--device", "cuda:0", "--clips", *clips], evaluation / "receipt.json",
                "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED")
            audit = base / ("eval_" + phase + "_audit.json")
            add(name + "_eval_" + phase + "_audit", [root / "script/supermimic_for_q1/audit_evaluation.py",
                "--root", root, "--evaluation", evaluation, "--output", audit], audit,
                "EVALUATION_TRACE_AUDIT_PASSED", True)
            tracking = base / ("tracking_" + phase + ".json")
            add(name + "_tracking_" + phase, [root / "script/supermimic_for_q1/inspect_tracking.py",
                "--root", root, "--evaluation", evaluation / "receipt.json", "--output", tracking],
                tracking, "SAVED_TRACKING_DIAGNOSTIC_COMPLETED", True)
            arm["evaluations"][phase] = {"receipt": str(evaluation / "receipt.json"),
                                          "audit": str(audit), "tracking": str(tracking)}
        arms.append(arm)
    return result, arms


def register(root, output, release_path, parent_experiment, diagnosis):
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent Q1 experiment directory")
    evidence = contract.validated_parent(parent_experiment, parent_experiment / "run")
    parent_plan = json.loads((parent_experiment / "plan.json").read_text())
    parent = json.loads(Path(evidence["admission"]["path"]).read_text())
    if parent["policy_trainer"]["horizon"] != 32 or parent["args"]["seed"] == SEED:
        raise ValueError("Expected H32 parent and a distinct fresh replay seed")
    completed = json.loads((diagnosis / "completion.json").read_text())
    if (completed["status"] != "R03_V2_ACTION_RESPONSE_AND_AUDIT_COMPLETED"
            or completed["frozen_verified_before_and_after"] is not True):
        raise ValueError("D08 diagnosis has not completed")
    for key in ("plan", "receipt", "audit"):
        contract.check_identity(completed[key])
    diagnostic_plan = json.loads(Path(completed["plan"]["path"]).read_text())
    if diagnostic_plan["checkpoint"] != evidence["checkpoint"]:
        raise ValueError("D08 checkpoint differs from the selected common parent")
    published = json.loads(release_path.read_text())
    if published["branch"] != "Q1" or len(published["code_commit"]) != 40:
        raise ValueError("Expected a Q1 release")
    deployed = []
    for item in published["files"]:
        observed = contract.identity(root / item["path"])
        if (observed["bytes"], observed["sha256"]) != (item["bytes"], item["sha256"]):
            raise ValueError("Release deployment differs")
        deployed.append(observed)
    core_set = {str(p.resolve()) for p in runner.source_paths(root)}
    extension_set = {str(p.resolve()) for p in continuation.extension_paths(root)}
    horizon_set = {str(p.resolve()) for p in sources(root)}
    inputs = {item["path"]: item for item in parent_plan["inputs"]}
    for item in [*deployed, *[v for v in evidence.values() if isinstance(v, dict) and "sha256" in v],
                 contract.identity(parent_experiment / "plan.json"),
                 contract.identity(parent_experiment / "events.jsonl"),
                 contract.identity(diagnosis / "completion.json"),
                 *[completed[k] for k in ("plan", "receipt", "audit")]]:
        inputs[item["path"]] = item
    evaluation = json.loads(Path(evidence["evaluation"]["path"]).read_text())
    clips = [row["clip"] for row in evaluation["rows"]]
    plan = {"status": "REGISTERED_NOT_STARTED", "registered_utc": runner.utc(), "root": str(root),
        "output": str(output), "logs": str(root / "logs" / ("supermimic_q1_" + output.name)),
        "code_commit": published["code_commit"], "release": contract.identity(release_path),
        "sources": [row for row in deployed if row["path"] in core_set],
        "extension_sources": [row for row in deployed if row["path"] in extension_set],
        "horizon_sources": [row for row in deployed if row["path"] in horizon_set],
        "inputs": list(inputs.values()), "python": contract.identity(sys.executable),
        "packages": parent_plan["packages"], "boot_id": parent_plan["boot_id"],
        "baseline_health": continuation.runner.health(root), "parent": evidence,
        "parent_experiment": str(parent_experiment), "evaluation_clips": clips,
        "choice": {"steps_per_arm": STEPS, "seed": SEED, "sole_setting": "policy_trainer.horizon",
            "reason": "D07 root prediction drift; D08 walking H8 16/16 versus H32 11/16 positive root responses. Test, not assume, shorter imagination."},
        "automatic_restart": False, "storage_status": "LOCAL_ONLY"}
    plan["stages"], plan["arms"] = stages(root, output, parent, clips)
    verify_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    Path(plan["logs"]).mkdir(parents=True, exist_ok=False)
    runner.write_json(output / "plan.json", plan, exclusive=True)
    print(json.dumps({"status": plan["status"], "plan": contract.identity(output / "plan.json")}))


def load_delegate(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(plan, arm):
    from train_horizon import run

    run(plan, arm)


def audit(plan, arm):
    root = Path(plan["root"])
    worker = load_delegate(root / "script/supermimic_q1_extension/audit.py", "q1_preserved_audit")
    worker.validated_parent = contract.validated_parent
    worker.validate_contract = contract.validate_contract
    report = worker.audit_run(Path(arm["run"]))
    report["horizon_audit_entry"] = contract.identity(__file__)
    report["delegated_audit_entry"] = contract.identity(root / "script/supermimic_q1_extension/audit.py")
    path = Path(arm["run"]).parent / "training_audit.json"
    runner.write_json(path, report, exclusive=True)
    print(json.dumps(report, allow_nan=False), flush=True)
    if not all(report["gates"].values()):
        raise SystemExit(2)


def compare(plan):
    import numpy as np

    events = [json.loads(line) for line in (Path(plan["output"]) / "events.jsonl").read_text().splitlines()]
    for stage in plan["stages"]:
        record = contract.identity(stage["result"])
        if sum(e.get("stage") == stage["name"] and e.get("result") == record
               and e.get("exit_code") == 0 for e in events) != 1:
            raise ValueError("Missing or repeated successful stage event")
    settings, rows, initials, audits = [], [], [], []
    for arm in plan["arms"]:
        run = Path(arm["run"])
        admission = json.loads((run / "admission.json").read_text())
        if (admission["policy_trainer"]["horizon"] != arm["horizon"]
                or admission["horizon_experiment"]["plan"] != contract.identity(Path(plan["output"]) / "plan.json")):
            raise ValueError("Arm differs from the admitted intervention")
        settings.append(contract.matched_settings(admission))
        saved = json.loads((run.parent / "training_audit.json").read_text())
        if not saved["gates"] or not all(saved["gates"].values()):
            raise ValueError("Arm save audit did not pass")
        audits.append(contract.identity(run.parent / "training_audit.json"))
        initial = []
        for phase, paths in arm["evaluations"].items():
            evaluation = contract.verified_evaluation(contract.identity(paths["receipt"]),
                                                       contract.identity(paths["audit"]))
            if evaluation["checkpoint"] != saved[phase]:
                raise ValueError("Evaluation does not use the audited checkpoint")
            tracking = {r["clip"]: r for r in json.loads(Path(paths["tracking"]).read_text())["rows"]}
            if [r["clip"] for r in evaluation["rows"]] != plan["evaluation_clips"]:
                raise ValueError("Evaluation clip coverage differs")
            for row in evaluation["rows"]:
                with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                    state = trace["state"].copy()
                with np.load(row["reference"]["path"], allow_pickle=False) as ref:
                    target = ref["qpos"][:len(state)]
                if phase == "initial":
                    initial.append((row["clip"], row["trace"]["sha256"]))
                diagnostic = tracking[row["clip"]]
                rows.append({"arm": arm["name"], "horizon": arm["horizon"], "phase": phase,
                    "clip": row["clip"], "split": "training" if row["split"] == "training" else "reused_development",
                    "executed_joint_rmse_rad": row["executed_joint_rmse_rad"],
                    "leg_joint_rmse_rad": diagnostic["group_joint_rmse_rad"]["legs"],
                    "root_horizontal_rmse_executed_m": float(np.sqrt(np.mean(np.sum((state[:, :2] - target[:, :2])**2, axis=-1)))),
                    "actual_max_horizontal_excursion_m": diagnostic["actual_max_horizontal_excursion_m"],
                    "reference_full_max_horizontal_excursion_m": diagnostic["reference_full_max_horizontal_excursion_m"],
                    "completion": row["full_clip_completion"], "failed": row["failed"],
                    "normalized_return": row["normalized_return"]})
            if phase == "initial":
                initials.append(initial)
    if settings[0] != settings[1] or initials[0] != initials[1]:
        raise ValueError("Arms differ beyond horizon/output or their initial physics traces differ")
    return {"status": "MATCHED_HORIZON_COMPARISON_COMPLETED", "sole_setting_change_verified": True,
        "initial_traces_identical": True, "parent": plan["parent"], "training_audits": audits,
        "rows": rows, "plan": contract.identity(Path(plan["output"]) / "plan.json"),
        "storage_status": "LOCAL_ONLY",
        "limits": "One seed, fixed order, reused development data. Different imagination work per update; matched real interactions/updates, not wall time. No automatic winner or extra training."}


def execute(plan_path):
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    locks = []
    try:
        for name in (".q1_batch_pair.lock", ".q1_extension.lock"):
            lock = (Path(plan["root"]) / "logs" / name).open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_plan(plan)
        state = {"status": "RUNNING", "runner": runner.process_identity(os.getpid()),
                 "plan": contract.identity(plan_path), "utc": runner.utc()}
        runner.write_json(output / "started.json", state, exclusive=True)
        try:
            for stage in plan["stages"]:
                verify_plan(plan)
                runner.run_stage(plan, stage, state)
            verify_plan(plan)
            runner.write_json(output / "comparison.json", compare(plan), exclusive=True)
            state.update({"status": "HORIZON_PAIR_COMPLETED", "comparison": contract.identity(output / "comparison.json")})
        except BaseException as exc:
            state.update({"status": "STOPPED_SIGNAL" if runner.STOP_REQUESTED else "FAILED",
                          "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            state.update({"utc": runner.utc(), "storage_status": "LOCAL_ONLY"})
            runner.write_json(output / "status.json", state)
            runner.write_json(output / "receipt.json", state, exclusive=True)
            print(json.dumps(state, allow_nan=False), flush=True)
    finally:
        for lock in reversed(locks):
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="mode", required=True)
    publication = commands.add_parser("release")
    for key in ("root", "output"):
        publication.add_argument("--" + key, type=Path, required=True)
    registration = commands.add_parser("register")
    for key in ("root", "output", "release", "parent-experiment", "diagnosis"):
        registration.add_argument("--" + key, type=Path, required=True)
    for mode in ("execute", "train", "audit"):
        sub = commands.add_parser(mode)
        sub.add_argument("--plan", type=Path, required=True)
        if mode != "execute":
            sub.add_argument("--arm", choices=("A_H32", "B_H8"), required=True)
    args = parser.parse_args()
    if args.mode == "release":
        release(args.root.resolve(), args.output.resolve())
    elif args.mode == "register":
        register(args.root.resolve(), args.output.resolve(), args.release.resolve(),
                 args.parent_experiment.resolve(), args.diagnosis.resolve())
    elif args.mode == "execute":
        signal.signal(signal.SIGTERM, runner.request_stop)
        signal.signal(signal.SIGINT, runner.request_stop)
        execute(args.plan.resolve())
    else:
        plan = json.loads(args.plan.read_text())
        verify_plan(plan)
        arm = next(a for a in plan["arms"] if a["name"] == args.arm)
        (train if args.mode == "train" else audit)(plan, arm)


if __name__ == "__main__":
    main()
