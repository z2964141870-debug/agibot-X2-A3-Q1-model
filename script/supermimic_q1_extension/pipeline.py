#!/usr/bin/env python3
"""Register and execute one bounded continuation after the Q1 matched pair."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import initialization

CORE = Path(__file__).resolve().parents[1] / "supermimic_for_q1"
sys.path.insert(0, str(CORE))
import run_batch_pair as runner


EXTENSION = Path("script/supermimic_q1_extension")


def extension_paths(root):
    return sorted((root / EXTENSION).rglob("*.py"))


def release(root, output):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    if branch != "Q1":
        raise ValueError("Continuation release must come from Q1")
    files = []
    for path in [*runner.source_paths(root), *extension_paths(root)]:
        relative = str(path.relative_to(root))
        try:
            published = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Unpublished source: {relative}") from exc
        observed = initialization.identity(path)
        if hashlib.sha256(published).hexdigest() != observed["sha256"]:
            raise ValueError(f"Unpublished source: {relative}")
        files.append({**observed, "path": relative})
    runner.write_json(output, {"code_commit": commit, "branch": branch, "files": files}, exclusive=True)


def make_stages(root, output, parent, evaluation_clips, pair, parent_run, steps, seed, commit):
    run = output / "run"
    python = sys.executable
    stages = []

    def add(name, arguments, result, expected, cpu=False):
        stages.append({"name": name, "argv": [python, "-u", *map(str, arguments)],
                       "result": str(result), "expected_status": expected, "cpu": cpu})

    add("train", [root / EXTENSION / "train.py", "--root", root, "--pair", pair,
        "--parent-run", parent_run, "--output", run, "--steps", steps, "--seed", seed,
        "--code-commit", commit], run / "receipt.json", "TRAINING_FINISHED_PENDING_EVALUATION")
    add("save_audit", [root / EXTENSION / "audit.py", "--run", run,
        "--output", output / "training_audit.json"], output / "training_audit.json",
        "CONTINUATION_SAVE_AUDIT_PASSED", cpu=True)
    entry = root / "script/supermimic_for_q1/run_contact_corrected.py"
    for label, checkpoint in (("initial", "step_000000.pt"), ("final", "final.pt")):
        evaluation = output / ("eval_" + label)
        add("eval_" + label, [entry, "evaluate", "--root", root,
            "--motions", Path(parent["motion_manifest"]["path"]).parent,
            "--checkpoint", run / checkpoint, "--output", evaluation, "--device", "cuda:0",
            "--clips", *evaluation_clips], evaluation / "receipt.json", "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED")
        audit = output / ("eval_" + label + "_audit.json")
        add("eval_" + label + "_audit", [root / runner.PACKAGE / "audit_evaluation.py", "--root", root,
            "--evaluation", evaluation, "--output", audit], audit, "EVALUATION_TRACE_AUDIT_PASSED", cpu=True)
        tracking = output / ("tracking_" + label + ".json")
        add("tracking_" + label, [root / runner.PACKAGE / "inspect_tracking.py", "--root", root,
            "--evaluation", evaluation / "receipt.json", "--output", tracking],
            tracking, "SAVED_TRACKING_DIAGNOSTIC_COMPLETED", cpu=True)
    return stages


def verify_plan(plan):
    runner.verify_frozen(plan)
    if ({str(path.resolve()) for path in extension_paths(Path(plan["root"]))}
            != {item["path"] for item in plan["extension_sources"]}):
        raise ValueError("Continuation source file set changed")


def register(root, output, release_path, pair, parent_run, steps, seed, reason):
    if steps < 3000 or steps % 500 or not reason.strip():
        raise ValueError("Specify a fixed additional budget >=3000, a multiple of 500, and selection reasoning")
    if not output.is_relative_to(root / "data/supermimic_for_q1") or output.is_relative_to(pair):
        raise ValueError("Use an independent experiment directory under data/supermimic_for_q1")
    if output.exists():
        raise FileExistsError(output)
    evidence = initialization.validated_parent(pair, parent_run)
    parent = json.loads((parent_run / "admission.json").read_text())
    if seed == parent["args"]["seed"]:
        raise ValueError("Continuation requires an explicit different seed for fresh replay")
    published = json.loads(release_path.read_text())
    if published["branch"] != "Q1" or len(published["code_commit"]) != 40:
        raise ValueError("Expected a frozen Q1 release")
    deployed = []
    for item in published["files"]:
        observed = initialization.identity(root / item["path"])
        if (observed["bytes"], observed["sha256"]) != (item["bytes"], item["sha256"]):
            raise ValueError(f"Deployment differs from release: {item['path']}")
        deployed.append(observed)
    core_set = {str(path.resolve()) for path in runner.source_paths(root)}
    extension_set = {str(path.resolve()) for path in extension_paths(root)}
    if {item["path"] for item in deployed} != core_set | extension_set:
        raise ValueError("Published source file set differs from deployment")
    parent_plan = json.loads((pair / "plan.json").read_text())
    inputs = {item["path"]: item for item in parent_plan["inputs"]}
    for item in evidence.values():
        if isinstance(item, dict) and {"path", "bytes", "sha256"} <= item.keys():
            inputs[item["path"]] = item
    for path in (pair / "plan.json", pair / "events.jsonl"):
        inputs[str(path)] = initialization.identity(path)
    extension_sources = [item for item in deployed if item["path"] in extension_set]
    for item in extension_sources:
        inputs[item["path"]] = item
    plan = {"status": "REGISTERED_NOT_STARTED", "registered_utc": runner.utc(),
        "root": str(root), "output": str(output), "logs": str(root / "logs" / ("supermimic_q1_" + output.name)),
        "code_commit": published["code_commit"], "release": initialization.identity(release_path),
        "sources": [item for item in deployed if item["path"] in core_set],
        "extension_sources": extension_sources, "inputs": list(inputs.values()),
        "python": initialization.identity(sys.executable),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "mujoco")},
        "boot_id": runner.process_identity(os.getpid())["boot_id"],
        "baseline_health": runner.health(root), "parent": evidence,
        "choice": {"parent_run": str(parent_run), "reason": reason,
                   "additional_collection_rounds": steps, "fresh_replay_seed": seed},
        "automatic_restart": False, "storage_status": "LOCAL_ONLY"}
    evaluation = json.loads(Path(evidence["evaluation"]["path"]).read_text())
    evaluation_clips = [row["clip"] for row in evaluation["rows"]]
    if not evaluation_clips or len(set(evaluation_clips)) != len(evaluation_clips):
        raise ValueError("Parent evaluation clip set is empty or duplicated")
    plan["evaluation_clips"] = evaluation_clips
    plan["stages"] = make_stages(root, output, parent, evaluation_clips, pair, parent_run,
                                  steps, seed, published["code_commit"])
    verify_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    Path(plan["logs"]).mkdir(parents=True, exist_ok=False)
    runner.write_json(output / "plan.json", plan, exclusive=True)
    print(json.dumps({"status": plan["status"], "plan": initialization.identity(output / "plan.json")}))


def summarize(plan):
    output = Path(plan["output"])
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    for stage in plan["stages"]:
        expected = initialization.identity(stage["result"])
        matching = [event for event in events if event.get("stage") == stage["name"]
                    and event.get("result") == expected and event.get("exit_code") == 0]
        if len(matching) != 1:
            raise ValueError(f"Missing or ambiguous successful event: {stage['name']}")
    saved = json.loads((output / "training_audit.json").read_text())
    if (saved["status"] != "CONTINUATION_SAVE_AUDIT_PASSED"
            or not saved["gates"] or not all(saved["gates"].values())):
        raise ValueError("Continuation save audit did not pass")
    evaluations = {}
    for label in ("initial", "final"):
        receipt_path = output / ("eval_" + label) / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        audit = json.loads((output / ("eval_" + label + "_audit.json")).read_text())
        if audit["verified"] is not True or audit["receipt"] != initialization.identity(receipt_path):
            raise ValueError("Evaluation does not match its independent audit")
        if (receipt["checkpoint"] != saved[label]
                or [row["clip"] for row in receipt["rows"]] != plan["evaluation_clips"]):
            raise ValueError("Evaluation checkpoint or clip coverage differs from registration")
        evaluations[label] = {"receipt": initialization.identity(receipt_path),
            "audit": initialization.identity(output / ("eval_" + label + "_audit.json")),
            "tracking": initialization.identity(output / ("tracking_" + label + ".json")),
            "usability_gates": receipt["usability_gates"], "rows": receipt["rows"]}
    return {"status": "CONTINUATION_PIPELINE_COMPLETED", "plan": initialization.identity(output / "plan.json"),
        "parent": plan["parent"], "choice": plan["choice"],
        "training_receipt": initialization.identity(output / "run/receipt.json"),
        "training_audit": initialization.identity(output / "training_audit.json"),
        "evaluations": evaluations, "storage_status": "LOCAL_ONLY",
        "limits": "Execution completion is separate from usability. Fresh replay warm start; reused development clips; no hardware validation."}


def execute(plan_path):
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    locks = []
    try:
        for name in (".q1_batch_pair.lock", ".q1_extension.lock"):
            lock = (Path(plan["root"]) / "logs" / name).open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "RUNNING", "runner": runner.process_identity(os.getpid()),
                 "plan": initialization.identity(plan_path), "utc": runner.utc()}
        runner.write_json(output / "started.json", state, exclusive=True)
        try:
            for stage in plan["stages"]:
                verify_plan(plan)
                runner.run_stage(plan, stage, state)
            verify_plan(plan)
            runner.write_json(output / "summary.json", summarize(plan), exclusive=True)
            state.update({"status": "CONTINUATION_PIPELINE_COMPLETED",
                          "summary": initialization.identity(output / "summary.json")})
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
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publishing = commands.add_parser("release")
    publishing.add_argument("--root", type=Path, required=True)
    publishing.add_argument("--output", type=Path, required=True)
    registration = commands.add_parser("register")
    for name in ("root", "output", "release", "pair", "parent-run"):
        registration.add_argument("--" + name, type=Path, required=True)
    registration.add_argument("--steps", type=int, required=True)
    registration.add_argument("--seed", type=int, required=True)
    registration.add_argument("--reason", required=True)
    execution = commands.add_parser("execute")
    execution.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, runner.request_stop)
    signal.signal(signal.SIGTERM, runner.request_stop)
    if args.command == "release":
        release(args.root.resolve(), args.output.resolve())
    elif args.command == "register":
        register(args.root.resolve(), args.output.resolve(), args.release.resolve(), args.pair.resolve(),
                 args.parent_run.resolve(), args.steps, args.seed, args.reason)
    else:
        execute(args.plan.resolve())


if __name__ == "__main__":
    main()
