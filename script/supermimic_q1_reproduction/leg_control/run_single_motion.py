#!/usr/bin/env python3
"""One RF06 leg-cap trial, using the preserved single-motion pipeline and guards."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "project_tools"))
import run_q1_single_motion as baseline

RF05 = "BONES_RF05_UP01_E4096_T02H_20261025"
RF06 = "BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026"
SOFT_TRAINING_SECONDS = 3000.
MAXIMUM_TRAINING_SECONDS = 3600.
MAXIMUM_TRAIN_PHASE_SECONDS = 3500.


def completed_trial(run, num_envs, maximum_steps, training_seconds,
                    validator=baseline.completed_training):
    result = validator(run, num_envs, maximum_steps, training_seconds)
    if result["training_seconds"] > MAXIMUM_TRAINING_SECONDS:
        raise ValueError("Actual training exceeded the user's one-hour maximum")
    return result


def limited_guard(runner, command, workdir, budget, status_path, log_path, **options):
    if not (len(command) > 3 and Path(command[2]) == HERE / "entry.py" and command[3] == "train"):
        return runner(command, workdir, budget, status_path, log_path, **options)
    started = time.monotonic()
    original_stop = options.get("stop_requested")
    reached_limit = False

    def stop():
        nonlocal reached_limit
        if original_stop and original_stop():
            return True
        reached_limit = time.monotonic() - started >= MAXIMUM_TRAIN_PHASE_SECONDS
        return reached_limit

    options["stop_requested"] = stop
    result = runner(command, workdir, budget, status_path, log_path, **options)
    result["training_phase_time_limit_seconds"] = MAXIMUM_TRAIN_PHASE_SECONDS
    result["training_phase_time_limit_reached"] = reached_limit
    if reached_limit:
        result["error"] = "User one-hour trial limit: training phase stopped; no automatic extension"
    baseline.save(status_path, result)
    return result


def require_baseline(root):
    run = root / "data/supermimic_for_q1" / RF05
    status = json.loads((run / "status.json").read_text())
    guard = json.loads((run / "train_guard.json").read_text())
    if (status["status"] != "COMPLETED" or guard["status"] != "WORKER_COMPLETED"
            or not guard.get("cleanup", {}).get("worker_reaped")
            or guard["cleanup"].get("worker_return_code") != 0):
        raise ValueError("RF05 must finish its entire pipeline and release its training worker first")
    baseline.completed_training(run / "run", 4096, 12000, 7200.)
    expected = status["process"]
    path = Path("/proc") / str(expected["pid"]) / "stat"
    if path.exists():
        fields = path.read_text().rsplit(") ", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if (boot == expected["boot_id"] and int(fields[19]) == expected["start_ticks"]
                and fields[0] != "Z"):
            raise ValueError("RF05 controller has not exited yet")
    return {name: baseline.identity(run / name) for name in (
        "status.json", "train_guard.json", "run/receipt.json", "comparison.json")}


def require_idle_gpu():
    sample = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,temperature.gpu",
                                      "--format=csv,noheader,nounits"], text=True).strip()
    memory, temperature = map(int, sample.split(","))
    if memory >= 1024 or temperature >= 86:
        raise RuntimeError("Wait for GPU release before the single authorized RF06 trial")


def variant_commands(args, original_commands):
    phases, library, backend = original_commands(args)
    tasks = {"train": "train", "evaluate_initial": "evaluate", "evaluate_final": "evaluate",
             "audit_initial": "audit", "audit_final": "audit", "fullbody": "fullbody",
             "render_initial": "render", "render_final": "render"}
    for phase in phases:
        if phase["name"] in tasks:
            old = phase["command"]
            phase["command"] = [*old[:2], str(HERE / "entry.py"), tasks[phase["name"]], *old[3:]]
    phases.insert(next(i for i, item in enumerate(phases) if item["name"] == "render_initial"),
        {"name": "legs", "command": [sys.executable, "-u", str(HERE / "analyze.py"),
            "--root", str(args.root), "--evaluation", str(args.output / "evaluation_final/receipt.json"),
            "--output", str(args.output / "leg_analysis")]})
    return phases, library, backend


def execute(args, baseline_evidence):
    original_commands, original_save = baseline.commands, baseline.save
    original_guard, original_completed = baseline.guarded_run, baseline.completed_training
    original_entries = [Path(item["command"][2]) for item in original_commands(args)[0]]

    def write(path, value):
        if path == args.output / "plan.json":
            value["base_orchestrator"] = value["source"]
            value["source"] = baseline.identity(__file__)
            value["main_variable"] = "LEG_ACTUATOR_FORCE_CAPS_AND_USER_SHORTENED_TIME_BUDGET"
            value["preserved"] = "RF05 environments, batches, algorithm, PD50/2, original 25%-based action scale, waist/arm caps, reference, seed and from-zero initialization"
            value["control_variant"] = "Q1_LEGS_CAP50_OTHER25_SCALE25_V1"
            value["maximum_training_seconds"] = MAXIMUM_TRAINING_SECONDS
            value["maximum_train_phase_seconds"] = MAXIMUM_TRAIN_PHASE_SECONDS
            value["time_budget_comparison"] = "RF05 >=7200 seconds; RF06 <=3600 seconds with a 3000-second soft target; not a matched-duration ablation"
            value["baseline_evidence"] = baseline_evidence
            sources = {item["path"]: item for item in value["entries"]}
            for entry in [*original_entries, *HERE.glob("*.py")]:
                record = baseline.identity(entry)
                sources[record["path"]] = record
            value["entries"] = [sources[key] for key in sorted(sources)]
        return original_save(path, value)

    baseline.commands = lambda value: variant_commands(value, original_commands)
    baseline.save = write
    baseline.guarded_run = lambda *values, **options: limited_guard(original_guard, *values, **options)
    baseline.completed_training = completed_trial
    try:
        baseline.execute(args)
    finally:
        baseline.commands, baseline.save = original_commands, original_save
        baseline.guarded_run, baseline.completed_training = original_guard, original_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    args.output = (args.output or args.root / "data/supermimic_for_q1" / RF06).resolve()
    if (not args.output.is_relative_to(args.root / "data/supermimic_for_q1")
            or args.output == args.root / "data/supermimic_for_q1"
            or len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit)
            or (args.detach and args.worker)):
        parser.error("Use a new Q1 experiment and the published 40-character commit")
    args.steps, args.training_seconds, args.num_envs = 12000, SOFT_TRAINING_SECONDS, 4096
    args.threads, args.model_slots, args.wm_batch, args.batch = 8, 8, 2048, 4096
    args.dashboard_label = "RF06-E4096-LEGS50-MAX1H"
    evidence = require_baseline(args.root)
    require_idle_gpu()
    if not args.worker:
        args.output.mkdir()
    if args.detach:
        log = args.root / "logs" / (args.output.name + "_controller.log")
        forwarded = [value for value in sys.argv[1:] if value != "--detach"]
        with log.open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                *forwarded, "--worker"], cwd=args.root, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {"status": "LAUNCHED_PENDING_OBSERVATION", "process": baseline.process_identity(child.pid),
            "started_utc": baseline.now(), "output": str(args.output), "log_path": str(log),
            "source": baseline.identity(__file__), "code_commit": args.code_commit}
        baseline.save(args.output / "launch.json", launch)
        print(json.dumps(launch), flush=True)
        return
    execute(args, evidence)


if __name__ == "__main__":
    main()
