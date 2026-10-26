#!/usr/bin/env python3
"""One bounded RF07 root-feedback trial on the completed RF06 control baseline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "project_tools"))
import run_q1_single_motion as baseline

RF06 = "BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026"
RF07 = "BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026"
SOFT_SECONDS, MAXIMUM_SECONDS, PHASE_SECONDS = 3000., 3600., 3500.


def require_parent(root):
    path = root / "data/supermimic_for_q1" / RF06
    status = json.loads((path / "status.json").read_text())
    guard = json.loads((path / "train_guard.json").read_text())
    if (status["status"] != "COMPLETED" or guard["status"] != "WORKER_COMPLETED"
            or not guard["cleanup"].get("worker_reaped")
            or guard["cleanup"].get("worker_return_code") != 0):
        raise ValueError("Finish and reap RF06 before the independent RF07 trial")
    receipt = baseline.completed_training(path / "run", 4096, 12000, SOFT_SECONDS)
    if receipt["training_seconds"] > MAXIMUM_SECONDS:
        raise ValueError("RF06 did not obey the one-hour budget")
    expected = status["process"]
    proc = Path("/proc") / str(expected["pid"]) / "stat"
    if proc.exists():
        fields = proc.read_text().rsplit(") ", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if (boot == expected["boot_id"] and int(fields[19]) == expected["start_ticks"]
                and fields[0] != "Z"):
            raise RuntimeError("RF06 controller still owns the pipeline")
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,temperature.gpu",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    memory, temperature = map(int, gpu.split(","))
    if memory >= 1024 or temperature >= 86:
        raise RuntimeError("RF07 requires a released GPU")
    return {name: baseline.identity(path / name) for name in
            ("status.json", "train_guard.json", "run/receipt.json", "comparison.json")}


def variant_commands(args, original):
    phases, library, backend = original(args)
    tasks = {"train": "train", "evaluate_initial": "evaluate", "evaluate_final": "evaluate",
             "audit_initial": "audit", "audit_final": "audit", "fullbody": "fullbody",
             "render_initial": "render", "render_final": "render"}
    for phase in phases:
        if phase["name"] in tasks:
            command = phase["command"]
            phase["command"] = [*command[:2], str(HERE / "entry_root_feedback.py"), tasks[phase["name"]], *command[3:]]
    phases.insert(next(i for i, item in enumerate(phases) if item["name"] == "render_initial"),
        {"name": "legs", "command": [sys.executable, "-u", str(HERE.parent / "leg_control/analyze.py"),
            "--root", str(args.root), "--evaluation", str(args.output / "evaluation_final/receipt.json"),
            "--output", str(args.output / "leg_analysis")]})
    return phases, library, backend


def execute(args, parent):
    original_commands, original_save = baseline.commands, baseline.save
    original_guard, original_completed = baseline.guarded_run, baseline.completed_training

    def write(path, value):
        if path == args.output / "plan.json":
            value.update(base_orchestrator=value["source"], source=baseline.identity(__file__),
                main_variable="CURRENT_REFERENCE_ANCHOR_POSITION_AND_ROTATION_FEEDBACK",
                preserved="RF06 controls, reward, reference, seed, from-zero initialization and 3000-second soft target",
                parent_evidence=parent, maximum_training_seconds=MAXIMUM_SECONDS,
                maximum_train_phase_seconds=PHASE_SECONDS,
                time_comparison="Same soft target as RF06; record actual time and update counts; no equal-update claim")
            entries = {item["path"]: item for item in value["entries"]}
            for source in HERE.glob("*.py"):
                item = baseline.identity(source)
                entries[item["path"]] = item
            value["entries"] = [entries[key] for key in sorted(entries)]
        return original_save(path, value)

    def limited_run(command, *values, **options):
        if len(command) < 4 or command[2:4] != [str(HERE / "entry_root_feedback.py"), "train"]:
            return original_guard(command, *values, **options)
        started = time.monotonic()
        previous_stop = options.get("stop_requested")
        options["stop_requested"] = lambda: bool(previous_stop and previous_stop()) or time.monotonic() - started >= PHASE_SECONDS
        return original_guard(command, *values, **options)

    def completed(*values):
        result = original_completed(*values)
        if result["training_seconds"] > MAXIMUM_SECONDS:
            raise ValueError("RF07 exceeded the user's one-hour limit")
        return result

    baseline.commands = lambda value: variant_commands(value, original_commands)
    baseline.save, baseline.guarded_run, baseline.completed_training = write, limited_run, completed
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
    args.output = (args.output or args.root / "data/supermimic_for_q1" / RF07).resolve()
    if (not args.output.is_relative_to(args.root / "data/supermimic_for_q1")
            or args.output == args.root / "data/supermimic_for_q1"
            or len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit)
            or (args.detach and args.worker)):
        parser.error("Use a new Q1 output and the published 40-character commit")
    args.steps, args.training_seconds, args.num_envs = 12000, SOFT_SECONDS, 4096
    args.threads, args.model_slots, args.wm_batch, args.batch = 8, 8, 2048, 4096
    args.dashboard_label = "RF07-E4096-ROOTFEEDBACK-MAX1H"
    parent = require_parent(args.root)
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
    execute(args, parent)


if __name__ == "__main__":
    main()
