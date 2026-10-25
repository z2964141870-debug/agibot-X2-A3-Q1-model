#!/usr/bin/env python3
"""Run the fixed RF03-UP01 single-motion experiment and its physical evaluation."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_reproduction"))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from bounded_archive_io import bounded_identity as identity
from guard_phuma_batch import MemoryBoundBudget
from pipeline_baseline import now, process_identity
from runtime_guard import StorageBudget, guarded_run, save

CLIP = "train_3419f09ba93c91d75fc1"
REFERENCE_SHA = "f9bce4c3452720cda88a5adc4111fe2354440dd04e8c8ecb50cbf94d7181b8c3"
STEPS = 12000
ENVS = 64


def commands(args):
    library = args.root / "data/supermimic_for_q1/BONES_PF02_1024_20261025/library"
    backend = args.root / "data/supermimic_for_q1/PHUMA_P05_slots_V01_20261024/receipt.json"
    run = args.output / "run"
    reproduction = args.root / "script/supermimic_q1_reproduction"
    tools = args.root / "script/project_tools"

    def phase(name, entry, *arguments):
        return {"name": name, "command": [sys.executable, "-u", str(entry),
                "--root", str(args.root), *map(str, arguments)]}

    phases = [phase("train", tools / "train_q1_with_dashboard.py",
        "--output", run, "--dashboard-label", "RF03-UP01", "--motions", library / "references",
        "--reference-receipt", library / "admission.json", "--backend-receipt", backend,
        "--steps", STEPS, "--num-envs", ENVS, "--threads", 4, "--batch", 128,
        "--wm-batch", 64, "--wm-batches", 10, "--warmup-steps", 128,
        "--model-warmup-updates", 50, "--checkpoint-every", 1000,
        "--seed", 2026102513, "--device", "cuda:0", "--clips", CLIP,
        "--code-commit", args.code_commit),
        phase("save_check", tools / "audit_q1_checkpoint.py", "--run", run,
            "--checkpoint", "final.pt", "--output", args.output / "training_audit.json",
            "--expected-oom-kill", args.expected_oom_kill)]
    for label, checkpoint in (("initial", "step_000000.pt"), ("final", "final.pt")):
        output = args.output / ("evaluation_" + label)
        phases.append(phase("evaluate_" + label, reproduction / "evaluate_reproduction.py",
            "--motions", library / "references", "--checkpoint", run / checkpoint,
            "--output", output, "--device", "cpu", "--count", 1, "--clips", CLIP))
        phases.append(phase("audit_" + label, reproduction / "audit_reproduction_evaluation.py",
            "--evaluation", output, "--output", output / "audit.json"))
    phases.append(phase("fullbody", args.root / "script/supermimic_q1_analysis/diagnose_fullbody.py",
        "analyze", "--evaluation", "initial", args.output / "evaluation_initial/receipt.json",
        "--evaluation", "final", args.output / "evaluation_final/receipt.json",
        "--output", args.output / "fullbody_analysis.json"))
    for label in ("initial", "final"):
        phases.append(phase("render_" + label, reproduction / "render_reproduction.py",
            "--evaluation", args.output / ("evaluation_" + label) / "receipt.json",
            "--clip", CLIP, "--output", args.output / ("render_" + label), "--max-seconds", 8.3))
    return phases, library, backend


def completed_training(run):
    receipt = json.loads((run / "receipt.json").read_text())
    admission = json.loads((run / "admission.json").read_text())
    if (receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION"
            or receipt["step"] != STEPS or receipt["env_steps"] != STEPS * ENVS
            or receipt["error"] is not None
            or [Path(row["path"]).stem for row in admission["train_clips"]] != [CLIP]):
        raise ValueError("The fixed single-motion budget did not finish; no final evaluation")
    return receipt


def matched_comparison(output):
    receipts = {label: json.loads((output / ("evaluation_" + label) / "receipt.json").read_text())
                for label in ("initial", "final")}
    rows = {label: receipt["rows"] for label, receipt in receipts.items()}
    if (any(len(value) != 1 or value[0]["clip"] != CLIP for value in rows.values())
            or rows["initial"][0]["reference"] != rows["final"][0]["reference"]):
        raise ValueError("The fixed initial/final physical reference identity differs")
    return {"status": "MATCHED_SINGLE_MOTION_PHYSICS_COMPARISON_COMPLETED",
        "clip": CLIP, "pairs": [{"clip": CLIP, **{label: value[0] for label, value in rows.items()}}],
        "groups": {label: receipt["groups"] for label, receipt in receipts.items()},
        "evaluations": {label: identity(output / ("evaluation_" + label) / "receipt.json")
                        for label in receipts},
        "performance_thresholds_used": False, "storage_status": "LOCAL_ONLY",
        "limits": "One trained reference, deterministic MuJoCo; no held-out or hardware claim."}


def execute(args):
    phases, library, backend = commands(args)
    entry_paths = {Path(phase["command"][2]) for phase in phases}
    entry_paths.update((Path(__file__).resolve(), HERE / "plot_training_dashboard.py"))
    reference = identity(library / "references" / (CLIP + ".npz"))
    if reference["sha256"] != REFERENCE_SHA:
        raise ValueError("The selected existing reference changed")
    plan = {"status": "PLANNED", "source": identity(__file__), "code_commit": args.code_commit,
        "reference": reference, "reference_frames": 365, "reference_fps": 50,
        "library": identity(library / "admission.json"), "backend": identity(backend),
        "entries": [identity(path) for path in sorted(entry_paths)], "phases": phases,
        "training_steps": STEPS, "num_envs": ENVS, "main_variable": "TRAINING_REFERENCE_SCOPE_ONLY",
        "preserved": "RF02 world model, H32 BPTT, reward, control, network, optimizer, seed and budget",
        "evaluation_selection": "Fixed same single reference and frame-zero start, initial and final",
        "no_automatic_retry_or_budget_extension": True, "storage_status": "LOCAL_ONLY"}
    save(args.output / "plan.json", plan)
    path = args.output / "status.json"
    state = {"status": "RUNNING", "started_utc": now(), "process": process_identity(os.getpid()),
             "phase": None, "events": [], "error": None, "storage_status": "LOCAL_ONLY"}
    with path.open("x") as stream:
        json.dump(state, stream)
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    budget = MemoryBoundBudget(StorageBudget(args.output,
        args.root / "data/references/bones_seed/archives", 100 * 1024 ** 3,
        12 * 1024 ** 3, 250 * 1024 ** 3), 30 * 1024 ** 3, args.expected_oom_kill)
    try:
        for phase in phases:
            if requested:
                raise InterruptedError("Pipeline stop requested")
            event = {"phase": phase["name"], "started_utc": now()}
            state.update(phase=phase["name"])
            state["events"].append(event)
            save(path, state)
            record = guarded_run(phase["command"], args.root, budget,
                args.output / (phase["name"] + "_guard.json"),
                args.root / "logs" / (args.output.name + "_" + phase["name"] + ".log"),
                grace_s=55., stop_requested=lambda: requested,
                progress_path=args.output / "run/status.json" if phase["name"] == "train" else None,
                maximum_idle_s=900.)
            event.update(ended_utc=now(), status=record["status"], error=record["error"])
            save(path, state)
            if requested or record["status"] == "STOPPED_SIGNAL":
                raise InterruptedError("Pipeline stop requested")
            if record["status"] != "WORKER_COMPLETED":
                raise RuntimeError(f"{phase['name']}: {record['status']}; no automatic retry")
            if phase["name"] == "train":
                completed_training(args.output / "run")
        comparison = matched_comparison(args.output)
        save(args.output / "comparison.json", comparison)
        if any(identity(row["path"]) != row for row in plan["entries"]):
            raise ValueError("A frozen orchestration, reporting or evaluation entry changed")
        state.update(status="COMPLETED", training=identity(args.output / "run/receipt.json"),
                     comparison=identity(args.output / "comparison.json"),
                     fullbody=identity(args.output / "fullbody_analysis.json"),
                     dashboard=identity(args.output / "run_dashboard/receipt.json"))
    except BaseException as exc:
        state.update(status="STOPPED_SIGNAL" if isinstance(exc, InterruptedError) else "FAILED",
                     error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["ended_utc"] = now()
        save(path, state)
    print(json.dumps(state, allow_nan=False), flush=True)


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
    args.output = (args.output or args.root /
        "data/supermimic_for_q1/BONES_RF03_UP01_12000_20261025").resolve()
    if (not args.output.is_relative_to(args.root / "data/supermimic_for_q1")
            or args.output == args.root / "data/supermimic_for_q1"
            or len(args.code_commit) != 40 or (args.detach and args.worker)):
        parser.error("Use an independent Q1 experiment and a published commit")
    if not args.worker:
        args.output.mkdir()
    if args.detach:
        log_path = args.root / "logs" / (args.output.name + "_controller.log")
        forwarded = [value for value in sys.argv[1:] if value != "--detach"]
        with log_path.open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                *forwarded, "--worker"], cwd=args.root, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launched = {"status": "LAUNCHED_PENDING_OBSERVATION", "process": process_identity(child.pid),
                    "started_utc": now(), "output": str(args.output), "log_path": str(log_path),
                    "source": identity(__file__), "code_commit": args.code_commit}
        save(args.output / "launch.json", launched)
        print(json.dumps(launched), flush=True)
        return
    execute(args)


if __name__ == "__main__":
    main()
