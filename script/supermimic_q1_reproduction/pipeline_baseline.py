#!/usr/bin/env python3
"""Run the BONES Q1 baseline through preparation, training and physical replay."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from bounded_archive_io import bounded_identity as identity
from runtime_guard import save


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def commands(args):
    data = args.root / "data/supermimic_for_q1"
    entry = HERE / "run_reproduction.py"
    base = [sys.executable, "-u", str(entry)]
    shared = ["--root", str(args.root), "--expected-oom-kill", str(args.expected_oom_kill)]
    libraries = [data / "BONES_PF01_128_20261025/library"]
    phases = []
    for offset in range(128, 1024, 128):
        output = data / f"BONES_PF02_{offset:04d}_20261025"
        libraries.append(output / "library")
        phases.append({"name": f"prepare_{offset:04d}", "output": str(output),
            "command": [*base, "prepare", *shared, "--intake", str(args.intake),
                        "--output", str(output), "--offset", str(offset), "--count", "128"]})
    library = data / "BONES_PF02_1024_20261025/library"
    phases.append({"name": "merge", "output": str(library),
        "command": [sys.executable, "-u", str(HERE / "merge_library.py"),
                    "--root", str(args.root), "--libraries", *map(str, libraries), "--output", str(library)]})
    training = data / "BONES_RF02_12000_20261025"
    phases.append({"name": "train", "output": str(training),
        "command": [*base, "train", *shared, "--library", str(library),
                    "--backend-receipt", str(args.backend_receipt), "--output", str(training),
                    "--steps", "12000", "--code-commit", args.code_commit]})
    phases.append({"name": "save_check", "output": str(training / "training_audit.json"),
        "command": [sys.executable, str(HERE.parent / "supermimic_for_q1/audit_training.py"),
                    "--run", str(training / "run"), "--output", str(training / "training_audit.json")]})
    for label, filename in (("initial", "step_000000.pt"), ("final", "final.pt")):
        output = args.output / ("evaluation_" + label)
        phases.append({"name": "evaluate_" + label, "output": str(output),
            "command": [sys.executable, str(HERE / "evaluate_reproduction.py"),
                        "--root", str(args.root), "--motions", str(library / "references"),
                        "--checkpoint", str(training / "run" / filename),
                        "--output", str(output), "--device", "cpu", "--count", "8"]})
        phases.append({"name": "recompute_" + label, "output": str(output / "audit.json"),
            "command": [sys.executable, str(HERE / "audit_reproduction_evaluation.py"),
                        "--root", str(args.root), "--evaluation", str(output),
                        "--output", str(output / "audit.json")]})
    return phases, library, training


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=55)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def execute(args):
    path = args.output / "status.json"
    phases, library, training = commands(args)
    plan = {"status": "PLANNED", "source": identity(__file__), "code_commit": args.code_commit,
            "intake": identity(args.intake), "backend_receipt": identity(args.backend_receipt),
            "phases": phases, "training_steps": 12000, "num_envs": 64,
            "reference_policy": "USER_20261025_PAPER_FIRST_NO_LOCAL_GEOMETRY_GATES",
            "evaluation_selection": "Same lexicographically first eight train references for both checkpoints",
            "render_selection": "The first of those references, initial and final; no result-based selection",
            "storage_status": "LOCAL_ONLY"}
    state = {"status": "RUNNING", "started_utc": now(), "process": process_identity(os.getpid()),
             "phase": None, "events": [], "error": None, "storage_status": "LOCAL_ONLY"}
    with path.open("x") as stream:
        json.dump(state, stream)
    save(args.output / "plan.json", plan)
    requested = False

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    child = None

    def run(phase):
        nonlocal child
        if requested:
            raise InterruptedError("Pipeline stop requested")
        log_path = args.root / "logs" / (args.output.name + "_" + phase["name"] + ".log")
        event = {"phase": phase["name"], "started_utc": now(), "output": phase["output"],
                 "command": phase["command"], "log_path": str(log_path), "exit_code": None}
        state["phase"] = phase["name"]
        state["events"].append(event)
        save(path, state)
        with log_path.open("x") as log:
            child = subprocess.Popen(phase["command"], cwd=args.root, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            event["process"] = process_identity(child.pid)
            save(path, state)
            while child.poll() is None:
                if requested:
                    stop_child(child)
                    break
                time.sleep(1)
            event.update(exit_code=child.wait(), ended_utc=now())
        event["log"] = identity(log_path)
        save(path, state)
        child = None
        if requested:
            raise InterruptedError("Pipeline stop requested")
        if event["exit_code"]:
            raise RuntimeError(f"{phase['name']} exited {event['exit_code']}; no automatic retry")

    try:
        for phase in phases:
            run(phase)
        results = {label: json.loads((args.output / ("evaluation_" + label) / "receipt.json").read_text())
                   for label in ("initial", "final")}
        by_label = {label: {row["clip"]: row for row in receipt["rows"]}
                    for label, receipt in results.items()}
        if not by_label["initial"] or by_label["initial"].keys() != by_label["final"].keys():
            raise ValueError("Initial and final evaluation reference coverage differs")
        names = sorted(by_label["initial"])
        comparison = {"status": "MATCHED_PHYSICS_COMPARISON_COMPLETED",
                      "pairs": [{"clip": name, **{label: by_label[label][name]
                                 for label in ("initial", "final")}} for name in names],
                      "groups": {label: receipt["groups"] for label, receipt in results.items()},
                      "evaluations": {label: identity(args.output / ("evaluation_" + label) / "receipt.json")
                                      for label in results},
                      "performance_thresholds_used": False, "optimizer_updates": 0,
                      "limits": "Training-library coverage on fixed references; no held-out generalization or hardware claim.",
                      "storage_status": "LOCAL_ONLY"}
        save(args.output / "comparison.json", comparison)
        for label in ("initial", "final"):
            output = args.output / ("render_" + label)
            seconds = by_label[label][names[0]]["expected_steps"] / 50. + 1.
            run({"name": "render_" + label, "output": str(output),
                 "command": [sys.executable, str(HERE / "render_reproduction.py"),
                    "--root", str(args.root), "--evaluation",
                    str(args.output / ("evaluation_" + label) / "receipt.json"),
                    "--clip", names[0], "--output", str(output), "--max-seconds", str(seconds)]})
        state.update(status="COMPLETED", library=identity(library / "admission.json"),
                     training=identity(training / "run/receipt.json"),
                     comparison=identity(args.output / "comparison.json"))
    except BaseException as exc:
        state.update(status="STOPPED_SIGNAL" if isinstance(exc, InterruptedError) else "FAILED",
                     error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_child(child)
        state["ended_utc"] = now()
        save(path, state)
    print(json.dumps(state, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--backend-receipt", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    args.output = (args.output or args.root / "data/supermimic_for_q1/BONES_BASELINE_G02_20261025").resolve()
    if (not args.output.is_relative_to(args.root / "data/supermimic_for_q1")
            or len(args.code_commit) != 40 or (args.detach and args.worker)):
        parser.error("Use a new Q1 output directory and a published code commit")
    if not args.worker:
        args.output.mkdir()
    if args.detach:
        log_path = args.root / "logs" / (args.output.name + "_controller.log")
        forwarded = [value for value in sys.argv[1:] if value != "--detach"]
        with log_path.open("x") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                      *forwarded, "--worker"], cwd=args.root,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        launched = {"status": "LAUNCHED_PENDING_OBSERVATION", "process": process_identity(child.pid),
                    "output": str(args.output), "log_path": str(log_path), "started_utc": now(),
                    "source": identity(__file__), "code_commit": args.code_commit}
        save(args.output / "launch.json", launched)
        print(json.dumps(launched), flush=True)
        return
    execute(args)


if __name__ == "__main__":
    main()
