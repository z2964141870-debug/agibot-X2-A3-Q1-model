#!/usr/bin/env python3
"""A finite expanded-reference Q1 run with verified 64-environment collection."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHUMA = HERE.parent / "supermimic_q1_phuma"
CORE = HERE.parent / "supermimic_for_q1"
sys.path.insert(0, str(PHUMA))

from pipeline import register as register_legacy_template, save, verify_bindings
from run_training import identity


def register(args):
    completion_path = args.library / "completion.json"
    library = json.loads(completion_path.read_text())
    if library["status"] != "PHUMA_EXPANDED_LIBRARY_ASSEMBLED_AND_TRAINER_ADMISSION_PASSED":
        raise ValueError("Complete composed library and preserved trainer admission are required")
    for key in ("manifest", "reference_check", "scan_check"):
        if identity(library[key]["path"]) != library[key]:
            raise ValueError("Composed library contract changed")
    backend = json.loads(args.backend_receipt.read_text())
    if (backend["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
            or backend["model_slots"] != args.model_slots
            or backend["resource_probe64"]["environments"] != 64):
        raise ValueError("The model-slot numerical/gradient/resource verification is required")
    args.motions = args.library / "references"
    args.reference_receipt = args.library / "reference_check.json"
    args.scan_check = args.library / "scan_check.json"
    register_legacy_template(args)
    spec_path = args.experiment / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec.update({"num_envs": 64, "threads": 4, "model_slots": args.model_slots,
                 "library_completion": identity(completion_path),
                 "registration_entry": identity(__file__),
                 "training_entry": identity(HERE / "train_expanded.py"),
                 "training_method": "Learned differentiable world model, imagined actor BPTT, critic; H32 unchanged.",
                 "runtime_backend": "VERIFIED_MUJOCO_MODEL_SLOTS_NO_DISCRETIZED_DR"})
    files = {Path(row["path"]): row for row in spec["frozen_files"]}
    for path in [completion_path, *HERE.glob("*.py")]:
        files[path.resolve()] = identity(path)
    spec["frozen_files"] = [files[path] for path in sorted(files)]
    save(spec_path, spec)
    verify_bindings(spec)
    print(json.dumps({"status": "EXPANDED_RUN_REGISTERED_NOT_STARTED", "spec": identity(spec_path),
                      "num_envs": 64, "model_slots": args.model_slots, "steps": args.steps,
                      "training_transitions": args.steps * 64,
                      "split_counts": {key: len(value) for key, value in spec["split_clips"].items()}}), flush=True)


def tasks_for(spec, experiment):
    common = ["--root", spec["root"], "--motions", spec["motions"]]
    training = [str(HERE / "train_expanded.py"), "--model-slots", str(spec["model_slots"]),
                "--slot-binding-output", str(experiment / "slot_binding.json"),
                "--scan-check", spec["scan_check"], "--binding-output", str(experiment / "phuma_binding.json"),
                *common, "--output", str(experiment / "run"), "--backend-receipt", spec["backend_receipt"],
                "--reference-receipt", spec["reference_receipt"], "--steps", str(spec["steps"]),
                "--num-envs", str(spec["num_envs"]), "--threads", str(spec["threads"]),
                "--batch", str(spec["actor_batch"]), "--wm-batch", str(spec["world_model_batch"]),
                "--wm-batches", "10", "--checkpoint-every", "1000", "--seed", str(spec["seed"]),
                "--code-commit", spec["code_commit"]]
    tasks = [("train", training), ("train_audit", [str(CORE / "audit_training.py"),
              "--run", str(experiment / "run"), "--output", str(experiment / "training_audit.json")])]
    for label, checkpoint in (("initial", "step_000000.pt"), ("final", "final.pt")):
        destination = experiment / ("evaluation_" + label)
        tasks.append(("evaluate_" + label, [str(CORE / "run_contact_corrected.py"), "evaluate", *common,
                      "--checkpoint", str(experiment / "run" / checkpoint), "--output", str(destination),
                      "--clips", *spec["evaluation_clips"]]))
        tasks.append(("audit_" + label, [str(CORE / "audit_evaluation.py"), "--root", spec["root"],
                      "--evaluation", str(destination), "--output", str(experiment / ("evaluation_audit_" + label + ".json"))]))
    tasks.append(("compare", [str(PHUMA / "compare.py"), "--experiment", str(experiment)]))
    return tasks


def run(experiment):
    spec = json.loads((experiment / "spec.json").read_text())
    if (experiment / "controller_status.json").exists():
        raise FileExistsError("This independent experiment was already launched; no automatic restart")
    verify_bindings(spec)
    child, requested = None, False

    def stop(signum, frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process = {"pid": os.getpid(), "pgid": os.getpgid(0),
               "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    events, status, error, phase = [], "FAILED", None, None
    try:
        for phase, arguments in tasks_for(spec, experiment):
            if requested:
                status = "STOPPED_SIGNAL"
                break
            verify_bindings(spec)
            log = Path(spec["log_directory"]) / (phase + ".log")
            started = datetime.now(timezone.utc).isoformat()
            with log.open("x") as stream:
                child = subprocess.Popen([sys.executable, *arguments], cwd=spec["root"], stdout=stream, stderr=subprocess.STDOUT)
                if requested:
                    child.send_signal(signal.SIGTERM)
                save(experiment / "controller_status.json", {"status": "RUNNING", "phase": phase,
                     "process": process, "child_pid": child.pid, "started_utc": started,
                     "events": events, "log": str(log)})
                changed, marker = time.monotonic(), None
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        if phase == "train":
                            progress = experiment / "run/status.json"
                            new_marker = progress.stat().st_mtime_ns if progress.exists() else None
                            if new_marker != marker:
                                changed, marker = time.monotonic(), new_marker
                            if time.monotonic() - changed > 900:
                                child.terminate()
                                try:
                                    child.wait(timeout=90)
                                except subprocess.TimeoutExpired:
                                    child.kill()
                                    child.wait()
                                raise RuntimeError("Expanded training stopped after 900 seconds without progress")
            events.append({"phase": phase, "exit_code": code, "started_utc": started,
                           "ended_utc": datetime.now(timezone.utc).isoformat(), "log": identity(log)})
            print(json.dumps(events[-1]), flush=True)
            if requested:
                status = "STOPPED_SIGNAL"
                break
            if code:
                raise RuntimeError(f"Expanded phase {phase} exited {code}")
            if phase == "train":
                receipt = json.loads((experiment / "run/receipt.json").read_text())
                if receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION":
                    status = receipt["status"]
                    break
        else:
            verify_bindings(spec)
            status = "EXPANDED_Q1_TRAINING_AND_MATCHED_EVALUATION_COMPLETED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(experiment / "controller_status.json", {"status": status, "phase": phase,
             "process": process, "events": events, "error": error,
             "ended_utc": datetime.now(timezone.utc).isoformat(), "storage_status": "LOCAL_ONLY"})
        print(json.dumps({"status": status, "error": error}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--backend-receipt", type=Path)
    parser.add_argument("--model-slots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.phase == "register":
        if not all((args.root, args.library, args.backend_receipt, args.code_commit)):
            parser.error("register requires root, library, backend receipt and published code commit")
        register(args)
    else:
        run(args.experiment.resolve())
