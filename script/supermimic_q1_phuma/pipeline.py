#!/usr/bin/env python3
"""One finite PHUMA Q1 training run followed by matched complete-clip evaluations."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from run_training import admitted_clips, identity


HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "supermimic_for_q1"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify_bindings(spec):
    for record in spec["frozen_files"]:
        if identity(record["path"]) != record:
            raise ValueError("Frozen PHUMA run input changed: " + record["path"])


def register(args):
    if args.experiment.exists() or args.steps < 300 or len(args.code_commit) != 40:
        raise ValueError("Use a new experiment with an explicit finite budget and published code commit")
    admitted = admitted_clips(args.motions, args.reference_receipt, args.scan_check)
    manifest_path = args.motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    backend = json.loads(args.backend_receipt.read_text())
    if backend["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED":
        raise ValueError("Verified Q1 backend is required")
    files = {Path(row["path"]) for row in backend["source_files"] + manifest["sources"]}
    for record in backend["source_files"]:
        if identity(record["path"]) != record:
            raise ValueError("Q1 backend changed since verification")
    files.update(HERE.glob("*.py"))
    files.update(HERE.glob("*.json"))
    files.update(CORE.glob("*.py"))
    files.update([manifest_path, args.reference_receipt, args.scan_check, args.backend_receipt])
    files.update(Path(row["output"]["path"]) for row in manifest["motions"] if row["clip"] in sum(admitted.values(), []))
    spec = {"status": "REGISTERED_NOT_STARTED", "root": str(args.root.resolve()),
            "motions": str(args.motions.resolve()), "reference_receipt": str(args.reference_receipt.resolve()),
            "scan_check": str(args.scan_check.resolve()), "backend_receipt": str(args.backend_receipt.resolve()),
            "code_commit": args.code_commit, "steps": args.steps, "seed": args.seed,
            "num_envs": 16, "actor_batch": 128, "world_model_batch": 64, "horizon": 32,
            "split_clips": admitted, "frozen_files": [identity(p) for p in sorted(files)],
            "evaluation_clips": sum(admitted.values(), []), "auto_restart": False,
            "initialization": "RANDOM_NO_PARENT_CHECKPOINT", "storage_status": "LOCAL_ONLY"}
    args.experiment.mkdir(parents=True)
    save(args.experiment / "spec.json", spec)
    print(json.dumps({"status": spec["status"], "spec": identity(args.experiment / "spec.json"),
                      "split_counts": {k: len(v) for k, v in admitted.items()}}), flush=True)


def run(experiment):
    spec = json.loads((experiment / "spec.json").read_text())
    if (experiment / "controller_status.json").exists():
        raise FileExistsError("This experiment has already been launched; no automatic restart")
    verify_bindings(spec)
    stop_requested, child = False, None

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process_identity = {"pid": os.getpid(), "pgid": os.getpgid(0),
                        "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    common = ["--root", spec["root"], "--motions", spec["motions"]]
    training = [str(HERE / "run_training.py"), "--scan-check", spec["scan_check"],
                "--binding-output", str(experiment / "phuma_binding.json"), *common,
                "--output", str(experiment / "run"), "--backend-receipt", spec["backend_receipt"],
                "--reference-receipt", spec["reference_receipt"], "--steps", str(spec["steps"]),
                "--num-envs", "16", "--threads", "4", "--batch", "128", "--wm-batch", "64",
                "--wm-batches", "10", "--checkpoint-every", "500", "--seed", str(spec["seed"]),
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
    events, status, error = [], "FAILED", None
    active_phase = None
    try:
        for phase, arguments in tasks:
            if stop_requested:
                status = "STOPPED_SIGNAL"
                break
            verify_bindings(spec)
            active_phase = phase
            log = experiment / (phase + ".log")
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with log.open("x") as stream:
                child = subprocess.Popen([sys.executable, *arguments], cwd=spec["root"], stdout=stream, stderr=subprocess.STDOUT)
                if stop_requested:
                    child.send_signal(signal.SIGTERM)
                save(experiment / "controller_status.json", {"status": "RUNNING", "phase": phase,
                     "process": process_identity, "child_pid": child.pid, "started_utc": started,
                     "events": events, "log": str(log)})
                code = child.wait()
            events.append({"phase": phase, "exit_code": code, "started_utc": started,
                           "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "log": identity(log)})
            print(json.dumps(events[-1]), flush=True)
            if stop_requested:
                status = "STOPPED_SIGNAL"
                break
            if code:
                raise RuntimeError(f"PHUMA phase {phase} exited {code}")
            if phase == "train":
                receipt = json.loads((experiment / "run/receipt.json").read_text())
                if receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION":
                    status = receipt["status"]
                    break
        else:
            verify_bindings(spec)
            status = "PHUMA_TRAINING_AND_MATCHED_EVALUATION_COMPLETED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(experiment / "controller_status.json", {"status": status, "phase": active_phase,
             "process": process_identity, "events": events, "error": error,
             "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "storage_status": "LOCAL_ONLY"})
        print(json.dumps({"status": status, "error": error}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--scan-check", type=Path)
    parser.add_argument("--backend-receipt", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026102406)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.phase == "register":
        for required in ("root", "motions", "reference_receipt", "scan_check", "backend_receipt", "code_commit"):
            if getattr(args, required) is None:
                parser.error("register requires --" + required.replace("_", "-"))
        register(args)
    else:
        run(args.experiment.resolve())
