#!/usr/bin/env python3
"""Wait for one identified training job, then evaluate and audit it exactly once."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def health():
    memory = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
    temperature = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                                               "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if memory >= 30 * 1024 ** 3 or temperature >= 86:
        raise RuntimeError(f"Resource protection: memory={memory}, GPU={temperature}")
    return {"cgroup_memory_bytes": memory, "gpu_temperature_c": temperature}


def process_snapshot(launch, start_ticks=None):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != launch["boot_id"]:
        raise RuntimeError("Training host boot changed")
    path = Path("/proc") / str(launch["pid"])
    try:
        stat = (path / "stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return {"state": "MISSING", "pid": launch["pid"], "start_ticks": start_ticks}
    ticks = int(stat[19])
    if start_ticks is not None and ticks != start_ticks:
        raise RuntimeError("Training PID was reused")
    if stat[0] != "Z":
        command = (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        if command != launch["command"]:
            raise RuntimeError("Training PID command differs from its launch record")
    return {"pid": launch["pid"], "state": stat[0], "start_ticks": ticks,
            "native_wait_status": int(stat[49]) if stat[0] == "Z" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    launch = json.loads(args.launch.read_text())
    admission = json.loads((args.run / "admission.json").read_text())
    worker = process_snapshot(launch)
    start_ticks = worker["start_ticks"]
    source = Path(__file__).resolve().parent
    entries = {name: identity(source / (name + ".py")) for name in ("complete_run", "evaluate", "audit_run", "assess_policy")}
    plan = {"status": "ADMITTED_NOT_COMPLETED", "pid": os.getpid(), "training_launch": identity(args.launch),
            "training_admission": identity(args.run / "admission.json"), "initial_worker": worker,
            "entries": entries, "automatic_training_restart": False,
            "output": str(args.output.resolve()), "logs": str(args.log_dir.resolve())}
    atomic_json(args.output / "admission.json", plan)
    stage_results = []
    child = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def progress(phase, **values):
        row = {"status": "RUNNING", "phase": phase, "pid": os.getpid(),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}
        atomic_json(args.output / "status.json", row)
        print(json.dumps(row, allow_nan=False), flush=True)

    def run_stage(name, command):
        nonlocal child
        for entry in entries.values():
            if identity(entry["path"]) != entry:
                raise ValueError("Completion pipeline source changed after admission")
        if stop_requested:
            raise InterruptedError("Completion stop requested")
        progress(name, health=health(), command=command)
        log = args.log_dir / (name + ".log")
        started = time.monotonic()
        with log.open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                while child.poll() is None:
                    if stop_requested:
                        raise InterruptedError("Completion stop requested")
                    snapshot = health()
                    progress(name, child_pid=child.pid, seconds=time.monotonic() - started, health=snapshot)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
            result = {"stage": name, "native_exit_code": child.returncode,
                      "seconds": time.monotonic() - started, "log": str(log.resolve())}
        stage_results.append(result)
        if child.returncode != 0:
            raise RuntimeError(f"Completion stage failed: {name}, exit {child.returncode}")
        child = None

    status, error = "FAILED", None
    try:
        last_step = None
        while worker["state"] not in ("Z", "MISSING"):
            if stop_requested:
                raise InterruptedError("Completion stop requested")
            path = args.run / "status.json"
            training = json.loads(path.read_text()) if path.exists() else {}
            if training.get("step") != last_step:
                progress("waiting_for_training", worker=worker, training=training)
                last_step = training.get("step")
            time.sleep(10)
            worker = process_snapshot(launch, start_ticks)
        if worker.get("native_wait_status") not in (None, 0):
            raise RuntimeError("Training process exited unsuccessfully")
        receipt = json.loads((args.run / "receipt.json").read_text())
        if receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION" or receipt["step"] != admission["args"]["steps"]:
            raise ValueError("Training did not complete its admitted budget")
        if receipt["env_steps"] != admission["args"]["steps"] * admission["sim"]["num_envs"]:
            raise ValueError("Training interaction count differs")
        clips = [Path(row["path"]).stem for row in admission["train_clips"]] + admission["heldout_clips"]
        if len(clips) != len(set(clips)) or len(clips) != 8:
            raise ValueError("Expected four training and four development clips")
        baseline, final = args.output / "eval_step0", args.output / "eval_final"
        for label, checkpoint, directory in (("evaluate_step0", "step_000000.pt", baseline),
                                               ("evaluate_final", "final.pt", final)):
            command = [sys.executable, "-u", str(source / "evaluate.py"), "--root", admission["args"]["root"],
                       "--motions", admission["args"]["motions"], "--checkpoint", str(args.run / checkpoint),
                       "--output", str(directory), "--clips", *clips]
            run_stage(label, command)
        audit = args.output / "matched_audit.json"
        run_stage("audit", [sys.executable, "-u", str(source / "audit_run.py"), "--run", str(args.run),
                            "--baseline", str(baseline / "receipt.json"), "--trained", str(final / "receipt.json"),
                            "--output", str(audit)])
        assessment = args.output / "policy_assessment.json"
        run_stage("assess", [sys.executable, "-u", str(source / "assess_policy.py"),
                              "--audit", str(audit), "--output", str(assessment)])
        status = "COMPLETION_PIPELINE_FINISHED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result = {"status": status, "error": error, "worker": worker, "stages": stage_results,
                  "admission": identity(args.output / "admission.json"), "storage_status": "LOCAL_ONLY",
                  "automatic_training_restart": False,
                  "outputs": [identity(path) for path in sorted(args.output.glob("*.json"))
                              if path.name not in ("status.json", "receipt.json")],
                  "not_a_goal_completion_claim": True}
        atomic_json(args.output / "receipt.json", result)
        atomic_json(args.output / "status.json", result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
