"""Guard one owned HP-Q1 phase; keep the one-hour training budget external."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from runtime_guard import StorageBudget, guarded_run, save
from guard_phuma_batch import MemoryBoundBudget
from worker import now, process_identity, identity


class Q1Budget:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.inner = MemoryBoundBudget(StorageBudget(args.output, args.runtime, 100*1024**3, 12*1024**3, 2*1024**3), 30*1024**3, args.expected_oom_kill)

    def check(self):
        evidence = self.inner.check()
        raw = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.used,temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip()
        uuid, memory, temperature = (x.strip() for x in raw.split(","))
        evidence["gpu"] = {"uuid": uuid, "memory_mib": int(memory), "temperature_c": int(temperature)}
        if int(temperature) >= 86:
            raise RuntimeError("GPU temperature reserve reached")
        clock = self.args.output / "run/training_clock.json"
        if self.args.mode == "train":
            if clock.exists():
                elapsed = time.monotonic() - json.loads(clock.read_text())["started_monotonic"]
                evidence["training_elapsed_seconds"] = elapsed
                if elapsed >= 3540:
                    raise RuntimeError("External training limit reached; no automatic extension")
            elif time.monotonic() - self.started >= 900:
                raise RuntimeError("Training initialization made no progress within 900 seconds")
        return evidence


def execute(args):
    status_path = args.output / "status.json"
    status = {"status": "RUNNING", "phase": args.mode, "started_utc": now(), "process": process_identity(), "code_commit": args.code_commit, "error": None}
    save(status_path, status)
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, "-u", str(HERE / "activate.py"), "-u", str(HERE / "worker.py"), "--mode", args.mode, "--runtime", str(args.runtime), "--library", str(args.library), "--output", str(args.output / "run")]
    if args.code_commit:
        command += ["--code-commit", args.code_commit]
    source_paths = set(HERE.glob("*.py"))
    upstream = json.loads((args.runtime / "upstream_source.json").read_text())
    for name, expected in upstream["files"].items():
        path = args.runtime / name
        if identity(path)["sha256"] != expected["sha256"]:
            raise ValueError(f"Pinned HP source changed: {name}")
        source_paths.add(path)
    source_paths.update((args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic").glob("*"))
    source_paths.update((args.runtime / "q1_assets").glob("*.xml"))
    source_paths.update(Path("/opt/data/private/ZHY/script/supermimic_for_q1/supermimic/kinematics").glob("*.py"))
    source_paths.update((args.runtime / "q1_config.yaml", args.runtime / "q1_adapter.json"))
    adapter = json.loads((args.runtime / "q1_adapter.json").read_text())
    source_paths.update(Path(adapter["source_model"]).parent.parent.rglob("*"))
    plan = {"phase": args.mode, "command": command, "code_commit": args.code_commit, "source": [identity(p) for p in sorted(source_paths) if p.is_file()], "library": identity(args.library), "runtime_source": identity(args.runtime / "upstream_source.json"), "config": identity(args.runtime / "q1_config.yaml"), "q1_adapter": identity(args.runtime / "q1_adapter.json"), "no_automatic_retry": True, "maximum_training_seconds": 3600, "soft_training_seconds": 3500, "external_training_stop_seconds": 3540, "training_grace_seconds": 45, "main_change": "HP multi-motion SOMA, root feedback/history/future targets, Q1 model-based full-effort implicit PD and GPU Warp backend"}
    save(args.output / "plan.json", plan)
    try:
        guard = guarded_run(command, Path("/opt/data/private/ZHY"), Q1Budget(args), args.output / "train_guard.json", Path("/opt/data/private/ZHY/logs") / (args.output.name + "_" + args.mode + ".log"), check_interval_s=2, grace_s=45, stop_requested=lambda: requested, progress_path=args.output / "run/status.json", maximum_idle_s=900)
        if guard["status"] != "WORKER_COMPLETED" or not guard["cleanup"]["worker_reaped"]:
            raise RuntimeError(f"{args.mode} guard: {guard['status']}")
        receipt = json.loads((args.output / "run/receipt.json").read_text())
        expected = "TRAINING_COMPLETED" if args.mode == "train" else "PROFILE_COMPLETED"
        if receipt["status"] != expected:
            raise RuntimeError("Worker did not complete the requested phase")
        if args.mode == "train" and not 3500 <= receipt["training_seconds"] <= 3600:
            raise RuntimeError("Training time contract failed")
        for item in plan["source"]:
            if identity(item["path"]) != item:
                raise ValueError("Frozen trial entry changed during execution")
        status.update(status="TRAINING_COMPLETED_PENDING_EVALUATION" if args.mode == "train" else "PROFILE_COMPLETED", receipt=identity(args.output / "run/receipt.json"))
        save(status_path, status)
        if args.mode == "train":
            phases = [
                ("evaluation", ["--runtime", str(args.runtime), "--library", str(args.library), "--run", str(args.output/"run"), "--output", str(args.output/"evaluation")], "evaluate_trial.py"),
                ("dashboard", ["--run", str(args.output/"run"), "--output", str(args.output/"dashboard")], "plot_trial.py"),
            ]
            for phase, flags, entry in phases:
                status.update(phase=phase)
                save(status_path, status)
                options = copy.copy(args)
                options.mode = phase
                post_command = [sys.executable, "-u", str(HERE/"activate.py"), "-u", str(HERE/entry), *flags]
                post = guarded_run(post_command, Path("/opt/data/private/ZHY"), Q1Budget(options), args.output/(phase+"_guard.json"), Path("/opt/data/private/ZHY/logs")/(args.output.name+"_"+phase+".log"), grace_s=45, stop_requested=lambda: requested)
                if post["status"] != "WORKER_COMPLETED":
                    raise RuntimeError(f"Training completed but {phase} failed; preserve training outcome")
            status.update(status="COMPLETED", evaluation=identity(args.output/"evaluation/receipt.json"), dashboard=identity(args.output/"dashboard/receipt.json"))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["ended_utc"] = now()
        save(status_path, status)
    print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("profile", "train"), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-oom-kill", type=int, required=True)
    p.add_argument("--code-commit")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--controller", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path("/opt/data/private/ZHY/data/supermimic_for_q1")):
        p.error("Use a new experiment inside the sole Q1 root")
    if args.mode == "train" and (not args.code_commit or len(args.code_commit) != 40):
        p.error("Formal training requires the independently published 40-digit commit")
    if not args.controller:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        forwarded = [x for x in sys.argv[1:] if x != "--detach"]
        log = Path("/opt/data/private/ZHY/logs") / (args.output.name + "_controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(HERE / "run_trial.py"), *forwarded, "--controller"], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", {"status": "LAUNCHED_PENDING_OBSERVATION", "controller_pid": child.pid, "started_utc": now(), "log": str(log)})
        print(json.dumps({"controller_pid": child.pid, "output": str(args.output)}))
        return
    execute(args)


if __name__ == "__main__":
    main()
