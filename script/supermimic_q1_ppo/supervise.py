"""Own exactly one RF12 worker, enforce cumulative guards, and reap it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE.parent / "supermimic_q1_sources"), str(HERE.parent / "supermimic_q1_hp")]
from runtime_guard import StorageBudget, guarded_run
from guard_phuma_batch import MemoryBoundBudget
from worker import process_identity
from resources import save, stamp, identity


class Budget:
    def __init__(self, args, boot, gpu):
        self.args, self.boot, self.gpu = args, boot, gpu
        self.inner = MemoryBoundBudget(StorageBudget(args.output, args.runtime,
                                                    100 * 1024**3, 12 * 1024**3, 2 * 1024**3),
                                       30 * 1024**3, args.expected_oom_kill)

    def check(self):
        result = self.inner.check()
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.boot:
            raise RuntimeError("Boot identity changed")
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,temperature.gpu",
                                         "--format=csv,noheader,nounits"], text=True).strip()
        uuid, memory, temperature = (x.strip() for x in value.split(","))
        if uuid != self.gpu or int(temperature) >= 86:
            raise RuntimeError("GPU identity or temperature protection reached")
        result["gpu"] = {"uuid": uuid, "memory_mib": int(memory), "temperature_c": int(temperature)}
        clock_path = self.args.output / "run/training_clock.json"
        if clock_path.exists():
            clock = json.loads(clock_path.read_text())
            cap = clock["current_cap_seconds"]
            if not 0 < cap <= 28800 or clock["absolute_maximum_seconds"] != 28800:
                raise RuntimeError("Invalid cumulative training admission")
            if clock["phase"] == "A" and cap > 3600 or clock["phase"] == "B" and cap > 10800:
                raise RuntimeError("Training admission exceeded A/B budget")
            if clock["phase"] == "C":
                decisions = json.loads((self.args.output / "run/stage_decisions.json").read_text())
                qualified = [d for d in decisions if d["gate"]["extension_qualified"]]
                if not qualified or cap > min(28800, qualified[0]["training_seconds"] + 18000):
                    raise RuntimeError("Extension has no matching physical admission")
            elapsed = clock["accumulated_training_seconds"]
            if clock["active_started_monotonic"] is not None:
                elapsed += time.monotonic() - clock["active_started_monotonic"]
                if elapsed >= cap - 55:
                    raise RuntimeError("External cumulative training limit; no retry or budget reset")
            result["actual_training_seconds"] = elapsed
            result["current_cap_seconds"] = cap
        return result


def sources(args):
    paths = set(HERE.glob("*.py"))
    paths.update((HERE.parent / "supermimic_q1_hp").glob("*.py"))
    paths.update((HERE.parent / "supermimic_q1_sources").joinpath(name)
                 for name in ("runtime_guard.py", "guard_phuma_batch.py"))
    upstream = json.loads((args.runtime / "upstream_source.json").read_text())
    for name, expected in upstream["files"].items():
        path = args.runtime / name
        if identity(path)["sha256"] != expected["sha256"]:
            raise ValueError("Pinned HP runtime changed: " + name)
        paths.add(path)
    paths.update((args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic").glob("*"))
    paths.update((args.runtime / "q1_assets").rglob("*"))
    paths.update(args.runtime / name for name in ("q1_config.yaml", "q1_adapter.json", "candidate_source.json", "upstream_source.json"))
    paths.update((ROOT / "script/supermimic_for_q1/supermimic/kinematics").glob("*.py"))
    adapter = json.loads((args.runtime / "q1_adapter.json").read_text())
    paths.update(Path(adapter["source_model"]).parent.parent.rglob("*"))
    rsl = Path(sysconfig.get_paths()["purelib"]) / "rsl_rl"
    paths.update(rsl.rglob("*.py"))
    paths.update((args.library, args.checkpoint))
    for path in (args.evaluation_library, args.video_suite):
        if path is not None:
            paths.add(path)
    for library_path in (args.library, args.evaluation_library):
        if library_path is not None:
            library = json.loads(library_path.read_text())
            for rows in library["splits"].values():
                for row in rows:
                    path = Path(row["path"])
                    if identity(path)["sha256"] != row["sha256"]:
                        raise ValueError("Reference content changed: " + str(path))
                    paths.add(path)
    return [identity(path) for path in sorted(paths) if path.is_file()]


def execute(args):
    status = {"status": "INITIALIZING", "phase": args.mode, "started_utc": stamp(),
              "process": process_identity(), "code_commit": args.code_commit, "no_automatic_retry": True}
    save(args.output / "status.json", status)
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if boot != args.expected_boot or gpu != args.expected_gpu:
        raise RuntimeError("Container identity differs from the admitted preflight")
    entry = "train.py" if args.mode == "train" else "preflight.py"
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(HERE / entry), "--runtime", str(args.runtime), "--library", str(args.library),
               "--checkpoint", str(args.checkpoint), "--output", str(args.output / "run")]
    if args.mode == "train":
        command += ["--evaluation-library", str(args.evaluation_library), "--video-suite", str(args.video_suite),
                    "--code-commit", args.code_commit]
    else:
        command += ["--envs", "4096" if args.mode == "preflight4096" else "16"]
    plan = {"started_utc": stamp(), "command": command, "source": sources(args),
            "process": process_identity(), "code_commit": args.code_commit, "boot": boot, "gpu": gpu,
            "mode": args.mode, "maximum_total_training_seconds": 28800,
            "checkpoint_seconds": [3500, 7100, 10700], "extension_seconds": 18000,
            "environment_count": 4096 if args.mode != "preflight16" else 16,
            "new_training_authorization": "One RF12 experiment only; in-process continuation conditional on fixed physical gates; no automatic retries."}
    save(args.output / "plan.json", plan)
    budget = Budget(args, boot, gpu)
    try:
        status.update(status="RUNNING", boot=boot, gpu=gpu)
        save(args.output / "status.json", status)
        guard = guarded_run(command, ROOT, budget, args.output / "train_guard.json",
                            ROOT / "logs" / (args.output.name + "_" + args.mode + ".log"),
                            check_interval_s=2, grace_s=45, stop_requested=lambda: requested,
                            progress_path=args.output / "run/status.json", maximum_idle_s=900)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("RF12 worker failed or was stopped; preserve terminal outcome")
        receipt = json.loads((args.output / "run/receipt.json").read_text())
        expected = "TRAINING_COMPLETED" if args.mode == "train" else "ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED"
        if receipt["status"] != expected:
            raise RuntimeError("Worker completion receipt mismatch")
        for item in plan["source"]:
            if identity(item["path"]) != item:
                raise RuntimeError("Frozen execution source changed: " + item["path"])
        if args.mode == "train":
            status.update(status="TRAINING_COMPLETED_PENDING_DELIVERY", phase="deliver")
            save(args.output / "status.json", status)
            post = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
                    str(HERE / "deliver.py"), "--runtime", str(args.runtime), "--trial", str(args.output),
                    "--video-suite", str(args.video_suite)]
            delivered = guarded_run(post, ROOT, budget, args.output / "delivery_guard.json",
                                    ROOT / "logs" / (args.output.name + "_delivery.log"),
                                    stop_requested=lambda: requested, progress_path=args.output / "delivery_status.json",
                                    maximum_idle_s=900, grace_s=45)
            if delivered["status"] != "WORKER_COMPLETED":
                raise RuntimeError("Training completed but nontraining delivery failed")
            for item in plan["source"]:
                if identity(item["path"]) != item:
                    raise RuntimeError("Frozen source changed during delivery")
        status.update(status="COMPLETED" if args.mode == "train" else "PREFLIGHT_COMPLETED", completed_utc=stamp(),
                      receipt=identity(args.output / "run/receipt.json"))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=stamp())
        raise
    finally:
        save(args.output / "status.json", status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight16", "preflight4096", "train"), required=True)
    for name in ("runtime", "library", "checkpoint", "output", "evaluation-library", "video-suite"):
        parser.add_argument("--" + name, type=Path, required=name in ("runtime", "library", "checkpoint", "output"))
    parser.add_argument("--code-commit")
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--expected-boot", required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and (not args.code_commit or len(args.code_commit) != 40
                                  or args.evaluation_library is None or args.video_suite is None):
        parser.error("Formal RF12 requires a published 40-digit commit, fixed evaluation and video suite")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use the sole ZHY data tree")
    if not args.controller:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        forwarded = [value for value in sys.argv[1:] if value != "--detach"]
        log = ROOT / "logs" / (args.output.name + "_controller.log")
        with log.open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(HERE / "supervise.py"), *forwarded, "--controller"],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", {"status": "LAUNCHED_PENDING_OBSERVATION", "controller_pid": child.pid,
                                           "started_utc": stamp(), "log": str(log)})
        print(json.dumps({"controller_pid": child.pid, "output": str(args.output)}))
    else:
        try:
            execute(args)
        except BaseException as error:
            path = args.output / "status.json"
            status = json.loads(path.read_text()) if path.exists() else {"process": process_identity()}
            status.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=stamp())
            save(path, status)
            raise


if __name__ == "__main__":
    main()
