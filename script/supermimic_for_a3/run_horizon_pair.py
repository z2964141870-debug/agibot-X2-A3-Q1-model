#!/usr/bin/env python3
"""Run one preregistered H32/H8 pair sequentially, retaining failed stages."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from complete_run import atomic_json, health, identity


BOOT = "d994e71f-bae0-46ba-a5f7-6c2163771c6c"
GPU = "GPU-2225f49c-1e19-9891-6cae-d084e5f828cf"
ENTRY_NAMES = ("run_horizon_pair", "train", "initialization", "complete_run", "evaluate",
               "audit_run", "assess_policy", "launch")


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_files(items):
    for item in items:
        if identity(item["path"]) != item:
            raise ValueError(f"Frozen input/source changed: {item['path']}")


def check_host():
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT:
        raise RuntimeError("Registered host boot changed")
    actual = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if actual != GPU:
        raise RuntimeError("Registered GPU changed")
    return health()


def prepare(args):
    import torch

    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).resolve().parent
    parent_path = args.parent.resolve()
    parent_id = identity(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False, mmap=True)
    manifest = parent["manifest"]
    training_receipt = parent_path.parent / "receipt.json"
    receipt = json.loads(training_receipt.read_text())
    if receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION" or parent_id not in receipt["artifacts"]:
        raise ValueError("Parent training not complete")
    audit = json.loads(args.parent_audit.read_text())
    if audit["status"] != "TRAINING_AND_MATCHED_EVALUATION_VERIFIED" or audit["final_checkpoint"] != parent_id:
        raise ValueError("Parent has no matched training/evaluation audit")
    gradient = json.loads(args.gradient_check.read_text())
    if gradient["status"] != "H8_FROZEN_MODEL_ACTOR_GRADIENT_PASSED" or gradient["checkpoint"] != parent_id:
        raise ValueError("H8 graph control not passed on parent")
    diagnostic = json.loads(args.diagnostic_audit.read_text())
    if diagnostic["status"] != "ACTION_RESPONSE_EVIDENCE_INDEPENDENTLY_VERIFIED" or not diagnostic["all_numerical_controls_passed"]:
        raise ValueError("Action-response diagnostic not verified")
    diagnostic_receipt = json.loads(Path(diagnostic["diagnostic"]["path"]).read_text())
    diagnostic_plan = json.loads(Path(diagnostic_receipt["plan"]["path"]).read_text())
    if diagnostic_plan["checkpoint"] != parent_id:
        raise ValueError("Diagnostic belongs to another parent")
    a = manifest["args"]
    if (a["num_envs"], a["warmup_steps"], manifest["policy_trainer"]["horizon"]) != (16, 128, 32):
        raise ValueError("Unexpected parent trial contract")
    backend = json.loads(Path(manifest["backend_receipt"]["path"]).read_text())
    sources = [identity(source / (name + ".py")) for name in ENTRY_NAMES] + backend["source_files"]
    sources += [gradient["entry"], diagnostic["entry"], diagnostic_plan["entry"]]
    inputs = [parent_id, identity(training_receipt), identity(args.parent_audit), manifest["backend_receipt"],
              identity(args.gradient_check), identity(args.diagnostic_audit), diagnostic["diagnostic"],
              diagnostic_receipt["plan"], *audit["evaluation_receipts"]]
    inputs += manifest["train_clips"]
    inputs += [identity(Path(a["motions"]) / (clip + ".npz")) for clip in manifest["heldout_clips"]]
    inputs.append(identity(Path(a["motions"]) / "manifest.json"))
    assets = manifest["assets"]
    inputs += [assets["mjcf"], assets["control_source"]]
    verify_files(sources + inputs)
    arms = []
    for label, horizon in (("R05A_H32", 32), ("R05B_H8", 8)):
        run = Path("data/supermimic_for_a3") / (label + "_20261023")
        completion = run.with_name(run.name + "_completion")
        train_command = [sys.executable, "-u", str(source / "train.py"), "--root", a["root"],
                         "--motions", a["motions"], "--output", str(run), "--backend-receipt", a["backend_receipt"],
                         "--steps", "3000", "--num-envs", "16", "--threads", str(a["threads"]),
                         "--batch", str(a["batch"]), "--wm-batch", str(a["wm_batch"]),
                         "--wm-batches", str(a["wm_batches"]), "--warmup-steps", "128",
                         "--model-warmup-updates", "0", "--checkpoint-every", "500",
                         "--seed", "8723", "--device", a["device"], "--initialize-from", str(parent_path),
                         "--policy-horizon", str(horizon), "--clips", *[Path(x["path"]).stem for x in manifest["train_clips"]]]
        if a["compile"]:
            train_command.append("--compile")
        if horizon == 8:
            train_command.append("--allow-policy-horizon-change")
        launch_record = args.output / (label + "_train_launch.json")
        log_dir = Path("logs") / ("supermimic_a3_" + label + "_completion_20261023")
        complete_command = [sys.executable, "-u", str(source / "complete_run.py"), "--run", str(run),
                            "--launch", str(launch_record), "--output", str(completion), "--log-dir", str(log_dir)]
        for path in (run, completion, log_dir):
            if path.exists():
                raise FileExistsError(path)
        arms.append({"label": label, "horizon": horizon, "run": str(run), "completion": str(completion),
                     "train_command": train_command, "completion_command": complete_command,
                     "training_launch": str(launch_record), "log_dir": str(log_dir),
                     "expected_env_steps": 48000, "expected_wm_updates": 1149, "expected_policy_updates": 1149,
                     "imagined_transitions": horizon * a["batch"] * 1149})
    plan = {"status": "PREPARED_NOT_EXECUTED", "root": str(Path.cwd()), "output": str(args.output.resolve()),
            "boot": BOOT, "gpu": GPU, "sources": sources, "inputs": inputs, "arms": arms,
            "parent": parent_id, "seed": 8723, "order": "H32 then H8, sequential; any failed stage stops the pair",
            "matched": "Same R04 parent networks/Adam, seed, reference clips, physical/new-update budgets, batch and all settings except policy_trainer.horizon.",
            "not_matched": "Horizon changes imagined transition count, critic target/minibatch size, and objective normalization; compute and random-stream consumption are not equal after divergence. One seed is exploratory.",
            "decision": "Report all per-clip outcomes and existing unchanged usability gates. Exploratory preference only if one arm improves both training full-clip return and completion over the other; no generalization or causal certainty claim.",
            "max_new_training_env_steps": 96000, "total_expected_policy_updates": 2298,
            "resource_limits": {"memory_bytes": 30 * 1024 ** 3, "gpu_c": 86,
                                "each_training_wall_seconds": 10800, "each_completion_wall_seconds": 3600},
            "automatic_restart": False, "health_at_preparation": check_host(), "storage_status": "LOCAL_ONLY"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "plan.json", plan)
    print(json.dumps({"status": plan["status"], "plan": identity(args.output / "plan.json"),
                      "arms": arms, "max_new_training_env_steps": 96000}, indent=2), flush=True)


def execute(args):
    plan_path = args.output / "plan.json"
    plan = json.loads(plan_path.read_text())
    if str(args.output.resolve()) != plan["output"] or str(Path.cwd()) != plan["root"]:
        raise ValueError("Registered output or working directory differs")
    if (args.output / "admission.json").exists():
        raise FileExistsError("Pair already admitted; no automatic resume or retry")
    verify_files(plan["sources"] + plan["inputs"])
    initial_health = check_host()
    write_new(args.output / "admission.json", {"plan": identity(plan_path), "pid": os.getpid(),
                                              "boot": BOOT, "health": initial_health})
    child = None
    stop = False
    stages = []
    started = time.monotonic()

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def run_stage(arm, kind):
        nonlocal child
        if stop:
            raise InterruptedError("Pair stop requested")
        verify_files(plan["sources"] + plan["inputs"])
        health_snapshot = check_host()
        command = arm["train_command" if kind == "train" else "completion_command"]
        log = Path("logs") / ("supermimic_a3_" + arm["label"] + "_" + kind + "_20261023.log")
        deadline = plan["resource_limits"]["each_training_wall_seconds" if kind == "train" else "each_completion_wall_seconds"]
        stage_start = time.monotonic()
        maximum_memory, maximum_temperature = 0, 0
        with log.open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            launch = {"pid": child.pid, "pgid": child.pid, "command": command, "log": str(log.resolve()),
                      "boot_id": BOOT, "automatic_restart": False, "status": "LAUNCHED_NOT_COMPLETED"}
            if kind == "train":
                write_new(arm["training_launch"], launch)
            try:
                while child.poll() is None:
                    if stop:
                        raise InterruptedError("Pair stop requested")
                    health_snapshot = health()
                    maximum_memory = max(maximum_memory, health_snapshot["cgroup_memory_bytes"])
                    maximum_temperature = max(maximum_temperature, health_snapshot["gpu_temperature_c"])
                    if time.monotonic() - stage_start >= deadline:
                        raise TimeoutError("Pair stage wall-time protection")
                    atomic_json(args.output / "status.json", {"status": "RUNNING", "arm": arm["label"],
                                "phase": kind, "child": launch, "health": health_snapshot,
                                "stage_seconds": time.monotonic() - stage_start})
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
                stages.append({"arm": arm["label"], "phase": kind, "native_exit_code": child.returncode,
                               "child_pid": child.pid, "seconds": time.monotonic() - stage_start,
                               "max_sampled_memory_bytes": maximum_memory,
                               "max_sampled_gpu_c": maximum_temperature, "log": identity(log)})
            if child.returncode or stop:
                raise RuntimeError(f"Pair stage failed or stopped: {arm['label']}/{kind}")
            child = None

    result = {"status": "FAILED", "plan": identity(plan_path), "stages": stages,
              "storage_status": "LOCAL_ONLY", "goal_achieved": False}
    error = None
    try:
        outcomes = {}
        for arm in plan["arms"]:
            run_stage(arm, "train")
            train = json.loads((Path(arm["run"]) / "receipt.json").read_text())
            if train["status"] != "TRAINING_FINISHED_PENDING_EVALUATION" or any(
                    train[key] != arm["expected_" + key] for key in ("env_steps", "wm_updates", "policy_updates")):
                raise ValueError("Arm did not finish its registered budget")
            run_stage(arm, "complete")
            completion = Path(arm["completion"])
            evaluation = json.loads((completion / "eval_final/receipt.json").read_text())
            assessment = json.loads((completion / "policy_assessment.json").read_text())
            outcomes[arm["label"]] = {"groups": evaluation["groups"], "rows": evaluation["rows"],
                                      "gate": assessment["passed_four_training_clip_gate"],
                                      "assessment": identity(completion / "policy_assessment.json"),
                                      "audit": identity(completion / "matched_audit.json"),
                                      "evaluation": identity(completion / "eval_final/receipt.json")}
        control, candidate = outcomes["R05A_H32"], outcomes["R05B_H8"]
        keys = ("mean_normalized_return", "mean_completion")
        differences = {key: candidate["groups"]["training"][key] - control["groups"]["training"][key] for key in keys}
        preference = "H8" if all(x > 0 for x in differences.values()) else "H32" if all(x < 0 for x in differences.values()) else "MIXED"
        result.update(status="HORIZON_PAIR_COMPLETED", outcomes=outcomes, training_h8_minus_h32=differences,
                      exploratory_preference=preference, any_four_clip_gate_passed=control["gate"] or candidate["gate"])
    except Exception as exc:
        error = repr(exc)
        result["error"] = error
    finally:
        result["seconds"] = time.monotonic() - started
        atomic_json(args.output / "receipt.json", result)
        atomic_json(args.output / "status.json", result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-audit", type=Path)
    parser.add_argument("--gradient-check", type=Path)
    parser.add_argument("--diagnostic-audit", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if any(x is None for x in (args.parent, args.parent_audit, args.gradient_check, args.diagnostic_audit)):
            parser.error("Preparation requires parent and all three verification receipts")
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
