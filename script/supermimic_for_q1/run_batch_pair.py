#!/usr/bin/env python3
"""Register and run the bounded S01 corrected-contact batch comparison once."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys


PACKAGE = Path("script/supermimic_for_q1")
STOP_REQUESTED = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value, exclusive=False):
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(data)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def source_paths(root):
    return sorted([*root.joinpath(PACKAGE).rglob("*.py"),
                   root / PACKAGE / "improvement_plan_S01.json"])


def release(root, output):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    if branch != "Q1":
        raise ValueError("Release must come from Q1")
    records = []
    for path in source_paths(root):
        relative = str(path.relative_to(root))
        published = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        record = identity(path)
        if hashlib.sha256(published).hexdigest() != record["sha256"]:
            raise ValueError(f"Uncommitted source: {relative}")
        records.append({**record, "path": relative})
    write_json(output, {"code_commit": commit, "branch": branch, "files": records}, exclusive=True)


def health(root, oom_baseline=None):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu,memory.free",
                                   "--format=csv,noheader,nounits"], text=True, timeout=15)
    temperature, free_mib = map(int, raw.splitlines()[0].split(","))
    cg = Path("/sys/fs/cgroup/memory")
    memory = int((cg / "memory.usage_in_bytes").read_text())
    limit = int((cg / "memory.limit_in_bytes").read_text())
    counters = dict(line.split() for line in (cg / "memory.oom_control").read_text().splitlines())
    oom = int(counters["oom_kill"])
    if temperature >= 86 or memory >= 30 * 1024 ** 3 or free_mib < 2048:
        raise RuntimeError(f"Resource protection: GPU {temperature} C, free {free_mib} MiB, memory {memory}")
    if oom_baseline is not None and oom != oom_baseline:
        raise RuntimeError("Container OOM counter changed since registration")
    disk_free = shutil.disk_usage(root).free
    if disk_free < 10 * 1024 ** 3:
        raise RuntimeError("Less than 10 GiB disk space available")
    return {"utc": utc(), "gpu_temperature_c": temperature, "gpu_free_mib": free_mib,
            "cgroup_memory_bytes": memory, "cgroup_limit_bytes": limit,
            "oom_kill": oom, "disk_free_bytes": disk_free}


def process_identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return {"pid": pid, "start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def reclaim_checkpoint_cache(root):
    base = (root / "data/supermimic_for_q1").resolve()
    memory = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    before = int(memory.read_text()) if memory.exists() else None
    count = advised_bytes = 0
    for path in sorted(base.rglob("*.pt")):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base):
            continue
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        count += 1
        advised_bytes += path.stat().st_size
    after = int(memory.read_text()) if memory.exists() else None
    return {"scope": str(base), "files": count, "advised_file_bytes": advised_bytes,
            "cgroup_before_bytes": before, "cgroup_after_bytes": after,
            "contents_changed": False, "files_deleted": 0}


def verify_frozen(plan):
    root = Path(plan["root"])
    if process_identity(os.getpid())["boot_id"] != plan["boot_id"]:
        raise RuntimeError("Host boot changed; this registered experiment cannot restart")
    for item in [plan["release"], plan["python"], *plan["sources"], *plan["inputs"]]:
        if identity(item["path"]) != item:
            raise ValueError(f"Frozen file changed: {item['path']}")
    if {str(p.resolve()) for p in source_paths(root)} != {row["path"] for row in plan["sources"]}:
        raise ValueError("Source file set changed")
    versions = {name: importlib.metadata.version(name) for name in plan["packages"]}
    if versions != plan["packages"]:
        raise ValueError("Runtime package versions changed")


def make_stages(root, output, proposal, commit):
    common = proposal["common"]
    if common["collection_rounds"] != 3000 or [a["actor_batch"] for a in proposal["arms"]] != [32, 128]:
        raise ValueError("This runner implements only the registered S01 3000-round batch32/128 pair")
    python = sys.executable
    entry = root / PACKAGE / "run_contact_corrected.py"
    motions = root / proposal["motions"]
    stages, arms = [], []

    def add(name, args, result, expected, cpu=False):
        stages.append({"name": name, "argv": [python, "-u", *map(str, args)],
                       "result": str(result), "expected_status": expected, "cpu": cpu})

    for letter, arm in zip(("A", "B"), proposal["arms"]):
        name = f"R02{letter}_batch{arm['actor_batch']}"
        run = output / name
        add(name + "_train", [entry, "train", "--root", root, "--motions", motions,
            "--output", run, "--backend-receipt", root / proposal["backend_receipt"],
            "--reference-receipt", root / proposal["reference_receipt"],
            "--steps", common["collection_rounds"], "--num-envs", common["num_envs"],
            "--threads", common["threads"], "--batch", arm["actor_batch"],
            "--wm-batch", common["wm_batch"], "--wm-batches", common["wm_batches_per_update"],
            "--warmup-steps", common["warmup_steps"], "--model-warmup-updates", common["model_warmup_updates"],
            "--checkpoint-every", common["checkpoint_every"], "--seed", common["seed"],
            "--device", "cuda:0", "--clips", *proposal["train_clips"], "--code-commit", commit],
            run / "receipt.json", "TRAINING_FINISHED_PENDING_EVALUATION")
        audit = output / (name + "_training_audit.json")
        add(name + "_save_audit", [root / PACKAGE / "audit_training.py", "--run", run,
            "--output", audit], audit, "TRAINING_SAVE_AUDIT_PASSED", cpu=True)
        evaluations = {}
        for label, checkpoint in (("initial", "step_000000.pt"), ("final", "final.pt")):
            evaluation = output / (name + "_eval_" + label)
            add(name + "_eval_" + label, [entry, "evaluate", "--root", root, "--motions", motions,
                "--checkpoint", run / checkpoint, "--output", evaluation, "--device", "cuda:0",
                "--clips", *proposal["train_clips"], *proposal["development_clips"]],
                evaluation / "receipt.json", "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED")
            audit = output / (name + "_eval_" + label + "_audit.json")
            add(name + "_eval_" + label + "_audit", [root / PACKAGE / "audit_evaluation.py",
                "--root", root, "--evaluation", evaluation, "--output", audit],
                audit, "EVALUATION_TRACE_AUDIT_PASSED", cpu=True)
            tracking = output / (name + "_tracking_" + label + ".json")
            add(name + "_tracking_" + label, [root / PACKAGE / "inspect_tracking.py", "--root", root,
                "--evaluation", evaluation / "receipt.json", "--output", tracking],
                tracking, "SAVED_TRACKING_DIAGNOSTIC_COMPLETED", cpu=True)
            evaluations[label] = {"receipt": str(evaluation / "receipt.json"), "tracking": str(tracking)}
        analysis = output / (name + "_learning.json")
        add(name + "_learning", [root / PACKAGE / "analyze_learning.py", "--root", root,
            "--evaluation", evaluations["initial"]["receipt"], evaluations["final"]["receipt"],
            "--training-log", run / "metrics.jsonl", "--output", analysis],
            analysis, "SAVED_LEARNING_ANALYSIS_COMPLETED", cpu=True)
        arms.append({"name": name, "batch": arm["actor_batch"], "run": str(run), "evaluations": evaluations})
    return stages, arms


def register(root, output, release_path):
    published = json.loads(release_path.read_text())
    sources = []
    for item in published["files"]:
        observed = identity(root / item["path"])
        if (observed["bytes"], observed["sha256"]) != (item["bytes"], item["sha256"]):
            raise ValueError(f"Deployment differs from published code: {item['path']}")
        sources.append(observed)
    proposal = json.loads((root / PACKAGE / "improvement_plan_S01.json").read_text())
    paths = [root / proposal[key] for key in ("backend_receipt", "reference_receipt")]
    paths += [root / proposal["motions"] / "manifest.json"]
    paths += [root / proposal["motions"] / (clip + ".npz")
              for clip in proposal["train_clips"] + proposal["development_clips"]]
    model = root / "data/references/q1/model_20261023_P01"
    paths += sorted(p for p in model.rglob("*") if p.is_file())
    if not any(p.suffix == ".xml" for p in paths):
        raise FileNotFoundError("Q1 model files not found")
    stages, arms = make_stages(root, output, proposal, published["code_commit"])
    plan = {"status": "REGISTERED_NOT_STARTED", "registered_utc": utc(), "root": str(root),
            "output": str(output), "logs": str(root / "logs" / ("supermimic_q1_" + output.name)),
            "code_commit": published["code_commit"], "release": identity(release_path), "sources": sources,
            "inputs": [identity(p) for p in sorted(set(paths))], "python": identity(sys.executable),
            "python_version": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "mujoco")},
            "boot_id": process_identity(os.getpid())["boot_id"], "baseline_health": health(root),
            "proposal": proposal, "stages": stages, "arms": arms, "automatic_restart": False,
            "cache_policy": "Advise DONTNEED on project-local completed .pt files before stages and health samples; no content changes or global cache drops.",
            "storage_status": "LOCAL_ONLY"}
    verify_frozen(plan)
    output.mkdir(parents=True, exist_ok=False)
    Path(plan["logs"]).mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan, exclusive=True)
    print(json.dumps({"status": plan["status"], "plan": identity(output / "plan.json")}), flush=True)


def run_stage(plan, stage, state):
    if STOP_REQUESTED:
        raise InterruptedError("Stop requested before next stage")
    verify_frozen(plan)
    cache_advice = reclaim_checkpoint_cache(Path(plan["root"]))
    sample = health(Path(plan["root"]), plan["baseline_health"]["oom_kill"])
    log = Path(plan["logs"]) / (stage["name"] + ".log")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    if stage["cpu"]:
        env["CUDA_VISIBLE_DEVICES"] = ""
    child = None
    event = {"stage": stage["name"], "start_utc": utc(), "log": str(log), "cache_advice": cache_advice}
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(stage["argv"], cwd=plan["root"], env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            event["child"] = process_identity(child.pid)
            while True:
                state.update({"utc": utc(), "stage": stage["name"], "child": event["child"],
                              "log": str(log), "health": sample, "cache_advice": cache_advice})
                write_json(Path(plan["output"]) / "status.json", state)
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if STOP_REQUESTED:
                        raise InterruptedError("Stop requested during stage")
                    cache_advice = reclaim_checkpoint_cache(Path(plan["root"]))
                    sample = health(Path(plan["root"]), plan["baseline_health"]["oom_kill"])
            if code != 0:
                raise RuntimeError(f"{stage['name']} exited {code}; see {log}")
            result = json.loads(Path(stage["result"]).read_text())
            if result["status"] != stage["expected_status"]:
                raise RuntimeError(f"Unexpected result for {stage['name']}: {result['status']}")
            event["result"] = identity(stage["result"])
            reclaim_checkpoint_cache(Path(plan["root"]))
            health(Path(plan["root"]), plan["baseline_health"]["oom_kill"])
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        event.update({"end_utc": utc(), "exit_code": child.returncode if child is not None else None})
        with (Path(plan["output"]) / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)


def run_pipeline(plan, state):
    for stage in plan["stages"]:
        run_stage(plan, stage, state)


def compare(plan):
    import numpy as np

    admissions, rows = [], []
    for arm in plan["arms"]:
        run = Path(arm["run"])
        admission = json.loads((run / "admission.json").read_text())
        settings = {key: admission[key] for key in ("args", "assets", "sim", "world_model", "policy",
                    "critic", "wm_trainer", "policy_trainer", "train_clips", "source_files", "code_commit")}
        settings["args"] = {key: val for key, val in settings["args"].items() if key not in ("batch", "output")}
        settings["policy_trainer"] = {key: val for key, val in settings["policy_trainer"].items() if key != "batch_size"}
        admissions.append(settings)
        for label, paths in arm["evaluations"].items():
            evaluation = json.loads(Path(paths["receipt"]).read_text())
            tracking = {r["clip"]: r for r in json.loads(Path(paths["tracking"]).read_text())["rows"]}
            for row in evaluation["rows"]:
                with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                    state = trace["state"]
                with np.load(row["reference"]["path"], allow_pickle=False) as ref:
                    target = ref["qpos"][:len(state)]
                rows.append({"arm": arm["name"], "batch": arm["batch"], "checkpoint": label,
                    "clip": row["clip"], "split": "training" if row["clip"] in plan["proposal"]["train_clips"] else "reused_development",
                    "executed_joint_rmse_rad": row["executed_joint_rmse_rad"],
                    "leg_joint_rmse_rad": tracking[row["clip"]]["group_joint_rmse_rad"]["legs"],
                    "root_horizontal_rmse_executed_m": float(np.sqrt(np.mean(np.sum((state[:, :2] - target[:, :2]) ** 2, axis=-1)))),
                    "actual_max_horizontal_excursion_m": tracking[row["clip"]]["actual_max_horizontal_excursion_m"],
                    "reference_full_max_horizontal_excursion_m": tracking[row["clip"]]["reference_full_max_horizontal_excursion_m"],
                    "completion": row["full_clip_completion"], "failed": row["failed"],
                    "normalized_return": row["normalized_return"]})
    if admissions[0] != admissions[1]:
        raise ValueError("Between-arm settings differ beyond actor batch and output")
    return {"status": "MATCHED_BATCH_COMPARISON_COMPLETED", "rows": rows, "sole_setting_change_verified": True,
            "plan": identity(Path(plan["output"]) / "plan.json"), "storage_status": "LOCAL_ONLY",
            "limits": "Single seed and reused development data. Executed-step errors must be read with completion/failure. No automatic winner or longer training is selected."}


def execute(plan_path):
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    with (Path(plan["root"]) / "logs/.q1_batch_pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "RUNNING", "runner": process_identity(os.getpid()), "plan": identity(plan_path), "utc": utc()}
        write_json(output / "started.json", state, exclusive=True)
        try:
            run_pipeline(plan, state)
            verify_frozen(plan)
            write_json(output / "comparison.json", compare(plan), exclusive=True)
            state.update({"status": "PAIR_COMPLETED", "comparison": identity(output / "comparison.json")})
        except BaseException as exc:
            state.update({"status": "STOPPED_SIGNAL" if STOP_REQUESTED else "FAILED",
                          "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            state.update({"utc": utc(), "storage_status": "LOCAL_ONLY"})
            write_json(output / "status.json", state)
            write_json(output / "receipt.json", state, exclusive=True)
            print(json.dumps(state, allow_nan=False), flush=True)


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("release", "register"):
        sub = actions.add_parser(action)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if action == "register":
            sub.add_argument("--release", type=Path, required=True)
    actions.add_parser("execute").add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "release":
        release(args.root.resolve(), args.output.resolve())
    elif args.action == "register":
        register(args.root.resolve(), args.output.resolve(), args.release.resolve())
    else:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        execute(args.plan.resolve())


if __name__ == "__main__":
    main()
