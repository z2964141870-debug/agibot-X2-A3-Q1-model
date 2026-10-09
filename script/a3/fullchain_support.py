"""Durable task evidence and bounded official artifact acquisition."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "script/vendor/sonic_for_a3"
DATA = ROOT / "data/experiments/a3_fullchain_20261010"
LOGS = ROOT / "logs/a3_fullchain_20261010"
TASKS = ["official_pt", "input_audit", "retarget", "mujoco_baseline",
         "isaac_dependency", "isaac_baseline", "split", "finetune",
         "paired_evaluation", "reference_buffer", "onnx", "rknn"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, payload):
    from script.a3.checkpoint_store import atomic_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)


def mark(name, status, **evidence):
    path = DATA / "tasks.json"
    if path.exists():
        ledger = json.loads(path.read_text())
    else:
        ledger = {"experiment": "a3_fullchain_20261010", "backup_status": "LOCAL_ONLY",
                  "tasks": {key: {"status": "pending"} for key in TASKS},
                  "old_training_restart": False, "hardware_control": False}
    row = ledger["tasks"][name]
    row.update(status=status, **evidence,
               updated_utc=datetime.now(timezone.utc).isoformat())
    ledger["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    write_json(path, ledger)
    print(json.dumps({"task": name, **row}), flush=True)


def download():
    import requests
    manifest = json.loads((VENDOR / "a3_hf_manifest.json").read_text())
    target_root = ROOT / "data/models/a3_official_035"
    entries = [entry for entry in manifest["files"] if entry["group"] == "pt"]
    evidence = []
    mark("official_pt", "running")
    for entry in entries:
        path = target_root / entry["destination"]
        row = {**entry, "target": str(path), "revision": manifest["revision"],
               "source": f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{entry['remote']}"}
        evidence.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == entry["size"] and digest(path) == entry["sha256"]:
            row["status"] = "verified"
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            # Never replace a published artifact before independent size/hash checks.
            started = last_progress = time.monotonic()
            window_time, window_bytes = started, 0
            transferred = 0
            with requests.get(row["source"], stream=True, timeout=(10, 15)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 256):
                        now = time.monotonic()
                        if now - last_progress >= 60:
                            raise TimeoutError("No download progress for 60 seconds")
                        if not chunk:
                            continue
                        stream.write(chunk)
                        transferred += len(chunk)
                        last_progress = now
                        if now - window_time >= 60:
                            rate = (transferred - window_bytes) / (now - window_time)
                            remaining = (entry["size"] - transferred) / max(rate, 1)
                            print(json.dumps({"file": entry["remote"], "bytes": transferred,
                                              "remaining_seconds": remaining}), flush=True)
                            if remaining > 1200:
                                raise TimeoutError("Stable transfer estimate exceeds 20 minutes")
                            window_time, window_bytes = now, transferred
            if partial.stat().st_size != entry["size"] or digest(partial) != entry["sha256"]:
                raise ValueError("Official artifact size or SHA-256 mismatch")
            partial.replace(path)
            row["status"] = "verified"
        except Exception as error:
            row.update(status="deferred", error=f"{type(error).__name__}: {error}")
            row["partial_bytes"] = partial.stat().st_size if partial.exists() else 0
    write_json(DATA / "official_download.json", evidence)
    if any(item["status"] != "verified" for item in evidence):
        mark("official_pt", "deferred", evidence=str(DATA / "official_download.json"),
             user_action="Transfer listed PT/config to server targets; independently verify size/SHA afterwards")
        return 2
    import torch
    torch.set_num_threads(1)
    pt = next(Path(item["target"]) for item in evidence if item["remote"].endswith(".pt"))
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    networks = {}
    for key in ("policy_state_dict", "value_state_dict"):
        tensors = payload[key]
        networks[key] = {"tensor_count": len(tensors), "finite": all(
            not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()) for tensor in tensors.values())}
    if not all(row["finite"] for row in networks.values()):
        raise ValueError("Nonfinite official network")
    step = int(payload["state"].global_step) if "state" in payload else None
    write_json(DATA / "official_reload.json", {"checkpoint": str(pt), "networks": networks,
                                               "source_step": step, "fine_tune_start_step": 0})
    mark("official_pt", "passed", checkpoint=str(pt), evidence=str(DATA / "official_reload.json"))
    return 0


def dependencies():
    import subprocess
    modules = {}
    for name in ("mujoco", "mink", "quadprog", "smplx", "onnx", "onnxruntime", "rknn"):
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    result = subprocess.run(["/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python", "-c",
                             "from smpl_sim.smpllib.smpl_eval import compute_metrics_lite; print('METRICS_IMPORT_OK')"],
                            capture_output=True, text=True, timeout=30)
    evidence = {"modules": modules, "isaac_metrics_returncode": result.returncode,
                "isaac_metrics_stdout": result.stdout, "isaac_metrics_stderr": result.stderr[-3000:]}
    write_json(DATA / "dependencies.json", evidence)
    mark("isaac_dependency", "passed" if result.returncode == 0 else "deferred",
         evidence=str(DATA / "dependencies.json"),
         user_action=None if result.returncode == 0 else "Supply compatible smpl_sim metrics dependency; no substitute success metrics")
    return evidence


def split():
    files = sorted((VENDOR / "a3_data/agibot_a3").glob("*.csv"))
    if len(files) != 20:
        raise ValueError("Expected pinned selected20")
    random.Random(42).shuffle(files)
    report = {"seed": 42, "unit": "whole_motion", "source_fps": 120,
              "fine_tune_usage": "DIAGNOSTIC_OFFICIAL_SAMPLE_ONLY", "train": [], "heldout": []}
    for role, selections in (("train", files[:16]), ("heldout", files[16:])):
        target = DATA / "split" / role
        target.mkdir(parents=True, exist_ok=True)
        for path in selections:
            destination = target / path.name
            if destination.is_symlink() and destination.resolve() == path.resolve():
                pass
            elif destination.exists():
                raise FileExistsError(destination)
            else:
                destination.symlink_to(path)
            report[role].append({"path": str(path), "name": path.name, "sha256": digest(path)})
    write_json(DATA / "split.json", report)
    mark("split", "passed", train_count=16, heldout_count=4, evidence=str(DATA / "split.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "download", "dependencies", "split", "mark"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--status", choices=("pending", "running", "passed", "failed", "deferred"))
    parser.add_argument("--reason")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.action == "init":
        if not (DATA / "tasks.json").exists():
            mark("official_pt", "pending")
    elif args.action == "download":
        return download()
    elif args.action == "dependencies":
        print(json.dumps(dependencies(), indent=2))
    elif args.action == "split":
        split()
    else:
        if not args.task or not args.status:
            parser.error("mark requires --task and --status")
        mark(args.task, args.status, reason=args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
