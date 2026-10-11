"""Guarded supervisor for a single ordinary Ultralytics YOLO training run.

The supervisor uses only stdlib telemetry and stops on a new boot, unreadable
telemetry, thermal protection, or a worker failure. It never restarts a worker.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def cpu_temp() -> float | None:
    vals = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            v = float(p.read_text().strip()) / 1000.0
        except Exception:
            continue
        if 0 < v < 150:
            vals.append(v)
    return max(vals) if vals else None


def health() -> dict:
    rec: dict = {"at": time.time(), "boot_id": boot_id(), "cpu_temp_c": cpu_temp()}
    try:
        load = os.getloadavg()
        rec["load_average"] = list(load)
    except Exception:
        rec["load_average"] = None
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemAvailable:", "MemTotal:")):
                k, v = line.split()[:2]
                mem[k] = int(v) * 1024
        rec["memory_bytes"] = mem
    except Exception:
        rec["memory_bytes"] = None
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        rec["gpu_query"] = {"returncode": q.returncode, "stdout": q.stdout.strip(), "stderr": q.stderr.strip()}
        if q.returncode == 0 and q.stdout.strip():
            parts = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
            rec["gpu"] = {"temp_c": float(parts[0]), "power_w": float(parts[1]), "power_limit_w": float(parts[2]), "util_pct": float(parts[3]), "memory_used_mib": float(parts[4])}
        else:
            rec["gpu"] = None
    except Exception as exc:
        rec["gpu_query"] = {"error": repr(exc)}
        rec["gpu"] = None
    return rec


def append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    health_path = out / "health.jsonl"
    state_path = out / "state.json"
    console = (out / "worker.console.log").open("a", encoding="utf-8", buffering=1)
    initial_boot = boot_id()
    state = {"status": "RUNNING", "initial_boot_id": initial_boot, "started_at": time.time(), "batch": args.batch, "epochs": args.epochs, "imgsz": args.imgsz}
    save_state(state_path, state)
    env = os.environ.copy()
    env.update({"YOLO_PROBE_DIR": str(out), "YOLO_MODEL": args.model, "YOLO_DATA": args.data, "YOLO_EPOCHS": str(args.epochs), "YOLO_BATCH": str(args.batch), "YOLO_IMGSZ": str(args.imgsz), "YOLO_PROJECT": args.project, "YOLO_NAME": args.name})
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [args.python, args.worker]
    proc = subprocess.Popen(cmd, stdout=console, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    state["worker_pid"] = proc.pid
    save_state(state_path, state)
    hot_cpu = hot_gpu = 0
    try:
        while True:
            rec = health()
            append(health_path, rec)
            if rec["boot_id"] != initial_boot:
                state.update(status="STOPPED_REBOOT", reason="CONFIRMED_NEW_BOOT", observed_boot_id=rec["boot_id"], stopped_at=time.time())
                save_state(state_path, state)
                terminate(proc)
                return 2
            ct = rec.get("cpu_temp_c")
            gt = (rec.get("gpu") or {}).get("temp_c")
            hot_cpu = hot_cpu + 1 if ct is not None and ct >= 90 else 0
            hot_gpu = hot_gpu + 1 if gt is not None and gt >= 85 else 0
            if rec.get("memory_bytes", {}).get("MemAvailable", 10**18) < 4 * 1024**3:
                state.update(status="STOPPED_PROTECTION", reason="MEMORY_RESERVE", stopped_at=time.time())
                save_state(state_path, state)
                terminate(proc)
                return 3
            if hot_cpu >= 2 or hot_gpu >= 2:
                state.update(status="STOPPED_PROTECTION", reason="TEMPERATURE", stopped_at=time.time(), cpu_temp_c=ct, gpu_temp_c=gt)
                save_state(state_path, state)
                terminate(proc)
                return 3
            rc = proc.poll()
            if rc is not None:
                state.update(status="COMPLETED" if rc == 0 else "FAILED", worker_exit=rc, ended_at=time.time())
                save_state(state_path, state)
                return int(rc)
            time.sleep(5)
    finally:
        console.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--pythonpath", default="")
    p.add_argument("--worker", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
