"""Read a single owned experiment and verify process identities without mutations."""

import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess


def observe(trial):
    files = {}
    for name in ("launch.json", "status.json", "plan.json", "train_guard.json", "run/status.json",
                 "run/training_clock.json", "run/stage_decisions.json", "run/checkpoint_selection.json",
                 "delivery_status.json", "delivery_guard.json"):
        path = trial / name
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if name == "plan.json":
            value = {k:v for k,v in value.items() if k != "source"}
        files[name] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "content": value}
    pids = set()
    for name, item in files.items():
        value = item["content"]
        if isinstance(value, dict):
            process = value.get("process", {})
            if process.get("pid"):
                pids.add(process["pid"])
            pids.update(value[k] for k in ("controller_pid", "worker_pid") if value.get(k))
    processes = {}
    for pid in sorted(pids):
        path = Path("/proc") / str(pid)
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            processes[str(pid)] = {"exists": True, "start_ticks": int(stat[19]),
                                   "state": stat[0], "command": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode()}
        except FileNotFoundError:
            processes[str(pid)] = {"exists": False}
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,temperature.gpu,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    return {"captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "trial": str(trial), "files": files, "processes": processes,
            "gpu": gpu, "compute_processes": compute, "free_bytes": shutil.disk_usage(trial).free}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    print(json.dumps(observe(parser.parse_args().trial), ensure_ascii=False, allow_nan=False))
