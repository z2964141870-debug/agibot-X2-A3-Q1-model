"""Read-only, fsynced host telemetry for diagnosing interrupted training."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def command(args, timeout=8):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error": str(error)}


def sample():
    result = {"at_utc": datetime.now(timezone.utc).isoformat(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    result["uptime_seconds"] = float(Path("/proc/uptime").read_text().split()[0])
    result["load_average"] = os.getloadavg()
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[name] = int(value.split()[0]) * 1024
    result["memory_bytes"] = memory
    result["temperatures_c"] = {}
    for directory in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            kind = (directory / "type").read_text().strip()
            result["temperatures_c"][kind] = int((directory / "temp").read_text()) / 1000
        except (OSError, ValueError):
            pass
    fields = ["temperature.gpu", "power.draw", "power.limit", "utilization.gpu", "memory.used"]
    gpu = command(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                   "--format=csv,noheader,nounits"])
    result["gpu_query"] = gpu
    result["gpus"] = []
    if gpu["returncode"] == 0:
        for row in csv.reader(gpu["stdout"].splitlines()):
            values = {}
            for field, value in zip(fields, row):
                try:
                    values[field] = float(value.strip())
                except ValueError:
                    values[field] = None
            result["gpus"].append(values)
    return result


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def collect_snapshot(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checks = {
        "kernel": ["uname", "-a"],
        "boots": ["journalctl", "--list-boots", "--no-pager"],
        "previous_kernel": ["journalctl", "-b", "-1", "-k", "--no-pager", "-n", "250"],
        "previous_system_tail": ["journalctl", "-b", "-1", "--no-pager", "-n", "120"],
        "pstore": ["ls", "-la", "/sys/fs/pstore"],
        "archived_pstore": ["ls", "-la", "/var/lib/systemd/pstore"],
        "gpu": ["nvidia-smi", "-q"],
        "mount": ["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "/media/yu/FAFF-E9771"],
    }
    record = {"health": sample(), "checks": {name: command(args) for name, args in checks.items()}}
    append_record(directory / "snapshot.jsonl", record)
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    snapshot = collect_snapshot(options.directory)
    print(json.dumps({"directory": str(options.directory), "health": snapshot["health"]}, indent=2))
