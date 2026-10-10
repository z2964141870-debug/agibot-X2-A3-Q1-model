"""Read-only R05 progress, with an SSH watcher that survives connection loss."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from script.a3.fullchain_support import DATA


def status():
    state = json.loads((DATA / "finetune_R05/state.json").read_text())
    latest = 0
    for attempt in state["attempts"]:
        for sidecar in Path(attempt["run_dir"]).glob("model_step_*.json"):
            record = json.loads(sidecar.read_text())
            if record.get("writer") == "yuanqi_durable_v1":
                latest = max(latest, record["step"])
    return {"status": state["status"], "reason": state.get("reason"), "published_step": latest,
            "attempts": len(state["attempts"]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "uptime_s": int(float(Path("/proc/uptime").read_text().split()[0])),
            "scope": "PUBLISHED_SIDECAR_PROGRESS_FINAL_RELOAD_IS_SEPARATE"}


def watch(host):
    previous = None
    command = "cd /media/yu/FAFF-E9771/YUANQI_A3 && data/environments/a3-sonic/bin/python -m script.a3.training_status"
    while True:
        try:
            result = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=2", host, command], capture_output=True, text=True, timeout=20)
            if result.returncode:
                payload = {"status": "UNKNOWN", "error": result.stderr.strip()}
            else:
                payload = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            payload = {"status": "UNKNOWN", "error": str(error)}
        signature = {key: value for key, value in payload.items() if key != "uptime_s"}
        if signature != previous:
            print(json.dumps(payload), flush=True)
            previous = signature
        if payload["status"] in ("complete", "blocked"):
            return
        time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-host")
    args = parser.parse_args()
    if args.watch_host:
        watch(args.watch_host)
    else:
        print(json.dumps(status()))
