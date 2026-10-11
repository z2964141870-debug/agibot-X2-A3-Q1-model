"""Summarize recorded load trials; never starts or resumes workloads."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from script.a3.idle_reboot_watch import save_json, utc
from script.diagnostics.host_loadcheck import LOGS, OUTPUT, ROOT


def main():
    job = json.loads((OUTPUT / "job.json").read_text())
    state = json.loads((OUTPUT / "state.json").read_text())
    rows = []
    malformed = []
    health_path = LOGS / "health.jsonl"
    health_bytes = health_path.read_bytes()
    for index, line in enumerate(health_bytes.decode().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed.append(index)
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    interrupted = state["status"] in {"RUNNING", "COOLDOWN"} and current_boot != job["initial_boot"]
    if interrupted:
        state.update(status="INTERRUPTED_REBOOT", observed_new_boot=current_boot,
                     checked_at_utc=utc(time.time()), stop_reason="CONFIRMED_NEW_BOOT_NO_AUTO_RESUME")
        running = state.get("current_stage")
        if running and state["stages"][running]["status"] == "RUNNING":
            state["stages"][running].update(status="INTERRUPTED_REBOOT",
                last_old_sample_utc=rows[-1]["health"]["at_utc"], observed_new_boot=current_boot,
                new_boot_observed_at_utc=utc(time.time()))
        save_json(OUTPUT / "state.json", state)
    stages = {}
    for stage in job["stages"]:
        for duration in stage["durations_seconds"]:
            name = f"{stage['id']}_{duration}s"
            records = [row for row in rows if row["stage"] == name]
            health = [row["health"] for row in records]
            result = dict(state["stages"].get(name, dict(status="NOT_RUN")))
            result.update(samples=len(records))
            if health:
                cpu = [h["temperatures_c"].get("x86_pkg_temp") for h in health]
                cpu = [value for value in cpu if value is not None]
                gpu = [g.get("temperature.gpu") for h in health for g in h.get("gpus", [])]
                gpu = [value for value in gpu if value is not None]
                busy = [row.get("cpu_busy_percent", {}).get("cpu") for row in records]
                busy = [value for value in busy if value is not None]
                result.update(first_sample_utc=health[0]["at_utc"], last_sample_utc=health[-1]["at_utc"],
                    max_cpu_c=max(cpu) if cpu else None, max_gpu_c=max(gpu) if gpu else None,
                    min_available_ram_gib=min(h["memory_bytes"]["MemAvailable"] for h in health)/1024**3,
                    mean_whole_host_cpu_busy_percent=sum(busy)/len(busy) if busy else None,
                    critical_kernel_lines=[line for row in records for line in row.get("critical_kernel_lines", [])])
            path = LOGS / f"{name}.log"
            if path.exists():
                data = path.read_bytes()
                result.update(actual_log_bytes=len(data), actual_log_sha256=hashlib.sha256(data).hexdigest())
                if result.get("log_sha256") and result["log_sha256"] != result["actual_log_sha256"]:
                    result.update(log_integrity_passed=False)
                elif result.get("log_sha256"):
                    result.update(log_integrity_passed=True)
            stages[name] = result
    output = dict(experiment=job["experiment"], summarized_at_utc=utc(time.time()), status=state["status"],
        initial_boot=job["initial_boot"], current_boot=current_boot, confirmed_boot_change=current_boot != job["initial_boot"],
        samples=len(rows), malformed_log_lines=malformed, stages=stages,
        health_log=dict(path=str(health_path.relative_to(ROOT)), bytes=len(health_bytes),
            sha256=hashlib.sha256(health_bytes).hexdigest()),
        interpretation="LIMITED_WORKLOAD_WINDOW_NOT_HARDWARE_CAUSE_PROOF", backup_status="LOCAL_ONLY")
    save_json(ROOT / "data/manifests/host_loadcheck_result_20261011_E01.json", output)
    print(json.dumps(dict(status=output["status"], boot_changed=output["confirmed_boot_change"],
        samples=output["samples"], results={name:result["status"] for name,result in stages.items()}), indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
