"""Two-hour passive host observation; never loads models or launches compute."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "data/manifests/a3_idle_watch_20261011.json"
LOG = ROOT / "logs/a3_idle_watch_20261011/samples.jsonl"
SUMMARY = ROOT / "data/manifests/a3_idle_watch_summary_20261011.json"


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def command(argv):
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        return dict(returncode=completed.returncode, stdout=completed.stdout.strip(),
                    stderr=completed.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(returncode=None, error=str(error))


def compute_processes(output):
    result = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, executable, argv = fields
        if (re.fullmatch(r"(?:python[\d.]*|accelerate|torchrun)", executable)
                and re.search(r"script\.a3\.(?:run_controller_adapter|controller_adapter_trial|sim_entry|"
                              r"run_|autoresume|.*train)|(?:train_agent|eval_agent)_trl\.py", argv)):
            result.append(dict(pid=int(pid), argv=argv))
    return result


def sample(job):
    now = time.time()
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "SwapFree"):
            memory[name] = int(value.split()[0]) * 1024
    temperatures = {}
    for directory in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            temperatures[(directory / "type").read_text().strip()] = int((directory / "temp").read_text()) / 1000
        except (OSError, ValueError):
            pass
    processes = command(["ps", "-eo", "pid=,comm=,args="])
    gpu = command(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                   "--format=csv,noheader,nounits"])
    gpu_processes = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    units = command(["systemctl", "--user", "is-enabled", "yuanqi-a3-autoresume.service", "yuanqi-a3-longtrain.service"])
    return dict(type="sample", at_utc=utc(now), epoch=now,
                elapsed_seconds=now-job["started_epoch"],
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                uptime_seconds=float(Path("/proc/uptime").read_text().split()[0]),
                load_average=os.getloadavg(), memory_bytes=memory, temperatures_c=temperatures,
                gpu_query=gpu, gpu_rows=list(csv.reader(gpu.get("stdout", "").splitlines())),
                gpu_compute_query=gpu_processes,
                a3_compute_processes=compute_processes(processes.get("stdout", "")),
                process_query_readable=processes["returncode"] == 0,
                training_autostart=units, expires_at_utc=job["expires_at_utc"])


def append(payload):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    rows, invalid = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                invalid.append(number)
    return rows, invalid


def summarize(job, rows):
    samples = [row for row in rows if row.get("type") == "sample"]
    transitions = []
    for previous, current in zip(samples, samples[1:]):
        if current["boot_id"] != previous["boot_id"]:
            transitions.append(dict(last_old_sample_utc=previous["at_utc"],
                                    first_new_sample_utc=current["at_utc"],
                                    old_boot=previous["boot_id"], new_boot=current["boot_id"]))
    contamination = [row["at_utc"] for row in samples if row["a3_compute_processes"]
                     or row["gpu_compute_query"].get("stdout", "")]
    unknown = [row["at_utc"] for row in samples if not row["process_query_readable"]
               or row["gpu_compute_query"]["returncode"] != 0]
    return dict(started_at_utc=job["started_at_utc"], expires_at_utc=job["expires_at_utc"],
                summarized_at_utc=utc(time.time()), samples=len(samples),
                deadline_reached=time.time() >= job["expires_epoch"],
                observed_boot_transitions=transitions, observed_reboot_count_lower_bound=len(transitions),
                compute_seen_sample_times_utc=contamination, unreadable_sample_times_utc=unknown,
                interpretation="OBSERVATIONAL_ONLY_NO_TRAINING_CAUSALITY_PROOF",
                monitoring_background="STDLIB_PROCESS_AND_DRIVER_STATUS_READS_EVERY_60_SECONDS",
                sample_errors=sum(row.get("type") == "sample_error" for row in rows))


def main(initialize=False, summary_only=False):
    if initialize:
        if JOB.exists():
            raise FileExistsError("Existing observation window must not be reset")
        now = time.time()
        job = dict(watch="A3_IDLE_20261011", started_epoch=now, expires_epoch=now+7200,
                   started_at_utc=utc(now), expires_at_utc=utc(now+7200), sample_seconds=60,
                   training_allowed=False, evaluation_allowed=False, duration_seconds=7200,
                   service="yuanqi-a3-idle-watch.service", automation_id="yuanqi-a3",
                   log=str(LOG.relative_to(ROOT)))
        save_json(JOB, job)
        append(sample(job))
        print(json.dumps(job, indent=2), flush=True)
        return
    job = json.loads(JOB.read_text())
    if job["training_allowed"] or job["evaluation_allowed"] or job["duration_seconds"] != 7200:
        raise ValueError("Observation must stay passive and keep its original deadline")
    if not summary_only:
        while time.time() < job["expires_epoch"]:
            try:
                append(sample(job))
            except Exception as error:
                append(dict(type="sample_error", at_utc=utc(time.time()), error=str(error)))
            remaining = job["expires_epoch"] - time.time()
            if remaining > 0:
                time.sleep(min(job["sample_seconds"], remaining))
        append(sample(job))
    rows, invalid = read_rows(LOG)
    result = summarize(job, rows)
    result["log_parse_error_lines"] = invalid
    save_json(SUMMARY, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--initialize", action="store_true")
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    main(args.initialize, args.summarize)
