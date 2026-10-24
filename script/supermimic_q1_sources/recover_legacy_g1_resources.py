#!/usr/bin/env python3
"""Inspect or reap the six identified, failed October 21 G1 evaluations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


BOOT = "d994e71f-bae0-46ba-a5f7-6c2163771c6c"
BASE = Path("/opt/data/private/yuanqi_ops/data/g1_leg")
EXPECTED = [
    (399565, 3587246244, "onset_walk_onset_f0", "onset_pilot/walk_onset_f0.log", "KeyError: 'fps'"),
    (400254, 3587250492, "onset_walk_onset_f50", "onset_pilot/walk_onset_f50.log", "KeyError: 'fps'"),
    (400945, 3587254861, "onset_walk_onset_f100", "onset_pilot/walk_onset_f100.log", "KeyError: 'fps'"),
    (401666, 3587259231, "onset_walk_onset_f200", "onset_pilot/walk_onset_f200.log", "KeyError: 'fps'"),
    (439955, 3588563843, "phuma_test1", "phuma_test1.log", "AssertionError"),
    (452846, 3588655310, "phuma_Ways_to_Catch_A_Cold_clip1_chunk_0000",
     "phuma_Ways_to_Catch_A_Cold_clip1_chunk_0000.log", "AssertionError"),
]


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def process(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        row = {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
               "start_ticks": int(fields[19]), "cpu_ticks": int(fields[11]) + int(fields[12])}
        if row["state"] != "Z":
            row.update(cmdline=(path / "cmdline").read_bytes().split(b"\0")[:-1],
                       cwd=os.readlink(path / "cwd"), stdout=os.readlink(path / "fd/1"),
                       io=(path / "io").read_text(),
                       rss_bytes=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
            row["cmdline"] = [value.decode() for value in row["cmdline"]]
        return row
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "missing": True}


def resources():
    cg = Path("/sys/fs/cgroup/memory")
    disk = os.statvfs("/opt/data/private/YuanQi_Q1")
    return {"memory": {name: (cg / name).read_text() for name in
                       ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.oom_control", "memory.stat")},
            "disk_available_bytes": disk.f_bavail * disk.f_frsize,
            "gpu": subprocess.check_output([
                "nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                "--format=csv,noheader"], text=True).strip()}


def write_new(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminate-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot != BOOT:
        raise RuntimeError("Host boot changed; this narrowly scoped recovery no longer applies")
    before = resources()
    rows = []
    for pid, ticks, name, rel, error in EXPECTED:
        row = process(pid)
        log = BASE / rel
        text = log.read_text()
        row.update(log=identity(log), log_tail=text[-2400:], log_mtime=log.stat().st_mtime)
        row["identity_matches"] = (
            row.get("start_ticks") == ticks and row.get("ppid") == 1
            and row.get("cwd") == str(BASE / "run") and row.get("stdout") == str(log)
            and row.get("cmdline", [])[:2] == ["/opt/data/private/hechengzhu/ZHY/data/env/bin/python", "eval_smoke.py"]
            and "eval_name=" + name in row.get("cmdline", [])
            and "+run_once=true" in row.get("cmdline", [])
            and error in text[-2400:] and "HYDRA_FULL_ERROR" in text[-2400:]
            and datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).date().isoformat() == "2026-10-21")
        rows.append(row)
    time.sleep(3)
    for row in rows:
        second = process(row["pid"])
        row["second"] = second
        row["no_progress"] = all(second.get(key) == row.get(key)
                                 for key in ("start_ticks", "cpu_ticks", "io", "cmdline", "cwd", "stdout"))
        row["no_progress"] = row["no_progress"] and identity(Path(row["log"]["path"])) == row["log"]
    record = {"captured_utc": datetime.now(timezone.utc).isoformat(), "boot": boot,
              "source": identity(Path(__file__).resolve()), "before": before, "processes": rows,
              "all_confirmed_failed_residuals": all(r["identity_matches"] and r["no_progress"] for r in rows),
              "signals": [], "file_deletions": 0, "storage_status": "LOCAL_ONLY"}
    if args.terminate_from:
        previous = json.loads(args.terminate_from.read_text())
        if not (record["all_confirmed_failed_residuals"] and previous["all_confirmed_failed_residuals"]
                and previous["boot"] == boot and previous["source"] == record["source"]
                and [(r["pid"], r["start_ticks"], r["log"]) for r in previous["processes"]]
                == [(r["pid"], r["start_ticks"], r["log"]) for r in rows]):
            raise RuntimeError("Saved evidence or current process identity does not admit termination")
        descriptors = []
        try:
            for row in rows:
                fd = os.pidfd_open(row["pid"])
                descriptors.append((row, fd))
                if process(row["pid"]).get("start_ticks") != row["start_ticks"]:
                    raise RuntimeError("Process identity changed while opening pidfd")
            write_new(args.output.with_name(args.output.stem + "_before_signal.json"), record)
            for row, fd in descriptors:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                record["signals"].append({"pid": row["pid"], "signal": "SIGTERM"})
            time.sleep(10)
            for row, fd in descriptors:
                current = process(row["pid"])
                if not current.get("missing") and current.get("state") != "Z":
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    record["signals"].append({"pid": row["pid"], "signal": "SIGKILL"})
            time.sleep(2)
        finally:
            for _, fd in descriptors:
                os.close(fd)
        record["after_processes"] = [process(row["pid"]) for row in rows]
        record["all_released"] = all(r.get("missing") or r.get("state") == "Z"
                                     for r in record["after_processes"])
    record["after"] = resources()
    write_new(args.output, record)
    print(json.dumps({key: value for key, value in record.items() if key != "processes"}), flush=True)
    return 0 if record.get("all_released", record["all_confirmed_failed_residuals"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
