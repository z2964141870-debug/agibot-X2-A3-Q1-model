"""Sample Linux CPU/process counters and an owned tegrastats child; never infer."""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time


def parse_stat(text):
    head, tail = text.rsplit(")", 1)
    pid, name = head.split("(", 1)
    fields = tail.split()
    return dict(pid=int(pid), name=name, state=fields[0], ppid=int(fields[1]),
                ticks=int(fields[11])+int(fields[12]), starttick=int(fields[19]))


def parse_cpu(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu\d*", fields[0]):
            continue
        values = list(map(int, fields[1:]))
        result[fields[0]] = (sum(values[:8]), values[3]+values[4])
    return result


def cpu_delta(previous, current):
    result = {}
    for key in previous.keys() & current.keys():
        total = current[key][0]-previous[key][0]
        idle = current[key][1]-previous[key][1]
        if total > 0 and 0 <= idle <= total:
            result[key] = 100*(total-idle)/total
    return result


def parse_tegra(line):
    gpu = re.search(r"\bGR3D_FREQ\s+(\d+)%", line)
    ram = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
    emc = re.search(r"\bEMC_FREQ\s+(\d+)%@([\d.]+)", line)
    return dict(gpu_busy_percent=int(gpu[1]) if gpu else None,
                ram_used_mib=int(ram[1]) if ram else None,
                emc_busy_percent=int(emc[1]) if emc else None,
                emc_mhz=float(emc[2]) if emc else None,
                temperature_c={k: float(v) for k, v in re.findall(r"([\w]+)@([\d.]+)C", line)})


def descendants(rows, root_pid, root_starttick):
    root = rows.get(root_pid)
    if root is None or root["starttick"] != root_starttick:
        return set()
    selected = {root_pid}
    while True:
        added = {pid for pid, row in rows.items() if row["ppid"] in selected}-selected
        if not added:
            return selected
        selected.update(added)


def process_delta(previous, current, seconds, hz):
    result = []
    for pid, row in current.items():
        old = previous.get(pid)
        if old is None or old["starttick"] != row["starttick"]:
            continue
        ticks = row["ticks"]-old["ticks"]
        if ticks >= 0:
            result.append(dict(row, cpu_percent_one_core=100*ticks/hz/seconds))
    return sorted(result, key=lambda row: row["cpu_percent_one_core"], reverse=True)


def stat_rows(directory):
    rows = {}
    for item in directory.iterdir():
        if not item.name.isdigit():
            continue
        try:
            row = parse_stat((item/"stat").read_text())
            rows[row["pid"]] = row
        except (OSError, ValueError, IndexError):
            continue
    return rows


def memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "Cached", "SwapFree"):
            values[key+"_bytes"] = int(value.split()[0])*1024
    return values


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    at = (len(values)-1)*fraction
    low, high = math.floor(at), math.ceil(at)
    return values[low]+(values[high]-values[low])*(at-low)


class TegraReader:
    def __init__(self, interval):
        self.child = None
        self.thread = None
        self.lines = deque()
        binary = shutil.which("tegrastats")
        if binary:
            self.child = subprocess.Popen([binary, "--interval", str(round(interval*1000))],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()

    def _read(self):
        for line in self.child.stdout:
            self.lines.append(dict(received_monotonic_ns=time.monotonic_ns(), raw=line.rstrip(),
                                   **parse_tegra(line)))

    def drain(self):
        result = []
        while self.lines:
            result.append(self.lines.popleft())
        return result

    def close(self):
        if self.child is not None:
            if self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=2)
            self.thread.join(timeout=2)
            self.child.stdout.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--root-pid", type=int)
    parser.add_argument("--root-starttick", type=int)
    args = parser.parse_args()
    if (not 1 <= args.seconds <= 120 or not .5 <= args.interval <= 5
            or args.seconds < args.interval):
        parser.error("Require 1..120 seconds, 0.5..5 second interval <= duration")
    if (args.root_pid is None) != (args.root_starttick is None):
        parser.error("Specify the root PID and original starttick together")
    proc = Path("/proc")
    boot = (proc/"sys/kernel/random/boot_id").read_text().strip()
    previous_processes = stat_rows(proc)
    if args.root_pid is not None and not descendants(previous_processes, args.root_pid, args.root_starttick):
        parser.error("Requested PID/starttick is not alive; no monitor started")
    args.output.mkdir(parents=True, exist_ok=False)
    os.nice(19)
    started = previous_at = time.monotonic()
    previous_cpu = parse_cpu((proc/"stat").read_text())
    previous_threads = {}
    hz = os.sysconf("SC_CLK_TCK")
    samples, tegra = [], None
    status, error = "COMPLETED", None
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__), Path(__file__).with_name("__init__.py"))}
    began_cpu = time.process_time()
    try:
        tegra = TegraReader(args.interval)
        with (args.output/"samples.jsonl").open("x") as stream:
            for index in range(1, math.ceil(args.seconds/args.interval)+1):
                until = started+min(index*args.interval, args.seconds)
                time.sleep(max(0, until-time.monotonic()))
                before = time.monotonic()
                cpu = parse_cpu((proc/"stat").read_text())
                processes = stat_rows(proc)
                at = time.monotonic()
                owned = descendants(processes, args.root_pid, args.root_starttick) if args.root_pid else set()
                threads = {}
                for pid in owned:
                    try:
                        threads.update(stat_rows(proc/str(pid)/"task"))
                    except OSError:
                        pass
                row = dict(monotonic_ns=round(at*1e9), utc_ns=time.time_ns(), interval_s=at-previous_at,
                    cpu_busy_percent=cpu_delta(previous_cpu, cpu), memory=memory(),
                    top_processes=process_delta(previous_processes, processes, at-previous_at, hz)[:16],
                    owned_pids=sorted(owned), owned_processes=[r for r in
                        process_delta(previous_processes, processes, at-previous_at, hz) if r["pid"] in owned],
                    owned_threads=process_delta(previous_threads, threads, at-previous_at, hz),
                    tegrastats=tegra.drain(), sampler_read_wall_ms=(time.monotonic()-before)*1000)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                samples.append(row)
                previous_cpu, previous_processes, previous_threads, previous_at = cpu, processes, threads, at
                if args.root_pid is not None and not owned:
                    status = "TARGET_EXITED"
                    break
    except BaseException as exc:
        status, error = "FAILED", repr(exc)
    finally:
        if tegra is not None:
            tegra.close()
        tegra_rows = [t for row in samples for t in row["tegrastats"]]
        trailing = tegra.drain() if tegra else []
        tegra_rows.extend(trailing)
        gpu = [r["gpu_busy_percent"] for r in tegra_rows if r["gpu_busy_percent"] is not None]
        result = dict(status=status, error=error, boot_id=boot,
            boot_id_after=(proc/"sys/kernel/random/boot_id").read_text().strip(),
            wall_s=time.monotonic()-started, own_cpu_s=time.process_time()-began_cpu,
            source_sha256=source, interval_s=args.interval, duration_bound_s=args.seconds,
            sample_count=len(samples), tegrastats_trailing=trailing,
            tegrastats_pid=tegra.child.pid if tegra and tegra.child else None,
            tegrastats_returncode=tegra.child.returncode if tegra and tegra.child else None,
            tegrastats_valid_samples=len(gpu), gpu_busy_p50=percentile(gpu,.5), gpu_busy_max=max(gpu,default=None),
            max_temperature_c=max((v for r in tegra_rows for v in r["temperature_c"].values()),default=None),
            min_available_bytes=min((r["memory"]["MemAvailable_bytes"] for r in samples),default=None),
            sampler_read_p95_ms=percentile([r["sampler_read_wall_ms"] for r in samples],.95),
            cpu_mean_percent={key:sum(r["cpu_busy_percent"].get(key,0) for r in samples)/len(samples)
                for key in sorted({k for r in samples for k in r["cpu_busy_percent"]})},
            root_pid=args.root_pid,root_starttick=args.root_starttick,
            hardware_commands=0,feedback_subscriptions=0,model_calls=0,training_updates=0,
            limits="Read timestamps align counters; GPU utilization is whole-device, not per-process attribution")
        (args.output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(result,allow_nan=False))
    if status == "FAILED":
        raise SystemExit(1)


def interrupted(*_):
    raise KeyboardInterrupt()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    main()
