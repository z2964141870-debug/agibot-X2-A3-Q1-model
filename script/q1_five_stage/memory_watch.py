"""Bounded read-only cgroup/process sampling for one verified controller handle."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path("/opt/data/private/ZHY")
MEMORY_FIELDS = ("total_cache", "total_rss", "total_shmem", "total_dirty", "total_writeback")
PROCESS_FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM")


def process(proc, pid):
    try:
        directory = proc / str(pid)
        raw = (directory / "stat").read_text()
        head, _, tail = raw.rpartition(") ")
        values = tail.split()
        row = dict(pid=int(pid), name=head.partition("(")[2], state=values[0],
                   ppid=int(values[1]), start_ticks=values[19])
        for line in (directory / "status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in PROCESS_FIELDS:
                number, unit = value.split()
                if unit != "kB":
                    raise ValueError("Unexpected proc memory unit")
                row[key + "_bytes"] = int(number) * 1024
        return row
    except (FileNotFoundError, ProcessLookupError):
        return None


def subtree(proc, controller):
    rows = {}
    for path in proc.iterdir():
        if path.name.isdigit():
            row = process(proc, int(path.name))
            if row is not None:
                rows[row["pid"]] = row
    selected = {controller["pid"]}
    while True:
        extended = selected | {pid for pid, row in rows.items() if row["ppid"] in selected}
        if extended == selected:
            break
        selected = extended
    return [rows[pid] for pid in sorted(selected) if pid in rows]


def memory(cgroup):
    fields = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    oom = dict(line.split() for line in (cgroup / "memory.oom_control").read_text().splitlines())
    return dict(total_bytes=int((cgroup / "memory.usage_in_bytes").read_text()),
                **{name + "_bytes": int(fields[name]) for name in MEMORY_FIELDS},
                oom_kill=int(oom["oom_kill"]), under_oom=int(oom["under_oom"]))


def sample(proc, cgroup, pid, start_ticks):
    controller = process(proc, pid)
    if controller is None:
        return dict(status="OBSERVED_PROCESS_EXIT")
    if controller["start_ticks"] != str(start_ticks):
        return dict(status="OBSERVED_PID_REUSE")
    processes = subtree(proc, controller)
    current = process(proc, pid)
    if current is None:
        return dict(status="OBSERVED_PROCESS_EXIT")
    if current["start_ticks"] != str(start_ticks):
        return dict(status="OBSERVED_PID_REUSE")
    return dict(status="OBSERVED_LIVE", utc=datetime.now(timezone.utc).isoformat(),
                memory=memory(cgroup), processes=processes)


def watch(proc, cgroup, pid, start_ticks, output, seconds, interval):
    if not 1 <= seconds <= 10800 or not .5 <= interval <= 10:
        raise ValueError("Use a 1..10800s bound and 0.5..10s sampling interval")
    first = sample(proc, cgroup, pid, start_ticks)
    if first["status"] != "OBSERVED_LIVE":
        raise RuntimeError("Controller handle must be verified live before sampling")
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Preserve existing observations")
    started = time.monotonic()
    summary = dict(status="OBSERVING", controller_pid=pid, controller_start_ticks=str(start_ticks),
        samples=0, maximum_total_bytes=0, maximum_cache_bytes=0, maximum_anonymous_rss_bytes=0,
        training_status_inferred=False, signals_sent=0, optimizer_updates=0,
        note="Cgroup total/cache/rss are distinct samples; process RSS can count shared mappings repeatedly.")
    try:
        with output.open("x") as stream:
            row = first
            while True:
                if row["status"] != "OBSERVED_LIVE":
                    summary["status"] = row["status"]
                    break
                elapsed = time.monotonic() - started
                row["elapsed_seconds"] = elapsed
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                summary["samples"] += 1
                for source, target in (("total_bytes", "maximum_total_bytes"),
                        ("total_cache_bytes", "maximum_cache_bytes"),
                        ("total_rss_bytes", "maximum_anonymous_rss_bytes")):
                    summary[target] = max(summary[target], row["memory"][source])
                if elapsed >= seconds:
                    summary["status"] = "OBSERVATION_TIME_BOUND_REACHED"
                    break
                if stream.tell() >= 16 * 1024**2:
                    summary["status"] = "OBSERVATION_SIZE_BOUND_REACHED"
                    break
                time.sleep(min(interval, seconds - elapsed))
                row = sample(proc, cgroup, pid, start_ticks)
    except BaseException as error:
        summary.update(status="OBSERVER_FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        with summary_path.open("x") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", required=True)
    parser.add_argument("--expected-boot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10800)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if (ROOT.resolve() != ROOT or Path.cwd().resolve() != ROOT
            or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != args.expected_boot):
        parser.error("Run on the explicitly identified Q1 server and physical root")
    if (not args.output.is_absolute() or not args.output.parent.resolve().is_relative_to(ROOT / "logs")
            or args.output.is_symlink() or args.output.suffix != ".jsonl"):
        parser.error("Use a fresh JSONL under the Q1 logs directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(watch(Path("/proc"), Path("/sys/fs/cgroup/memory"), args.pid,
                           args.start_ticks, args.output, args.seconds, args.interval)))


if __name__ == "__main__":
    main()
