"""Read-only per-process memory evidence; never captures arguments or environment."""

import os
from pathlib import Path

from script.a3.host_health import append_record
from script.diagnostics.host_loadcheck import protection


def fields(path):
    return dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)


def process_memory(directory):
    status = fields(directory / 'status')
    rollup = fields(directory / 'smaps_rollup')
    kb = lambda value: int(value.split()[0]) * 1024
    return dict(pid=int(status['Pid']), ppid=int(status['PPid']),
        comm=status['Name'].strip(), rss_bytes=kb(status['VmRSS']),
        peak_rss_bytes=kb(status['VmHWM']), pss_bytes=kb(rollup['Pss']),
        swap_bytes=kb(rollup['Swap']))


def processes(proc=Path('/proc')):
    rows, errors = [], []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            rows.append(process_memory(directory))
        except (OSError, ValueError, KeyError) as exc:
            if directory.exists():
                errors.append(dict(pid=int(directory.name), error=type(exc).__name__))
    return dict(rows=sorted(rows, key=lambda row: row['rss_bytes'], reverse=True), errors=errors)


def with_processes(base_health):
    value = base_health()
    value['process_memory'] = processes()
    group = Path('/sys/fs/cgroup/memory')
    value['cgroup_memory_stat_bytes'] = {
        key: int(count) for key, count in
        (line.split() for line in (group / 'memory.stat').read_text().splitlines())
        if key in ('total_rss', 'total_cache', 'total_inactive_file', 'total_active_file',
                   'total_mapped_file', 'total_swap')}
    own = [row for row in value['process_memory']['rows'] if row['pid'] == os.getpid()]
    if len(own) != 1:
        raise RuntimeError('Reliable own-process RSS/PSS required')
    value['observer_memory'] = own[0]
    return value


def stage_event(path, name, health, process_budget=None):
    own = health['observer_memory']
    reason, hot = protection(health, 0)
    append_record(path, dict(phase=name, pid=os.getpid(), health=health))
    if reason or hot:
        raise RuntimeError(reason or 'HIGH_STAGE_TEMPERATURE')
    if process_budget is not None and own['rss_bytes'] > process_budget:
        raise RuntimeError('PREPARE_PROCESS_MEMORY_BUDGET')


def summarize(path):
    import json

    values = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [value.get('health', value) for value in values]
    rows = [row for row in rows if 'process_memory' in row]
    peaks = {}
    for value in rows:
        for row in value['process_memory']['rows']:
            peak = peaks.setdefault(str(row['pid']), dict(row, first_utc=value['at_utc']))
            for key in ('rss_bytes', 'peak_rss_bytes', 'pss_bytes', 'swap_bytes'):
                peak[key] = max(peak[key], row[key])
            peak['last_utc'] = value['at_utc']
    return dict(samples=len(rows), process_peaks=sorted(peaks.values(),
        key=lambda row: row['rss_bytes'], reverse=True),
        reading_errors=[e for value in rows for e in value['process_memory']['errors']],
        scope='SAMPLED_VISIBLE_PROCESSES_NOT_PROOF_OF_E37_TRANSIENT_CAUSE')
