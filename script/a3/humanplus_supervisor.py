"""Fail-closed, standard-library stage supervision for the humanplus A3 root."""

import csv
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from script.a3 import humanplus_health as health
from script.a3.checkpoint_store import atomic_json

ROOT = Path('/home/humanplus/YUANQI_A3')
INTERVAL = 5.0
SAMPLE_TIMEOUT = 4.0
STAGE_SECONDS = 900.0


def query(fields, kind='gpu'):
    result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields,
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=2)
    if result.returncode:
        raise health.TelemetryError('GPU query failed')
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def process_rows(proc=Path('/proc')):
    rows = []
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            text = (path / 'stat').read_text()
            name = text[text.index('(') + 1:text.rindex(')')]
            fields = text[text.rindex(')') + 2:].split()
            status = dict(line.split(':', 1) for line in
                          (path / 'status').read_text().splitlines() if ':' in line)
            row = dict(pid=int(path.name), name=name, state=fields[0],
                       ppid=int(fields[1]), pgid=int(fields[2]), start_ticks=int(fields[19]),
                       rss_bytes=int(status.get('VmRSS', '0').split()[0]) * 1024)
        except (OSError, ValueError, KeyError, IndexError):
            if path.exists():
                raise health.TelemetryError('Process metadata unreadable: ' + path.name)
            continue
        if row['state'] != 'Z':
            rows.append(row)
    return rows


def collect(baseline):
    started = time.monotonic()
    evidence = health.snapshot(baseline)
    cpu = []
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        if (zone / 'type').read_text().strip() == 'x86_pkg_temp':
            cpu.append(int((zone / 'temp').read_text()) / 1000)
    if len(cpu) != 1 or not 0 < cpu[0] < 150:
        raise health.TelemetryError('Reliable CPU package temperature required')
    gpu = query('uuid,temperature.gpu,utilization.gpu,memory.used')
    if len(gpu) != 1 or len(gpu[0]) != 4:
        raise health.TelemetryError('Exactly one readable GPU required')
    uuid, temperature, utilization, memory = gpu[0]
    values = list(map(float, (temperature, utilization, memory)))
    if (not all(math.isfinite(value) for value in values)
            or not 0 < values[0] < 150 or not 0 <= values[1] <= 100 or values[2] < 0):
        raise health.TelemetryError('Invalid GPU telemetry')
    apps = query('pid', kind='compute-apps')
    if any(len(row) != 1 or not row[0].isdecimal() for row in apps):
        raise health.TelemetryError('GPU compute PID unreadable')
    evidence.update(cpu_celsius=cpu[0], gpu_uuid=uuid, gpu_celsius=values[0],
                    gpu_utilization=values[1], gpu_memory_mib=values[2],
                    gpu_pids=[int(row[0]) for row in apps], processes=process_rows(),
                    monotonic=time.monotonic(), collection_seconds=time.monotonic() - started)
    return evidence


def sample(baseline_path):
    # Isolate blocking journal/NVML/proc reads so a stuck sensor cannot stall supervision.
    result = subprocess.run([sys.executable, '-m', __name__, str(baseline_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=SAMPLE_TIMEOUT)
    if result.returncode:
        message = result.stderr.strip()
        raise health.TelemetryError(message or 'Telemetry subprocess failed')
    return json.loads(result.stdout)


def issues(row, *, startup, monitor_pid, worker_group=None, previous=None, gpu_uuid=None):
    result = health.reasons(row, startup=startup)
    if gpu_uuid is not None and row['gpu_uuid'] != gpu_uuid:
        result.append('GPU_IDENTITY_CHANGED')
    processes = {item['pid']: item for item in row['processes']}
    own = {monitor_pid, os.getpid()}
    # The sampler itself is a child of the monitor; workers have a separate group.
    own.update(pid for pid, item in processes.items() if item['ppid'] == monitor_pid)
    if worker_group is not None:
        own.update(pid for pid, item in processes.items() if item['pgid'] == worker_group)
    other = [item['pid'] for item in row['processes'] if item['pid'] not in own and
             (item['rss_bytes'] > health.GIB or item['name'].lower().startswith(
                 ('python', 'kit', 'isaac', 'stress', 'train')))]
    if other:
        result.append('OTHER_COMPUTE:' + ','.join(map(str, sorted(other))))
    if any(pid not in own for pid in row['gpu_pids']):
        result.append('OTHER_GPU_COMPUTE')
    if startup:
        if row['gpu_utilization'] != 0:
            result.append('GPU_NOT_IDLE')
        if row['cpu_celsius'] >= 90 or row['gpu_celsius'] >= 85:
            result.append('STARTUP_TEMPERATURE_HIGH')
    elif previous is not None:
        if row['cpu_celsius'] >= 90 and previous['cpu_celsius'] >= 90:
            result.append('CPU_TEMPERATURE_TWO_SAMPLES')
        if row['gpu_celsius'] >= 85 and previous['gpu_celsius'] >= 85:
            result.append('GPU_TEMPERATURE_TWO_SAMPLES')
    return result


def stop_group(process):
    if process is None or process.pid == os.getpgrp():
        return
    # Popen(start_new_session=True) owns this group, even if the leader already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(0.5)
    process.wait(timeout=5)


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run_stage(directory, argv, baseline_path, *, env=None):
    """One launch only. A terminal directory is never resumed or overwritten."""
    if Path.cwd().resolve() != ROOT or Path.home() != ROOT.parent:
        raise ValueError('Only authorized humanplus root supported')
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='STARTUP', argv=argv, started_at=time.time(),
                 supervisor_pid=os.getpid(), seconds_limit=STAGE_SECONDS,
                 interval_seconds=INTERVAL, no_automatic_retry=True)
    state_path = directory / 'state.json'
    atomic_json(state_path, state)
    process = None
    last = None
    uuid = None
    def interrupted(signum, frame):
        raise RuntimeError('SUPERVISOR_SIGNAL:' + str(signum))
    handlers = {sig: signal.signal(sig, interrupted)
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        for ordinal in range(3):
            if ordinal:
                time.sleep(max(0, next_sample - time.monotonic()))
            next_sample = time.monotonic() + INTERVAL
            row = sample(baseline_path)
            append(directory / 'health.jsonl', dict(phase='startup', ordinal=ordinal, **row))
            errors = issues(row, startup=True, monitor_pid=os.getpid(), gpu_uuid=uuid)
            if errors:
                raise RuntimeError(';'.join(errors))
            uuid, last = row['gpu_uuid'], row
        with (directory / 'console.log').open('x') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stdin=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='RUNNING', worker_pid=process.pid,
                         launched_at=time.time())
            atomic_json(state_path, state)
            process.stdin.write(b'READY\n')
            process.stdin.close()
            while True:
                remaining = STAGE_SECONDS - (time.monotonic() - started)
                if remaining <= SAMPLE_TIMEOUT:
                    raise RuntimeError('STAGE_DEADLINE_900_SECONDS')
                time.sleep(max(0, min(next_sample - time.monotonic(), remaining - SAMPLE_TIMEOUT)))
                next_sample = time.monotonic() + INTERVAL
                row = sample(baseline_path)
                append(directory / 'health.jsonl', dict(phase='runtime', **row))
                errors = issues(row, startup=False, monitor_pid=os.getpid(),
                                worker_group=process.pid, previous=last, gpu_uuid=uuid)
                if errors:
                    raise RuntimeError(';'.join(errors))
                last = row
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError('WORKER_EXIT:' + str(code))
                    break
        receipt = json.loads((directory / 'result.json').read_text())
        if receipt['status'] != 'COMPLETE':
            raise RuntimeError('MISSING_COMPLETE_WORKER_RECEIPT')
        state.update(status='COMPLETE', returncode=0)
    except BaseException as exc:
        state.update(status='STOPPED', reason=type(exc).__name__ + ':' + str(exc),
                     launched=process is not None)
        raise
    finally:
        try:
            stop_group(process)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            state.update(finished_at=time.time(), elapsed_seconds=time.monotonic() - started)
            atomic_json(state_path, state)
    return state


if __name__ == '__main__':
    baseline = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(collect(baseline)))
