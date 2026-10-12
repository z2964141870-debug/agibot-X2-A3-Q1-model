"""A single new-host A3 trial with container-aware telemetry and no recovery."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command, sample
from script.diagnostics.host_loadcheck import FATAL, protection

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_20261012_E01'
LOG = ROOT / 'logs/a3_zhy_20261012_E01'
TRAINING = ROOT / 'data/training/a3_zhy_20261012_E01'
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
TRIAL = 'R11'
CGROUP = Path('/sys/fs/cgroup/memory')
TARGET = Path('/opt/data/private/hechengzhu/ZHY')
OFFICIAL_SHA = '9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d'


def read(path):
    return json.loads(path.read_text())


def container_memory(cgroup=CGROUP):
    stat = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
    limit = int((cgroup / 'memory.limit_in_bytes').read_text())
    usage = int((cgroup / 'memory.usage_in_bytes').read_text())
    reclaimable = int(stat['total_inactive_file'])
    available = max(0, min(limit, limit - usage + reclaimable))
    oom = dict(line.split() for line in (cgroup / 'memory.oom_control').read_text().splitlines())
    return dict(limit_bytes=limit, usage_bytes=usage,
        reclaimable_inactive_file_bytes=reclaimable, available_estimate_bytes=available,
        rss_bytes=int(stat['total_rss']), oom_kill=int(oom.get('oom_kill', 0)))


def health():
    result = sample()
    temperatures = []
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        if (zone / 'type').read_text().strip() == 'x86_pkg_temp':
            temperatures.append(int((zone / 'temp').read_text()) / 1000)
    if temperatures:
        result['temperatures_c']['x86_pkg_temp'] = max(temperatures)
    result['container_memory'] = container_memory()
    result['memory_bytes']['MemAvailable'] = result['container_memory']['available_estimate_bytes']
    return result


def preflight():
    if ROOT != TARGET:
        raise RuntimeError('This trial is authorized only inside the new ZHY workspace')
    vendor = command(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'])
    if vendor['returncode'] or vendor['stdout'] != 'fe6868ba37034f89b912f0fb851bce19f120266d':
        raise RuntimeError('Vendor revision mismatch')
    gpu = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if gpu['returncode'] or gpu['stdout']:
        raise RuntimeError('GPU occupied or unreadable')
    value = health()
    reason, hot = protection(value, 0)
    if reason or hot:
        raise RuntimeError(reason or 'HIGH_PREFLIGHT_TEMPERATURE')
    return value


def register():
    if OUT.exists() or TRAINING.exists():
        raise FileExistsError('This experiment must not be reset or reused')
    baseline = preflight()
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unreadable')
    official = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
    if sha256(official) != OFFICIAL_SHA:
        raise RuntimeError('Official checkpoint identity mismatch')
    job = dict(experiment='A3_ZHY_20261012_E01', trial=TRIAL, registered_at=time.time(),
        boot_id=baseline['boot_id'], source=str(official), source_sha256=sha256(official),
        updates=2, num_envs=16, minibatches=4, epochs=1, lr_scale=.1, seed=0,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False,
        repeat_allowed=False, synchronized_diagnostic=False,
        baseline_oom_kill=baseline['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), runner_sha256=sha256(Path(__file__)),
        code_sha256={name:sha256(ROOT / name) for name in
            ('script/a3/corrected_trial.py', 'script/a3/verified_finetune.py',
             'script/a3/resumable_sonic.py', 'script/a3/checkpoint_store.py')},
        backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'launch_state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', baseline)
    print(json.dumps({k:v for k,v in job.items() if k != 'kernel_baseline'}, indent=2))


def choose_checkpoint(attempts, trial=TRIAL):
    from script.a3.checkpoint_store import select_checkpoint
    records = []
    for attempt in attempts:
        directory = Path(attempt['run_dir']).resolve()
        if directory.parent != TRAINING.resolve() or not directory.name.startswith(TRIAL + '_'):
            raise ValueError('Checkpoint escaped the new experiment')
        try:
            path, payload, _ = select_checkpoint(directory)
        except RuntimeError:
            continue
        step = int(payload['state'].global_step)
        if not 0 <= step <= 2 or payload['state'].cur_episode_length.numel() != 16:
            raise ValueError('Checkpoint escaped the two-update contract')
        records.append((step, path))
    return max(records, key=lambda r:r[0]) if records else (0, None)


def guarded_monitor(child, log_dir, config):
    job = read(OUT / 'registration.json')
    started, hot, stopped, reason = time.monotonic(), 0, False, None
    seen = set(job['kernel_baseline'])
    def handle(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {s:signal.signal(s, handle) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            try:
                value = health()
            except (OSError, ValueError, KeyError) as exc:
                reason = 'TELEMETRY_UNREADABLE'
                append_record(LOG / 'health.jsonl', dict(error=repr(exc), at_unix=time.time()))
                break
            kernel = command(['dmesg', '--color=never'])
            lines = kernel.get('stdout', '').splitlines()
            fresh = [line for line in lines if line not in seen and FATAL.search(line)]
            seen.update(lines)
            append_record(LOG / 'health.jsonl', dict(health=value, kernel_errors=fresh))
            append_record(log_dir / 'health.jsonl', value)
            reason, hot = protection(value, hot)
            if value['boot_id'] != job['boot_id']:
                reason = 'CONFIRMED_NEW_BOOT'
            elif value['container_memory']['oom_kill'] > job['baseline_oom_kill']:
                reason = 'CGROUP_OOM_KILL'
            elif stopped:
                reason = 'OPERATOR_STOP'
            elif kernel['returncode']:
                reason = 'KERNEL_TELEMETRY_UNREADABLE'
            elif fresh:
                reason = 'CRITICAL_KERNEL_EVENT'
            elif time.monotonic() - started >= 900:
                reason = 'RUNTIME_LIMIT'
            if reason:
                break
            if child.poll() is not None:
                break
            time.sleep(5)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return child.wait(), reason


def configure():
    from script.a3 import corrected_trial as trial
    trial.configure_trial('R09')
    trial.TRIAL, trial.OUTPUT, trial.LOGS, trial.TRAINING = TRIAL, OUT, LOG, TRAINING
    trial.preflight = preflight
    trial.monitor_child = guarded_monitor
    trial.select_trial_checkpoint = choose_checkpoint
    return trial


def launch_allowed(state, job, boot):
    return state['status'] == 'READY' and boot == job['boot_id']


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read(OUT / 'registration.json'), read(OUT / 'launch_state.json')
        if not launch_allowed(state, job, health()['boot_id']):
            raise RuntimeError('No repeat or reboot recovery is allowed')
        if sha256(Path(__file__)) != job['runner_sha256']:
            raise RuntimeError('Registered runner changed')
        if any(sha256(ROOT / name) != digest for name, digest in job['code_sha256'].items()):
            raise RuntimeError('Registered training dependencies changed')
        preflight()
        state.update(status='RUNNING', started_at=time.time())
        atomic_json(OUT / 'launch_state.json', state)
        try:
            code = configure().train()
            actual = read(OUT / 'state.json')
            state.update(status='TRAINING_COMPLETE' if code == 0 and actual['status'] == 'complete'
                         else 'STOPPED', returncode=code, verified_step=actual.get('verified_step', 0),
                         reason=actual.get('reason'))
            return code
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'launch_state.json', state)


def verify():
    if read(OUT / 'launch_state.json')['status'] != 'TRAINING_COMPLETE':
        raise RuntimeError('Training must finish before independent verification')
    if health()['boot_id'] != read(OUT / 'registration.json')['boot_id']:
        raise RuntimeError('Boot changed; stop further computation')
    return configure().verify()


def report():
    job, state = read(OUT / 'registration.json'), read(OUT / 'launch_state.json')
    verification = read(OUT / 'verification.json') if (OUT / 'verification.json').exists() else {}
    rows = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    samples = [row.get('health', row) for row in rows if 'health' in row or 'boot_id' in row]
    gpu = [g for value in samples for g in value['gpus']]
    times = [datetime.fromisoformat(value['at_utc']).timestamp() for value in samples]
    result = dict(experiment=job['experiment'], status=state['status'], launch=state,
        verification_status=verification.get('status', 'NOT_VERIFIED'),
        source_sha256=job['source_sha256'], parameters={key:job[key] for key in
            ('updates', 'num_envs', 'minibatches', 'epochs', 'lr_scale', 'seed', 'auto_resume')},
        checkpoints=verification.get('checkpoints', {}),
        actual_optimizer_steps=len(verification.get('actual_optimizer_steps', [])),
        undefined_statistics=verification.get('undefined_episode_statistics', []),
        health=dict(samples=len(samples), first_utc=samples[0]['at_utc'], last_utc=samples[-1]['at_utc'],
            max_sample_gap_seconds=max((b-a for a,b in zip(times,times[1:])), default=0),
            boot_ids=sorted({value['boot_id'] for value in samples}),
            cpu_max_c=max(value['temperatures_c']['x86_pkg_temp'] for value in samples),
            gpu_max_c=max(g['temperature.gpu'] for g in gpu),
            gpu_power_max_w=max(g['power.draw'] for g in gpu),
            gpu_memory_max_mib=max(g['memory.used'] for g in gpu),
            container_rss_max_bytes=max(value['container_memory']['rss_bytes'] for value in samples),
            container_available_min_bytes=min(value['container_memory']['available_estimate_bytes'] for value in samples),
            oom_kill_counts=sorted({value['container_memory']['oom_kill'] for value in samples}),
            critical_kernel_events=[line for row in rows for line in row.get('kernel_errors', [])],
            telemetry_errors=[row for row in rows if 'error' in row]),
        scope='NEW_HOST_SHORT_TRAINING_NOT_POLICY_QUALITY_OR_HARDWARE_ROOT_CAUSE', backup_status='LOCAL_ONLY')
    atomic_json(ROOT / 'data/manifests/a3_zhy_20261012_E01.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'run', 'verify', 'report'))
    result = dict(register=register, run=run, verify=verify, report=report)[p.parse_args().action]()
    raise SystemExit(result if isinstance(result, int) else 0)
