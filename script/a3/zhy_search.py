"""Matched new-host evaluation of R11 and the completed E21 adapters."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.a3.zhy_trial import ROOT, TARGET, health, preflight
from script.diagnostics.host_loadcheck import FATAL, protection

OUT = ROOT / 'data/experiments/a3_zhy_search_20261012_E02'
LOG = ROOT / 'logs/a3_zhy_search_20261012_E02'
OLD = ROOT / 'data/experiments/a3_controller_adapter_20261010_E21'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
R11 = ROOT / 'data/training/a3_zhy_20261012_E01/R11_001_s0/model_step_000002.pt'
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
SOURCES = ('script/a3/zhy_search.py', 'script/a3/verify_zhy_search.py',
    'script/a3/zhy_trial.py', 'script/a3/evaluate_mujoco.py', 'script/a3/sim_entry.py',
    'script/a3/joint_forecast_replay.py', 'script/a3/causal_orientation.py')
FIELDS = ('root_pos_error_m_mean', 'root_quat_error_deg_mean', 'torso_world_mean_m',
    'wrists_world_mean_m', 'legs_world_mean_m', 'torso_anchor_local_mean_m',
    'wrists_anchor_local_mean_m', 'legs_anchor_local_mean_m')


def read(path):
    return json.loads(path.read_text())


def checked_source(row):
    path = VENDOR / 'a3_data/agibot_a3' / Path(row['path']).name
    if sha256(path) != row['sha256']:
        raise ValueError('Motion differs from the historical fixed split')
    return dict(path=str(path), sha256=row['sha256'], name=path.name)


def register():
    if ROOT != TARGET or OUT.exists():
        raise RuntimeError('New-host experiment cannot be reset or run elsewhere')
    value = preflight()
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    from script.a3.evaluation_common import checkpoint_metadata
    split = read(ROOT / 'data/experiments/a3_fullchain_20261010/split.json')
    training = [checked_source(r) for r in split['train']]
    diagnostic = [checked_source(r) for r in split['heldout']]
    if len(training) != 16 or len(diagnostic) != 4:
        raise ValueError('Unexpected motion split')
    receipt = read(OLD / 'independent_verification.json')
    if receipt['status'] != 'passed' or len(receipt['checkpoints']) != 44:
        raise ValueError('E21 training was not verified')
    if sha256(OLD / 'training_windows.npz') != receipt['dataset_sha256']:
        raise ValueError('Verified E21 dataset changed')
    adapters = {}
    for objective in ('mse', 'latent'):
        for step in (0, 200):
            sidecar = read(OLD / objective / f'adapter_step_{step:06d}.json')
            path = OLD / objective / f'adapter_step_{step:06d}.npz'
            if sha256(path) != sidecar['export_sha256']:
                raise ValueError('E21 exported adapter identity mismatch')
            adapters[f'{objective}_{step}'] = dict(path=str(path), sha256=sha256(path))
    tasks = [dict(key='smoke_cv', mode='cv', sigma=0., smoke=True)]
    tasks += [dict(key=f'smoke_{o}', mode='cv_residual', objective=o, step=0, sigma=0., smoke=True)
              for o in ('mse', 'latent')]
    for sigma in (0., .01):
        tasks.append(dict(key=f'cv_noise{sigma:g}_seed0', mode='cv', sigma=sigma))
        tasks += [dict(key=f'{o}_noise{sigma:g}_seed0', mode='cv_residual', objective=o,
                       step=200, sigma=sigma) for o in ('mse', 'latent')]
    tasks += [dict(key='official_oracle', mode=None, checkpoint=str(OFFICIAL)),
              dict(key='r11_oracle', mode=None, checkpoint=str(R11))]
    job = dict(experiment='A3_ZHY_SEARCH_20261012_E02', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), tasks=tasks, adapters=adapters,
        official=checkpoint_metadata(OFFICIAL), r11=checkpoint_metadata(R11),
        training_motions=training, diagnostic_motions=diagnostic,
        code_sha256={p:sha256(ROOT / p) for p in SOURCES},
        source_e21_receipt_sha256=sha256(OLD / 'independent_verification.json'),
        source_e21_data_sha256=sha256(OLD / 'training_windows.npz'),
        gate=dict(no_new_falls=True, each_motion_joint_no_worse=True,
                  mean_joint_improvement_min_percent=1., body_and_root_regression_max_percent=5.,
                  action_delta_regression_max_percent=5., both_noise_conditions_required=True),
        scope='REPEATED_OFFICIAL_DIAGNOSTICS_NOT_FRESH_FINAL_OR_REAL_ROBOT_ACCEPTANCE',
        matched_new_host_baselines=True, e21_retrained=False, auto_resume=False,
        repeat_allowed=False, sample_seconds=5, max_cpu_c=90, max_gpu_c=85,
        consecutive_hot_samples=2, memory_reserve_gib=4, per_motion_limit_seconds=900,
        total_limit_seconds=5400, backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    for name, rows in (('diagnostic', diagnostic), ('selected20', training+diagnostic)):
        directory = OUT / 'references' / name
        directory.mkdir(parents=True)
        for row in rows:
            (directory / row['name']).symlink_to(os.path.relpath(row['path'], directory))
    atomic_json(OUT / 'job.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY', tasks={}))
    append_record(LOG / 'health.jsonl', dict(phase='registered', health=value))
    print(json.dumps(dict(experiment=job['experiment'], tasks=len(tasks), status='READY')))


def monitor(child, log_dir, unused):
    job = read(OUT / 'job.json')
    started, hot, interrupted, reason = time.monotonic(), 0, False, None
    seen = set(job['kernel_baseline'])
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            try:
                value = health()
                kernel = command(['dmesg', '--color=never'])
                lines = kernel.get('stdout', '').splitlines()
                fresh = [line for line in lines if line not in seen and FATAL.search(line)]
                seen.update(lines)
                append_record(LOG / 'health.jsonl', dict(task=log_dir.parts[-3], health=value,
                                                         kernel_errors=fresh))
                append_record(log_dir / 'health.jsonl', value)
                reason, hot = protection(value, hot)
                if value['boot_id'] != job['boot_id']:
                    reason = 'CONFIRMED_NEW_BOOT'
                elif value['container_memory']['oom_kill'] > job['baseline_oom']:
                    reason = 'CGROUP_OOM_KILL'
                elif interrupted:
                    reason = 'OPERATOR_STOP'
                elif kernel['returncode']:
                    reason = 'KERNEL_TELEMETRY_UNREADABLE'
                elif fresh:
                    reason = 'CRITICAL_KERNEL_EVENT'
                elif time.monotonic()-started >= job['per_motion_limit_seconds']:
                    reason = 'MOTION_RUNTIME_LIMIT'
                elif time.time()-read(OUT / 'state.json')['started_at'] >= job['total_limit_seconds']:
                    reason = 'TOTAL_RUNTIME_LIMIT'
            except (OSError, ValueError, KeyError) as exc:
                reason = 'TELEMETRY_UNREADABLE'
                append_record(LOG / 'health.jsonl', dict(error=repr(exc), at_unix=time.time()))
            if reason or child.poll() is not None:
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


def args_for(task, job):
    mode = task['mode']
    predictor = job['adapters'][f"{task['objective']}_{task['step']}"]['path'] if 'objective' in task else None
    motion = (VENDOR / 'a3_data/agibot_a3/001_walk_front_slow.csv' if task.get('smoke')
              else OUT / 'references' / ('diagnostic' if mode else 'selected20'))
    return SimpleNamespace(output=OUT / task['key'], logs=LOG / task['key'],
        checkpoint=Path(task.get('checkpoint', str(OFFICIAL))), motion=motion,
        reference_fps=30., frame_stride=4, max_policy_steps=10 if task.get('smoke') else None,
        action_delay_ms=0., reference_buffer_ms=0., capture_inputs=True, capture_count=100000,
        summarize_only=False, joint_reference_mode=mode, joint_predictor=Path(predictor) if predictor else None,
        joint_noise_std=task.get('sigma', 0.), joint_noise_seed=0, causal_orientation=bool(mode))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read(OUT / 'job.json'), read(OUT / 'state.json')
        if state['status'] != 'READY' or health()['boot_id'] != job['boot_id']:
            raise RuntimeError('Repeat or reboot recovery is not allowed')
        if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
            raise RuntimeError('Registered evaluation code changed')
        if any(sha256(Path(r['path'])) != r['sha256'] for r in job['adapters'].values()):
            raise RuntimeError('Registered adapter changed')
        if sha256(OFFICIAL) != job['official']['sha256'] or sha256(R11) != job['r11']['sha256']:
            raise RuntimeError('Registered policy changed')
        preflight()
        from script.a3 import evaluate_mujoco as evaluation
        evaluation.preflight = preflight
        evaluation.monitor_child = monitor
        state.update(status='RUNNING', started_at=time.time(), pid=os.getpid())
        atomic_json(OUT / 'state.json', state)
        try:
            for task in job['tasks']:
                key = task['key']
                state['tasks'][key] = dict(status='RUNNING', started_at=time.time())
                atomic_json(OUT / 'state.json', state)
                code = evaluation.evaluate_explicit(args_for(task, job))
                if code:
                    raise RuntimeError(f'Evaluation stopped at {key}, exit={code}')
                state['tasks'][key].update(status='COMPLETE', finished_at=time.time())
                atomic_json(OUT / 'state.json', state)
                print(json.dumps(dict(task=key, status='COMPLETE')), flush=True)
            state['status'] = 'EVALUATION_COMPLETE'
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)


def aggregate(rows):
    weights = np.asarray([r['windows']['full_rollout']['policy_steps'] for r in rows])
    result = dict(policy_steps=int(weights.sum()), falls=sum(r['fall'] for r in rows))
    result['joint_rmse_rad'] = float(np.sqrt(np.average(
        [r['windows']['full_rollout']['joint_rmse_rad']**2 for r in rows], weights=weights)))
    for field in FIELDS:
        result[field] = float(np.average([r['windows']['full_rollout'][field] for r in rows], weights=weights))
    return result


def paired(candidate_dir, baseline_dir, required_names):
    current = read(candidate_dir / 'explicit_summary.json')['motions']
    old = {Path(r['motion']).stem:r for r in read(baseline_dir / 'explicit_summary.json')['motions']}
    if {Path(r['motion']).stem for r in current} != set(required_names) or set(old) != set(required_names):
        raise ValueError('Mismatched coverage')
    reasons, rows = [], []
    for row in current:
        name = Path(row['motion']).stem
        prior = old[name]
        metric, previous = row['windows']['full_rollout'], prior['windows']['full_rollout']
        if not row['full_replay_completed'] or not prior['full_replay_completed'] or metric['policy_steps'] != previous['policy_steps']:
            raise ValueError('Incomplete or unmatched rollout duration')
        if row['fall'] and not prior['fall']:
            reasons.append(f'{name}:new_fall')
        if metric['joint_rmse_rad'] > previous['joint_rmse_rad']+1e-12:
            reasons.append(f'{name}:joint_regression')
        for field in FIELDS:
            if metric[field] > previous[field]*1.05+1e-12:
                reasons.append(f'{name}:{field}_regression')
        actions = [np.asarray(read(d / f'{name}.timeseries.json')['motions'][0]['raw_action_29'])
                   for d in (candidate_dir, baseline_dir)]
        delta, previous_delta = [float(np.sqrt(np.mean(np.diff(a, axis=0)**2))) for a in actions]
        if delta > previous_delta*1.05+1e-12:
            reasons.append(f'{name}:action_delta_regression')
        rows.append(dict(motion=name, metrics=metric, baseline_metrics=previous, fall=row['fall'],
            baseline_fall=prior['fall'], first_fall_s=row['first_fall_s'],
            joint_change_percent=100*(metric['joint_rmse_rad']/previous['joint_rmse_rad']-1),
            action_delta_rms=delta, baseline_action_delta_rms=previous_delta))
    totals, baseline = aggregate(current), aggregate(list(old.values()))
    change = 100*(totals['joint_rmse_rad']/baseline['joint_rmse_rad']-1)
    if change > -1.:
        reasons.append('mean_joint_improvement_below_1_percent')
    return dict(aggregate=totals, baseline_aggregate=baseline, joint_change_percent=change,
                diagnostic_gate_passed=not reasons, gate_failures=reasons, motions=rows)


def report():
    job = read(OUT / 'job.json')
    if read(OUT / 'state.json')['status'] not in ('EVALUATION_COMPLETE', 'VERIFIED'):
        raise RuntimeError('Incomplete evaluation cannot be accepted')
    keys = [Path(r['path']).stem for r in job['diagnostic_motions']]
    results = {}
    for objective in ('mse', 'latent'):
        for sigma in (0., .01):
            key = f'{objective}_noise{sigma:g}_seed0'
            results[key] = paired(OUT / key, OUT / f'cv_noise{sigma:g}_seed0', keys)
    all_keys = [Path(r['path']).stem for r in job['training_motions']+job['diagnostic_motions']]
    results['r11_oracle'] = paired(OUT / 'r11_oracle', OUT / 'official_oracle', all_keys)
    retained = [o for o in ('mse', 'latent') if all(results[f'{o}_noise{s:g}_seed0']['diagnostic_gate_passed'] for s in (0., .01))]
    samples = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    values = [r['health'] for r in samples if 'health' in r]
    result = dict(experiment=job['experiment'], status=read(OUT / 'state.json')['status'],
        results=results, retained_adapter_candidates=retained,
        r11_retained=results['r11_oracle']['diagnostic_gate_passed'],
        independent_verification=read(OUT / 'independent_verification.json') if (OUT / 'independent_verification.json').exists() else None,
        health=dict(samples=len(values), boot_ids=sorted({r['boot_id'] for r in values}),
            cpu_max_c=max(r['temperatures_c']['x86_pkg_temp'] for r in values),
            gpu_max_c=max(g['temperature.gpu'] for r in values for g in r['gpus']),
            gpu_power_max_w=max(g['power.draw'] for r in values for g in r['gpus']),
            available_min_bytes=min(r['container_memory']['available_estimate_bytes'] for r in values),
            oom_counts=sorted({r['container_memory']['oom_kill'] for r in values}),
            critical_kernel_errors=[e for r in samples for e in r.get('kernel_errors', [])]),
        scope=job['scope'], backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'result.json', result)
    small = dict(experiment=job['experiment'], status=result['status'],
        job_sha256=sha256(OUT / 'job.json'), result_sha256=sha256(OUT / 'result.json'),
        health_sha256=sha256(LOG / 'health.jsonl'), health=result['health'],
        compared={k:{f:v[f] for f in ('aggregate', 'baseline_aggregate', 'joint_change_percent',
                                     'diagnostic_gate_passed', 'gate_failures')} for k,v in results.items()},
        retained_adapter_candidates=retained, r11_retained=result['r11_retained'],
        scope=job['scope'], backup_status='LOCAL_ONLY')
    atomic_json(ROOT / 'data/manifests/a3_zhy_search_20261012_E02.json', small)
    print(json.dumps(small, indent=2))


def verify():
    if read(OUT / 'state.json')['status'] != 'EVALUATION_COMPLETE':
        raise RuntimeError('Verify only a fully completed evaluation')
    preflight()
    directory = LOG / 'independent_verification'
    directory.mkdir(exist_ok=False)
    with (directory / 'console.log').open('w') as stream:
        child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
            '-m', 'script.a3.verify_zhy_search'], cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        code, reason = monitor(child, directory, {})
    if code or reason:
        raise RuntimeError(f'Independent verification stopped: exit={code}, reason={reason}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'verify', 'report'))
    value = dict(register=register, run=run, verify=verify, report=report)[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
