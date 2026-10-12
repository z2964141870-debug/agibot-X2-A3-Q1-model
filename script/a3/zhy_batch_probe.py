"""Replay E05 with its actual runtime metadata to isolate minibatch behavior."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import command
from script.a3 import zhy_trial as guard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_batch_20261012_E06'
LOG = ROOT / 'logs/a3_zhy_batch_20261012_E06'
SOURCE = ROOT / 'data/experiments/a3_zhy_likelihood_20261012_E05'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
SOURCES = ('script/a3/zhy_batch_probe.py', 'script/a3/zhy_trial.py',
           'script/a3/likelihood_probe.py', 'script/a3/gradient_probe.py')


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('No reset or repeat of the batch probe')
    source = read(SOURCE / 'independent_verification.json')
    if read(SOURCE / 'state.json')['status'] != 'VERIFIED' or source['probability_contract_passed']:
        raise RuntimeError('Requires the completed failed E05 contract')
    value = guard.preflight()
    previous = read(SOURCE / 'registration.json')
    if value['boot_id'] != previous['boot_id']:
        raise RuntimeError('Boot changed: stop computation')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    diagnostic = read(SOURCE / 'likelihood_diagnostic.json')
    paths = {str(SOURCE / 'runtime_capture.pt'):source['capture_sha256'],
             str(SOURCE / 'runtime_model_config.json'):diagnostic['runtime_metadata_sha256'],
             str(SOURCE / 'actual_first_minibatch.pt'):source['precision']['actual_first_minibatch_sha256'],
             str(SOURCE / 'initial_batches.pt'):read(SOURCE / 'initial_batches.json')['capture_sha256'],
             str(OFFICIAL):guard.OFFICIAL_SHA}
    if any(sha256(Path(p)) != h for p,h in paths.items()):
        raise ValueError('Verified source changed')
    job = dict(experiment='A3_ZHY_BATCH_20261012_E06', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), input_sha256=paths,
        code_sha256={p:sha256(ROOT / p) for p in SOURCES}, optimizer_steps_allowed=0,
        tf32=False, reconstruction_action_rmse_max=1e-5, reconstruction_ratio_deviation_max=.001,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4,
        auto_resume=False, repeat_allowed=False, runtime_limit_seconds=900,
        scope='CAPTURE_REPLAY_WITH_RUNTIME_CONFIG_NO_ENVIRONMENT_OR_TRAINING', backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def replay_layout(actor, obs, groups, framewise, auxiliary):
    import torch
    size, steps = next(iter(obs.values())).shape[:2]
    coverage = [int(i) for indices in groups for i in indices.cpu()]
    if sorted(coverage) != list(range(size)):
        raise ValueError('Replay must cover each environment exactly once')
    result = torch.empty(size, steps, 29)
    for indices in groups:
        selected = {k:v[indices] for k,v in obs.items()}
        if framewise:
            mean = torch.cat([actor.forward({k:v[:, t:t+1] for k,v in selected.items()},
                is_training=auxiliary) for t in range(steps)], dim=1)
        else:
            mean = actor.forward(selected, is_training=auxiliary)
        if mean.shape != (len(indices), steps, 29):
            raise ValueError('Unexpected replay dimensions')
        result[indices.cpu()] = mean.detach().cpu()
    return result


def worker():
    state = read(OUT / 'state.json')
    if (state['status'] != 'RUNNING' or state.get('monitor_pid') != os.getppid()
        or os.environ.get('YUANQI_E06_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires its active guarded monitor')
    job = read(OUT / 'registration.json')
    if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
        raise ValueError('Registered code changed')
    import torch
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate
    from script.a3.gradient_probe import parameter_digest
    from script.a3.likelihood_probe import compare
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if any(sha256(Path(p)) != h for p,h in job['input_sha256'].items()):
        raise ValueError('Registered source changed')
    capture = torch.load(SOURCE / 'runtime_capture.pt', map_location='cpu', weights_only=False)
    source = capture['rollout']
    metadata = OmegaConf.create(read(SOURCE / 'runtime_model_config.json'))
    actor = custom_instantiate(metadata.algo_config.actor, env_config=metadata.env_config,
        algo_config=metadata.algo_config, _resolve=False)
    payload = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    actor.load_state_dict(payload['policy_state_dict'], strict=True)
    del payload
    actor.get_std
    before = parameter_digest(list(actor.named_parameters()))
    std = actor.get_std.detach().cpu().clone()
    batches = torch.load(SOURCE / 'initial_batches.pt', map_location='cpu', weights_only=False)
    actual = torch.load(SOURCE / 'actual_first_minibatch.pt', map_location='cpu', weights_only=False)
    rows, means = [], {}
    with torch.no_grad():
        actor.eval()
        mean = actor.forward(source['all_obs_dict'], is_training=False).cpu()
        baseline = compare(mean, source['old_mu_batch'], std, source['old_sigma_batch'],
            source['actions'], source['logprobs'])
        atomic_json(OUT / 'reconstruction.json', baseline)
        if (baseline['action_rmse'] > job['reconstruction_action_rmse_max']
            or baseline['max_abs_ratio_minus_one'] > job['reconstruction_ratio_deviation_max']):
            raise ValueError('Runtime reconstruction does not reproduce full16: no attribution allowed')
        for device in ('cpu', 'cuda'):
            actor.to(device)
            obs = {k:v.to(device) for k,v in source['all_obs_dict'].items()}
            all_envs = torch.arange(16, device=device)
            groups = [r['indices'].to(device) for r in batches]
            for auxiliary in (False, True):
                actor.train(auxiliary)
                for layout, selection, frames in (('full16', [all_envs], False),
                    ('chunks4', groups, False), ('chunks4_framewise', groups, True)):
                    label = f'{device}_aux{auxiliary}_{layout}'
                    mean = replay_layout(actor, obs, selection, frames, auxiliary)
                    means[label] = mean
                    row = dict(label=label, **compare(mean, source['old_mu_batch'], std,
                        source['old_sigma_batch'], source['actions'], source['logprobs']))
                    if layout == 'chunks4':
                        difference = mean[actual['indices']]-actual['mean']
                        row['versus_actual_first_mean_rmse'] = float(difference.square().mean().sqrt())
                        row['versus_actual_first_mean_max_abs'] = float(difference.abs().max())
                    rows.append(row)
        actor.cpu()
        after = parameter_digest(list(actor.named_parameters()))
        if before != after:
            raise ValueError('Replay changed actor parameters')
    saved = OUT / 'replayed_means.pt'
    torch.save(dict(means=means, std=std, actual_first=actual), saved)
    atomic_json(OUT / 'result.json', dict(experiment=job['experiment'], optimizer_steps=0,
        reconstruction=baseline, rows=rows, parameters_before_sha256=before,
        parameters_after_sha256=after, capture_sha256=sha256(saved),
        source_capture_sha256=sha256(SOURCE / 'runtime_capture.pt'),
        scope=job['scope'], backup_status='LOCAL_ONLY'))
    print(json.dumps(dict(status='REPLAY_COMPLETE', variants=len(rows))))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read(OUT / 'registration.json'), read(OUT / 'state.json')
        if state['status'] != 'READY' or guard.health()['boot_id'] != job['boot_id']:
            raise RuntimeError('No repeat or reboot recovery')
        if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
            raise ValueError('Registered code changed')
        guard.preflight()
        guard.OUT, guard.LOG = OUT, LOG
        logs = LOG / 'worker'
        logs.mkdir(exist_ok=False)
        state.update(status='RUNNING', started_at=time.time(), monitor_pid=os.getpid())
        atomic_json(OUT / 'state.json', state)
        try:
            with (logs / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.zhy_batch_probe', 'worker'], cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, YUANQI_E06_MONITOR_PID=str(os.getpid())))
                code, reason = guard.guarded_monitor(child, logs, job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED',
                         returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        return 0 if state['status'] == 'COMPLETE' else 2


def verify():
    import math
    import torch
    from script.a3.likelihood_probe import compare
    torch.set_num_threads(1)
    state, job, result = read(OUT / 'state.json'), read(OUT / 'registration.json'), read(OUT / 'result.json')
    if state['status'] != 'COMPLETE' or guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Requires completed same-boot replay')
    path = OUT / 'replayed_means.pt'
    if sha256(path) != result['capture_sha256'] or any(sha256(Path(p)) != h for p,h in job['input_sha256'].items()):
        raise ValueError('Capture or input changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source = torch.load(SOURCE / 'runtime_capture.pt', map_location='cpu', weights_only=False)['rollout']
    for row in result['rows']:
        mean = saved['means'][row['label']]
        recomputed = compare(mean, source['old_mu_batch'], saved['std'], source['old_sigma_batch'],
                             source['actions'], source['logprobs'])
        if any(not math.isclose(row[k], v, rel_tol=1e-6, abs_tol=1e-7) for k,v in recomputed.items()):
            raise ValueError('Independent probability recomputation failed')
        if 'versus_actual_first_mean_rmse' in row:
            actual = saved['actual_first']
            difference = mean[actual['indices']]-actual['mean']
            if (float(difference.square().mean().sqrt()) != row['versus_actual_first_mean_rmse']
                or float(difference.abs().max()) != row['versus_actual_first_mean_max_abs']):
                raise ValueError('Independent actual-minibatch comparison failed')
    raw = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    samples = [r['health'] for r in raw if 'health' in r]
    health = dict(samples=len(samples), first_utc=samples[0]['at_utc'], last_utc=samples[-1]['at_utc'],
        boot_ids=sorted({v['boot_id'] for v in samples}),
        cpu_max_c=max(v['temperatures_c']['x86_pkg_temp'] for v in samples),
        gpu_max_c=max(g['temperature.gpu'] for v in samples for g in v['gpus']),
        available_min_bytes=min(v['container_memory']['available_estimate_bytes'] for v in samples),
        oom_counts=sorted({v['container_memory']['oom_kill'] for v in samples}),
        critical_kernel_errors=[e for r in raw for e in r.get('kernel_errors', [])],
        telemetry_errors=[r for r in raw if 'error' in r])
    result.update(status='VERIFIED', independent_cpu_recomputation=True,
        registration_sha256=sha256(OUT / 'registration.json'), code_sha256=job['code_sha256'],
        state=state, health=health, health_sha256=sha256(LOG / 'health.jsonl'))
    atomic_json(OUT / 'independent_verification.json', result)
    atomic_json(ROOT / 'data/manifests/a3_zhy_batch_20261012_E06.json', result)
    state.update(status='VERIFIED', verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = dict(register=register, run=run, worker=worker, verify=verify)[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
