"""Replay captured observations through the actual PPO forward methods."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import command
from script.a3.zhy_batch_probe import SOURCE, OFFICIAL, read

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_wrapper_20261012_E07'
LOG = ROOT / 'logs/a3_zhy_wrapper_20261012_E07'
SOURCES = ('script/a3/zhy_wrapper_probe.py', 'script/a3/zhy_batch_probe.py',
           'script/a3/zhy_trial.py', 'script/a3/gradient_probe.py', 'script/a3/likelihood_probe.py')


def input_identity(actual, expected):
    import torch
    if actual.keys() != expected.keys():
        raise ValueError('Observation keys differ')
    rows = {}
    for key, value in actual.items():
        if value.shape != expected[key].shape:
            raise ValueError('Observation shapes differ')
        rows[key] = dict(exact=torch.equal(value, expected[key]),
            max_abs=float((value-expected[key]).abs().max()), shape=list(value.shape),
            dtype=str(value.dtype), device=str(value.device))
    return rows


def register():
    if OUT.exists():
        raise FileExistsError('No repeat or reset')
    prior = read(ROOT / 'data/experiments/a3_zhy_batch_20261012_E06/independent_verification.json')
    value = guard.preflight()
    if prior['status'] != 'VERIFIED' or prior['health']['boot_ids'] != [value['boot_id']]:
        raise RuntimeError('Requires verified same-boot E06')
    e05 = read(SOURCE / 'registration.json')
    e06 = read(ROOT / 'data/experiments/a3_zhy_batch_20261012_E06/registration.json')
    inputs = e06['input_sha256']
    if any(sha256(Path(p)) != h for p,h in inputs.items()):
        raise ValueError('Verified input changed')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode'] or value['container_memory']['oom_kill'] != e05['baseline_oom_kill']:
        raise RuntimeError('Kernel unavailable or OOM count changed')
    job = dict(experiment='A3_ZHY_WRAPPER_20261012_E07', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), input_sha256=inputs,
        code_sha256={p:sha256(ROOT / p) for p in SOURCES}, optimizer_steps_allowed=0,
        tf32=False, accelerator_mixed_precision='no', seed=0,
        reconstruction_action_rmse_max=1e-5, reconstruction_ratio_deviation_max=.001,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False,
        repeat_allowed=False, scope='SAVED_ROLLOUT_FORWARD_WRAPPER_NO_ENV_OR_OPTIMIZER',
        backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state.get('monitor_pid') != os.getppid()
        or os.environ.get('YUANQI_E07_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires active guarded monitor')
    if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
        raise ValueError('Registered code changed')
    if any(sha256(Path(p)) != h for p,h in job['input_sha256'].items()):
        raise ValueError('Registered input changed')
    import torch
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper, TRLPPOTrainer
    from script.a3.gradient_probe import parameter_digest
    from script.a3.likelihood_probe import compare

    torch.set_num_threads(1)
    torch.manual_seed(job['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(SOURCE / 'runtime_capture.pt', map_location='cpu', weights_only=False)['rollout']
    metadata = OmegaConf.create(read(SOURCE / 'runtime_model_config.json'))
    networks = [custom_instantiate(metadata.algo_config[key], env_config=metadata.env_config,
                algo_config=metadata.algo_config, _resolve=False) for key in ('actor', 'critic')]
    actor, critic = networks
    payload = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    for net, key in zip(networks, ('policy_state_dict', 'value_state_dict')):
        net.load_state_dict(payload[key], strict=True)
    del payload
    actor.get_std
    std = actor.get_std.detach().cpu().clone()
    with torch.no_grad():
        reconstruction = compare(actor.forward(source['all_obs_dict']).detach().cpu(),
            source['old_mu_batch'], std, source['old_sigma_batch'], source['actions'], source['logprobs'])
    atomic_json(OUT / 'reconstruction.json', reconstruction)
    if (reconstruction['action_rmse'] > job['reconstruction_action_rmse_max']
        or reconstruction['max_abs_ratio_minus_one'] > job['reconstruction_ratio_deviation_max']):
        raise ValueError('Runtime reconstruction failed; no attribution allowed')
    wrapper = PolicyAndValueWrapper(actor, critic, sample_actions_in_policy_update=
        metadata.algo_config.get('sample_actions_in_policy_update', True)).cuda().train()
    before = parameter_digest(list(wrapper.named_parameters()))
    buffers_before = parameter_digest(list(wrapper.named_buffers()))
    actor_buffers_before = parameter_digest(list(actor.named_buffers()))
    cuda_source = {k:({name:t.cuda() for name,t in v.items()} if isinstance(v,dict)
                     else v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in source.items()}
    batches = torch.load(SOURCE / 'initial_batches.pt', map_location='cpu', weights_only=False)
    actual = torch.load(SOURCE / 'actual_first_minibatch.pt', map_location='cpu', weights_only=False)
    dummy = SimpleNamespace(use_symmetry=False, compute_imgaug_bc_loss=False)
    rows, means, calls, identities = [], {}, [], []
    original_forward = actor.forward
    def traced(obs, *args, **kwargs):
        if not all(r['exact'] for r in input_identity(obs, expected).values()):
            raise ValueError('Actor receives different inputs')
        calls.append(dict(label=label, indices=indices.tolist(), grad_enabled=torch.is_grad_enabled(),
            autocast_enabled=torch.is_autocast_enabled(), tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, input_identity=input_identity(obs, expected)))
        return original_forward(obs, *args, **kwargs)
    actor.forward = traced
    accelerator = None
    for path in ('direct', 'wrapper', 'accelerated_wrapper'):
        if path == 'accelerated_wrapper':
            accelerator = Accelerator(mixed_precision='no')
            wrapper = accelerator.prepare(wrapper)
            if accelerator.mixed_precision != 'no':
                raise ValueError('Unexpected accelerator precision')
        for grad in (False, True):
            label = f'{path}_grad{grad}'
            mean = torch.empty_like(source['old_mu_batch'])
            for batch in batches:
                indices = batch['indices'].cuda()
                mb = TRLPPOTrainer._get_mb_rollout_data(dummy, cuda_source, indices)
                expected = {k:v[indices] for k,v in cuda_source['all_obs_dict'].items()}
                identity = input_identity(mb['mb_obs_dict'], expected)
                if not all(r['exact'] for r in identity.values()):
                    raise ValueError('Training slice changes input')
                identities.append(dict(label=label, indices=indices.tolist(), inputs=identity))
                with torch.set_grad_enabled(grad):
                    if path == 'direct':
                        output = actor.forward(mb['mb_obs_dict'], is_training=True)
                    else:
                        result = TRLPPOTrainer._forward_model(dummy, wrapper, mb)
                        output = result['policy_results']['action_mean']
                    mean[indices.cpu()] = output.detach().cpu()
                    del output
                    if path != 'direct':
                        del result
            means[label] = mean
            difference = mean[actual['indices']]-actual['mean']
            rows.append(dict(label=label,
                **compare(mean, source['old_mu_batch'], std, source['old_sigma_batch'],
                    source['actions'], source['logprobs']),
                versus_actual_first_mean_rmse=float(difference.square().mean().sqrt()),
                versus_actual_first_mean_max_abs=float(difference.abs().max())))
    actor.forward = original_forward
    wrapper = accelerator.unwrap_model(wrapper) if accelerator is not None else wrapper
    after = parameter_digest(list(wrapper.named_parameters()))
    buffers_after = parameter_digest(list(wrapper.named_buffers()))
    actor_buffers_after = parameter_digest(list(actor.named_buffers()))
    if before != after or actor_buffers_before != actor_buffers_after:
        raise ValueError('Replay changed parameters or actor buffers')
    capture = OUT / 'wrapper_means.pt'
    torch.save(dict(means=means, std=std, actual_first=actual), capture)
    atomic_json(OUT / 'result.json', dict(experiment=job['experiment'], optimizer_steps=0,
        reconstruction=reconstruction, rows=rows, calls=calls, input_slices=identities,
        parameters_before_sha256=before, parameters_after_sha256=after,
        buffers_before_sha256=buffers_before, buffers_after_sha256=buffers_after,
        actor_buffers_before_sha256=actor_buffers_before, actor_buffers_after_sha256=actor_buffers_after,
        critic_running_mean_std=critic.running_mean_std is not None,
        critic_normalizer_buffers_may_update_without_optimizer=True,
        accelerator_mixed_precision=accelerator.mixed_precision,
        capture_sha256=sha256(capture), registration_sha256=sha256(OUT / 'registration.json'),
        code_sha256=job['code_sha256'], scope=job['scope'], backup_status='LOCAL_ONLY'))
    print(json.dumps(dict(status='REPLAY_COMPLETE', variants=len(rows))))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY' or guard.health()['boot_id'] != job['boot_id']:
            raise RuntimeError('No repeat or reboot recovery')
        if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
            raise ValueError('Registered code changed')
        guard.preflight()
        guard.OUT, guard.LOG = OUT, LOG
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.zhy_wrapper_probe', 'worker'], cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, YUANQI_E07_MONITOR_PID=str(os.getpid())))
                code, reason = guard.guarded_monitor(child, LOG, job)
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
    import torch
    from script.a3.likelihood_probe import compare
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE' or guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Requires same-boot completed replay')
    result = read(OUT / 'result.json')
    path = OUT / 'wrapper_means.pt'
    if sha256(path) != result['capture_sha256']:
        raise ValueError('Capture changed')
    if any(sha256(Path(p)) != h for p,h in job['input_sha256'].items()):
        raise ValueError('Input changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source = torch.load(SOURCE / 'runtime_capture.pt', map_location='cpu', weights_only=False)['rollout']
    for row in result['rows']:
        mean = saved['means'][row['label']]
        value = compare(mean, source['old_mu_batch'], saved['std'], source['old_sigma_batch'],
                        source['actions'], source['logprobs'])
        if any(row[k] != v for k,v in value.items()):
            raise ValueError('Independent probability mismatch')
        actual = saved['actual_first']
        difference = mean[actual['indices']]-actual['mean']
        if (float(difference.square().mean().sqrt()) != row['versus_actual_first_mean_rmse']
            or float(difference.abs().max()) != row['versus_actual_first_mean_max_abs']):
            raise ValueError('Independent actual-minibatch comparison failed')
    raw = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    samples = [r['health'] for r in raw if 'health' in r]
    result.update(status='VERIFIED', independent_cpu_recomputation=True, state=dict(state),
        health=dict(samples=len(samples), first_utc=samples[0]['at_utc'], last_utc=samples[-1]['at_utc'],
            boot_ids=sorted({v['boot_id'] for v in samples}),
            cpu_max_c=max(v['temperatures_c']['x86_pkg_temp'] for v in samples),
            gpu_max_c=max(g['temperature.gpu'] for v in samples for g in v['gpus']),
            available_min_bytes=min(v['container_memory']['available_estimate_bytes'] for v in samples),
            oom_counts=sorted({v['container_memory']['oom_kill'] for v in samples}),
            critical_kernel_errors=[e for r in raw for e in r.get('kernel_errors', [])],
            telemetry_errors=[r for r in raw if 'error' in r]))
    atomic_json(OUT / 'independent_verification.json', result)
    atomic_json(ROOT / 'data/manifests/a3_zhy_wrapper_20261012_E07.json', result)
    state.update(status='VERIFIED', verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(dict(status='VERIFIED', rows=result['rows'], health=result['health']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = dict(register=register, run=run, worker=worker, verify=verify)[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value,int) else 0)
