"""Isolate saved encoder/decoder effects on fixed, verified FP32 observations."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_modules_20261012_E12'
LOG = ROOT / 'logs/a3_zhy_modules_20261012_E12'
GRADIENT = ROOT / 'data/experiments/a3_zhy_gradient_20261012_E11'
PRECISION = ROOT / 'data/experiments/a3_zhy_consistency_20261012_E09'
RUN = ROOT / 'data/training/a3_zhy_fp32_20261012_E10/R12_001_s0'
VARIANTS = ('initial', 'update1', 'update2', 'encoder_only', 'decoder_only', 'std_only')
SOURCES = ('script/a3/zhy_module_probe.py', 'script/a3/zhy_trial.py',
    'script/a3/checkpoint_store.py', 'script/a3/gradient_probe.py',
    'script/a3/likelihood_probe.py', 'script/a3/zhy_likelihood_trainer.py')


def read(path):
    return json.loads(path.read_text())


def module_swap(initial, updated, family):
    if initial.keys() != updated.keys():
        raise ValueError('Actor state names differ')
    prefix = {'encoder': 'actor_module.encoders.', 'decoder': 'actor_module.decoders.', 'std': 'std'}[family]
    selected = [name for name in initial if name == prefix or name.startswith(prefix) and family != 'std']
    if not selected:
        raise ValueError('Requested Actor module is absent')
    return {name: updated[name] if name in selected else value for name, value in initial.items()}


def drift(base, current):
    import torch
    if base['mean'].shape != current['mean'].shape or base['tokens'].shape != current['tokens'].shape:
        raise ValueError('Replay dimensions differ')
    difference = current['mean'] - base['mean']
    changed = (current['tokens'] != base['tokens']).reshape(*difference.shape[:-1], -1)
    frames = changed.any(dim=-1)
    frame_rms = difference.square().mean(dim=-1).sqrt()
    return dict(action_rmse=float(difference.square().mean().sqrt()),
        action_max_abs=float(difference.abs().max()),
        token_changed_coordinates=int(changed.sum()), token_coordinates=changed.numel(),
        token_changed_frames=int(frames.sum()), frames=frames.numel(),
        changed_token_frames_action_rmse=float(frame_rms[frames].square().mean().sqrt()) if frames.any() else None,
        unchanged_token_frames_action_rmse=float(frame_rms[~frames].square().mean().sqrt()) if (~frames).any() else None,
        action_changed_frames_over_1e_4=int((frame_rms > 1e-4).sum()))


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Boot changed; no recovery')
    if any(sha256(ROOT / name) != value for name, value in job['code_sha256'].items()):
        raise ValueError('Registered code changed')
    if any(sha256(Path(name)) != value for name, value in job['input_sha256'].items()):
        raise ValueError('Verified input changed')


def register():
    if OUT.exists():
        raise FileExistsError('No repeat or reset of E12')
    prior = read(GRADIENT / 'independent_verification.json')
    value = guard.preflight()
    if (prior['status'] != 'VERIFIED' or not prior['probability_contract_passed']
        or prior['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot FP32 E11')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    inputs = [GRADIENT / 'independent_verification.json', GRADIENT / 'rollout.pt',
        GRADIENT / 'forward_evidence.pt', PRECISION / 'runtime_model_config.json']
    inputs += [RUN / f'model_step_{step:06d}.pt' for step in range(3)]
    job = dict(experiment='A3_ZHY_MODULES_20261012_E12', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(),
        code_sha256={name: sha256(ROOT / name) for name in SOURCES},
        input_sha256={str(path): sha256(path) for path in inputs},
        variants=VARIANTS, optimizer_steps_allowed=0, new_rollouts=0,
        fp32=True, expected_frames=384, reconstruction_action_rmse_max=1e-5,
        reconstruction_ratio_deviation_max=.001, max_cpu_c=90, max_gpu_c=85,
        consecutive_hot_samples=2, sample_seconds=5, memory_reserve_gib=4,
        runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        scope='FIXED_STATE_OFFLINE_WEIGHT_SWAPS_NOT_CLOSED_LOOP_OR_NEW_TRAINING', backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state.get('monitor_pid') != os.getppid()
        or os.environ.get('YUANQI_E12_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires its active guarded monitor')
    unchanged(job)
    import torch
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate
    from script.a3.gradient_probe import parameter_digest
    from script.a3.likelihood_probe import compare
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled('cuda'):
        raise ValueError('Unexpected CUDA autocast')
    source = torch.load(GRADIENT / 'rollout.pt', map_location='cpu', weights_only=False)
    evidence = torch.load(GRADIENT / 'forward_evidence.pt', map_location='cpu', weights_only=False)
    metadata = OmegaConf.create(read(PRECISION / 'runtime_model_config.json'))
    actor = custom_instantiate(metadata.algo_config.actor, env_config=metadata.env_config,
        algo_config=metadata.algo_config, _resolve=False).cuda().eval()
    states = []
    for step in range(3):
        payload = torch.load(RUN / f'model_step_{step:06d}.pt', map_location='cpu', weights_only=False)
        states.append(payload['policy_state_dict'])
        del payload
    alternatives = dict(zip(VARIANTS[:3], states, strict=True))
    for family in ('encoder', 'decoder', 'std'):
        alternatives[family + '_only'] = module_swap(states[0], states[2], family)
    captured, digests = {}, {}
    with torch.no_grad():
        for label in VARIANTS:
            actor.load_state_dict(alternatives[label], strict=True)
            actor.get_std
            before = parameter_digest(list(actor.named_parameters()))
            means, tokens = torch.empty_like(source['old_mu_batch']), None
            for forward in evidence['forwards']:
                inds = forward['indices']
                obs = {name: value[inds].cuda() for name, value in source['all_obs_dict'].items()}
                mean = actor.forward(obs, is_training=False).cpu()
                codes = actor.actor_module._last_full_latent_flat.detach().cpu().clone()
                if codes.shape[:-1] != mean.shape[:-1] or not torch.isfinite(codes).all():
                    raise ValueError('Assembled quantized token shape or values differ')
                if tokens is None:
                    tokens = torch.empty((*means.shape[:-1], codes.shape[-1]), dtype=codes.dtype)
                means[inds], tokens[inds] = mean, codes
            after = parameter_digest(list(actor.named_parameters()))
            if before != after:
                raise ValueError('Offline forward changed Actor parameters')
            captured[label] = dict(mean=means, tokens=tokens, std=actor.get_std.cpu().clone())
            digests[label] = dict(before=before, after=after)
            if label == 'initial':
                reconstruction = compare(means, source['old_mu_batch'], captured[label]['std'],
                    source['old_sigma_batch'], source['actions'], source['logprobs'])
                atomic_json(OUT / 'reconstruction.json', reconstruction)
                if (reconstruction['action_rmse'] > job['reconstruction_action_rmse_max']
                    or reconstruction['max_abs_ratio_minus_one'] > job['reconstruction_ratio_deviation_max']):
                    raise ValueError('Initial Actor does not reconstruct E11; no attribution')
    if not torch.equal(captured['initial']['mean'], captured['std_only']['mean']):
        raise ValueError('Std-only swap changed deterministic action mean')
    if not torch.equal(captured['initial']['tokens'], captured['decoder_only']['tokens']):
        raise ValueError('Decoder-only swap changed encoder tokens')
    path = OUT / 'module_outputs.pt'
    torch.save(captured, path)
    rows = [dict(variant=label, **drift(captured['initial'], captured[label])) for label in VARIANTS]
    atomic_json(OUT / 'result.json', dict(rows=rows, reconstruction=reconstruction,
        parameters_unchanged_during_replay=digests, capture_sha256=sha256(path),
        optimizer_steps=0, new_rollouts=0, scope=job['scope'], backup_status='LOCAL_ONLY'))
    print(json.dumps(dict(status='REPLAY_COMPLETE', rows=rows), indent=2))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery')
        unchanged(job)
        guard.preflight()
        guard.OUT, guard.LOG = OUT, LOG
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        stage = LOG / 'worker'
        stage.mkdir()
        try:
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.zhy_module_probe', 'worker'], cwd=ROOT,
                    env=dict(os.environ, YUANQI_E12_MONITOR_PID=str(os.getpid())),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
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
    from script.a3.gradient_report import finite
    from script.a3.likelihood_probe import compare
    from script.a3 import zhy_fp32_trial as prior
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Requires complete E12')
    unchanged(job)
    result = read(OUT / 'result.json')
    path = OUT / 'module_outputs.pt'
    if sha256(path) != result['capture_sha256'] or result['optimizer_steps'] or result['new_rollouts']:
        raise ValueError('Output identity or no-training contract failed')
    captured = torch.load(path, map_location='cpu', weights_only=False)
    finite(captured)
    rows = [dict(variant=label, **drift(captured['initial'], captured[label])) for label in VARIANTS]
    if rows != result['rows'] or any(row['frames'] != job['expected_frames'] for row in rows):
        raise ValueError('Independent drift or coverage recomputation failed')
    source = torch.load(GRADIENT / 'rollout.pt', map_location='cpu', weights_only=False)
    initial = captured['initial']
    reconstruction = compare(initial['mean'], source['old_mu_batch'], initial['std'],
        source['old_sigma_batch'], source['actions'], source['logprobs'])
    if reconstruction != result['reconstruction']:
        raise ValueError('Independent probability reconstruction failed')
    previous = prior.LOG
    prior.LOG = LOG
    try:
        health = prior.health_summary()
    finally:
        prior.LOG = previous
    receipt = dict(result, status='VERIFIED', experiment=job['experiment'], rows=rows,
        independent_cpu_recomputation=True, health=health, code_sha256=job['code_sha256'],
        input_sha256=job['input_sha256'], state=dict(state),
        captures=dict(module_outputs=dict(bytes=path.stat().st_size, sha256=sha256(path))),
        policy_effect_acceptance=False, closed_loop_causal_proof=False)
    atomic_json(OUT / 'independent_verification.json', receipt)
    atomic_json(ROOT / 'data/manifests/a3_zhy_modules_20261012_E12.json', receipt)
    state.update(status='VERIFIED', verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(dict(status='VERIFIED', rows=rows, health=health), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = {'register': register, 'run': run, 'worker': worker, 'verify': verify}[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
