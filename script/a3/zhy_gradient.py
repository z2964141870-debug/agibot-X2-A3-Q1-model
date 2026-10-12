"""Audit saved R12 changes, then diagnose one FP32 rollout without updating it."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_gradient_20261012_E11'
LOG = ROOT / 'logs/a3_zhy_gradient_20261012_E11'
PRIOR = ROOT / 'data/experiments/a3_zhy_consistency_20261012_E09'
TRIAL = ROOT / 'data/experiments/a3_zhy_fp32_20261012_E10'
RUN = ROOT / 'data/training/a3_zhy_fp32_20261012_E10/R12_001_s0'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
SOURCES = ('script/a3/zhy_gradient.py', 'script/a3/zhy_gradient_trainer.py',
    'script/a3/gradient_probe.py', 'script/a3/consistent_precision.py',
    'script/a3/verified_finetune.py', 'script/a3/finetune_trainer.py',
    'script/a3/resumable_sonic.py', 'script/a3/checkpoint_store.py',
    'script/a3/zhy_trial.py', 'script/a3/likelihood_probe.py',
    'script/a3/zhy_likelihood_trainer.py')


def read(path):
    return json.loads(path.read_text())


def checkpoint_rows(receipt):
    records = receipt['checkpoints']
    rows = list(records.values()) if isinstance(records, dict) else records
    if len(rows) != 3 or sorted(row['step'] for row in rows) != [0, 1, 2]:
        raise ValueError('Requires the three verified R12 saving points')
    return rows


def buffer_changes(before, after):
    import torch
    if before.keys() != after.keys():
        raise ValueError('Buffer names changed')
    return [name for name in before if not torch.equal(before[name], after[name])]


def state_delta(before, after, parameter_names):
    import torch
    from script.a3.gradient_probe import group
    if before.keys() != after.keys():
        raise ValueError('State tensor names changed')
    buckets = {}
    changed_parameters, changed_buffers = [], []
    for name, old in before.items():
        new = after[name]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError('State tensor shape or dtype changed: ' + name)
        if not torch.isfinite(new).all():
            raise ValueError('Nonfinite saved state tensor')
        parameter = name in parameter_names
        bucket = group(name) if parameter else 'buffers'
        row = buckets.setdefault(bucket, dict(elements=0, changed_elements=0,
            delta_squared=0., original_squared=0., max_abs_delta=0., changed_tensors=[]))
        delta = new.double() - old.double()
        count = int((new != old).sum())
        row['elements'] += old.numel()
        row['changed_elements'] += count
        row['delta_squared'] += float(delta.square().sum())
        row['original_squared'] += float(old.double().square().sum())
        if delta.numel():
            row['max_abs_delta'] = max(row['max_abs_delta'], float(delta.abs().max()))
        if count:
            row['changed_tensors'].append(name)
            (changed_parameters if parameter else changed_buffers).append(name)
    for row in buckets.values():
        row['delta_l2'] = row.pop('delta_squared') ** .5
        original = row.pop('original_squared') ** .5
        row['relative_l2'] = row['delta_l2'] / original if original else None
    return dict(groups=buckets, changed_parameters=changed_parameters,
                changed_buffers=changed_buffers)


def recompute_terms(row, source, config):
    import torch
    inds, policy = row['indices'], row['policy']
    advantage = source['advantages'][inds]
    weights = config['multi_critic_advantage_weights']
    if weights is not None:
        advantage = advantage * torch.as_tensor(weights).to(advantage)[None, None, :]
    ratio = torch.exp(policy['logprobs'] - source['logprobs'][inds]).unsqueeze(-1)
    pg = torch.maximum(-advantage * ratio,
        -advantage * ratio.clamp(1-config['cliprange'], 1+config['cliprange'])).sum(dim=-1)
    values, returns = source['values'][inds], source['returns'][inds]
    predicted = row['value']
    clipped = predicted.clamp(values-config['cliprange_value'], values+config['cliprange_value'])
    vf = torch.maximum((predicted-returns).square(), (clipped-returns).square()).mean(dim=-1)
    valid_policy, valid_value = ~source['padding_mask'][inds], ~source['padding_mask_p1'][inds]
    masked = lambda value, mask: (value * mask).sum() / mask.sum()
    coefficient = config['ppo_loss_coef']
    aux = sum(float(policy['aux_loss_coef'].get(name, 0.)) * float(value)
              for name, value in policy['aux_losses'].items()) * config['aux_loss_scale']
    return dict(policy=float(masked(pg, valid_policy)) * coefficient,
        value=float(masked(vf, valid_value)) * config['vf_coef'] * coefficient,
        entropy=-float(masked(policy['entropy'], valid_policy)) * config['entropy_coef'] * coefficient,
        aux=aux)


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Boot changed; no recovery')
    for relative, expected in job['code_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('Registered code changed')
    for path, expected in job['input_sha256'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('Registered input changed')


def register():
    if OUT.exists():
        raise FileExistsError('No repeat or reset of E11')
    previous, trial = read(PRIOR / 'independent_verification.json'), read(TRIAL / 'independent_verification.json')
    value = guard.preflight()
    if (previous['status'] != 'VERIFIED' or not previous['probability_contract_passed']
        or trial['status'] != 'VERIFIED' or trial['policy_effect']['diagnostic_gate_passed']
        or previous['health']['boot_ids'] != [value['boot_id']]
        or trial['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot E09 fix and failed R12 effect gate')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode'] or sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise ValueError('Kernel telemetry or official model identity failed')
    inputs = [OFFICIAL, PRIOR / 'independent_verification.json', TRIAL / 'independent_verification.json']
    inputs += [RUN / f'model_step_{step:06d}.pt' for step in range(3)]
    inputs += [RUN / 'config.yaml']
    for row in checkpoint_rows(trial):
        path = Path(row['checkpoint'])
        if sha256(path) != row['sha256']:
            raise ValueError('Previously verified R12 checkpoint changed')
    job = dict(experiment='A3_ZHY_GRADIENT_20261012_E11', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(),
        code_sha256={name: sha256(ROOT / name) for name in SOURCES},
        input_sha256={str(path): sha256(path) for path in inputs},
        num_envs=16, policy_steps=24, minibatches=4, seed=0,
        optimizer_steps_allowed=0, fresh_rollouts=1, fp32=True,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900,
        auto_resume=False, repeat_allowed=False,
        probability_mean_rmse_max=1e-5, initial_ratio_deviation_max=.001,
        scope='SAVED_CHECKPOINT_AUDIT_AND_ACTUAL_FP32_ZERO_UPDATE_GRADIENTS',
        historical_causality=False, policy_effect_acceptance=False, backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def audit_worker():
    state = read(OUT / 'state.json')
    if (state['status'] != 'RUNNING' or state.get('phase') != 'CHECKPOINT_AUDIT'
        or state.get('monitor_pid') != os.getppid()
        or os.environ.get('YUANQI_E11_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Audit worker requires its active guarded monitor')
    unchanged(read(OUT / 'registration.json'))
    import torch
    torch.set_num_threads(1)
    initial = torch.load(RUN / 'model_step_000000.pt', map_location='cpu', weights_only=False)
    source = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    for key in ('policy_state_dict', 'value_state_dict'):
        if initial[key].keys() != source[key].keys() or any(
            not torch.equal(initial[key][name], source[key][name]) for name in source[key]):
            raise ValueError('Initial R12 networks differ from official')
    del source
    rows = []
    for step in range(3):
        path = RUN / f'model_step_{step:06d}.pt'
        payload = initial if step == 0 else torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != step:
            raise ValueError('Checkpoint counter mismatch')
        counters = sorted({int(value['step']) for value in payload['optimizer_state_dict']['state'].values()})
        rows.append(dict(step=step, path=str(path), sha256=sha256(path),
            optimizer_parameter_counters=counters,
            std=payload['policy_state_dict']['std'].tolist()))
        if step:
            del payload
    atomic_json(OUT / 'checkpoint_audit.json', dict(initial_networks_match_official=True,
        rows=rows, scope='READ_ONLY_SAVED_CHECKPOINT_AUDIT_NO_OPTIMIZER_OR_CUDA'))
    print(json.dumps(dict(status='CHECKPOINT_AUDIT_COMPLETE', counters=[r['optimizer_parameter_counters'] for r in rows])))


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
        base_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(ROOT), YUANQI_E11_MONITOR_PID=str(os.getpid()))
        try:
            stage = LOG / 'checkpoint_audit'
            stage.mkdir()
            state.update(phase='CHECKPOINT_AUDIT')
            atomic_json(OUT / 'state.json', state)
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.zhy_gradient', 'audit-worker'], cwd=ROOT, env=base_env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            if code or reason or not (OUT / 'checkpoint_audit.json').exists():
                raise RuntimeError(f'Checkpoint audit stopped: {code}, {reason}')
            unchanged(job)
            guard.preflight()
            initial = OUT / 'initial_rollout'
            initial.mkdir(exist_ok=False)
            stage = LOG / 'gradient_probe'
            stage.mkdir()
            env = dict(base_env, ISAAC_PYTHON=str(ROOT / 'data/environments/a3-sonic/bin/python'),
                REPO_DIR=str((ROOT / 'script/vendor/sonic_for_a3').resolve()),
                MOTION_FILE=str(ROOT / 'data/experiments/a3_fullchain_20261010/motionlib/train'),
                EXPERIMENT_DIR=str(initial), CHECKPOINT=str(OFFICIAL), NUM_ENVS='16',
                NUM_MINI_BATCHES='4', NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1',
                WANDB_MODE='disabled', ACCELERATE_MIXED_PRECISION='no',
                YUANQI_RESUME_LOG_DIR=str(stage), YUANQI_PROBE_OUTPUT=str(OUT))
            argv = ['bash', str(ROOT / 'script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh'),
                'seed=0', 'trainer._target_=script.a3.zhy_gradient_trainer.ZHYGradientTrainer',
                'algo.config.num_learning_epochs=1', 'algo.config.actor_learning_rate=2e-6',
                'algo.config.adaptive_lr_min=1e-6', 'algo.config.adaptive_lr_max=2e-5',
                'callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback',
                'callbacks.model_save.save_frequency=1', '+callbacks.model_save.target_global_step=2',
                '++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']
            state.update(phase='ZERO_UPDATE_GRADIENT', argv=argv, probe_started_at=time.time())
            atomic_json(OUT / 'state.json', state)
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            complete = code == 0 and reason is None and (OUT / 'forward_evidence.json').exists()
            state.update(status='COMPLETE' if complete else 'STOPPED', returncode=code, reason=reason)
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
    from script.a3.gradient_report import finite
    from script.a3.likelihood_probe import compare
    from script.a3.zhy_fp32_trial import health_summary
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Requires completed E11')
    unchanged(job)
    result, evidence_meta = read(OUT / 'gradient_diagnostic.json'), read(OUT / 'forward_evidence.json')
    if (result['optimizer_steps'] or result['global_step']
        or result['parameters_before_sha256'] != result['parameters_after_sha256']):
        raise ValueError('Diagnostic changed parameters or optimizer counter')
    steps = LOG / 'gradient_probe/actual_optimizer_steps.jsonl'
    if steps.exists() and steps.stat().st_size:
        raise ValueError('Actual optimizer step recorded')
    for filename, receipt, key in (('rollout.pt', result, 'rollout_sha256'),
                                  ('forward_evidence.pt', evidence_meta, 'sha256')):
        if sha256(OUT / filename) != receipt[key]:
            raise ValueError('Diagnostic capture changed')
    source = torch.load(OUT / 'rollout.pt', map_location='cpu', weights_only=False)
    evidence = torch.load(OUT / 'forward_evidence.pt', map_location='cpu', weights_only=False)
    finite(source)
    finite(evidence)
    coverage, rows = [], []
    for index, (forward, gradient) in enumerate(zip(evidence['forwards'], result['batches'], strict=True)):
        coverage += forward['indices'].tolist()
        if forward['indices'].tolist() != gradient['environment_indices']:
            raise ValueError('Forward and gradient batch identity differs')
        terms = recompute_terms(forward, source, evidence['objective'])
        if any(not math.isclose(value, gradient['losses'][key], rel_tol=2e-5, abs_tol=1e-6)
               for key, value in terms.items()):
            raise ValueError('Independent CPU loss decomposition failed')
        inds, policy = forward['indices'], forward['policy']
        probability = compare(policy['action_mean'], source['old_mu_batch'][inds],
            policy['action_std'], source['old_sigma_batch'][inds],
            source['actions'][inds], source['logprobs'][inds])
        if (probability['action_rmse'] > job['probability_mean_rmse_max']
            or probability['max_abs_ratio_minus_one'] > job['initial_ratio_deviation_max']):
            raise ValueError('Corrected precision probability contract regressed')
        rows.append(dict(batch=index, probability=probability, cpu_recomputed_losses=terms,
            gradient_geometry=gradient['gradient_geometry'],
            initial_value_explained_variance=gradient['initial_value_explained_variance'],
            normalized_advantage=gradient['normalized_advantage'],
            critic_buffers_changed_during_forward=buffer_changes(forward['buffers_before'], forward['buffers_after'])))
    if len(rows) != 4 or sorted(coverage) != list(range(16)):
        raise ValueError('Require exact four-batch coverage')
    precision = read(LOG / 'gradient_probe/precision_contract.json')
    if precision['actor_calls'] != dict(rollout=24, training=4):
        raise ValueError('Unexpected actual Actor call counts')
    for row in precision['first_call_per_mode']:
        if row['mixed_precision'] != 'no' or any(value for key, value in row.items()
                if key not in ('training', 'mixed_precision')):
            raise ValueError('Actual precision contract failed')
    named = set(evidence['parameter_names'])
    import yaml
    saved_config = yaml.safe_load((RUN / 'config.yaml').read_text())['algo']['config']
    initial = torch.load(RUN / 'model_step_000000.pt', map_location='cpu', weights_only=False)
    base = {prefix+name: tensor for key, prefix in (
        ('policy_state_dict', 'policy.'), ('value_state_dict', 'value_model.')) for name, tensor in initial[key].items()}
    audits = []
    for step in (1, 2):
        payload = torch.load(RUN / f'model_step_{step:06d}.pt', map_location='cpu', weights_only=False)
        current = {prefix+name: tensor for key, prefix in (
            ('policy_state_dict', 'policy.'), ('value_state_dict', 'value_model.')) for name, tensor in payload[key].items()}
        clamped_std = base['policy.std'].clamp(saved_config['std_clamp_min'], saved_config['std_clamp_max'])
        audits.append(dict(step=step, versus_initial=state_delta(base, current, named),
            initial_std_clamp_changed=not torch.equal(base['policy.std'], clamped_std),
            std_delta_after_expected_clamp_l2=float((current['policy.std']-clamped_std).double().norm()),
            scope='SAVED_STATE_DELTA_INCLUDES_EXPECTED_INITIAL_STD_CLAMP'))
        del payload, current
    import script.a3.zhy_fp32_trial as old
    previous_log = old.LOG
    old.LOG = LOG
    try:
        health = health_summary()
    finally:
        old.LOG = previous_log
    receipt = dict(status='VERIFIED', experiment=job['experiment'], optimizer_steps=0,
        probability_contract_passed=True, parameters_unchanged_during_gradients=True,
        cpu_independent_loss_recomputation=True, gradient_geometry_independently_recomputed=False,
        fresh_rollouts=1, batches=rows, checkpoint_audit=read(OUT / 'checkpoint_audit.json'),
        saved_update_deltas=audits,
        buffers_changed_during_rollout=buffer_changes(evidence['loaded_buffers'], evidence['rollout_buffers']),
        buffers_changed_during_gradient_forwards=buffer_changes(evidence['rollout_buffers'], evidence['final_buffers']),
        actor_has_normalizer=evidence['actual_actor_normalizer'],
        critic_has_normalizer=evidence['actual_critic_normalizer'], precision=precision, health=health,
        captures={name: dict(sha256=sha256(OUT / name), bytes=(OUT / name).stat().st_size)
                  for name in ('rollout.pt', 'forward_evidence.pt', 'gradient_diagnostic.json')},
        state=dict(state), code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
        historical_causal_proof=False, policy_effect_acceptance=False, backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'independent_verification.json', receipt)
    atomic_json(ROOT / 'data/manifests/a3_zhy_gradient_20261012_E11.json', receipt)
    state.update(status='VERIFIED', verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(dict(status='VERIFIED', batches=rows, health=health), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'audit-worker', 'verify'))
    actions = {'register': register, 'run': run, 'audit-worker': audit_worker, 'verify': verify}
    value = actions[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
