"""Guarded real-entry diagnosis following the completed E07 replay."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.a3.zhy_wrapper_probe import input_identity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_entry_20261012_E08'
LOG = ROOT / 'logs/a3_zhy_entry_20261012_E08'
PREVIOUS = ROOT / 'data/experiments/a3_zhy_wrapper_20261012_E07'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
SOURCES = ('script/a3/zhy_entry_probe.py', 'script/a3/zhy_entry_trainer.py',
    'script/a3/zhy_precision_trainer.py', 'script/a3/zhy_likelihood_trainer.py',
    'script/a3/runtime_likelihood_probe.py', 'script/a3/gradient_probe.py',
    'script/a3/likelihood_probe.py', 'script/a3/verified_finetune.py',
    'script/a3/finetune_trainer.py', 'script/a3/resumable_sonic.py',
    'script/a3/zhy_trial.py', 'script/a3/zhy_wrapper_probe.py')


def read(path):
    return json.loads(path.read_text())


def evaluate_capture(capture):
    import torch
    from script.a3.likelihood_probe import compare
    source, indices, entry = capture['rollout'], capture['indices'], capture['entry']
    expected = {key:value[indices] for key,value in source['all_obs_dict'].items()}
    identities = dict(actor_vs_stored=input_identity(entry['inputs'], expected),
        batch_vs_stored=input_identity(capture['batch_input_before'], expected),
        batch_after_vs_before=input_identity(capture['batch_input_after'], capture['batch_input_before']),
        live_after_vs_entry=input_identity(capture['live_input_after'], entry['inputs']))
    actual = capture['wrapper_mean']
    std, old_mean, old_std = capture['wrapper_std'], source['old_mu_batch'][indices], source['old_sigma_batch'][indices]
    actions, logprobs = source['actions'][indices], source['logprobs'][indices]
    rows = {'actual_first_minibatch':compare(actual, old_mean, std, old_std, actions, logprobs)}
    differences = {}
    for label,mean in dict(raw_actor_output=entry['mean'], **capture['replays']).items():
        rows[label] = compare(mean, old_mean, std, old_std, actions, logprobs)
        delta = mean-actual
        differences[label] = dict(rmse=float(delta.square().mean().sqrt()), max_abs=float(delta.abs().max()))
    states = [capture['rollout_actor_state'], capture['wrapper_actor_state_before'],
        entry['state_before'], entry['state_after'], capture['wrapper_actor_state_after'],
        capture['actor_state_after_replays']]
    flags = dict(actor_is_sampling_actor=capture['actor_is_sampling_actor'],
        actor_parameters_unchanged=len({value['parameters'] for value in states}) == 1,
        actor_buffers_unchanged=len({value['buffers'] for value in states}) == 1,
        raw_actor_matches_wrapper=torch.equal(entry['mean'], actual),
        loss_entry_matches_wrapper=torch.equal(capture['mean_at_loss_entry'], actual),
        actual_input_matches_stored=all(value['exact'] for value in identities['actor_vs_stored'].values()),
        immediate_same_call_matches=torch.equal(capture['replays']['same_live_inputs_and_options'], entry['mean']),
        sampling_and_training_tf32_off=not (entry['flags']['matmul_tf32'] or entry['flags']['cudnn_tf32']))
    passed = (all(flags.values()) and rows['actual_first_minibatch']['action_rmse'] <= 1e-5
              and rows['actual_first_minibatch']['max_abs_ratio_minus_one'] <= .001)
    return dict(probability_contract_passed=passed, flags=flags, input_identity=identities,
        rows=rows, differences_from_actual=differences, actual_forward_flags=entry['flags'],
        actual_input_metadata=entry['input_metadata'],
        actual_option_keys=sorted(entry['options']),
        accelerator_mixed_precision=capture['accelerator_mixed_precision'],
        actor_states=states, parameter_changes_from_load=capture['parameter_changes_from_load'],
        scope='ACTUAL_FIRST_ACTOR_ENTRY_ZERO_UPDATE_NOT_POLICY_IMPROVEMENT')


def register():
    if OUT.exists():
        raise FileExistsError('No reset or repeat of E08')
    previous = read(PREVIOUS / 'independent_verification.json')
    health = guard.preflight()
    if previous['status'] != 'VERIFIED' or previous['health']['boot_ids'] != [health['boot_id']]:
        raise RuntimeError('Requires verified same-boot E07')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode'] or sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise RuntimeError('Kernel unavailable or official checkpoint changed')
    job = dict(experiment='A3_ZHY_ENTRY_20261012_E08', registered_at=time.time(),
        boot_id=health['boot_id'], baseline_oom_kill=health['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), official_sha256=sha256(OFFICIAL),
        code_sha256={name:sha256(ROOT / name) for name in SOURCES},
        num_envs=16, policy_steps=24, minibatches=4, seed=0,
        optimizer_steps_allowed=0, immediate_replays=3, tf32=False,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        gate=dict(action_mean_rmse_max=1e-5, initial_ratio_deviation_max=.001),
        diagnostic_cpu_copies_and_rng_replays_change_timing=True, backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', health)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], optimizer_steps_allowed=0)))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY' or guard.health()['boot_id'] != job['boot_id']:
            raise RuntimeError('No repeat or reboot recovery')
        if any(sha256(ROOT / name) != value for name,value in job['code_sha256'].items()):
            raise RuntimeError('Registered sources changed')
        guard.preflight()
        run_dir = OUT / 'initial_rollout'
        run_dir.mkdir(exist_ok=False)
        environment = dict(os.environ, ISAAC_PYTHON=str(ROOT / 'data/environments/a3-sonic/bin/python'),
            REPO_DIR=str((ROOT / 'script/vendor/sonic_for_a3').resolve()),
            MOTION_FILE=str(ROOT / 'data/experiments/a3_fullchain_20261010/motionlib/train'),
            EXPERIMENT_DIR=str(run_dir), CHECKPOINT=str(OFFICIAL), NUM_ENVS='16',
            NUM_MINI_BATCHES='4', NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(ROOT), WANDB_MODE='disabled', YUANQI_RESUME_LOG_DIR=str(LOG),
            YUANQI_PROBE_OUTPUT=str(OUT))
        argv = ['bash', str(ROOT / 'script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh'),
            'seed=0', 'trainer._target_=script.a3.zhy_entry_trainer.ZHYEntryTrainer',
            'algo.config.num_learning_epochs=1', 'algo.config.actor_learning_rate=2e-6',
            'algo.config.adaptive_lr_min=1e-6', 'algo.config.adaptive_lr_max=2e-5',
            'callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback',
            'callbacks.model_save.save_frequency=1', '+callbacks.model_save.target_global_step=2']
        state.update(status='RUNNING', started_at=time.time(), argv=argv)
        atomic_json(OUT / 'state.json', state)
        guard.OUT, guard.LOG = OUT, LOG
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, LOG, job)
            state.update(status='COMPLETE' if code == 0 and reason is None and (OUT / 'entry_result.json').exists()
                         else 'STOPPED', returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        return 0 if state['status'] == 'COMPLETE' else 2


def verify():
    import torch
    from script.a3.zhy_likelihood_trainer import sampling_identity
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE' or guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Requires a completed same-boot E08')
    result = read(OUT / 'entry_result.json')
    path = OUT / 'entry_capture.pt'
    if sha256(path) != result['capture_sha256'] or result['optimizer_steps']:
        raise ValueError('Capture changed or optimizer update observed')
    steps = LOG / 'actual_optimizer_steps.jsonl'
    if steps.exists() and steps.stat().st_size:
        raise ValueError('Unexpected optimizer step')
    capture = torch.load(path, map_location='cpu', weights_only=False)
    independent = evaluate_capture(capture)
    if any(result[key] != value for key,value in independent.items()):
        raise ValueError('Independent entry evidence recomputation failed')
    sampling = read(OUT / 'sampling_identity.json')
    sampling_path = OUT / 'sampled_inputs.pt'
    if sha256(sampling_path) != sampling['capture_sha256']:
        raise ValueError('Sampling capture changed')
    identity = sampling_identity(torch.load(sampling_path, map_location='cpu', weights_only=False), capture['rollout'])
    if any(sampling[key] != value for key,value in identity.items()):
        raise ValueError('Independent sampling identity recomputation failed')
    raw = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    unique = {}
    for row in raw:
        value = row.get('health', row)
        if 'boot_id' in value:
            unique[value['at_utc']] = value
    samples = list(unique.values())
    times = [datetime.fromisoformat(value['at_utc']).timestamp() for value in samples]
    result.update(status='VERIFIED', experiment=job['experiment'], independent_cpu_recomputation=True,
        sampling_identity=sampling, state=dict(state), code_sha256=job['code_sha256'],
        health=dict(samples=len(samples), first_utc=samples[0]['at_utc'], last_utc=samples[-1]['at_utc'],
            max_sample_gap_seconds=max((b-a for a,b in zip(times,times[1:])), default=0),
            boot_ids=sorted({value['boot_id'] for value in samples}),
            cpu_max_c=max(value['temperatures_c']['x86_pkg_temp'] for value in samples),
            gpu_max_c=max(g['temperature.gpu'] for value in samples for g in value['gpus']),
            available_min_bytes=min(value['container_memory']['available_estimate_bytes'] for value in samples),
            oom_counts=sorted({value['container_memory']['oom_kill'] for value in samples}),
            critical_kernel_errors=[e for row in raw for e in row.get('kernel_errors', [])],
            telemetry_errors=[row for row in raw if 'error' in row]))
    result['probability_contract_passed'] = (result['probability_contract_passed']
        and sampling['all_inputs_exact'] and sampling['sampled_means_match_storage'])
    atomic_json(OUT / 'independent_verification.json', result)
    atomic_json(ROOT / 'data/manifests/a3_zhy_entry_20261012_E08.json', result)
    state.update(status='VERIFIED', verified_at=time.time(),
                 probability_contract_passed=result['probability_contract_passed'])
    atomic_json(OUT / 'state.json', state)
    print(json.dumps({key:result[key] for key in
        ('status', 'probability_contract_passed', 'flags', 'rows', 'differences_from_actual', 'health')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'verify'))
    value = dict(register=register, run=run, verify=verify)[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value,int) else 0)
