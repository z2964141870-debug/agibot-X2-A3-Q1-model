"""New-host zero-update probability diagnosis after the interrupted E10."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.a3 import zhy_trial as guard

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'E03'
OUT = ROOT / 'data/experiments/a3_zhy_likelihood_20261012_E03'
LOG = ROOT / 'logs/a3_zhy_likelihood_20261012_E03'
RUN = OUT / 'initial_rollout'
PREVIOUS = ROOT / 'data/experiments/a3_zhy_search_20261012_E02'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
SOURCES = ('script/a3/zhy_likelihood.py', 'script/a3/zhy_likelihood_trainer.py',
    'script/a3/runtime_likelihood_probe.py', 'script/a3/gradient_probe.py',
    'script/a3/likelihood_probe.py', 'script/a3/verified_finetune.py',
    'script/a3/finetune_trainer.py', 'script/a3/zhy_trial.py')


def read(path):
    return json.loads(path.read_text())


def configure(experiment):
    global EXPERIMENT, OUT, LOG, RUN
    EXPERIMENT = experiment
    OUT = ROOT / f'data/experiments/a3_zhy_likelihood_20261012_{experiment}'
    LOG = ROOT / f'logs/a3_zhy_likelihood_20261012_{experiment}'
    RUN = OUT / 'initial_rollout'


def register():
    if OUT.exists():
        raise FileExistsError('No reset of the probability diagnosis')
    if read(PREVIOUS / 'state.json')['status'] != 'VERIFIED':
        raise RuntimeError('Finish the matched E02 evaluation first')
    value = guard.preflight()
    previous = read(PREVIOUS / 'job.json')
    if value['boot_id'] != previous['boot_id']:
        raise RuntimeError('New boot: stop computation and inspect')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    if sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise ValueError('Official checkpoint changed')
    if EXPERIMENT == 'E04':
        failed = read(ROOT / 'data/experiments/a3_zhy_likelihood_20261012_E03/state.json')
        receipt = read(ROOT / 'data/manifests/a3_zhy_likelihood_20261012_E03.json')
        if (failed['status'] != 'STOPPED' or failed.get('reason') is not None
            or failed.get('returncode') != 1 or receipt['failure'] != 'SNAPSHOT_TENSORDICT_NOT_COPIED'
            or receipt['optimizer_steps_observed'] != 0
            or receipt['health']['boot_ids'] != [value['boot_id']]
            or receipt['health']['critical_kernel_errors']):
            raise RuntimeError('E04 follows only the reviewed E03 snapshot integration error')
    if EXPERIMENT == 'E05':
        prior = read(ROOT / 'data/experiments/a3_zhy_likelihood_20261012_E04/state.json')
        evidence = read(ROOT / 'data/manifests/a3_zhy_likelihood_20261012_E04.json')
        if (prior['status'] != 'VERIFIED' or evidence['probability_contract_passed']
            or not evidence['sampling_identity']['all_inputs_exact']
            or not evidence['precision']['cuda_matmul_tf32']):
            raise RuntimeError('E05 requires the verified E04 numerical inconsistency')
    sources = SOURCES + (('script/a3/zhy_precision_trainer.py',) if EXPERIMENT == 'E05' else ())
    job = dict(experiment=f'A3_ZHY_LIKELIHOOD_20261012_{EXPERIMENT}', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), source_sha256=sha256(OFFICIAL),
        code_sha256={p:sha256(ROOT / p) for p in sources}, num_envs=16, policy_steps=24,
        minibatches=4, seed=0, optimizer_steps_allowed=0, auto_resume=False,
        repeat_allowed=False, runtime_limit_seconds=900, sample_seconds=5,
        max_cpu_c=90, max_gpu_c=85, memory_reserve_gib=4,
        instrumentation='CPU_COPIES_AT_SAMPLING_FORWARD_ALTER_TIMING_DIAGNOSTIC_ONLY',
        e10_ledger_modified=False, new_condition='MIGRATED_NEW_HOST_R11_AND_E02_VERIFIED',
        preceding_diagnostic='E03_SNAPSHOT_TENSORDICT_INTEGRATION_ERROR' if EXPERIMENT == 'E04' else None,
        sampling_and_training_tf32=False if EXPERIMENT == 'E05' else True,
        additional_minibatches=3 if EXPERIMENT == 'E05' else 0,
        gate=dict(actual_initial_ratio_max_abs_deviation=.001, action_mean_rmse_max=1e-5,
                  exact_sampling_input_identity_required=True), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], optimizer_steps_allowed=0)))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read(OUT / 'registration.json'), read(OUT / 'state.json')
        if state['status'] != 'READY' or guard.health()['boot_id'] != job['boot_id']:
            raise RuntimeError('No repeat or reboot recovery')
        if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
            raise RuntimeError('Registered sources changed')
        guard.preflight()
        RUN.mkdir(exist_ok=False)
        environment = os.environ.copy()
        environment.update(ISAAC_PYTHON=str(ROOT / 'data/environments/a3-sonic/bin/python'),
            REPO_DIR=str(VENDOR.resolve()), MOTION_FILE=str(ROOT / 'data/experiments/a3_fullchain_20261010/motionlib/train'),
            EXPERIMENT_DIR=str(RUN), CHECKPOINT=str(OFFICIAL), NUM_ENVS='16',
            NUM_MINI_BATCHES='4', NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(ROOT), WANDB_MODE='disabled', YUANQI_RESUME_LOG_DIR=str(LOG),
            YUANQI_PROBE_OUTPUT=str(OUT))
        trainer = ('script.a3.zhy_precision_trainer.ZHYFullPrecisionTrainer' if EXPERIMENT == 'E05'
                   else 'script.a3.zhy_likelihood_trainer.ZHYRuntimeLikelihoodTrainer')
        argv = ['bash', str(VENDOR / 'train_a3_035_fromscratch.sh'), 'seed=0',
            'trainer._target_=' + trainer,
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
            state.update(status='COMPLETE' if code == 0 and reason is None and (OUT / 'likelihood_diagnostic.json').exists()
                         else 'STOPPED', returncode=code, reason=reason)
            return 0 if state['status'] == 'COMPLETE' else 2
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)


def verify():
    import torch
    from script.a3.likelihood_probe import compare
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE' or guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Diagnosis must finish without a boot change')
    diagnostic = read(OUT / 'likelihood_diagnostic.json')
    if diagnostic['optimizer_steps'] != 0 or diagnostic['parameters_before_sha256'] != diagnostic['parameters_after_sha256']:
        raise ValueError('Probe changed actor parameters')
    if (LOG / 'actual_optimizer_steps.jsonl').exists() and (LOG / 'actual_optimizer_steps.jsonl').stat().st_size:
        raise ValueError('Unexpected optimizer update')
    capture_path = OUT / 'runtime_capture.pt'
    if sha256(capture_path) != diagnostic['capture_sha256']:
        raise ValueError('Saved capture changed')
    capture = torch.load(capture_path, map_location='cpu', weights_only=False)
    source = capture['rollout']
    precision = read(OUT / 'runtime_precision_before.json')
    minibatch_path = OUT / 'actual_first_minibatch.pt'
    if sha256(minibatch_path) != precision['actual_first_minibatch_sha256']:
        raise ValueError('Actual first minibatch changed')
    minibatch = torch.load(minibatch_path, map_location='cpu', weights_only=False)
    for row in diagnostic['rows']:
        if row['label'] == 'actual_first_minibatch':
            indices = minibatch['indices']
            actual = compare(minibatch['mean'], source['old_mu_batch'][indices],
                minibatch['std'], source['old_sigma_batch'][indices],
                source['actions'][indices], source['logprobs'][indices])
        else:
            mean = source['old_mu_batch'] if row['label'] == 'stored_distribution_recomputed' else capture['means'][row['label']]
            actual = compare(mean, source['old_mu_batch'], source['old_sigma_batch'],
                source['old_sigma_batch'], source['actions'], source['logprobs'])
        for key, value in actual.items():
            if not __import__('math').isclose(value, row[key], rel_tol=1e-5, abs_tol=2e-5):
                raise ValueError(f'Independent probability recomputation mismatch: {row["label"]}/{key}')
    sampling = read(OUT / 'sampling_identity.json')
    sample_path = OUT / 'sampled_inputs.pt'
    if sha256(sample_path) != sampling['capture_sha256']:
        raise ValueError('Sampling capture changed')
    from script.a3.zhy_likelihood_trainer import sampling_identity
    independent_sampling = sampling_identity(
        torch.load(sample_path, map_location='cpu', weights_only=False), source)
    if any(independent_sampling[key] != sampling[key] for key in independent_sampling):
        raise ValueError('Independent sampling identity recomputation failed')
    first = next(r for r in diagnostic['rows'] if r['label'] == 'actual_first_minibatch')
    passed = (sampling['all_inputs_exact'] and sampling['sampled_means_match_storage']
              and first['max_abs_ratio_minus_one'] <= job['gate']['actual_initial_ratio_max_abs_deviation']
              and first['action_rmse'] <= job['gate']['action_mean_rmse_max'])
    batch_rows = None
    if EXPERIMENT == 'E05':
        from script.a3.zhy_precision_trainer import validate_initial_batches
        batches = read(OUT / 'initial_batches.json')
        batch_path = OUT / 'initial_batches.pt'
        if sha256(batch_path) != batches['capture_sha256']:
            raise ValueError('Initial minibatch capture changed')
        batch_rows = validate_initial_batches(torch.load(batch_path, map_location='cpu', weights_only=False), source)
        if batch_rows != batches['rows']:
            raise ValueError('CPU initial minibatch recomputation mismatch')
        if (precision['cuda_matmul_tf32'] or precision['cudnn_tf32']
            or any(r['cuda_matmul_tf32'] or r['cudnn_tf32'] for r in precision['sampling_precision'])):
            raise ValueError('Sampling and learning precision were not matched')
        passed = passed and all(r['max_abs_ratio_minus_one'] <= .001
                               and r['action_rmse'] <= 1e-5 for r in batch_rows)
    result = dict(experiment=job['experiment'], status='verified', optimizer_steps=0,
        probability_contract_passed=passed, actual_initial_minibatch=first,
        sampling_identity=sampling, variants=diagnostic['rows'],
        initial_batch_coverage=batch_rows,
        capture_sha256=sha256(capture_path), precision=precision,
        all_probability_variants_independently_recomputed=True,
        sampling_independently_recomputed=True,
        code_sha256=job['code_sha256'], scope='ZERO_UPDATE_DIAGNOSTIC_NOT_POLICY_IMPROVEMENT',
        backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'independent_verification.json', result)
    atomic_json(ROOT / f'data/manifests/a3_zhy_likelihood_20261012_{EXPERIMENT}.json', result)
    state.update(status='VERIFIED', verified_at=time.time(), probability_contract_passed=passed)
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(result, indent=2))


def report():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'STOPPED':
        raise RuntimeError('Use verify for a completed diagnosis')
    raw = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    unique = {}
    for row in raw:
        value = row.get('health', row)
        if 'boot_id' in value:
            unique[value['at_utc']] = value
    samples = list(unique.values())
    steps = LOG / 'actual_optimizer_steps.jsonl'
    console = LOG / 'console.log'
    capture = OUT / 'sampled_inputs.pt'
    error = 'Expected all tensors to be on the same device'
    result = dict(experiment=job['experiment'], status='STOPPED', state=state,
        registered_code_sha256=job['code_sha256'], registration_sha256=sha256(OUT / 'registration.json'),
        console_sha256=sha256(console), health_sha256=sha256(LOG / 'health.jsonl'),
        failure='SNAPSHOT_TENSORDICT_NOT_COPIED' if error in console.read_text() else 'UNCLASSIFIED',
        sampling_capture=dict(sha256=sha256(capture), bytes=capture.stat().st_size,
            validity='INVALID_INPUT_SNAPSHOTS_DO_NOT_INFER_STORAGE_IDENTITY') if capture.exists() else None,
        optimizer_steps_observed=len(steps.read_text().splitlines()) if steps.exists() else 0,
        probability_contract='NOT_EVALUATED', previous_ledger_preserved=True,
        health=dict(samples=len(samples), boot_ids=sorted({v['boot_id'] for v in samples}),
            first_utc=samples[0]['at_utc'], last_utc=samples[-1]['at_utc'],
            cpu_max_c=max(v['temperatures_c']['x86_pkg_temp'] for v in samples),
            gpu_max_c=max(g['temperature.gpu'] for v in samples for g in v['gpus']),
            available_min_bytes=min(v['container_memory']['available_estimate_bytes'] for v in samples),
            oom_counts=sorted({v['container_memory']['oom_kill'] for v in samples}),
            critical_kernel_errors=[e for r in raw for e in r.get('kernel_errors', [])]),
        scope='DIAGNOSTIC_INTEGRATION_FAILURE_NOT_TRAINING_OR_HOST_ROOT_CAUSE', backup_status='LOCAL_ONLY')
    atomic_json(ROOT / f'data/manifests/a3_zhy_likelihood_20261012_{EXPERIMENT}.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'verify', 'report'))
    parser.add_argument('--experiment', choices=('E03', 'E04', 'E05'), default='E03')
    args = parser.parse_args()
    configure(args.experiment)
    result = dict(register=register, run=run, verify=verify, report=report)[args.action]()
    raise SystemExit(result if isinstance(result, int) else 0)
