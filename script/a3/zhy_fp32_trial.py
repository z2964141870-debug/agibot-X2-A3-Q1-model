"""One corrected two-update trial, followed by four matched diagnostics."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_fp32_20261012_E10'
LOG = ROOT / 'logs/a3_zhy_fp32_20261012_E10'
TRAINING = ROOT / 'data/training/a3_zhy_fp32_20261012_E10'
RUN = TRAINING / 'R12_001_s0'
RUN_LOG = LOG / RUN.name
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
PRIOR = ROOT / 'data/experiments/a3_zhy_consistency_20261012_E09'
BASELINE = ROOT / 'data/experiments/a3_zhy_search_20261012_E02'
TRAINER = 'script.a3.consistent_precision.ConsistentPrecisionTrainer'
OVERRIDES = ['++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']
SOURCES = ('script/a3/zhy_fp32_trial.py', 'script/a3/consistent_precision.py',
    'script/a3/verified_finetune.py', 'script/a3/finetune_trainer.py',
    'script/a3/resumable_sonic.py', 'script/a3/checkpoint_store.py',
    'script/a3/corrected_trial.py', 'script/a3/zhy_trial.py',
    'script/a3/evaluate_mujoco.py', 'script/a3/verify_zhy_search.py', 'script/a3/sim_entry.py')


def read(path):
    return json.loads(path.read_text())


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Boot changed; no recovery or repeat')
    if any(sha256(ROOT / p) != h for p,h in job['code_sha256'].items()):
        raise ValueError('Registered code changed')
    if sha256(OFFICIAL) != job['official_sha256']:
        raise ValueError('Official checkpoint changed')
    if any(sha256(Path(p)) != h for p,h in job['baseline_sha256'].items()):
        raise ValueError('Verified baseline evidence changed')


def register():
    if OUT.exists() or TRAINING.exists():
        raise FileExistsError('No repeat or reset of E10/R12')
    previous, value = read(PRIOR / 'independent_verification.json'), guard.preflight()
    if (not previous['probability_contract_passed'] or previous['status'] != 'VERIFIED'
        or previous['health']['boot_ids'] != [value['boot_id']]):
        raise RuntimeError('Requires verified same-boot E09 precision fix')
    old, old_job = read(BASELINE / 'independent_verification.json'), read(BASELINE / 'job.json')
    if old['status'] != 'passed' or old_job['boot_id'] != value['boot_id']:
        raise ValueError('Requires verified same-boot official baseline')
    diagnostic = old_job['diagnostic_motions']
    names = {Path(row['path']).stem for row in diagnostic}
    verified = [r for r in old['metric_rows'] if r['condition'] == 'official_oracle' and r['motion'] in names]
    if len(diagnostic) != 4 or len(verified) != 4 or sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise ValueError('Diagnostic coverage or official identity failed')
    files = [BASELINE / 'independent_verification.json', BASELINE / 'official_oracle/explicit_summary.json']
    for row in verified:
        for suffix,key in (('metrics.json', 'metrics_sha256'), ('timeseries.json', 'timeseries_sha256')):
            path = BASELINE / 'official_oracle' / f"{row['motion']}.{suffix}"
            if sha256(path) != row[key]:
                raise ValueError('Verified baseline trace changed')
            files.append(path)
    for row in diagnostic:
        if sha256(Path(row['path'])) != row['sha256']:
            raise ValueError('Diagnostic motion changed')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unreadable')
    job = dict(experiment='A3_ZHY_FP32_20261012_E10', trial='R12', registered_at=time.time(),
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), official_sha256=sha256(OFFICIAL),
        code_sha256={p:sha256(ROOT / p) for p in SOURCES},
        baseline_sha256={str(p):sha256(p) for p in files}, diagnostic_motions=diagnostic,
        prior_verification_sha256=sha256(PRIOR / 'independent_verification.json'),
        trainer=TRAINER, updates=2, num_envs=16, minibatches=4, epochs=1, lr_scale=.1, seed=0,
        precision_overrides=OVERRIDES, max_cpu_c=90, max_gpu_c=85,
        consecutive_hot_samples=2, sample_seconds=5, memory_reserve_gib=4,
        runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        evaluation=dict(motions=4, reference_fps=30., frame_stride=4, max_policy_steps=None,
            action_delay_ms=0., reference_buffer_ms=0., gate=dict(no_new_falls=True,
                each_motion_joint_no_worse=True, mean_joint_improvement_min_percent=1.,
                root_body_and_action_regression_max_percent=5.),
            baseline_reused=True, fresh_final_test=False), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    references = OUT / 'references'
    references.mkdir()
    for row in diagnostic:
        (references / Path(row['path']).name).symlink_to(os.path.relpath(row['path'], references))
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'launch_state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], updates=2)))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'launch_state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery')
        unchanged(job)
        guard.preflight()
        if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 20 * 1024**3:
            raise RuntimeError('Less than 20GiB free')
        RUN.mkdir(parents=True, exist_ok=False)
        RUN_LOG.mkdir(parents=True, exist_ok=False)
        environment = dict(os.environ, ISAAC_PYTHON=str(ROOT / 'data/environments/a3-sonic/bin/python'),
            REPO_DIR=str((ROOT / 'script/vendor/sonic_for_a3').resolve()),
            MOTION_FILE=str(ROOT / 'data/experiments/a3_fullchain_20261010/motionlib/train'),
            EXPERIMENT_DIR=str(RUN), CHECKPOINT=str(OFFICIAL), NUM_ENVS='16', NUM_MINI_BATCHES='4',
            NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(ROOT),
            WANDB_MODE='disabled', ACCELERATE_MIXED_PRECISION='no', YUANQI_RESUME_LOG_DIR=str(RUN_LOG))
        argv = ['bash', str(ROOT / 'script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh'),
            'seed=0', 'trainer._target_=' + TRAINER, 'algo.config.num_learning_epochs=1',
            'algo.config.actor_learning_rate=2e-6', 'algo.config.adaptive_lr_min=1e-6',
            'algo.config.adaptive_lr_max=2e-5',
            'callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback',
            'callbacks.model_save.save_frequency=1', '+callbacks.model_save.target_global_step=2', *OVERRIDES]
        attempt = dict(start_step=0, run_dir=str(RUN), logs=str(RUN_LOG),
                       boot_id=job['boot_id'], started_at=time.time(), status='running', argv=argv)
        state.update(status='RUNNING', started_at=attempt['started_at'])
        atomic_json(OUT / 'launch_state.json', state)
        atomic_json(OUT / 'state.json', dict(status='running', attempts=[attempt]))
        guard.OUT, guard.LOG = OUT, LOG
        try:
            with (RUN_LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, RUN_LOG, job)
            from script.a3.checkpoint_store import select_checkpoint
            step, checkpoint = 0, None
            if code == 0 and reason is None:
                checkpoint, payload, _ = select_checkpoint(RUN)
                step = int(payload['state'].global_step)
            passed = code == 0 and reason is None and step == 2
            state.update(status='COMPLETE' if passed else 'STOPPED', returncode=code, reason=reason,
                         verified_step=step, checkpoint=str(checkpoint) if checkpoint else None)
            attempt.update(status='finished', returncode=code, reason=reason, finished_at=time.time())
            atomic_json(OUT / 'state.json', dict(status='complete' if passed else 'blocked',
                verified_step=step, checkpoint=state['checkpoint'], attempts=[attempt]))
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'launch_state.json', state)
        return 0 if state['status'] == 'COMPLETE' else 2


def health_summary():
    raw = [json.loads(line) for line in (LOG / 'health.jsonl').read_text().splitlines()]
    unique = {}
    for row in raw:
        value = row.get('health', row)
        if 'boot_id' in value:
            unique[value['at_utc']] = value
    samples = list(unique.values())
    times = [datetime.fromisoformat(r['at_utc']).timestamp() for r in samples]
    return dict(samples=len(samples), first_utc=samples[0]['at_utc'], last_utc=samples[-1]['at_utc'],
        max_sample_gap_seconds=max((b-a for a,b in zip(times,times[1:])), default=0),
        boot_ids=sorted({v['boot_id'] for v in samples}),
        cpu_max_c=max(v['temperatures_c']['x86_pkg_temp'] for v in samples),
        gpu_max_c=max(g['temperature.gpu'] for v in samples for g in v['gpus']),
        gpu_power_max_w=max(g['power.draw'] for v in samples for g in v['gpus']),
        available_min_bytes=min(v['container_memory']['available_estimate_bytes'] for v in samples),
        oom_counts=sorted({v['container_memory']['oom_kill'] for v in samples}),
        critical_kernel_errors=[e for row in raw for e in row.get('kernel_errors', [])],
        telemetry_errors=[row for row in raw if 'error' in row])


def verify():
    import yaml
    from script.a3 import corrected_trial as existing
    state, job = read(OUT / 'launch_state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Training has not completed')
    unchanged(job)
    existing.OUTPUT, existing.LOGS, existing.TRAINING, existing.TRIAL = OUT, LOG, TRAINING, 'R12'
    existing.EPOCHS, existing.LR_SCALE = 1, .1
    existing.configuration = lambda: dict(trainer=TRAINER)
    verification = existing.verify()
    config = yaml.safe_load((RUN / 'config.yaml').read_text())
    if any(config['algo']['trl'][key] is not False for key in ('bf16', 'fp16', 'tf32')):
        raise ValueError('Live training configuration differs from FP32 contract')
    precision = read(RUN_LOG / 'precision_contract.json')
    if (precision['actor_calls'] != dict(rollout=48, training=8)
        or sorted(r['training'] for r in precision['first_call_per_mode']) != [False, True]
        or any(r['mixed_precision'] != 'no' or any(v for k,v in r.items()
               if k not in ('training', 'mixed_precision')) for r in precision['first_call_per_mode'])):
        raise ValueError('Live Actor precision or call count differs from contract')
    if len(verification['actual_optimizer_steps']) != 8:
        raise ValueError('Two-update trial must execute exactly eight optimizer steps')
    result = dict(status='TRAINING_VERIFIED', experiment=job['experiment'],
        checkpoints=verification['checkpoints'], actual_optimizer_steps=verification['actual_optimizer_steps'],
        auxiliary_tags=verification['auxiliary_tags'], precision=precision, health=health_summary(),
        source_sha256=job['official_sha256'], code_sha256=job['code_sha256'], state=dict(state),
        independent_cpu_reload=True, policy_effect='NOT_EVALUATED', backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'independent_verification.json', result)
    atomic_json(ROOT / 'data/manifests/a3_zhy_fp32_20261012_E10.json', result)
    state.update(status='TRAINING_VERIFIED', verified_at=time.time())
    atomic_json(OUT / 'launch_state.json', state)
    print(json.dumps(dict(status=result['status'], optimizer_steps=8,
                         precision=precision, health=result['health']), indent=2))


def evaluate():
    from script.a3 import evaluate_mujoco as evaluation
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'launch_state.json'), read(OUT / 'registration.json')
        if state['status'] != 'TRAINING_VERIFIED':
            raise RuntimeError('Requires verified training; no repeat of evaluation')
        unchanged(job)
        guard.preflight()
        evaluation.preflight = guard.preflight
        guard.OUT, guard.LOG = OUT, LOG
        evaluation.monitor_child = guard.guarded_monitor
        state.update(status='EVALUATING', evaluation_started_at=time.time())
        atomic_json(OUT / 'launch_state.json', state)
        args = SimpleNamespace(output=OUT / 'evaluation', logs=LOG / 'evaluation',
            checkpoint=RUN / 'model_step_000002.pt', motion=OUT / 'references',
            reference_fps=30., frame_stride=4, max_policy_steps=None, action_delay_ms=0.,
            reference_buffer_ms=0., capture_inputs=True, capture_count=100000,
            summarize_only=False, joint_reference_mode=None, joint_predictor=None,
            joint_noise_std=0., joint_noise_seed=0, causal_orientation=False)
        try:
            code = evaluation.evaluate_explicit(args)
            state.update(status='EVALUATION_COMPLETE' if code == 0 else 'STOPPED', evaluation_returncode=code)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['evaluation_ended_at'] = time.time()
            atomic_json(OUT / 'launch_state.json', state)
        return 0 if state['status'] == 'EVALUATION_COMPLETE' else 2


def verify_evaluation():
    import numpy as np
    from script.a3.verify_zhy_search import verify_metrics
    state, job = read(OUT / 'launch_state.json'), read(OUT / 'registration.json')
    if state['status'] != 'EVALUATION_COMPLETE':
        raise RuntimeError('Evaluation incomplete')
    unchanged(job)
    records = verify_metrics(OUT / 'evaluation')
    new, old = read(OUT / 'evaluation/explicit_summary.json'), read(BASELINE / 'official_oracle/explicit_summary.json')
    for key in ('reference_fps_after_stride', 'frame_stride', 'max_policy_steps',
                'action_delay_ms', 'encoder', 'vendor_commit', 'capture_count'):
        if new['protocol'].get(key) != old['protocol'].get(key):
            raise ValueError('Evaluation differs from matched baseline')
    if len(records) != 4 or len(new['motions']) != 4:
        raise ValueError('Expected four full diagnostic motions')
    baseline = {Path(r['motion']).stem:r for r in old['motions']}
    rows, root_body_pass = [], True
    for row in new['motions']:
        name = Path(row['motion']).stem
        source = next(r for r in job['diagnostic_motions'] if Path(r['path']).stem == name)
        before = baseline[name]
        n, o = row['windows']['full_rollout'], before['windows']['full_rollout']
        if not row['full_replay_completed'] or n['policy_steps'] != o['policy_steps']:
            raise ValueError('Incomplete or mismatched evaluation duration')
        changes = {k:(n[k]/o[k]-1)*100 for k in n if k != 'policy_steps' and o[k] not in (None,0)}
        root_body_pass &= all(v <= 5. for k,v in changes.items() if k != 'joint_rmse_rad')
        deltas = []
        for directory in (BASELINE / 'official_oracle', OUT / 'evaluation'):
            trace = read(directory / f'{name}.timeseries.json')['motions'][0]
            action = np.asarray(trace['raw_action_29'])
            deltas.append(float(np.sqrt(np.mean(np.diff(action, axis=0)**2))))
        delta_change = (deltas[1]/deltas[0]-1)*100
        root_body_pass &= delta_change <= 5.
        rows.append(dict(motion=name, source_sha256=source['sha256'],
            baseline_joint_rmse=o['joint_rmse_rad'], corrected_joint_rmse=n['joint_rmse_rad'],
            changes_percent=changes, action_delta_rms_change_percent=delta_change,
            baseline_fall=before['fall'], corrected_fall=row['fall'], policy_steps=n['policy_steps']))
    old_mean = sum(r['baseline_joint_rmse'] for r in rows)/4
    new_mean = sum(r['corrected_joint_rmse'] for r in rows)/4
    improvement = (1-new_mean/old_mean)*100
    passed = (improvement >= 1. and root_body_pass
        and all(r['corrected_joint_rmse'] <= r['baseline_joint_rmse'] for r in rows)
        and not any(r['corrected_fall'] and not r['baseline_fall'] for r in rows))
    result = read(OUT / 'independent_verification.json')
    result.update(status='VERIFIED', policy_effect=dict(diagnostic_gate_passed=passed,
        mean_joint_improvement_percent=improvement, mean_joint_baseline_rad=old_mean,
        mean_joint_corrected_rad=new_mean, rows=rows, baseline_reused=True,
        fresh_final_test=False, independent_metrics=records,
        scope='FOUR_REUSED_ORACLE_DIAGNOSTICS_NOT_TELEOP_OR_ROBOT_ACCEPTANCE'),
        health=health_summary(), evaluation_summary_sha256=sha256(OUT / 'evaluation/explicit_summary.json'))
    atomic_json(OUT / 'independent_verification.json', result)
    atomic_json(ROOT / 'data/manifests/a3_zhy_fp32_20261012_E10.json', result)
    state.update(status='VERIFIED', evaluation_verified_at=time.time(), diagnostic_gate_passed=passed)
    atomic_json(OUT / 'launch_state.json', state)
    print(json.dumps(dict(status=result['status'], policy_effect=result['policy_effect'], health=result['health']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'verify', 'evaluate', 'verify_evaluation'))
    value = globals()[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value,int) else 0)
