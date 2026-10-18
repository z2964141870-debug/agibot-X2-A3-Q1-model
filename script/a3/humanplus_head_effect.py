"""HP16: complete first final-head replays from verified tensors and a sealed bundle."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

from script.a3 import humanplus_head_assessment as source
from script.a3 import humanplus_baseline as evaluation_base
from script.a3 import humanplus_runtime as runtime
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_kit_log import verify_log

ROOT, ENV, VENDOR = runtime.ROOT, runtime.ENV, runtime.VENDOR
NAME = 'a3_humanplus_head_effect_20261018_HP16'
OUT, LOG = ROOT / 'data/experiments' / NAME, ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
BUNDLE = OUT / 'checkpoint8'
RUN, OFFICIAL = BUNDLE, source.OFFICIAL
STAGES = ('cloth', 'official', 'effect')


def read(path):
    return json.loads(path.read_text())


def checkpoint(step):
    if step != 8:
        raise ValueError('Only the verified final8 may be replayed')
    return BUNDLE / 'model_step_000008.pt'


def materialize_bundle(directory, model, config, sidecar):
    """Make a new bundle without copying tensors or modifying their source."""
    if directory.exists():
        raise FileExistsError('Bundle cannot be overwritten')
    record = read(sidecar)
    if record['step'] != 8 or record['bytes'] != model.stat().st_size or record['sha256'] != sha256(model):
        raise ValueError('Only exact verified final8 can be bundled')
    directory.mkdir()
    target = directory / model.name
    os.link(model, target)
    shutil.copyfile(config, directory / 'config.yaml')
    shutil.copyfile(sidecar, target.with_suffix('.json'))
    for path in directory.iterdir():
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    from script.a3.checkpoint_store import fsync_directory
    fsync_directory(directory)
    if (target.is_symlink() or sha256(target) != record['sha256']
            or sha256(directory / 'config.yaml') != sha256(config)):
        raise ValueError('Bundle identity mismatch')
    return target


def source_evidence():
    original = source.checked()
    result = read(source.RESULT)
    if (source.NAME != 'a3_humanplus_head_assessment_20261018_HP15'
            or result['status'] != 'STOPPED' or result['current_stage'] != 'cloth'
            or result['reason'] != 'RuntimeError:WORKER_EXIT:1'
            or [r['stage'] for r in result['stages']] != ['verify-training']
            or result['stages'][0]['state']['status'] != 'COMPLETE'
            or result['stages'][0]['state']['returncode'] != 0):
        raise ValueError('Only completed HP15 model verification with pre-replay packaging fault is eligible')
    audit_path = source.PLAN.with_name(source.NAME + '_verify-training_audit.json')
    failure_path = source.PLAN.with_name(source.NAME + '_cloth_audit.json')
    audit, failure = read(audit_path), read(failure_path)
    console = (source.LOG / 'cloth/console.log').read_text()
    verified = read(source.OUT / 'training_verification.json')
    if (not audit['verified'] or audit['plan_sha256'] != sha256(source.PLAN)
            or failure['verified'] or failure['after_issues'] or failure['remaining_own_group_processes']
            or failure['result_sha256'] != sha256(source.RESULT)
            or not console.rstrip().endswith('ValueError: Missing matching checkpoint config.yaml')
            or console.count('Traceback (most recent call last):') != 1
            or (source.OUT / 'cloth/explicit_manifest.json').exists()
            or any((source.OUT / 'cloth').glob('*.metrics.json'))
            or verified['status'] != 'COMPLETE' or verified['scope'] != 'VERIFIED_HEAD_TRAINING_NOT_EFFECT'
            or verified['optimizer_steps'] != 32 or verified['updates'] != 8
            or verified['final_effect_step'] != 8):
        raise ValueError('Exact completed tensor verification and no prior policy replay required')
    inputs = dict(original['input_sha256'])
    for name, row in audit['artifacts'].items():
        path = ROOT / name
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Verified assessment artifact changed: ' + name)
        inputs[name] = row['sha256']
    for path in (source.PLAN, source.RESULT, audit_path, failure_path,
                 source.LOG / 'cloth/console.log', source.OUT / 'training_verification.json'):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    return original, inputs


def register():
    runtime.require_root()
    if any(p.exists() for p in (OUT, LOG, PLAN, RESULT)):
        raise FileExistsError('Final effect is single-use; no reset or resumption')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit reviewed implementation and records before registration')
    original, inputs = source_evidence()
    files = subprocess.check_output(['git', 'ls-files', 'script/a3', 'script/project_tools'], text=True).splitlines()
    code = {n: sha256(ROOT / n) for n in sorted(set(files)) if n.endswith('.py')}
    baseline = guard.health.snapshot(read(source.OUT / 'baseline.json'))
    live = guard.collect(baseline)
    issues = guard.issues(live, startup=True, monitor_pid=os.getpid())
    if issues:
        raise RuntimeError('Final effect registration resources rejected: ' + ';'.join(issues))
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    atomic_json(LOG / 'registration_health.json', live)
    model = source.checkpoint(8)
    target = materialize_bundle(BUNDLE, model, OFFICIAL.with_name('config.yaml'), model.with_suffix('.json'))
    from script.a3.evaluation_common import checkpoint_metadata
    metadata = checkpoint_metadata(target)
    if metadata['sha256'] != original['source_checkpoint8_sha256']:
        raise ValueError('Bundle differs from verified source checkpoint')
    for path in (*BUNDLE.iterdir(), OUT / 'baseline.json'):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    plan = dict(experiment=NAME.upper(), status='REGISTERED', registered_at_utc=runtime.stamp(),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        host='humanplus@100.101.30.68', root=str(ROOT), environment=str(ENV),
        vendor_commit=runtime.PIN, boot_id=baseline['boot_id'], code_sha256=code,
        input_sha256=inputs, input_bytes={n: (ROOT / n).stat().st_size for n in inputs},
        environment_source_sha256=original['environment_source_sha256'],
        source_assessment=source.NAME, source_trial_remains_stopped=True,
        source_model_verification_accepted_before_packaging_failure=True,
        source_checkpoint8_sha256=metadata['sha256'], new_optimizer_steps=0,
        new_training_interactions=0, new_model_verification_runs=0, resume=False,
        main_variable='FINAL8_BUNDLE_CONFIG_PROVENANCE_ONLY_NO_POLICY_OR_EVALUATION_CHANGE',
        checkpoint_bundle=str(BUNDLE.relative_to(ROOT)),
        bundle_config_source=str(OFFICIAL.with_name('config.yaml').relative_to(ROOT)),
        bundle_config_sha256=sha256(BUNDLE / 'config.yaml'),
        bundle_model_is_hardlink_not_symlink=True, bundle_metadata_verified=True,
        motions=original['motions'], protocol=original['protocol'], stages=list(STAGES),
        matched_baseline_contract=original['matched_baseline_contract'],
        budget=dict(runtime_seconds_per_stage=900, policy_effect_steps=8769, effect_rollouts=10,
            training_updates=0, optimizer_steps=0, training_interactions=0, model_verification_runs=0),
        bootstrap_environment=original['bootstrap_environment'], final_effect_step=8,
        early_checkpoint_effect_evaluation_allowed=False, no_automatic_retry=True,
        stopped_must_not_resume=True, backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', experiment=NAME, inputs=len(inputs), code_files=len(code),
                         checkpoint_sha256=metadata['sha256'], config_sha256=plan['bundle_config_sha256'])))



def checked():
    runtime.require_root()
    plan = read(PLAN)
    evaluation_base.clean_vendor()
    for field in ('code_sha256', 'input_sha256', 'environment_source_sha256'):
        for name, expected in plan[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if (sha256(path) != expected or (field == 'input_sha256'
                    and path.stat().st_size != plan['input_bytes'][name])):
                raise ValueError('Independent assessment identity changed: ' + name)
    return plan


def worker(stage):
    plan = checked()
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    state = read(LOG / stage / 'state.json')
    if (state['status'] != 'RUNNING' or state['worker_pid'] != os.getpid()
            or os.getppid() != state['supervisor_pid'] or os.getpgrp() != os.getpid()):
        raise ValueError('Registered assessment group required')
    if stage in ('cloth', 'official'):
        receipt = evaluate(plan, stage)
    else:
        receipt = effect(plan)
    atomic_json(LOG / stage / 'result.json', receipt)
    checked()


def evaluate(plan, stage):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import arguments
    output = OUT / stage
    if output.exists():
        raise FileExistsError('No motion-level resumption permitted')
    output.mkdir()
    references = OUT / (stage + '_references')
    references.mkdir()
    for row in plan['motions']:
        if row['group'] == stage:
            (references / (row['key'] + '.csv')).symlink_to(ROOT / row['path'])
    args = arguments(output, LOG / stage / 'motions', checkpoint(8), references)
    args.frame_stride = 1 if stage == 'cloth' else 4
    original = (evaluation.subprocess, evaluation.preflight, evaluation.monitor_child)
    evaluation.subprocess = evaluation_base.EvaluationLaunch()
    evaluation.preflight = lambda: dict(supervision='HUMANPLUS_PARENT_CONTINUOUS_5S')
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    try:
        code = evaluation.evaluate_explicit(args)
    finally:
        evaluation.subprocess, evaluation.preflight, evaluation.monitor_child = original
    if code:
        raise RuntimeError('Final head replay failed; no retry')
    summary = evaluation.summarize_explicit(output)
    verify_replays(plan, stage, summary)
    return dict(status='COMPLETE', stage=stage, motions=summary['motions'], optimizer_calls=0)


def verify_replays(plan, stage, summary):
    import numpy as np
    selected = [r for r in plan['motions'] if r['group'] == stage]
    if {r['motion'] for r in summary['motions']} != {r['key'] + '.csv' for r in selected}:
        raise ValueError('Incomplete motion set')
    for row in selected:
        output = OUT / stage
        metric = read(output / (row['key'] + '.metrics.json'))['motions'][0]
        trace = read(output / (row['key'] + '.timeseries.json'))['motions'][0]
        n = row['full_policy_steps']
        if metric['num_policy_steps'] != n or trace['policy_tick'] != list(range(n)):
            raise ValueError('Incomplete full replay')
        height, roll, pitch = (np.asarray(trace[k]) for k in ('root_height', 'root_roll_deg', 'root_pitch_deg'))
        falling = np.flatnonzero((height < .45) | (abs(roll) > 60) | (abs(pitch) > 60))
        if metric['fall_tick'] != (int(falling[0]) if len(falling) else None) or metric['fall'] != bool(len(falling)):
            raise ValueError('Independent fall check failed')
        with np.load(output / (row['key'] + '.inputs.npz'), allow_pickle=False) as captured:
            if captured['observations'].shape != (n, 1570) or captured['actions'].shape != (n, 29):
                raise ValueError('Policy input/action coverage mismatch')
            if not np.isfinite(captured['observations']).all():
                raise ValueError('Nonfinite policy inputs')
            np.testing.assert_array_equal(captured['actions'], trace['raw_action_29'])
        with np.load(output / (row['key'] + '.contact.npz'), allow_pickle=False) as contacts:
            if not np.array_equal(contacts['policy_tick'], np.arange(n)):
                raise ValueError('Contact coverage mismatch')
            for key in ('qpos', 'qvel', 'foot_floor_normal_force_n', 'foot_geometry_min_z_m'):
                if contacts[key].shape[0] != n or not np.isfinite(contacts[key]).all():
                    raise ValueError('Invalid contact/full-state capture')


def effect(plan):
    contract = read(ROOT / plan['matched_baseline_contract'])
    rows = []
    for old in contract['motions']:
        # Read the evaluator's persisted summary, including full world-space metrics.
        result = read(RESULT)
        stage = next(r for r in result['stages'] if r['stage'] == old['group'])
        row = next(r for r in stage['receipt']['motions'] if r['motion'] == old['key'] + '.csv')
        before, after = old['windows']['full_rollout'], row['windows']['full_rollout']
        ratios = {k: after[k] / before[k] - 1 for k in
                  ('joint_rmse_rad', 'root_pos_error_m_mean', 'wrists_world_mean_m')}
        gates = dict(no_new_fall=not row['fall'] if not old['fall'] else True,
            joint_regression=ratios['joint_rmse_rad'] <= .02 if old['group'] == 'official' or not old['fall'] else True,
            root_regression=ratios['root_pos_error_m_mean'] <= .05,
            wrist_regression=ratios['wrists_world_mean_m'] <= .05,
            clothing_no_fall=not row['fall'] if old['group'] == 'cloth' else True)
        rows.append(dict(key=old['key'], group=old['group'], fall=row['fall'], before=before,
                         after=after, relative_change=ratios, gates=gates))
    mean = sum(r['after']['joint_rmse_rad'] for r in rows if r['group'] == 'cloth') / 6
    improvement = 1 - mean / contract['baseline_clothing_arithmetic_mean_rmse_rad']
    passed = improvement >= .01 and all(all(r['gates'].values()) for r in rows)
    return dict(status='COMPLETE', matched_development_gate_passed=passed,
        baseline_clothing_arithmetic_mean_rmse_rad=contract['baseline_clothing_arithmetic_mean_rmse_rad'],
        final_clothing_arithmetic_mean_rmse_rad=mean, relative_mean_improvement=improvement,
        outcomes=rows, truly_unseen_final_test=False, deploy_allowed=False)


def run(stage):
    plan = checked()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = read(RESULT) if RESULT.exists() else dict(status='REGISTERED', stages=[], experiment=NAME.upper())
        if (result['status'] not in ('REGISTERED', 'IN_PROGRESS') or
                [r['stage'] for r in result['stages']] != list(STAGES[:STAGES.index(stage)])):
            raise ValueError('No repeat, resume or out-of-order stage')
        if result['stages']:
            previous = result['stages'][-1]['stage']
            if not read(PLAN.with_name(NAME + '_' + previous + '_audit.json'))['verified']:
                raise ValueError('Previous phase must have independent audit before proceeding')
        result.update(status='RUNNING', current_stage=stage, started_at_utc=runtime.stamp())
        atomic_json(RESULT, result)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=str(ROOT), CONDA_PREFIX=str(ENV), PATH=str(ENV / 'bin') + ':' + os.environ['PATH'],
                ACCELERATE_MIXED_PRECISION='no', WANDB_MODE='disabled',
                A3_LOOP_BRIDGE_CACHE=str(ROOT / 'data/experiments/a3_humanplus_baseline_20261018_HP10/solver_bridge'),
                **plan['bootstrap_environment'])
            state = guard.run_stage(LOG / stage, [str(ENV / 'bin/python'),
                '-B', '-u', '-m', 'script.a3.humanplus_head_effect', 'worker', '--stage', stage],
                OUT / 'baseline.json', env=env)
            result['stages'].append(dict(stage=stage, state=state, receipt=read(LOG / stage / 'result.json')))
            result['status'] = 'COMPLETE_MATCHED_EFFECT' if stage == 'effect' else 'IN_PROGRESS'
        except BaseException as exc:
            result.update(status='STOPPED', reason=type(exc).__name__ + ':' + str(exc))
            raise
        finally:
            result.update(updated_at_utc=runtime.stamp(), backup_status='LOCAL_ONLY', github_synced=False)
            atomic_json(RESULT, result)
        print(json.dumps(dict(status=result['status'], stage=stage)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=STAGES, default='cloth')
    args = parser.parse_args()
    {'register': register, 'run': lambda: run(args.stage), 'worker': lambda: worker(args.stage)}[args.action]()


