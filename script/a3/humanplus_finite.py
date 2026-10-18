"""HP12: independent humanplus registration of the fixed eight-update recipe."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from script.a3 import humanplus_baseline as evaluation_base
from script.a3 import humanplus_runtime as runtime
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.finite_retention_protocol import LIMITS, SCALES

ROOT, ENV, VENDOR = runtime.ROOT, runtime.ENV, runtime.VENDOR
NAME = 'a3_humanplus_finite_20261018_HP12'
OUT, LOG = ROOT / 'data/experiments' / NAME, ROOT / 'logs' / NAME
RUN = ROOT / 'data/training' / NAME / 'R21_001_s0'
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
REFERENCE = ROOT / 'data/manifests/a3_humanplus_baseline_reference_20261018_HP11'
OLD = ROOT / 'data/experiments/a3_zhy_cloth_failure_20261013_E45/prepare.json'
OLD_CONFIG = ROOT / 'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/config.yaml'
ANCHOR = ROOT / 'data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt'
TARGETS = ROOT / 'data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt'
OFFICIAL = evaluation_base.OFFICIAL
STAGES = ('train', 'verify-training', 'cloth', 'official', 'effect')
BOOTSTRAP = dict(OMNI_KIT_ACCEPT_EULA='YES', PRIVACY_CONSENT='Y',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                TORCH_NCCL_ASYNC_ERROR_HANDLING='1', TORCH_NCCL_BLOCKING_WAIT='1')
SHUTDOWN = ('SimulationApp.close: Stage closed',
            'SimulationApp.close: shutting down app and releasing framework')


def read(path):
    return json.loads(path.read_text())


def checkpoint(step):
    return RUN / ('milestone_' + str(step)) / f'model_step_{step:06d}.pt'


def event(phase, **fields):
    guard.append(LOG / 'train/progress.jsonl', dict(phase=phase, at_utc=runtime.stamp(),
        monotonic=time.monotonic(), pid=os.getpid(), **fields))


def launch_arguments():
    return ['+exp=manager/universal_token/all_modes/sonic_a3_035',
        'checkpoint=' + str(OFFICIAL), '+resume=false', 'experiment_dir=' + str(RUN),
        'headless=true', 'use_wandb=false', 'num_envs=16', 'algo.config.num_mini_batches=4',
        'algo.config.num_learning_iterations=200', 'algo.trl.report_to=tensorboard',
        '++manager_env.commands.motion.motion_lib_cfg.motion_file=[' + str(OUT / 'motionlib') + ']',
        '++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy',
        'exp_var=a3_035_finetune_selected20', 'seed=0',
        'trainer._target_=script.a3.finite_retention_trainer.FiniteRetentionTrainer',
        'algo.config.num_learning_epochs=1', 'algo.config.actor_learning_rate=2e-6',
        'algo.config.adaptive_lr_min=1e-6', 'algo.config.adaptive_lr_max=2e-5',
        'callbacks.model_save._target_=script.a3.finite_retention_trainer.FiniteCheckpointCallback',
        'callbacks.model_save.save_frequency=8', '+callbacks.model_save.target_global_step=8',
        '++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']


def register():
    runtime.require_root()
    if any(p.exists() for p in (OUT, LOG, RUN.parent, PLAN, RESULT)):
        raise FileExistsError('HP12 must never be repeated or reset')
    evaluation_base.clean_vendor()
    old_plan = read(Path(str(REFERENCE) + '_plan.json'))
    contract = read(Path(str(REFERENCE) + '_contract.json'))
    audit = read(Path(str(REFERENCE) + '_official_audit.json'))
    if (contract['status'] != 'VERIFIED_MATCHED_DEVELOPMENT_BASELINE' or not audit['verified']
            or contract['result_sha256'] != sha256(Path(str(REFERENCE) + '_result.json'))
            or contract['plan_sha256'] != sha256(Path(str(REFERENCE) + '_plan.json'))
            or contract['final_audit_sha256'] != sha256(Path(str(REFERENCE) + '_official_audit.json'))):
        raise ValueError('Completed unchanged HP11 baseline required')
    prior = read(OLD)
    from script.a3.cloth_failure_protocol import TRAINING
    if [r['sequence'] for r in prior['motions']] != list(TRAINING):
        raise ValueError('Original four development motions required')
    inputs = dict(old_plan['input_sha256'])
    for row in prior['motions']:
        inputs[row['path']] = row['sha256']
    for path in (OLD, OLD_CONFIG, ANCHOR, TARGETS, OFFICIAL,
                 Path(str(REFERENCE) + '_contract.json'), Path(str(REFERENCE) + '_official_audit.json'),
                 ROOT / 'data/manifests/a3_humanplus_runtime_20261018_HP05_result.json'):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in inputs.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered provenance changed: ' + name)
    # Pin all project Python sources without importing any tensor-dependent modules.
    code_paths = subprocess.check_output(['git', 'ls-files', 'script/a3', 'script/project_tools'], text=True).splitlines()
    code_paths += ['script/a3/humanplus_finite.py', 'script/a3/humanplus_training_entry.py',
                   'script/a3/test_humanplus_training_entry.py',
                   'script/a3/humanplus_finite_verification.py', 'script/project_tools/humanplus_finite_audit.py']
    code = {n: sha256(ROOT / n) for n in sorted(set(code_paths)) if n.endswith('.py')}
    baseline = guard.health.snapshot(read(ROOT / 'data/experiments/a3_humanplus_baseline_reference_20261018_HP11/baseline.json'))
    live = guard.collect(baseline)
    errors = guard.issues(live, startup=True, monitor_pid=os.getpid())
    if errors:
        raise RuntimeError('Registration health rejected: ' + ';'.join(errors))
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    atomic_json(LOG / 'registration_health.json', live)
    inputs[str((OUT / 'baseline.json').relative_to(ROOT))] = sha256(OUT / 'baseline.json')
    workspace = OUT / 'workspace'
    workspace.mkdir()
    (workspace / 'gear_sonic').symlink_to(VENDOR / 'gear_sonic', target_is_directory=True)
    motionlib = OUT / 'motionlib'
    motionlib.mkdir()
    for row in prior['motions']:
        (motionlib / Path(row['path']).name).symlink_to(ROOT / row['path'])
    environment_sources = dict(old_plan['environment_source_sha256'])
    lab = Path('/home/humanplus/x2_teleop_final/third_party/IsaacLab-v2.3.2/source/isaaclab/isaaclab')
    for path in (lab / 'sim/converters/urdf_converter.py', lab / 'sim/converters/asset_converter_base.py'):
        environment_sources[str(path)] = sha256(path)
    plan = dict(experiment=NAME.upper(), status='REGISTERED', registered_at_utc=runtime.stamp(),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        host='humanplus@100.101.30.68', root=str(ROOT), environment=str(ENV),
        vendor_commit=runtime.PIN, boot_id=baseline['boot_id'], code_sha256=code,
        input_sha256=inputs, environment_source_sha256=environment_sources,
        motions=contract['motions'], protocol=contract['protocol'], stages=list(STAGES),
        main_variable='FIXED_FINITE_NATIVE_ADAM_PROPOSAL_ACCEPTANCE_BEFORE_LIVE_COMMIT',
        source_state='FRESH_OFFICIAL_035_ACTOR_CRITIC_EMPTY_ADAM', resume=False,
        training_motions=prior['motions'], development_roles=prior['development_roles'],
        budget=dict(LIMITS, startup_samples=3, interval_seconds=5, startup_available_gib=8,
            runtime_reserve_gib=4, cpu_limit_c=90, gpu_limit_c=85, hot_samples=2,
            checkpoints=[0, 2, 4, 8], maximum_extra_forwards=1728, policy_effect_steps=8769),
        scales=list(SCALES), teacher_rmse_max=.05, teacher_max_abs_max=.5,
        same_batch_objective_tolerance=1e-7, launch_arguments=launch_arguments(),
        bootstrap_environment=BOOTSTRAP,
        native_adapter='THREE_AST_EDITS_ONLY_ENV_STORAGE_KIT_FILE_LOGGING_TERMINAL_SHUTDOWN',
        single_process_accelerator=True, accelerate_mixed_precision='no',
        generated_paths=dict(usd=str(OUT / 'usd'), workspace=str(workspace),
            training=str(RUN), logs=str(LOG), kit_default=str(ENV / 'lib/python3.11/site-packages/isaacsim/kit/logs')),
        matched_baseline_contract=str(Path(str(REFERENCE) + '_contract.json').relative_to(ROOT)),
        final_effect_step=8, early_checkpoint_effect_evaluation_allowed=False,
        no_automatic_retry=True, stopped_must_not_resume=True, backup_status='LOCAL_ONLY',
        scope='DEVELOPMENT_ADAPTATION_ONLY_NOT_UNSEEN_VALIDATION', github_synced=False)
    for name, digest in environment_sources.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Shared environment source changed')
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', experiment=plan['experiment'], budget=plan['budget'])))


def checked():
    runtime.require_root()
    plan = read(PLAN)
    evaluation_base.clean_vendor()
    for field in ('code_sha256', 'input_sha256', 'environment_source_sha256'):
        for name, digest in plan[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256(path) != digest:
                raise ValueError('HP12 registered identity changed: ' + name)
    return plan


def handshake(stage, native=False):
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    state = read(LOG / stage / 'state.json')
    leader = state['worker_pid']
    valid = os.getppid() == leader if native else os.getpid() == leader and os.getppid() == state['supervisor_pid']
    if state['status'] != 'RUNNING' or not valid or os.getpgrp() != leader:
        raise ValueError('Registered process group required')


def native():
    checked()
    handshake('train', native=True)
    from script.a3.humanplus_training_entry import execute
    workspace = OUT / 'workspace'
    os.chdir(workspace)
    sys.argv = [str(VENDOR / 'gear_sonic/train_agent_trl.py'), *launch_arguments()]
    event('vendor_training_begin', cwd=str(workspace))
    execute(VENDOR / 'gear_sonic/train_agent_trl.py')


def broker():
    directory = LOG / 'train'
    RUN.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, YUANQI_RESUME_LOG_DIR=str(directory),
        YUANQI_CLOTH_ANCHOR_ROLLOUT=str(ANCHOR), YUANQI_CLOTH_ANCHOR_OUTPUTS=str(TARGETS))
    with (directory / 'native_console.log').open('x') as stream:
        child = subprocess.run([str(ENV / 'bin/python'), '-B', '-u', '-m',
            'script.a3.humanplus_finite', 'native'], input='READY\n', text=True,
            env=environment, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    atomic_json(directory / 'native_exit.json', dict(returncode=child.returncode,
        at_utc=runtime.stamp(), broker_pid=os.getpid()))
    if child.returncode:
        raise RuntimeError('NATIVE_TRAINING_EXIT:' + str(child.returncode))
    receipt = read(directory / 'native_training.json')
    console = (directory / 'native_console.log').read_text()
    kit_root = ENV / 'lib/python3.11/site-packages/isaacsim/kit/logs'
    paths = [Path(p) for p in re.findall(r'Logging to file: (/[^\s\x1b]+\.log)', console)]
    if len(paths) != 1 or not paths[0].resolve().is_relative_to(kit_root.resolve()):
        raise ValueError('Exactly one authorized native Kit log required')
    kit_text = paths[0].read_text()
    phases = [r['phase'] for r in map(json.loads, (directory / 'progress.jsonl').read_text().splitlines())]
    required = ('training_environment_created', 'training_complete', 'context_clear_returned', 'native_close_begin')
    if (receipt['status'] != 'TRAINING_COMPLETE' or any(phases.count(p) != 1 for p in required)
            or [phases.index(p) for p in required] != sorted(phases.index(p) for p in required)
            or not all(marker in kit_text for marker in SHUTDOWN)):
        raise ValueError('Missing native training or shutdown evidence')
    atomic_json(directory / 'result.json', dict(receipt, status='COMPLETE',
        native_returncode=0, native_shutdown_verified=True, kit_log=str(paths[0]), kit_log_sha256=sha256(paths[0])))


def evaluate(plan, stage):
    evaluation_base.OUT, evaluation_base.LOG = OUT, LOG
    evaluation_base.OFFICIAL = checkpoint(8)
    return evaluation_base.evaluate(plan, stage)


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


def worker(stage):
    plan = checked()
    handshake(stage)
    if stage == 'train':
        broker()
    else:
        if stage == 'verify-training':
            from script.a3.humanplus_finite_verification import verify
            receipt = verify()
        elif stage in ('cloth', 'official'):
            receipt = evaluate(plan, stage)
        else:
            receipt = effect(plan)
        atomic_json(LOG / stage / 'result.json', receipt)
    checked()


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
            state = guard.run_stage(LOG / stage, ['/usr/bin/python3' if stage == 'train' else str(ENV / 'bin/python'),
                '-B', '-u', '-m', 'script.a3.humanplus_finite', 'worker', '--stage', stage],
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
    parser.add_argument('action', choices=('register', 'run', 'worker', 'native'))
    parser.add_argument('--stage', choices=STAGES, default='train')
    args = parser.parse_args()
    {'register': register, 'run': lambda: run(args.stage),
     'worker': lambda: worker(args.stage), 'native': native}[args.action]()
