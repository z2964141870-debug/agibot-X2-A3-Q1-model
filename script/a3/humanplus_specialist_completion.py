"""HP22: bounded intermediate zzx expert, with matched Isaac and MuJoCo checks."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from script.a3 import humanplus_baseline as base
from script.a3 import humanplus_runtime as runtime
from script.a3 import humanplus_recovered_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_kit_log import require_fresh, verify_log
from script.a3.humanplus_specialist_protocol import UPDATES, INTERACTIONS, OPTIMIZER_STEPS, OUTER_CALLS, outcome

ROOT, ENV, VENDOR = runtime.ROOT, runtime.ENV, runtime.VENDOR
NAME = 'a3_humanplus_specialist_completion_20261018_HP22'
MODULE = 'script.a3.humanplus_specialist_completion'
OUT, LOG = ROOT / 'data/experiments' / NAME, ROOT / 'logs' / NAME
SOURCE_NAME = 'a3_humanplus_specialist_repaired_20261018_HP21'
SOURCE_LOG = ROOT / 'logs' / SOURCE_NAME
SOURCE_PLAN = ROOT / 'data/manifests' / (SOURCE_NAME + '_plan.json')
SOURCE_RESULT = ROOT / 'data/manifests' / (SOURCE_NAME + '_result.json')
RUN = ROOT / 'data/training' / SOURCE_NAME / 'R27_001_s0'
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
DESIGN = ROOT / 'data/manifests/a3_humanplus_specialist_completion_design_20261018_D06.json'
CONTRACT = ROOT / 'data/manifests/a3_humanplus_baseline_reference_20261018_HP11_contract.json'
OFFICIAL = base.OFFICIAL
STAGES = ('verify-training', 'isaac-final', 'mujoco', 'effect')
BOOTSTRAP = dict(OMNI_KIT_ACCEPT_EULA='YES', PRIVACY_CONSENT='Y',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    TORCH_NCCL_ASYNC_ERROR_HANDLING='1', TORCH_NCCL_BLOCKING_WAIT='1')


def read(path):
    return json.loads(path.read_text())


def require_source():
    state = read(SOURCE_RESULT)
    audits = [read(SOURCE_PLAN.with_name(SOURCE_NAME + '_' + stage + '_audit.json'))
              for stage in ('isaac-baseline', 'train', 'verify-training')]
    if (state['status'] != 'STOPPED' or state['current_stage'] != 'verify-training'
            or [r['stage'] for r in state['stages']] != ['isaac-baseline', 'train']
            or not all(row['verified'] for row in audits[:2]) or audits[2]['verified']):
        raise ValueError('Only completed HP21 training with pre-load import failure accepted')
    console = (SOURCE_LOG / 'verify-training/console.log').read_text()
    if "ModuleNotFoundError: No module named 'gear_sonic'" not in console:
        raise ValueError('Different source failure cannot be adopted')
    for row in state['stages']:
        if row['state']['status'] != 'COMPLETE' or row['state']['returncode'] != 0:
            raise ValueError('Incomplete source stage')
    return state


def checkpoint(step):
    if step not in (0, UPDATES):
        raise ValueError('No early-point specialist evaluation')
    return RUN / f'milestone_{step}' / f'model_step_{step:06d}.pt'


def native_directory():
    stage = os.environ['YUANQI_SPECIALIST_STAGE']
    if stage not in ('train', 'isaac-baseline', 'isaac-final'):
        raise ValueError('Invalid native stage')
    return LOG / stage


def launch_arguments(stage):
    common = ['++headless=true', '++use_wandb=false', '++seed=0',
        '++manager_env.commands.motion.motion_lib_cfg.motion_file=[' + str(OUT / 'motionlib') + ']',
        '++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy',
        '++manager_env.commands.motion.encoder_sample_probs.g1=0.0']
    if stage == 'train':
        return ['+exp=manager/universal_token/all_modes/sonic_a3_035',
            'checkpoint=' + str(OFFICIAL), '+resume=false', 'experiment_dir=' + str(RUN),
            'num_envs=16', 'algo.config.num_mini_batches=4',
            'algo.config.num_learning_iterations=' + str(UPDATES),
            'algo.trl.report_to=tensorboard', 'exp_var=a3_035_finetune_selected20',
            'trainer._target_=script.a3.humanplus_specialist_trainer.SpecialistTrainer',
            'algo.config.num_learning_epochs=1', 'algo.config.actor_learning_rate=2e-6',
            'algo.config.adaptive_lr_min=1e-6', 'algo.config.adaptive_lr_max=2e-5',
            'callbacks.model_save._target_=script.a3.humanplus_specialist_trainer.SpecialistCheckpointCallback',
            'callbacks.model_save.save_frequency=' + str(UPDATES),
            '+callbacks.model_save.target_global_step=' + str(UPDATES),
            '++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false', *common]
    model = OUT / ('baseline_bundle' if stage == 'isaac-baseline' else 'final_bundle')
    name = OFFICIAL.name if stage == 'isaac-baseline' else checkpoint(UPDATES).name
    return ['checkpoint=' + str(model / name), '++num_envs=1', '++eval_callbacks=[]',
        '++run_eval_loop=true', '++export_onnx_only=false', *common]


def register():
    runtime.require_root()
    if any(p.exists() for p in (PLAN, RESULT, OUT, LOG)):
        raise FileExistsError('HP22 single registration only')
    base.clean_vendor()
    require_source()
    design = read(DESIGN)
    if design['status'] != 'IMPLEMENTED_CONTROLLED_READY_FOR_REGISTRATION':
        raise ValueError('Reviewed implementation and controls required')
    if sha256(CONTRACT) != '3be04e74cee457a8ad3b351347c4f1cc85b0233e6a1dbc4d4101c6105c7a7094':
        raise ValueError('Unchanged matched HP11 reference required')
    for field in ('code_identity', 'input_identity', 'environment_identity'):
        for name, identity in design[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if path.stat().st_size != identity['bytes'] or sha256(path) != identity['sha256']:
                raise ValueError('Reviewed identity changed: ' + name)
    baseline = read(guard.health.OLD)
    guard.health.snapshot(baseline)
    live = guard.collect(baseline)
    issues = guard.issues(live, startup=True, monitor_pid=os.getpid())
    if issues:
        raise RuntimeError('Registration resource rejected: ' + ';'.join(issues))
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    atomic_json(LOG / 'registration_health.json', live)
    workspace = OUT / 'workspace'
    workspace.mkdir()
    (workspace / 'gear_sonic').symlink_to(VENDOR / 'gear_sonic', target_is_directory=True)
    motionlib = OUT / 'motionlib'
    motionlib.mkdir()
    motion = design['training_motion']
    (motionlib / Path(motion['path']).name).symlink_to(ROOT / motion['path'])
    plan = dict(design, experiment=NAME, status='REGISTERED', registered_at_utc=runtime.stamp(),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        design_sha256=sha256(DESIGN), boot_id=baseline['boot_id'],
        baseline_sha256=sha256(OUT / 'baseline.json'), stages=list(STAGES),
        argv={stage: launch_arguments(stage) for stage in ('isaac-final',)},
        budget=dict(new_updates=0, new_interactions=0, new_optimizer_steps=0, source_updates=UPDATES,
                    source_interactions=INTERACTIONS, source_optimizer_steps=OPTIMIZER_STEPS,
                    training_outer_actor_decoder_calls=OUTER_CALLS,
                    isaac_baseline_steps=0, isaac_final_steps=499, mujoco_final_steps=499,
                    seconds_per_stage=900, saved_steps=[0, UPDATES]),
        no_automatic_retry=True, stopped_must_not_resume=True, github_synced=False,
        backup_status='LOCAL_ONLY', deploy_allowed=False)
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', experiment=NAME, budget=plan['budget'])))


def checked():
    runtime.require_root()
    base.clean_vendor()
    require_source()
    plan = read(PLAN)
    if sha256(DESIGN) != plan['design_sha256'] or sha256(OUT / 'baseline.json') != plan['baseline_sha256']:
        raise ValueError('Registered design or baseline changed')
    for field in ('code_identity', 'input_identity', 'environment_identity'):
        for name, identity in plan[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if path.stat().st_size != identity['bytes'] or sha256(path) != identity['sha256']:
                raise ValueError('Registered identity changed: ' + name)
    return plan


def handshake(stage, native=False):
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    state = read(LOG / stage / 'state.json')
    leader = state['worker_pid']
    valid = os.getppid() == leader if native else os.getpid() == leader and os.getppid() == state['supervisor_pid']
    if state['status'] != 'RUNNING' or not valid or os.getpgrp() != leader:
        raise ValueError('Registered process group required')


def bundle(directory, model):
    if directory.exists():
        raise FileExistsError('Bundle cannot be overwritten')
    directory.mkdir()
    target = directory / model.name
    os.link(model, target)
    shutil.copyfile(OFFICIAL.with_name('config.yaml'), directory / 'config.yaml')
    if model.with_suffix('.json').is_file():
        shutil.copyfile(model.with_suffix('.json'), target.with_suffix('.json'))
    if sha256(target) != sha256(model):
        raise ValueError('Bundle identity mismatch')
    return target


def native(stage):
    if stage != 'isaac-final':
        raise ValueError('Completion has no native training or repeated baseline')
    checked()
    handshake(stage, native=True)
    from script.a3.humanplus_specialist_completion_entry import execute
    os.chdir(OUT / 'workspace')
    training = stage == 'train'
    source = VENDOR / ('gear_sonic/train_agent_trl.py' if training else 'gear_sonic/eval_agent_trl.py')
    sys.argv = [str(source), *launch_arguments(stage)]
    execute(source, training)


def broker(stage):
    directory = LOG / stage
    if stage == 'train':
        RUN.mkdir(parents=True, exist_ok=False)
    else:
        bundle(OUT / ('baseline_bundle' if stage == 'isaac-baseline' else 'final_bundle'),
               OFFICIAL if stage == 'isaac-baseline' else checkpoint(UPDATES))
    require_fresh(directory)
    env = dict(os.environ, YUANQI_RESUME_LOG_DIR=str(directory), YUANQI_SPECIALIST_STAGE=stage)
    if stage != 'train':
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    with (directory / 'native_console.log').open('x') as stream:
        child = subprocess.Popen([str(ENV / 'bin/python'), '-B', '-u', '-m', MODULE, 'native',
            '--stage', stage], stdin=subprocess.PIPE, text=True, env=env, stdout=stream,
            stderr=subprocess.STDOUT, cwd=ROOT)
        child.communicate('READY\n')
    atomic_json(directory / 'native_exit.json', dict(returncode=child.returncode, native_pid=child.pid))
    if child.returncode:
        raise RuntimeError('Native specialist phase exit:' + str(child.returncode))
    complete = read(directory / 'native_complete.json')
    closing = read(directory / 'close_begin.json')
    if (complete['native_pid'] != child.pid or closing['native_pid'] != child.pid
            or not complete['cuda_synchronized'] or not closing['callbacks_cleared']):
        raise ValueError('Native completion or close identity mismatch')
    kit = verify_log(directory, child.pid)
    return dict(complete, native_returncode=0, native_shutdown_verified=True, **kit)


def mujoco(plan):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import arguments
    output, references = OUT / 'mujoco', OUT / 'mujoco_references'
    output.mkdir()
    references.mkdir()
    row = plan['baseline_motion']
    (references / (row['key'] + '.csv')).symlink_to(ROOT / row['path'])
    args = arguments(output, LOG / 'mujoco/motions', OUT / 'final_bundle' / checkpoint(UPDATES).name, references)
    original = evaluation.subprocess, evaluation.preflight, evaluation.monitor_child
    evaluation.subprocess = base.EvaluationLaunch()
    evaluation.preflight = lambda: dict(supervision='HUMANPLUS_PARENT_CONTINUOUS_5S')
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    try:
        code = evaluation.evaluate_explicit(args)
    finally:
        evaluation.subprocess, evaluation.preflight, evaluation.monitor_child = original
    if code:
        raise RuntimeError('Complete specialist MuJoCo replay failed')
    summary = evaluation.summarize_explicit(output)
    if len(summary['motions']) != 1 or summary['motions'][0]['num_policy_steps'] != 499:
        raise ValueError('Incomplete specialist full motion')
    return dict(status='COMPLETE', motion=summary['motions'][0], optimizer_calls=0)


def effect(plan):
    result = read(RESULT)
    receipts = {r['stage']: r['receipt'] for r in result['stages']}
    before = read(SOURCE_LOG / 'isaac-baseline/isaac_metrics.json')
    after = receipts['isaac-final']['isaac']
    isaac_pass = not after['fall'] and after['joint_rmse_rad'] <= before['joint_rmse_rad'] * .99
    assessment = outcome(plan['baseline_motion'], receipts['mujoco']['motion'])
    passed = assessment['specialist_mujoco_gate_passed'] and isaac_pass
    return dict(status='COMPLETE', **{k: v for k, v in assessment.items() if k != 'deploy_allowed'},
        isaac_baseline=before, isaac_final=after, isaac_gate_passed=isaac_pass,
        intermediate_expert_gate_passed=passed, deploy_allowed=False,
        next_step='REVIEW_OFF_DISTRIBUTION_CORRECTION_CONTRACT' if passed else 'REVIEW_TARGET_FEASIBILITY_AND_LEARNING_EXECUTION',
        converged=False, unified_distillation_started=False)


def worker(stage):
    plan = checked()
    handshake(stage)
    if stage not in STAGES:
        raise ValueError('No completion training or baseline')
    if stage in ('train', 'isaac-baseline', 'isaac-final'):
        receipt = broker(stage)
    elif stage == 'verify-training':
        from script.a3.humanplus_specialist_completion_verification import verify
        receipt = verify()
    elif stage == 'mujoco':
        receipt = mujoco(plan)
    else:
        receipt = effect(plan)
    atomic_json(LOG / stage / 'result.json', receipt)
    checked()


def run(stage):
    checked()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = read(RESULT) if RESULT.exists() else dict(status='REGISTERED', stages=[], experiment=NAME)
        index = STAGES.index(stage)
        if (result['status'] not in ('REGISTERED', 'IN_PROGRESS') or
                [r['stage'] for r in result['stages']] != list(STAGES[:index])):
            raise ValueError('No repeat, resume or out-of-order stage')
        if index and not read(PLAN.with_name(NAME + '_' + STAGES[index-1] + '_audit.json'))['verified']:
            raise ValueError('Previous independent audit required')
        result.update(status='RUNNING', current_stage=stage, started_at_utc=runtime.stamp())
        atomic_json(RESULT, result)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=str(ROOT), CONDA_PREFIX=str(ENV), PATH=str(ENV / 'bin') + ':' + os.environ['PATH'],
                ACCELERATE_MIXED_PRECISION='no', WANDB_MODE='disabled',
                A3_LOOP_BRIDGE_CACHE=str(ROOT / 'data/experiments/a3_humanplus_baseline_20261018_HP10/solver_bridge'),
                **BOOTSTRAP)
            state = guard.run_stage(LOG / stage, ['/usr/bin/python3' if stage in (
                'train', 'isaac-baseline', 'isaac-final') else str(ENV / 'bin/python'),
                '-B', '-u', '-m', MODULE, 'worker', '--stage', stage], OUT / 'baseline.json', env=env)
            result['stages'].append(dict(stage=stage, state=state, receipt=read(LOG / stage / 'result.json')))
            result['status'] = 'COMPLETE_SPECIALIST_ASSESSMENT' if stage == 'effect' else 'IN_PROGRESS'
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
     'worker': lambda: worker(args.stage), 'native': lambda: native(args.stage)}[args.action]()
