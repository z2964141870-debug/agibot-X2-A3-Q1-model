"""E45/R19: add one known failing development motion, keeping the E44 recipe."""

import argparse
import io
import json
import os
import subprocess
import time
import unittest

from script.a3 import cloth_cached_trial as cached
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.cloth_failure_protocol import (CHECKPOINTS, UPDATES, OPTIMIZER_STEPS, TRANSITIONS,
    outcome, ADDED_SEQUENCE, BASE_TRAINING, TRAINING, corpus_roles)

runtime, shared = cached.runtime, cached.shared
AUDIT = shared.ROOT / 'data/experiments/a3_zhy_cloth_effect_audit_20261013_E43'
PRIOR = shared.ROOT / 'data/experiments/a3_zhy_cloth_cached_20261013_E42'
OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_failure_20261013_E45'
shared.OUT = OUT
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_failure_20261013_E45'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_failure_20261013_E45.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_FAILURE_20261013_E45'
shared.TRIAL = 'R19'
shared.MODULE = 'script.a3.cloth_failure_trial'
shared.TRAINER = 'script.a3.cloth_curve_trainer.ClothCurveTrainer'
shared.CODE += ('script/a3/cloth_learning_curve.py', 'script/a3/cloth_curve_protocol.py',
                'script/a3/cloth_curve_trainer.py', 'script/a3/test_cloth_curve_protocol.py',
                'script/a3/test_cloth_curve_runtime.py')
shared.CODE += ('script/a3/cloth_failure_trial.py', 'script/a3/cloth_failure_protocol.py',
                'script/a3/cloth_failure_callback.py', 'script/a3/test_cloth_failure_protocol.py',
                'script/a3/test_cloth_failure_runtime.py')
STOPPED = shared.ROOT / 'data/experiments/a3_zhy_cloth_curve_20261013_E44'
STOP_RECEIPT = shared.ROOT / 'data/manifests/a3_zhy_cloth_curve_stop_20261013_E44.json'
PRIOR_RUN = shared.ROOT / 'data/training/a3_zhy_cloth_curve_20261013_E44/R18_001_s0'
EFFECT_VERIFY = shared.verify_evaluation


def register():
    if OUT.exists() or shared.RUN.parent.exists():
        raise FileExistsError('E45 must not be reset or repeated')
    stopped = shared.read(STOP_RECEIPT)
    if (shared.read(STOPPED / 'state.json')['status'] != 'STOPPED' or
            stopped['status'] != 'STOPPED_ORIGINAL_TEACHER_DRIFT_PROTECTION'):
        raise ValueError('Expected archived E44 quality stop; no automatic recovery')
    roles = corpus_roles(shared.read(shared.CORPUS / 'independent_verification.json')['sequences'])
    audit = shared.read(AUDIT / 'training_verification.json')
    if (shared.read(AUDIT / 'state.json')['status'] != 'COMPLETE' or
            audit['status'] != 'VERIFIED_SAVED_MATCHED_EFFECT_DECOMPOSITION' or
            not audit['matched_initial_inputs_and_reference_verified']):
        raise ValueError('Requires completed E43 comparison contract before registration')
    prior = shared.read(PRIOR / 'effect_verification.json')
    if prior['development_gate_passed']:
        raise ValueError('Expected rejected E42 result; reassess if history changed')
    cached.register()  # Includes the unchanged three snapshots and 8 GiB entry gate.
    job = shared.read(OUT / 'registration.json')
    job.update(parent_experiment='A3_ZHY_CLOTH_CURVE_20261013_E44', finalized_at=time.time(),
        main_variable='TRAINING_CORPUS_3_TO_4_KNOWN_FAILURE_DEVELOPMENT_ADAPTATION',
        training_sequences=list(TRAINING), development_roles=roles,
        added_sequence=ADDED_SEQUENCE, original_partition_files_unchanged=True,
        unseen_test_claim=False, reset_sampling_change=False,
        converted_reference_change=False, previous_three_motionlibs_reused=True,
        updates=UPDATES, optimizer_steps=OPTIMIZER_STEPS, sampled_transitions=TRANSITIONS,
        effect_checkpoints=list(CHECKPOINTS), evaluate_every_declared_checkpoint=True,
        runtime_controls_required=2,
        encoder_route_change=False, reward_change=False, anchor_constraint_change=False,
        selection_rule='LOWEST_MEAN_CLOTHING_RMSE_AMONG_WHOLE_CHECKPOINTS_PASSING_ALL_ORIGINAL_GATES',
        no_passing_checkpoint_action='STOP_THIS_CORPUS_BRANCH_NO_AUTOMATIC_EXTENSION',
        new_independent_seed_and_unseen_sessions_required=True)
    files = [STOPPED / 'registration.json', STOPPED / 'state.json', STOPPED / 'prepare.json',
             STOP_RECEIPT, PRIOR_RUN / 'config.yaml',
             shared.ROOT / 'data/manifests/a3_outcome_research_20261013.json',
             AUDIT / 'training_verification.json', AUDIT / 'state.json',
             PRIOR / 'effect_verification.json', PRIOR / 'registration.json',
             shared.ROOT / 'data/training/a3_zhy_cloth_cached_20261013_E42/R17_001_s0/config.yaml',
             shared.ROOT / 'script/vendor/sonic_for_a3/gear_sonic/envs/manager_env/mdp/commands.py']
    job['input_sha256'].update({str(p.relative_to(shared.ROOT)): sha256(p) for p in files})
    atomic_json(OUT / 'registration.json', job)


def prepare(job):
    shared.phase(job, 'prepare/before_imports')
    from script.a3 import test_cloth_failure_runtime as tests
    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import load_a3_flat_csv, convert_sequence, A3_29D_SPEC
    from script.a3.reference_contract import load_sim
    from scipy.spatial.transform import Rotation
    import joblib
    import numpy as np
    shared.phase(job, 'prepare/after_imports')
    roles = corpus_roles(shared.read(shared.CORPUS / 'independent_verification.json')['sequences'])
    if roles != job['development_roles'] or job['training_sequences'] != list(TRAINING):
        raise ValueError('Registered development roles or training corpus changed')
    prior = shared.read(STOPPED / 'prepare.json')
    if (not prior['tests_passed'] or prior['curve_runtime_controls'] != 2 or
            [r['sequence'] for r in prior['motions']] != list(BASE_TRAINING)):
        raise ValueError('The prior validated three-motion preparation is required')
    rows = []
    for row in prior['motions']:
        source = shared.ROOT / row['path']
        if sha256(source) != row['sha256']:
            raise ValueError('Previously accepted MotionLib changed')
        (OUT / 'motionlib' / source.name).symlink_to(os.path.relpath(source, OUT / 'motionlib'))
        rows.append(dict(row, original_partition='train', experiment_role='DEVELOPMENT_TRAIN', reused=True))
    stream = io.StringIO()
    checked = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not checked.wasSuccessful() or checked.testsRun != job['runtime_controls_required']:
        raise ValueError('Early checkpoint runtime controls failed: ' + stream.getvalue())
    shared.phase(job, 'prepare/after_runtime_controls')
    sim = load_sim()
    name = ADDED_SEQUENCE.replace('/', '__')
    path = shared.CORPUS / 'sequences' / name / 'reference.csv'
    entry = convert_sequence(load_a3_flat_csv(str(path), robot='a3_29'), fps=30)
    root, quat, joints, count = sim.load_a3_flat_csv(path, source_fps=30, frame_stride=1)
    expected_names = [sim.A3_CSV_JOINT_NAMES[i] for i in sim.A3_POLICY_TO_SDK_IDX]
    output_names = [A3_29D_SPEC['joint_names'][i] for i in A3_29D_SPEC['dof_output_indices']]
    remapped = entry['dof'][:, [output_names.index(n) for n in expected_names]]
    joint_error = float(np.max(np.abs(remapped - joints)))
    root_error = float(np.max(np.abs(entry['root_trans_offset'] - root)))
    angle_error = float((Rotation.from_quat(entry['root_rot']) *
        Rotation.from_quat(quat[:, [1, 2, 3, 0]]).inv()).magnitude().max())
    errors = (joint_error, root_error, angle_error)
    if (entry['dof'].shape != (300, 29) or count != 300 or
            not all(np.isfinite(v) and v <= 1e-6 for v in errors)):
        raise ValueError('New known-failure MotionLib ABI rejected')
    target = OUT / 'motionlib' / (name + '.pkl')
    joblib.dump({name: entry}, target, compress=True)
    restored = joblib.load(target)[name]
    if any(not np.array_equal(entry[k], restored[k]) for k in
            ('dof', 'pose_aa', 'root_rot', 'root_trans_offset')):
        raise ValueError('New MotionLib reload changed values')
    role = next(r for r in roles if r['sequence'] == ADDED_SEQUENCE)
    rows.append(dict(sequence=ADDED_SEQUENCE, original_partition=role['original_partition'],
        experiment_role='DEVELOPMENT_TRAIN', reused=False, frames=count, fps=30,
        joint_max_error=joint_error, root_max_error_m=root_error, rotation_max_error_rad=angle_error,
        path=str(target.relative_to(shared.ROOT)), sha256=sha256(target), bytes=target.stat().st_size))
    if ({p.name for p in (OUT / 'motionlib').iterdir()} !=
            {r['sequence'].replace('/', '__') + '.pkl' for r in rows}):
        raise ValueError('Training directory must contain exactly four declared motions')
    atomic_json(OUT / 'prepare.json', dict(motions=rows, development_roles=roles,
        previous_three_abi_receipt_sha256=sha256(STOPPED / 'prepare.json'),
        original_three_reused=True, added_motion_abi_verified=True,
        original_partition_files_unchanged=True, unseen_final_test=False,
        dynamics_acceptance=False, curve_runtime_controls=checked.testsRun,
        curve_runtime_output=stream.getvalue(), std_boundary_reused=prior['std_boundary']))
    shared.phase(job, 'prepare/complete')


def train(job):
    if shared.read(OUT / 'prepare.json').get('curve_runtime_controls') != 2:
        raise ValueError('Curve runtime controls have not passed')
    for row in shared.read(OUT / 'prepare.json')['motions']:
        if sha256(shared.ROOT / row['path']) != row['sha256']:
            raise ValueError('Prepared motion identity changed')
    shared.RUN.mkdir(parents=True, exist_ok=False)
    directory = shared.LOG / 'training'
    directory.mkdir(exist_ok=False)
    environment = dict(os.environ, ISAAC_PYTHON=str(shared.ROOT / 'data/environments/a3-sonic/bin/python'),
        REPO_DIR=str((shared.ROOT / 'script/vendor/sonic_for_a3').resolve()),
        MOTION_FILE=str(OUT / 'motionlib'), EXPERIMENT_DIR=str(shared.RUN),
        CHECKPOINT=str(shared.OFFICIAL), NUM_ENVS='16', NUM_MINI_BATCHES='4',
        NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(shared.ROOT),
        WANDB_MODE='disabled', ACCELERATE_MIXED_PRECISION='no',
        YUANQI_RESUME_LOG_DIR=str(directory), YUANQI_CLOTH_ANCHOR_ROLLOUT=str(shared.ANCHOR),
        YUANQI_CLOTH_ANCHOR_OUTPUTS=str(shared.TARGETS))
    argv = ['bash', str(shared.ROOT / 'script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh'),
        'seed=0', 'trainer._target_=' + shared.TRAINER, 'algo.config.num_learning_epochs=1',
        'algo.config.actor_learning_rate=2e-6', 'algo.config.adaptive_lr_min=1e-6',
        'algo.config.adaptive_lr_max=2e-5',
        'callbacks.model_save._target_=script.a3.cloth_failure_callback.FailureCheckpointCallback',
        'callbacks.model_save.save_frequency=8', '+callbacks.model_save.target_global_step=32',
        '++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']
    atomic_json(OUT / 'training_launch.json', dict(argv=argv, motion_file=str(OUT / 'motionlib')))
    with (directory / 'console.log').open('w') as stream:
        code = subprocess.run(argv, cwd=shared.ROOT, env=environment, stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError('Learning curve stopped; no restart, retry or relaxed protection')


def checkpoint(step):
    return shared.RUN / ('milestone_' + str(step)) / f'model_step_{step:06d}.pt'


def verify_training(job):
    import torch
    import yaml
    from script.a3.checkpoint_store import validate_payload
    from script.a3.cloth_constrained_trainer import DECODER, initial_probability
    from script.a3.cloth_std_contract import canonical_policy_state
    torch.set_num_threads(1)
    directory = shared.LOG / 'training'
    source = torch.load(shared.OFFICIAL, map_location='cpu', weights_only=False)
    official = canonical_policy_state(source['policy_state_dict'])
    records, changed = [], []
    for step in (0,) + CHECKPOINTS:
        path = checkpoint(step)
        record = shared.read(path.with_suffix('.json'))
        if record['sha256'] != sha256(path) or record['bytes'] != path.stat().st_size:
            raise ValueError('Checkpoint identity changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != step:
            raise ValueError('Checkpoint step differs from the declared curve')
        actor = payload['policy_state_dict']
        if actor.keys() != official.keys():
            raise ValueError('Actor keys changed')
        names = [n for n, v in actor.items() if not torch.equal(v, official[n])]
        if step == 0:
            critic = payload['value_state_dict']
            expected = source['value_state_dict']
            if (names or payload['optimizer_state_dict']['state'] or critic.keys() != expected.keys()
                    or any(not torch.equal(v, expected[n]) for n, v in critic.items())):
                raise ValueError('Initial Actor/Critic/optimizer is not the official warm start')
        else:
            if not names or any(not n.startswith(DECODER) for n in names):
                raise ValueError('Update escaped the frozen-codec dynamics decoder')
            steps = {int(v['step']) for v in payload['optimizer_state_dict']['state'].values() if 'step' in v}
            if steps != {step * 4}:
                raise ValueError('Saved optimizer counter does not match the curve')
        records.append(dict(record, path=str(path.relative_to(shared.ROOT))))
        changed.append(dict(step=step, changed_actor_tensors=names))
        del payload, actor
    contract = shared.read(directory / 'cloth_contract.json')
    batches = shared.read(directory / 'sampled_token_contract.json')['batches']
    audit = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    if (len(batches) != OPTIMIZER_STEPS or len(audit) != OPTIMIZER_STEPS or
            len(contract['optimizer_steps']) != OPTIMIZER_STEPS or
            [r['attempt_optimizer_step'] for r in audit] != list(range(1, OPTIMIZER_STEPS + 1)) or
            any(not r['decoder_gradient_enabled'] for r in batches) or
            any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5
                for r in contract['optimizer_steps'])):
        raise ValueError('Actual updates, cached gradients or teacher retention budget failed')
    repair = torch.load(directory / 'initial_token_repair.pt', map_location='cpu', weights_only=False)
    rollout = torch.load(directory / 'initial_sampled_tokens.pt', map_location='cpu', weights_only=False)
    batch = repair['batch']
    if not torch.equal(rollout['tokens'][batch['micro_batch_inds']], batch['mb_sampled_tokens']):
        raise ValueError('Actual sampling token identity changed')
    probability = initial_probability(repair['corrected'], batch)
    config = yaml.safe_load((shared.RUN / 'config.yaml').read_text())
    old = yaml.safe_load((PRIOR_RUN / 'config.yaml').read_text())
    # Only the corpus path differs here; sampler/reward/randomization stay exact.
    new_env, old_env = config['manager_env'], old['manager_env']
    new_env['commands']['motion']['motion_lib_cfg']['motion_file'] = []
    old_env['commands']['motion']['motion_lib_cfg']['motion_file'] = []
    if new_env != old_env or config['algo'] != old['algo']:
        raise ValueError('Corpus-only training recipe differs from E44')
    routes = shared.read(directory / 'reference_route_coverage.json')['rollouts']
    if len(routes) != UPDATES or sum(sum(r['routes'].values()) for r in routes) != TRANSITIONS:
        raise ValueError('Actual sampling coverage differs from registered exposure')
    atomic_json(OUT / 'training_verification.json', dict(status='VERIFIED_TRAINING_NOT_EFFECT',
        checkpoints=records, changes=changed, optimizer_steps=len(audit),
        actual_optimizer_steps=audit, contract=contract, initial_cached_probability=probability,
        sampled_reference_routes=routes, recipe_matches_e44_except_corpus=True, development_roles=job['development_roles'],
        source_sha256=sha256(shared.OFFICIAL), deploy_allowed=False))


def evaluate(step, kind):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    directory = OUT / ('checkpoint_' + str(step))
    args = arguments(directory / ('evaluation_' + kind),
        shared.LOG / ('checkpoint_' + str(step)) / ('evaluation_' + kind),
        checkpoint(step), OUT / (kind + '_references'))
    if kind == 'official':
        args.frame_stride = 4
    evaluation.preflight = shared.guard.preflight
    evaluation.monitor_child = lambda child, log, config: (child.wait(), None)
    evaluation.subprocess = LaunchProxy(evaluation.subprocess)
    if evaluation.evaluate_explicit(args):
        raise RuntimeError('Matched learning-curve evaluation failed; no retry')


def verify_effect(step):
    shared.OUT = OUT / ('checkpoint_' + str(step))
    try:
        EFFECT_VERIFY(shared.read(OUT / 'registration.json'))
    finally:
        shared.OUT = OUT
    if step == CHECKPOINTS[-1]:
        effects = {s: shared.read(OUT / ('checkpoint_' + str(s)) / 'effect_verification.json')
                   for s in CHECKPOINTS}
        atomic_json(OUT / 'effect_verification.json', outcome(effects))


shared.STAGES = {'prepare': ('REGISTERED', 'PREPARED'), 'train': ('PREPARED', 'TRAINED'),
                 'verify-training': ('TRAINED', 'TRAINING_VERIFIED')}
previous = 'TRAINING_VERIFIED'
for step in CHECKPOINTS:
    for operation in ('cloth', 'official', 'verify'):
        name = f'curve-{step}-{operation}'
        following = 'COMPLETE' if step == UPDATES and operation == 'verify' else name.upper() + '_DONE'
        shared.STAGES[name] = (previous, following)
        previous = following


def worker(stage):
    state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
    if (state['status'] != stage.upper() + '_RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E36_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires its registered guarded monitor')
    shared.unchanged(job)
    if stage == 'prepare':
        prepare(job)
    elif stage == 'train':
        train(job)
    elif stage == 'verify-training':
        verify_training(job)
    else:
        _, step, operation = stage.split('-')
        if operation == 'verify':
            verify_effect(int(step))
        else:
            evaluate(int(step), operation)
    shared.unchanged(job)


def plan():
    runtime.lightweight_controller()
    return dict(experiment=shared.EXPERIMENT, trial=shared.TRIAL,
        status='PLAN_ONLY_NOT_RUNTIME_STATE', prerequisites=['E44_ARCHIVED_TEACHER_STOP', 'UNCHANGED_RESOURCE_PREFLIGHT'],
        main_variable='TRAINING_CORPUS_3_TO_4', training_sequences=list(TRAINING),
        added_sequence_role='DEVELOPMENT_TRAIN_NOT_UNSEEN',
        updates=UPDATES, optimizer_steps=OPTIMIZER_STEPS, transitions=TRANSITIONS,
        effect_checkpoints=list(CHECKPOINTS), stages=shared.STAGES,
        maximum_evaluation_rollouts=len(CHECKPOINTS) * 10,
        preflight_extra_available_bytes=8 * 1024**3, runtime_limit_seconds_per_stage=900,
        controller_imported_tensor_runtime=False, model_runtime_validation='PENDING_ZHY_RESOURCE_GATE')


def run(stage):
    runtime.lightweight_controller()
    state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
    if state['status'] != shared.STAGES[stage][0]:
        raise RuntimeError('No repeat, recovery or out-of-order stage')
    from script.a3.host_health import append_record
    for i in range(3):
        value = shared.guard.preflight()
        if (value['boot_id'] != job['boot_id'] or
                value['container_memory']['oom_kill'] != job['baseline_oom_kill']):
            raise RuntimeError('New machine fault; no stage startup')
        if value['container_memory']['available_estimate_bytes'] < 8 * 1024**3:
            raise RuntimeError('Original 8 GiB stage startup gate rejected')
        if any(r['rss_bytes'] > 1024**3 for r in value['process_memory']['rows']):
            raise RuntimeError('Other large process; no parallel computation')
        append_record(shared.LOG / (stage + '_entry_preflight.jsonl'), value)
        if i < 2:
            time.sleep(5)
    return runtime.run(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='prepare')
    args = parser.parse_args()
    if args.action == 'plan':
        print(json.dumps(plan(), indent=2))
        raise SystemExit(0)
    result = register() if args.action == 'register' else worker(args.stage) if args.action == 'worker' else run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
