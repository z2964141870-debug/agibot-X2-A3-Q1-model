"""Registered E36 clothing decoder pilot, guarded and without repeat or recovery."""

import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time
import unittest

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_cloth_constrained_20261013_E36'
LOG = ROOT / 'logs/a3_zhy_cloth_constrained_20261013_E36'
RUN = ROOT / 'data/training/a3_zhy_cloth_constrained_20261013_E36/R13_001_s0'
MANIFEST = ROOT / 'data/manifests/a3_zhy_cloth_constrained_20261013_E36.json'
EXPERIMENT = 'A3_ZHY_CLOTH_CONSTRAINED_20261013_E36'
TRIAL = 'R13'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
CORPUS = ROOT / 'data/experiments/a3_zhy_smpl_hull_20261013_E23'
BASELINE = ROOT / 'data/experiments/a3_zhy_smpl_frozen_20261013_E26'
OLD = ROOT / 'data/experiments/a3_zhy_search_20261012_E02'
ANCHOR = ROOT / 'data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt'
TARGETS = ROOT / 'data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt'
TRAINER = 'script.a3.cloth_constrained_trainer.ClothConstrainedTrainer'
MODULE = 'script.a3.cloth_constrained_trial'
CANONICAL_STD = False
CODE = ('script/a3/cloth_constrained_trial.py', 'script/a3/cloth_constrained_trainer.py',
    'script/a3/test_cloth_constrained.py', 'script/a3/consistent_precision.py',
    'script/a3/verified_finetune.py', 'script/a3/finetune_trainer.py',
    'script/a3/resumable_sonic.py', 'script/a3/checkpoint_store.py',
    'script/a3/zhy_trial.py', 'script/a3/host_health.py', 'script/a3/likelihood_probe.py',
    'script/a3/gradient_probe.py', 'script/a3/zhy_likelihood_trainer.py',
    'script/a3/evaluate_mujoco.py', 'script/a3/frozen_evaluation_launch.py',
    'script/a3/smpl_frozen_entry.py', 'script/a3/sim_entry.py',
    'script/a3/verify_zhy_search.py', 'script/a3/zhy_fp32_trial.py')


def read(path):
    return json.loads(path.read_text())


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Boot changed; stop without recovery')
    for name, identity in {**job['code_sha256'], **job['input_sha256']}.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Registered input or code changed: ' + name)


def register():
    if OUT.exists() or RUN.parent.exists():
        raise FileExistsError('Do not reset or repeat E36/R13')
    value = guard.preflight()
    if sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise ValueError('Official identity changed')
    corpus, baseline = read(CORPUS / 'independent_verification.json'), read(BASELINE / 'registration.json')
    if corpus['status'] != 'VERIFIED_KINEMATIC_CORPUS':
        raise ValueError('Requires qualified clothing corpus')
    accepted = [r for r in corpus['sequences'] if r['pair_quality']['passed']]
    training = [r for r in accepted if r['partition'] == 'train']
    if len(accepted) != 6 or len(training) != 3:
        raise ValueError('Fixed qualified training coverage changed')
    inputs = [OFFICIAL, ANCHOR, TARGETS, CORPUS / 'independent_verification.json',
        BASELINE / 'independent_verification.json', BASELINE / 'registration.json',
        ROOT / 'data/experiments/a3_zhy_modules_20261012_E12/independent_verification.json',
        ROOT / 'data/experiments/a3_zhy_smpl_assembly_audit_20261013_E35/result.json',
        OLD / 'job.json', OLD / 'independent_verification.json']
    for row in accepted:
        path = CORPUS / 'sequences' / row['sequence'].replace('/', '__') / 'reference.csv'
        if sha256(path) != row['artifacts']['reference.csv']['sha256']:
            raise ValueError('Clothing reference no longer matches acceptance')
        inputs.append(path)
    for base in (BASELINE / 'evaluation', OLD / 'official_oracle'):
        inputs.extend(sorted(base.glob('*.metrics.json')))
        inputs.extend(sorted(base.glob('*.timeseries.json')))
        inputs.append(base / 'explicit_summary.json')
    official_motions = read(OLD / 'job.json')['diagnostic_motions']
    inputs.extend(Path(r['path']) for r in official_motions)
    vendor = ROOT / 'script/vendor/sonic_for_a3'
    inputs += [vendor / name for name in (
        'train_a3_035_fromscratch.sh', 'gear_sonic/data_process/convert_soma_csv_to_motion_lib.py',
        'gear_sonic/trl/modules/universal_token_modules.py',
        'gear_sonic/trl/trainer/ppo_trainer.py', 'gear_sonic/trl/trainer/ppo_trainer_aux_loss.py',
        'gear_sonic/scripts/sim2sim_a3_mujoco.py')]
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Reliable kernel telemetry required')
    job = dict(experiment=EXPERIMENT, trial=TRIAL,
        registered_at=time.time(), project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        boot_id=value['boot_id'], baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(),
        code_sha256={p: sha256(ROOT / p) for p in CODE},
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        main_variable='CONSTRAINED_DYNAMICS_DECODER_CLOTHING_ADAPTATION_RECIPE',
        isolated_freeze_causal_ablation=False,
        trainer=TRAINER, seed=0, updates=2, optimizer_steps=8, num_envs=16,
        epochs=1, minibatches=4, actor_learning_rate=2e-6,
        adaptive_lr_min=1e-6, adaptive_lr_max=2e-5, fp32=True,
        policy_update_prefix='actor_module.decoders.g1_dyn.', critic_updated=True,
        frozen_policy='ALL_OTHER_ACTOR_PARAMETERS_AND_BUFFERS',
        training_sequences=[r['sequence'] for r in training],
        qualified_reference_gate='E23_ORIGINAL_KINEMATIC_ONLY_NOT_DYNAMIC_PROOF',
        anchor_frames=384, teacher_coefficient=1., anchor_rmse_max=.05, anchor_max_abs=.5,
        initial_action_rmse_max=1e-5, initial_ratio_deviation_max=.001,
        selection=baseline['selection'], official_motions=official_motions,
        development_gate=dict(zero_clothing_falls=True, mean_clothing_improvement_percent=1.,
            each_previous_nonfall_clothing_regression_max_percent=2.,
            official_each_joint_regression_max_percent=2., no_new_official_falls=True,
            root_and_wrist_world_regression_max_percent=5.),
        fresh_final_test=False, final_test_sessions_not_yet_selected=True,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False,
        repeat_allowed=False, std_canonicalization=CANONICAL_STD,
        contract_tests=5 if CANONICAL_STD else 3, backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    for name in ('motionlib', 'cloth_references', 'official_references'):
        (OUT / name).mkdir()
    for row in job['selection']:
        source = ROOT / row['source'] / 'reference.csv'
        (OUT / 'cloth_references' / row['filename']).symlink_to(os.path.relpath(source, OUT / 'cloth_references'))
    for row in official_motions:
        path = Path(row['path'])
        (OUT / 'official_references' / path.name).symlink_to(os.path.relpath(path, OUT / 'official_references'))
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='REGISTERED'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='REGISTERED', experiment=job['experiment'], training_sequences=job['training_sequences'])))


def phase(job, name):
    if job.get('resource_evidence_required'):
        from script.a3.stage_resources import stage_event
        stage_event(LOG / 'stage_resources.jsonl', name, guard.health(),
                    job['prepare_process_budget_bytes'] if name.startswith('prepare/') else None)


def prepare(job):
    phase(job, 'prepare/before_contract_imports')
    from script.a3 import test_cloth_constrained as tests
    if job.get('std_canonicalization', False):
        from script.a3 import test_cloth_std_contract as tests
    phase(job, 'prepare/after_contract_imports')
    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import load_a3_flat_csv, convert_sequence
    from script.a3.reference_contract import load_sim
    from scipy.spatial.transform import Rotation
    import joblib
    import numpy as np
    import torch
    phase(job, 'prepare/after_conversion_imports')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful() or result.testsRun != job['contract_tests']:
        raise ValueError('Freeze/likelihood controls failed: ' + stream.getvalue())
    phase(job, 'prepare/after_contract_tests')
    checkpoint_tests = None
    if job.get('resource_evidence_required'):
        from script.a3 import test_checkpoint_store
        checked = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(test_checkpoint_store))
        if not checked.wasSuccessful() or checked.testsRun != 8:
            raise ValueError('Checkpoint transaction controls failed: ' + stream.getvalue())
        checkpoint_tests = checked.testsRun
        phase(job, 'prepare/after_checkpoint_tests')
    source = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    phase(job, 'prepare/after_official_load')
    std = source['policy_state_dict']['std']
    if not job.get('std_canonicalization', False) and not torch.equal(std, std.clamp(.001, .5)):
        raise ValueError('Official std needs clamping; register a separate baseline contract first')
    from script.a3.cloth_std_contract import canonical_policy_state
    canonical = canonical_policy_state(source['policy_state_dict'])
    std_boundary = dict(source_min=float(std.min()), source_max=float(std.max()),
        runtime_min=float(canonical['std'].min()), runtime_max=float(canonical['std'].max()),
        changed_coordinates=int((std != canonical['std']).sum()),
        semantics='OFFICIAL_GET_STD_CLAMP_BEFORE_FROZEN_DIGEST_NOT_OPTIMIZER_UPDATE',
        deterministic_mean_uses_std=False)
    del source
    sim, rows = load_sim(), []
    phase(job, 'prepare/after_sim_module_import')
    for sequence in job['training_sequences']:
        name = sequence.replace('/', '__')
        path = CORPUS / 'sequences' / name / 'reference.csv'
        entry = convert_sequence(load_a3_flat_csv(str(path), robot='a3_29'), fps=30)
        root, quat, joints, count = sim.load_a3_flat_csv(path, source_fps=30, frame_stride=1)
        axes = entry['dof']
        expected_names = [sim.A3_CSV_JOINT_NAMES[i] for i in sim.A3_POLICY_TO_SDK_IDX]
        from gear_sonic.data_process.convert_soma_csv_to_motion_lib import A3_29D_SPEC
        output_names = [A3_29D_SPEC['joint_names'][i] for i in A3_29D_SPEC['dof_output_indices']]
        remapped = axes[:, [output_names.index(n) for n in expected_names]]
        joint_error = float(np.max(np.abs(remapped - joints)))
        root_error = float(np.max(np.abs(entry['root_trans_offset'] - root)))
        angle_error = float((Rotation.from_quat(entry['root_rot']) *
            Rotation.from_quat(quat[:, [1, 2, 3, 0]]).inv()).magnitude().max())
        if axes.shape != (300, 29) or count != 300 or max(joint_error, root_error, angle_error) > 1e-6:
            raise ValueError('Clothing MotionLib ABI changed')
        target = OUT / 'motionlib' / (name + '.pkl')
        joblib.dump({name: entry}, target, compress=True)
        restored = joblib.load(target)[name]
        if any(not np.array_equal(entry[k], restored[k]) for k in
                ('dof', 'pose_aa', 'root_rot', 'root_trans_offset')):
            raise ValueError('MotionLib reload changed values')
        rows.append(dict(sequence=sequence, partition='train', frames=count, fps=30,
            joint_max_error=joint_error, root_max_error_m=root_error, rotation_max_error_rad=angle_error,
            path=str(target.relative_to(ROOT)), sha256=sha256(target), bytes=target.stat().st_size))
        phase(job, 'prepare/converted/' + name)
    atomic_json(OUT / 'prepare.json', dict(tests=result.testsRun, tests_passed=True, test_output=stream.getvalue(),
        checkpoint_tests=checkpoint_tests, motions=rows, excludes_other_partitions=True,
        std_boundary=std_boundary, dynamics_acceptance=False))
    phase(job, 'prepare/complete')


def train(job):
    phase(job, 'train/before_launch')
    for row in read(OUT / 'prepare.json')['motions']:
        if sha256(ROOT / row['path']) != row['sha256']:
            raise ValueError('Prepared training input changed')
    RUN.mkdir(parents=True, exist_ok=False)
    directory = LOG / 'training'
    directory.mkdir(exist_ok=False)
    environment = dict(os.environ, ISAAC_PYTHON=str(ROOT / 'data/environments/a3-sonic/bin/python'),
        REPO_DIR=str((ROOT / 'script/vendor/sonic_for_a3').resolve()),
        MOTION_FILE=str(OUT / 'motionlib'), EXPERIMENT_DIR=str(RUN), CHECKPOINT=str(OFFICIAL),
        NUM_ENVS='16', NUM_MINI_BATCHES='4', NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(ROOT),
        WANDB_MODE='disabled', ACCELERATE_MIXED_PRECISION='no', YUANQI_RESUME_LOG_DIR=str(directory),
        YUANQI_CLOTH_ANCHOR_ROLLOUT=str(ANCHOR), YUANQI_CLOTH_ANCHOR_OUTPUTS=str(TARGETS))
    argv = ['bash', str(ROOT / 'script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh'),
        'seed=0', 'trainer._target_=' + TRAINER, 'algo.config.num_learning_epochs=1',
        'algo.config.actor_learning_rate=2e-6', 'algo.config.adaptive_lr_min=1e-6',
        'algo.config.adaptive_lr_max=2e-5',
        'callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback',
        'callbacks.model_save.save_frequency=1', '+callbacks.model_save.target_global_step=2',
        '++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']
    atomic_json(OUT / 'training_launch.json', dict(argv=argv, motion_file=str(OUT / 'motionlib')))
    with (directory / 'console.log').open('w') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=False)
        if child.wait():
            raise RuntimeError('Clothing training exited unsuccessfully; no retry')
    phase(job, 'train/complete')


def verify_training(job):
    import torch
    from script.a3.checkpoint_store import validate_payload
    from script.a3.cloth_constrained_trainer import DECODER, initial_probability
    torch.set_num_threads(1)
    records = [read(RUN / f'model_step_{i:06d}.json') for i in range(3)]
    payloads = []
    for row in records:
        path = RUN / row['file']
        if sha256(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Saved checkpoint identity changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != row['step']:
            raise ValueError('Checkpoint step changed')
        payloads.append(payload)
    source = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    if job.get('std_canonicalization', False):
        from script.a3.cloth_std_contract import canonical_policy_state
        source['policy_state_dict'] = canonical_policy_state(source['policy_state_dict'])
    for key in ('policy_state_dict', 'value_state_dict'):
        if source[key].keys() != payloads[0][key].keys() or any(
                not torch.equal(v, payloads[0][key][n]) for n, v in source[key].items()):
            raise ValueError('Step0 differs from official warm start')
    if payloads[0]['optimizer_state_dict']['state']:
        raise ValueError('Step0 optimizer is not empty')
    changed = []
    for step in (1, 2):
        actor = payloads[step]['policy_state_dict']
        if actor.keys() != payloads[0]['policy_state_dict'].keys():
            raise ValueError('Actor keys changed')
        names = [n for n, v in actor.items() if not torch.equal(v, payloads[0]['policy_state_dict'][n])]
        if not names or any(not n.startswith(DECODER) for n in names):
            raise ValueError('Update escaped dynamics decoder or applied no update')
        changed.append(dict(step=step, changed_actor_tensors=names))
    contract = read(LOG / 'training/cloth_contract.json')
    audit = [json.loads(line) for line in (LOG / 'training/actual_optimizer_steps.jsonl').read_text().splitlines()]
    if (len(audit) != 8 or len(contract['optimizer_steps']) != 8 or
            any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5
                for r in contract['optimizer_steps'])):
        raise ValueError('Optimizer count or official anchor budget failed')
    captured = torch.load(LOG / 'training/initial_probability.pt', map_location='cpu', weights_only=False)
    probability = initial_probability(captured['policy'], captured['batch'])
    if probability != contract['initial_probability']:
        # Independent CPU reduction has a different summation order from CUDA.
        if abs(probability['action_rmse'] - contract['initial_probability']['action_rmse']) > 1e-7:
            raise ValueError('CPU initial probability reconstruction differs')
    import yaml
    config = yaml.safe_load((RUN / 'config.yaml').read_text())
    if any(config['algo']['trl'][k] is not False for k in ('bf16', 'fp16', 'tf32')):
        raise ValueError('Actual configuration violates FP32')
    atomic_json(OUT / 'training_verification.json', dict(status='VERIFIED_TRAINING_NOT_EFFECT',
        checkpoints=records, changes=changed, contract=contract, actual_optimizer_steps=audit,
        independent_cpu_initial_probability=probability, optimizer_steps=8, source_sha256=sha256(OFFICIAL),
        precision=read(LOG / 'training/precision_contract.json'), deploy_allowed=False))


def evaluate(job, kind):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    args = arguments(OUT / ('evaluation_' + kind), LOG / ('evaluation_' + kind),
        RUN / 'model_step_000002.pt', OUT / (kind + '_references'))
    if kind == 'official':
        args.frame_stride = 4
    evaluation.preflight = guard.preflight
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    evaluation.subprocess = LaunchProxy(evaluation.subprocess)
    if evaluation.evaluate_explicit(args):
        raise RuntimeError('Candidate evaluation failed; no repeat')


def verify_evaluation(job):
    import numpy as np
    from script.a3.verify_zhy_search import verify_metrics
    rows, gates = [], []
    for kind, baseline in (('cloth', BASELINE / 'evaluation'), ('official', OLD / 'official_oracle')):
        directory = OUT / ('evaluation_' + kind)
        for record in verify_metrics(directory):
            name = record['motion']
            before = read(baseline / (name + '.metrics.json'))['motions'][0]
            after = read(directory / (name + '.metrics.json'))['motions'][0]
            trace = read(directory / (name + '.timeseries.json'))['motions'][0]
            q, target = np.asarray(trace['q_state_29']), np.asarray(trace['reference_q_29'])
            old_rmse, new_rmse = before['tracking']['all_29_rmse'], float(np.sqrt(np.mean((q-target)**2)))
            falling = np.flatnonzero((np.asarray(trace['root_height']) < .45) |
                (np.abs(trace['root_roll_deg']) > 60.) | (np.abs(trace['root_pitch_deg']) > 60.))
            if after['fall_tick'] != (int(falling[0]) if len(falling) else None):
                raise ValueError('Independent fall reconstruction failed')
            fallen = bool(len(falling))
            old_fallen = bool(before['fall'])
            if fallen != after['fall']:
                raise ValueError('Saved fall flag differs from reconstruction')
            retention = (not fallen and new_rmse <= old_rmse * 1.02) if not old_fallen else None
            published = read(directory / 'explicit_summary.json')['motions']
            old_published = read(baseline / 'explicit_summary.json')['motions']
            current_window = next(r for r in published if Path(r['motion']).stem == name)['windows']['full_rollout']
            old_window = next(r for r in old_published if Path(r['motion']).stem == name)['windows']['full_rollout']
            world_ratios = {k: current_window[k] / max(old_window[k], 1e-12) for k in
                ('root_pos_error_m_mean', 'wrists_world_mean_m')}
            gates.append(all(r <= 1.05 for r in world_ratios.values()))
            rows.append(dict(kind=kind, motion=name, steps=len(q), baseline_rmse_rad=old_rmse,
                candidate_rmse_rad=new_rmse, ratio=new_rmse / old_rmse, baseline_fall=old_fallen,
                candidate_fall=fallen, candidate_fall_label_s=after['fall_time_s'],
                retention_passed=retention, world_error_ratios=world_ratios,
                metrics_sha256=record['metrics_sha256'], timeseries_sha256=record['timeseries_sha256']))
            if retention is not None:
                gates.append(retention)
    clothing = [r for r in rows if r['kind'] == 'cloth']
    old_mean = float(np.mean([r['baseline_rmse_rad'] for r in clothing]))
    new_mean = float(np.mean([r['candidate_rmse_rad'] for r in clothing]))
    passed = (len(clothing) == 6 and len(rows) == 10 and all(gates)
        and not any(r['candidate_fall'] for r in clothing) and new_mean <= old_mean * .99)
    atomic_json(OUT / 'effect_verification.json', dict(rows=rows, development_gate_passed=passed,
        baseline_clothing_mean_rad=old_mean, candidate_clothing_mean_rad=new_mean,
        fresh_final_test=False, unseen_generalization_proven=False, deploy_allowed=False,
        note='Development gate success still requires independent unseen session validation'))


def worker(stage):
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != stage.upper() + '_RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E36_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker needs its guarded monitor')
    unchanged(job)
    if stage == 'prepare':
        prepare(job)
    elif stage == 'train':
        train(job)
    elif stage == 'verify-training':
        verify_training(job)
    elif stage.startswith('evaluate-'):
        evaluate(job, stage.split('-')[1])
    else:
        verify_evaluation(job)


STAGES = {'prepare': ('REGISTERED', 'PREPARED'), 'train': ('PREPARED', 'TRAINED'),
    'verify-training': ('TRAINED', 'TRAINING_VERIFIED'),
    'evaluate-cloth': ('TRAINING_VERIFIED', 'CLOTH_EVALUATED'),
    'evaluate-official': ('CLOTH_EVALUATED', 'EVALUATED'),
    'verify-effect': ('EVALUATED', 'COMPLETE')}


def run(stage):
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != STAGES[stage][0]:
            raise RuntimeError('No repeat, recovery, or out-of-order stage')
        unchanged(job)
        guard.preflight()
        guard.OUT, guard.LOG = OUT, LOG
        directory = LOG / stage
        directory.mkdir(exist_ok=False)
        state.update(status=stage.upper() + '_RUNNING', monitor_pid=os.getpid(), stage=stage,
                     stage_started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (directory / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    MODULE, 'worker', '--stage', stage], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E36_MONITOR_PID=str(os.getpid()),
                        PYTHONPATH=str(ROOT / 'script/vendor/sonic_for_a3') + ':' + str(ROOT)),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, directory, job)
            state.update(status=STAGES[stage][1] if code == 0 and reason is None else 'STOPPED',
                         returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['stage_ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        receipt = dict(experiment=job['experiment'], status=state['status'], state=dict(state),
            code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
            backup_status='LOCAL_ONLY', fresh_final_test=False, deploy_allowed=False)
        from script.a3 import smpl_frozen_followup as shared
        receipt['health'] = shared.health_summary(LOG)
        health = receipt['health']
        if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
                or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
            state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
            atomic_json(OUT / 'state.json', state)
            receipt.update(status='STOPPED', state=dict(state))
        for name in ('prepare', 'training_verification', 'effect_verification'):
            if (OUT / (name + '.json')).exists():
                receipt[name] = read(OUT / (name + '.json'))
        atomic_json(MANIFEST, receipt)
        print(json.dumps(state))
        return 0 if state['status'] != 'STOPPED' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(STAGES), default='prepare')
    args = parser.parse_args()
    value = register() if args.action == 'register' else worker(args.stage) if args.action == 'worker' else run(args.stage)
    raise SystemExit(value if isinstance(value, int) else 0)
