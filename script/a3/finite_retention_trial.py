"""E51/R20 fresh finite-candidate training, followed by one final matched evaluation."""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import time
import unittest

from script.a3 import cloth_cached_trial as cached
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.cloth_failure_protocol import TRAINING, corpus_roles
from script.a3.finite_retention_protocol import CHECKPOINTS, UPDATES, STEPS, TRANSITIONS, LIMITS, SCALES
from script.a3.host_health import append_record, command
from script.a3.kernel_review import check_kernel

runtime, shared = cached.runtime, cached.shared
OUT = shared.ROOT / 'data/experiments/a3_zhy_finite_retention_20261014_E51'
shared.OUT = OUT
shared.LOG = shared.ROOT / 'logs/a3_zhy_finite_retention_20261014_E51'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_finite_retention_20261014_E51/R20_001_s0'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_finite_retention_20261014_E51.json'
shared.EXPERIMENT = 'A3_ZHY_FINITE_RETENTION_20261014_E51'
shared.TRIAL = 'R20'
shared.MODULE = 'script.a3.finite_retention_trial'
shared.TRAINER = 'script.a3.finite_retention_trainer.FiniteRetentionTrainer'
PLAN = shared.ROOT / 'data/manifests/a3_finite_retention_plan_20261017.json'
PRIOR = shared.ROOT / 'data/experiments/a3_zhy_cloth_failure_20261013_E45'
E50 = shared.ROOT / 'data/experiments/a3_zhy_candidate_retention_followup_20261014_E50'
OLD_RUN = shared.ROOT / 'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0'
shared.CODE += ('script/a3/cloth_curve_trainer.py', 'script/a3/cloth_curve_protocol.py',
    'script/a3/cloth_failure_protocol.py', 'script/a3/retention_direction_probe.py',
    'script/a3/retention_direction_protocol.py', 'script/a3/finite_adam.py',
    'script/a3/finite_retention_protocol.py', 'script/a3/finite_retention_trainer.py',
    'script/a3/finite_retention_trial.py', 'script/a3/test_finite_adam.py',
    'script/a3/test_finite_retention_protocol.py', 'script/a3/kernel_review.py',
    'script/a3/test_kernel_review.py')
shared.STAGES = {'prepare': ('REGISTERED', 'PREPARED'), 'train': ('PREPARED', 'TRAINED'),
    'verify-training': ('TRAINED', 'TRAINING_VERIFIED'),
    'evaluate-cloth': ('TRAINING_VERIFIED', 'CLOTH_EVALUATED'),
    'evaluate-official': ('CLOTH_EVALUATED', 'EVALUATED'),
    'verify-effect': ('EVALUATED', 'COMPLETE')}


def preflight(baseline):
    rows = []
    for index in range(3):
        value = shared.guard.preflight()
        if not value['gpus'] or any(g.get('utilization.gpu') != 0 for g in value['gpus']):
            raise RuntimeError('GPU not idle or utilization telemetry unavailable')
        if (value['boot_id'] != baseline['boot_id'] or
                value['container_memory']['oom_kill'] != baseline['baseline_oom_kill']):
            raise RuntimeError('New boot or OOM; no startup')
        if value['container_memory']['available_estimate_bytes'] < 8 * 1024**3:
            raise RuntimeError('Original 8 GiB headroom unavailable; wait without registration')
        if any(r['rss_bytes'] > 1024**3 for r in value['process_memory']['rows']):
            raise RuntimeError('Other large process; no parallel computation')
        if any(r['pid'] != os.getpid() and r['comm'].lower().startswith(('python', 'kit', 'isaac', 'stress'))
               for r in value['process_memory']['rows']):
            raise RuntimeError('Other compute process; no parallel computation')
        checks = []
        for error in value['process_memory']['errors']:
            path = Path('/proc') / str(error['pid']) / 'status'
            if not path.exists():
                checks.append(dict(pid=error['pid'], status='EXITED')); continue
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            state = fields['State'].strip()
            if state.startswith('Z'):
                checks.append(dict(pid=error['pid'], status='ZOMBIE')); continue
            rss, name = int(fields['VmRSS'].split()[0]) * 1024, fields['Name'].strip()
            if rss > 1024**3 or name not in ('pod', 'ssh-agent', 'gpg-agent'):
                raise RuntimeError('Unresolved process telemetry')
            checks.append(dict(pid=error['pid'], status=state, comm=name, rss_bytes=rss))
        value['missing_process_status_checks'] = checks
        rows.append(value)
        if index < 2:
            time.sleep(5)
    return rows


def register():
    runtime.lightweight_controller()
    if OUT.exists() or shared.LOG.exists() or shared.RUN.parent.exists():
        raise FileExistsError('E51 must not repeat or reset')
    if shared.read(E50 / 'state.json')['status'] != 'COMPLETE' or shared.read(PRIOR / 'state.json')['status'] != 'STOPPED':
        raise ValueError('Historical E50 and E45 states must remain preserved')
    plan = shared.read(PLAN)
    if plan['status'] != 'IMPLEMENTED_UNREGISTERED' or plan['limits'] != LIMITS or plan['scales'] != list(SCALES):
        raise ValueError('Fixed implementation plan differs')
    inputs = {r['path']: r['sha256'] for r in plan['inputs']}
    for name, identity in {**inputs, **plan['source_sha256']}.items():
        if sha256(shared.ROOT / name) != identity:
            raise ValueError('Fixed input/source changed: ' + name)
    baseline = shared.read(E50 / 'registration.json')
    rows = preflight(baseline)
    kernel = command(['dmesg', '--color=never'])
    kernel_review = check_kernel(shared.ROOT, kernel, baseline, plan.get('entry_kernel_review'),
        boot_id=rows[-1]['boot_id'], cgroup_text=Path('/proc/self/cgroup').read_text())
    shared.register()
    job = shared.read(OUT / 'registration.json')
    try:
        if (job['boot_id'] != baseline['boot_id'] or
                job['baseline_oom_kill'] != baseline['baseline_oom_kill']):
            raise RuntimeError('Boot or container OOM changed while registering')
        kernel_review = check_kernel(shared.ROOT,
            dict(returncode=0, stdout='\n'.join(job['kernel_baseline'])), baseline,
            plan.get('entry_kernel_review'), boot_id=job['boot_id'],
            cgroup_text=Path('/proc/self/cgroup').read_text())
    except Exception:
        state = shared.read(OUT / 'state.json')
        state.update(status='STOPPED', reason='POST_REGISTRATION_KERNEL_REVIEW_FAILED')
        atomic_json(OUT / 'state.json', state)
        raise
    roles = corpus_roles(shared.read(shared.CORPUS / 'independent_verification.json')['sequences'])
    job.update(main_variable='FINITE_ACTUAL_ADAM_CANDIDATE_ACCEPTANCE_BEFORE_COMMIT',
        updates=UPDATES, optimizer_steps=STEPS, sampled_transitions=TRANSITIONS,
        training_sequences=list(TRAINING), development_roles=roles,
        effect_checkpoints=[UPDATES], saved_checkpoints=[0, *CHECKPOINTS],
        limits=LIMITS, scales=list(SCALES), teacher_gate='ORIGINAL_ABSOLUTE_RMSE_0.05_MAXABS_0.5',
        same_batch_objective_tolerance=1e-7, e50_reinterpreted=False,
        source_state='FRESH_OFFICIAL_035_EMPTY_ADAM', no_old_training_resumed=True,
        resource_evidence_required=True, prepare_process_budget_bytes=3 * 1024**3,
        preflight_extra_available_bytes=8 * 1024**3, preflight_samples=3,
        controller_tensor_import_allowed=False, toy_transaction_controls=4,
        toy_native_optimizer_calls_max=20, preflight_samples_before_each_stage=3,
        selection_rule='FINAL_STEP8_ONLY_TEN_FULL_MATCHED_DEVELOPMENT_MOTIONS',
        historical_kernel_review=kernel_review)
    job['input_sha256'].update(inputs)
    job['input_sha256'].update(plan['source_sha256'])
    job['input_sha256'][str(PLAN.relative_to(shared.ROOT))] = sha256(PLAN)
    atomic_json(OUT / 'registration.json', job)
    for row in rows:
        append_record(shared.LOG / 'entry_preflight.jsonl', row)
    print(json.dumps(dict(status='REGISTERED', experiment=shared.EXPERIMENT, limits=LIMITS)))


def prepare(job):
    shared.phase(job, 'prepare/before_transaction_controls')
    from script.a3 import test_finite_adam
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_finite_adam))
    if result.testsRun != 4 or not result.wasSuccessful():
        atomic_json(OUT / 'prepare_controls.json', dict(tests=result.testsRun, passed=False, output=stream.getvalue()))
        raise ValueError('Finite transaction controls failed: ' + stream.getvalue())
    rows = shared.read(PRIOR / 'prepare.json')['motions']
    if [r['sequence'] for r in rows] != list(TRAINING):
        raise ValueError('Exactly four original development motions required')
    for row in rows:
        path = shared.ROOT / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('Validated MotionLib changed')
        (OUT / 'motionlib' / path.name).symlink_to(os.path.relpath(path, OUT / 'motionlib'))
    atomic_json(OUT / 'prepare.json', dict(tests=result.testsRun, tests_passed=True,
        test_output=stream.getvalue(), motions=rows, original_motion_abi_reused=True,
        no_reference_reconversion=True, no_robot_model_loaded=True,
        native_fused_adam_transaction_controls=True, development_roles=job['development_roles']))
    shared.phase(job, 'prepare/complete')


def train(job):
    if not shared.read(OUT / 'prepare.json')['tests_passed']:
        raise ValueError('Transaction runtime controls required')
    shared.RUN.mkdir(parents=True, exist_ok=False)
    directory = shared.LOG / 'training'; directory.mkdir(exist_ok=False)
    environment = dict(os.environ, ISAAC_PYTHON=str(shared.ROOT / 'data/environments/a3-sonic/bin/python'),
        REPO_DIR=str((shared.ROOT / 'script/vendor/sonic_for_a3').resolve()),
        MOTION_FILE=str(OUT / 'motionlib'), EXPERIMENT_DIR=str(shared.RUN),
        CHECKPOINT=str(shared.OFFICIAL), NUM_ENVS='16', NUM_MINI_BATCHES='4',
        NUM_LEARNING_ITERATIONS='200', NUM_PROCESSES='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(shared.ROOT), WANDB_MODE='disabled',
        ACCELERATE_MIXED_PRECISION='no', YUANQI_RESUME_LOG_DIR=str(directory),
        YUANQI_CLOTH_ANCHOR_ROLLOUT=str(shared.ANCHOR), YUANQI_CLOTH_ANCHOR_OUTPUTS=str(shared.TARGETS))
    argv = ['bash', str(shared.ROOT / 'script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh'),
        'seed=0', 'trainer._target_=' + shared.TRAINER, 'algo.config.num_learning_epochs=1',
        'algo.config.actor_learning_rate=2e-6', 'algo.config.adaptive_lr_min=1e-6',
        'algo.config.adaptive_lr_max=2e-5',
        'callbacks.model_save._target_=script.a3.finite_retention_trainer.FiniteCheckpointCallback',
        'callbacks.model_save.save_frequency=8', '+callbacks.model_save.target_global_step=8',
        '++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']
    atomic_json(OUT / 'training_launch.json', dict(argv=argv, limits=LIMITS))
    with (directory / 'console.log').open('w') as stream:
        code = subprocess.run(argv, cwd=shared.ROOT, env=environment, stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError('Finite training stopped; no retry, extension or effect evaluation')


def checkpoint(step):
    return shared.RUN / ('milestone_' + str(step)) / f'model_step_{step:06d}.pt'


def verify_training(job):
    import torch
    import yaml
    from script.a3.checkpoint_store import validate_payload
    from script.a3.cloth_constrained_trainer import DECODER, initial_probability
    from script.a3.cloth_std_contract import canonical_policy_state
    from script.a3.finite_retention_protocol import validate_recipe, accept
    torch.set_num_threads(1)
    original = torch.load(shared.OFFICIAL, map_location='cpu', weights_only=False)
    official = canonical_policy_state(original['policy_state_dict'])
    records = []
    for step in (0, *CHECKPOINTS):
        path = checkpoint(step); record = shared.read(path.with_suffix('.json'))
        if record['sha256'] != sha256(path) or record['bytes'] != path.stat().st_size:
            raise ValueError('Checkpoint identity changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != step:
            raise ValueError('Saved step mismatch')
        actor = payload['policy_state_dict']
        if actor.keys() != official.keys():
            raise ValueError('Actor key set changed')
        names = [n for n, v in actor.items() if not torch.equal(v, official[n])]
        if step == 0:
            critic = payload['value_state_dict']; target = original['value_state_dict']
            if names or payload['optimizer_state_dict']['state'] or critic.keys() != target.keys() or any(
                    not torch.equal(v, target[n]) for n, v in critic.items()):
                raise ValueError('Not a fresh official Actor/Critic/Adam warm start')
        else:
            if not names or any(not n.startswith(DECODER) for n in names):
                raise ValueError('Actor changes escaped the selected decoder')
            if {int(v['step']) for v in payload['optimizer_state_dict']['state'].values()} != {4 * step}:
                raise ValueError('Adam moment counters do not match commits')
        records.append(dict(record, path=str(path.relative_to(shared.ROOT)), changed_actor_names=names))
        del payload, actor
    directory = shared.LOG / 'training'
    finite = shared.read(directory / 'finite_contract.json')
    contract = shared.read(directory / 'cloth_contract.json')
    batches = shared.read(directory / 'sampled_token_contract.json')['batches']
    audit = [json.loads(l) for l in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    transactions = [json.loads(l) for l in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    if (finite['proposals'] != STEPS or finite['committed'] != STEPS or len(finite['rows']) != STEPS
            or len(transactions) != 2 * STEPS or len(audit) != STEPS or len(batches) != STEPS
            or len(contract['optimizer_steps']) != STEPS or
            [r['attempt_optimizer_step'] for r in audit] != list(range(1, STEPS + 1))):
        raise ValueError('Incomplete update/transaction/batch coverage')
    for i, row in enumerate(finite['rows']):
        candidates = row['candidates']
        pending, committed = transactions[2*i:2*i+2]
        if (row != committed or pending['status'] != 'ACCEPTED_PENDING_COMMIT' or
                committed['status'] != 'COMMITTED' or row['proposal'] != i+1 or
                pending['parameters_and_optimizer_now_sha256'] != pending['parameters_and_optimizer_before_sha256']
                or pending['candidates'] != candidates or
                [r['scale'] for r in candidates] != list(SCALES[:len(candidates)]) or
                not candidates[-1]['accepted'] or any(r['accepted'] for r in candidates[:-1]) or
                not accept(row['before']['objective'], candidates[-1])):
            raise ValueError('Candidate acceptance/commit audit failed')
    if any(not r['decoder_gradient_enabled'] for r in batches) or any(
            r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5 for r in contract['optimizer_steps']):
        raise ValueError('Original teacher or decoder-gradient contract failed')
    repair = torch.load(directory / 'initial_token_repair.pt', map_location='cpu', weights_only=False)
    rollout = torch.load(directory / 'initial_sampled_tokens.pt', map_location='cpu', weights_only=False)
    batch = repair['batch']
    if not torch.equal(rollout['tokens'][batch['micro_batch_inds']], batch['mb_sampled_tokens']):
        raise ValueError('Actual sampled token identity changed')
    probability = initial_probability(repair['corrected'], batch)
    validate_recipe(yaml.safe_load((shared.RUN / 'config.yaml').read_text()),
                    yaml.safe_load((OLD_RUN / 'config.yaml').read_text()))
    routes = shared.read(directory / 'reference_route_coverage.json')['rollouts']
    if len(routes) != UPDATES or sum(sum(r['routes'].values()) for r in routes) != TRANSITIONS:
        raise ValueError('Actual rollout budget differs')
    atomic_json(OUT / 'training_verification.json', dict(status='VERIFIED_TRAINING_NOT_EFFECT',
        checkpoints=records, optimizer_steps=STEPS, initial_probability=probability,
        finite_contract=finite, original_contract=contract, actual_optimizer_steps=audit,
        routes=routes, recipe_unchanged_except_registered_storage_and_optimizer_mechanism=True,
        deploy_allowed=False, final_effect_step=UPDATES))


def evaluate(kind):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    args = arguments(OUT / ('evaluation_' + kind), shared.LOG / ('evaluation_' + kind),
                     checkpoint(UPDATES), OUT / (kind + '_references'))
    if kind == 'official':
        args.frame_stride = 4
    evaluation.preflight = shared.guard.preflight
    evaluation.monitor_child = lambda child, log, config: (child.wait(), None)
    evaluation.subprocess = LaunchProxy(evaluation.subprocess)
    if evaluation.evaluate_explicit(args):
        raise RuntimeError('Final matched evaluation failed; no retry')


def worker(stage):
    state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
    if (state['status'] != stage.upper() + '_RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E36_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Requires registered guarded monitor')
    shared.unchanged(job)
    if stage == 'prepare': prepare(job)
    elif stage == 'train': train(job)
    elif stage == 'verify-training': verify_training(job)
    elif stage.startswith('evaluate-'): evaluate(stage.split('-')[1])
    else: shared.verify_evaluation(job)
    shared.unchanged(job)


def run(stage):
    runtime.lightweight_controller()
    state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
    if state['status'] != shared.STAGES[stage][0]:
        raise RuntimeError('No repeat, reset or out-of-order stage')
    for row in preflight(job):
        append_record(shared.LOG / (stage + '_entry_preflight.jsonl'), row)
    return runtime.run(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='prepare')
    args = parser.parse_args()
    if args.action == 'plan':
        runtime.lightweight_controller()
        print(json.dumps(dict(status='PLAN_ONLY_NOT_EXECUTED', limits=LIMITS, scales=SCALES,
                              stages=shared.STAGES, source='FRESH_OFFICIAL_035')))
    else:
        code = register() if args.action == 'register' else worker(args.stage) if args.action == 'worker' else run(args.stage)
        raise SystemExit(code if isinstance(code, int) else 0)
