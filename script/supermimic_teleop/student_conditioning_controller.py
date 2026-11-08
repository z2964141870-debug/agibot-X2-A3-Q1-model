"""Own the original GPU lock and cost ledger for one causal conditioning pair."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'q1_five_stage'), str(HERE.parent/'primebot_q1')]
from common import ROOT, identity, now, require, save
from delivery import read
from interim_review import TRAINING
from paused_runtime import own_gpu, paused_state
from preview_review import environment
from runtime_guard import allocated_tree, guarded_run
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_retained_sequence_training import SequenceBudget
from student_same_state import unique

DATA = ROOT/'data/supermimic_for_q1'
PRIOR = DATA/'Q1_LOW_LATENCY_CKPT32000_F0_TRAIN_OBJECTIVE_AUDIT01_20261108'
CHECKPOINT_AUDIT = DATA/'Q1_LOW_LATENCY_CKPT32000_F0_RETAINED_SEQUENCE_PAIR01_20261108/checkpoint_audit.json'
OUTPUT = DATA/'Q1_LOW_LATENCY_CKPT32000_F0_HUMAN_CONDITIONING_PAIR01_20261108'
PLAN = HERE/'student_conditioning_plan_20261108.json'
LEDGER = ROOT/'logs/session_records/q1_low_latency_gpu_cost_20261107.json'
FILES = ('student_conditioning_controller.py', 'student_conditioning_worker.py', 'student_conditioning_models.py',
    'student_conditioning_fit.py', 'student_conditioning_plan_20261108.json', 'test_student_conditioning.py')
WALL, OUTPUT_BYTES = 600, 128*(1 << 20)


def admission(args):
    require(args.output == OUTPUT and not OUTPUT.exists(), 'Use the single-use conditioning-pair root')
    sample = resources(True)
    paused_state(read(args.config), read(args.pause_receipt))
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set('0123456789abcdef')
        and identity(args.manifest)['sha256'] == args.manifest_sha256, 'Require the exact published code manifest')
    manifest, plan = read(args.manifest), read(PLAN)
    require(manifest.get('schema') == 'q1_f0_human_conditioning_code_v1'
        and manifest['evaluation_commit'] == args.evaluation_commit and len(manifest['files']) == len(FILES)
        and {r['path'] for r in manifest['files']} == {str(HERE/n) for n in FILES}, 'New source scope differs')
    require(plan.get('schema') == 'q1_f0_human_conditioning_plan_v1'
        and plan['arms'] == ['zero_human', 'real_human']
        and plan['fit'] == dict(seed=2026110806, maximum_updates=3000, windows_per_distribution=16,
            learning_rate=.0001, gradient_norm_limit=1., validation_every_updates=500, retention_weight=20.)
        and plan['human_real_future_frames'] == 0 and plan['heldout_events_opened'] is False
        and plan['resource']['maximum_wall_seconds'] == WALL
        and plan['resource']['maximum_output_bytes'] == OUTPUT_BYTES, 'Registered single-variable mechanism changed')
    require(identity(plan['base_plan']['path']) == plan['base_plan']
        and identity(plan['sequence_summary']['path']) == plan['sequence_summary'], 'Saved base plan or sequence audit changed')
    prior, guard = read(PRIOR/'result.json'), read(PRIOR/'audit.guard.json')
    require(prior['status'] == 'TRAIN_ONLY_OBJECTIVE_AUDIT_PASSED' and prior['source_integrity'] == 'PASSED'
        and prior['cost_settlement']['state'] == 'SETTLED' and prior['cumulative_gpu_charged_seconds'] == 4820
        and prior['actual_optimizer_updates'] == 0
        and guard['cleanup']['worker_reaped'] is True and guard['cleanup']['worker_return_code'] == 0
        and not Path('/proc', str(prior['controller']['pid'])).exists()
        and not Path('/proc', str(guard['cleanup']['worker_pid'])).exists(), 'Wait for the completed train-only audit cleanup')
    base = read(plan['base_plan']['path'])
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt, args.manifest,
        PRIOR/'result.json', PRIOR/'audit.guard.json', PRIOR/'authorization.json', PRIOR/'summary.json',
        CHECKPOINT_AUDIT, base['initial_checkpoint']['path'], base['initial_fit']['path'])]
    protected = unique(read(PRIOR/'authorization.json')['protected_inputs']+controls+manifest['files']
        +[plan['base_plan'], plan['sequence_summary']])
    require(str(LEDGER) not in {r['path'] for r in protected}
        and all(identity(r['path']) == r for r in protected), 'Old or published protected inputs changed')
    initial_fit = read(base['initial_fit']['path'])
    require(identity(base['initial_checkpoint']['path']) == base['initial_checkpoint']
        and initial_fit['selected_update'] == 3000 and initial_fit['selected_adapter_is_zero'] is False,
        'Keep the actual nonzero robot-history residual')
    ledger = checked(read(LEDGER))
    by_stage = {s:charged(dict(ledger, runs=[r for r in ledger['runs'] if r['stage'] == s])) for s in STAGES}
    require(charged(ledger) == plan['resource']['expected_cumulative_seconds'] == 4820
        and by_stage == plan['resource']['expected_by_stage']
        and not any(r['state'] == 'RESERVED' for r in ledger['runs'])
        and STAGES['r2']-by_stage['r2'] >= WALL and charged(ledger)+WALL <= LIMIT, 'Original cost ledger must not reset')
    require(allocated_tree(ROOT)['unique_allocated_bytes']+OUTPUT_BYTES < 500_000_000_000, 'Project500GB bound')
    with own_gpu():
        pass
    return sample, protected, plan


def execute(args):
    sample, protected, plan = admission(args)
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    with own_gpu() as lock:
        owner = dict(lock, boot=config['expected_boot'], gpu=config['expected_gpu'])
        args.output.mkdir(exist_ok=False)
        save(args.output/'checkpoint_audit.json', read(CHECKPOINT_AUDIT))
        budget = SequenceBudget(config, receipt, args.output, owner)
        budget.maximum_wall = WALL
        budget.check()
        previous, requested, reservation = dict(os.environ), False, None
        result = dict(stage='F0_HUMAN_CONDITIONING_PAIR', status='RUNNING', controller=owner, started_utc=now(),
            new_physical_steps=0, new_rendered_frames=0, user_accepted_checkpoint=None)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/'fit.guard.json'
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage='r2', owner=owner,
                gpu=config['expected_gpu'], commit=args.evaluation_commit)
            budget.maximum_wall = min(WALL, reservation['reserved_seconds']-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/'authorization.json', dict(stage=result['stage'], evaluation_commit=args.evaluation_commit,
                protected_inputs=protected, gpu_lock_owner=owner, conditioning_plan=identity(PLAN),
                maximum_wall_seconds=budget.maximum_wall, maximum_output_bytes=OUTPUT_BYTES,
                maximum_total_optimizer_updates=6000, maximum_human_future_frames=0,
                normalization_refitted=False, cache_reclaim_attempted=False, original_training_restarted=False,
                final_acceptance=False, user_accepted_checkpoint=None))
            selected = next(r['checkpoint'] for r in read(CHECKPOINT_AUDIT)['records'] if r['robot'] == 'q1')
            original = read(TRAINING/'e1024_train_plan.json')
            values = dict(plan=TRAINING/'e1024_train_plan.json', checkpoint=selected['path'],
                morph_source=original['source'], output=args.output, conditioning_plan=PLAN,
                expected_gpu=config['expected_gpu'], gpu_owner_pid=owner['pid'],
                gpu_owner_ticks=owner['start_ticks'], deadline_ns=deadline)
            command = [runtime['morph_python'], '-u', str(HERE/'student_conditioning_worker.py')]
            for key,value in values.items():
                command.extend(('--'+key.replace('_', '-'), str(value)))
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config['expected_gpu'])
            save(args.output/'result.json', result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/'fit.log',
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard['status'] == 'WORKER_COMPLETED' and guard['cleanup']['worker_reaped'] is True
                and guard['cleanup']['worker_return_code'] == 0, 'Bounded conditioning fit failed')
            fit = read(args.output/'fit_result.json')
            require(fit['status'] == 'HUMAN_CONDITIONING_PAIR_TRAINED_PENDING_CLOSED_LOOP'
                and fit['actual_optimizer_updates'] == 6000
                and [r['variant'] for r in fit['arms']] == plan['arms']
                and fit['matched_initialization_capacity_and_source_slots'] is True
                and fit['initial_action_audit']['status'] == 'CONDITIONING_INITIAL_ACTION_PARITY_PASSED'
                and all(identity(r['path']) == r for r in protected), 'Actual pair or source terminal gate failed')
            result.update(status=fit['status'], source_integrity='PASSED', actual_optimizer_updates=6000,
                arms=[{k:r[k] for k in ('variant', 'selected_update', 'selected_policy_unchanged',
                    'initial_validation_risk', 'selected_validation_risk')} for r in fit['arms']],
                fit_result=identity(args.output/'fit_result.json'), natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if reservation and guard_path.exists() and read(guard_path).get('cleanup', {}).get('worker_reaped') is True:
                result['cost_settlement'] = settle(LEDGER, run_id=str(args.output), owner=owner,
                    cleanup=read(guard_path)['cleanup'])
            result.update(ended_utc=now(), cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/'result.json', result)
            os.environ.clear()
            os.environ.update(previous)
            for signum,handler in handlers.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name,default in dict(config=ROOT/'script/q1_five_stage/conditional_shared_v04_20261104.json',
            runtime_config=ROOT/'data/manifests/primebot_q1_monitored_config_20261103.json',
            pause_receipt=ROOT/'logs/session_records/q1_shared_user_pause_20261106.json').items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=default)
    for name in ('output', 'manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('manifest-sha256', 'evaluation-commit'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--admit-only', action='store_true')
    args = parser.parse_args()
    if args.admit_only:
        sample,protected,plan = admission(args)
        print(json.dumps(dict(status='F0_HUMAN_CONDITIONING_PAIR_ADMITTED', protected_files=len(protected), resources=sample)))
    else:
        execute(args)
