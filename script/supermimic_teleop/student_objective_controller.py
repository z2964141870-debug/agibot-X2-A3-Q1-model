"""Own the original lock and ledger for one bounded train-only gradient audit."""

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
from student_same_state import unique
from student_training import TrainingBudget

DATA = ROOT/'data/supermimic_for_q1'
PRIOR = DATA/'Q1_LOW_LATENCY_CKPT32000_F0_RETAINED_SEQUENCE_PAIR01_20261108'
OUTPUT = DATA/'Q1_LOW_LATENCY_CKPT32000_F0_TRAIN_OBJECTIVE_AUDIT01_20261108'
PLAN = HERE/'student_objective_plan_20261108.json'
LEDGER = ROOT/'logs/session_records/q1_low_latency_gpu_cost_20261107.json'
FILES = ('student_objective_controller.py', 'student_objective_worker.py', 'student_objective_audit.py',
    'student_objective_plan_20261108.json', 'test_student_objective_audit.py')
WALL, OUTPUT_BYTES = 300, 64*(1 << 20)


def admission(args):
    require(args.output == OUTPUT and not OUTPUT.exists(), 'Use the single-use objective audit root')
    sample = resources(True)
    paused_state(read(args.config), read(args.pause_receipt))
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set('0123456789abcdef')
        and identity(args.manifest)['sha256'] == args.manifest_sha256, 'Require the exact published code manifest')
    manifest, plan = read(args.manifest), read(PLAN)
    require(manifest.get('schema') == 'q1_f0_train_objective_audit_code_v1'
        and manifest['evaluation_commit'] == args.evaluation_commit and len(manifest['files']) == len(FILES)
        and {r['path'] for r in manifest['files']} == {str(HERE/n) for n in FILES}, 'New source scope differs')
    require(plan.get('schema') == 'q1_f0_train_objective_audit_plan_v1'
        and plan['probe'] == dict(seed=2026110806, minibatches=4, windows_per_distribution=16,
            retention_weight=20., increment_weight=.5, maximum_forward_batches=1000,
            maximum_forward_rows=30000, maximum_autograd_calls=84)
        and plan['resource']['maximum_wall_seconds'] == WALL
        and plan['resource']['maximum_output_bytes'] == OUTPUT_BYTES
        and all(plan[k] == 0 for k in ('validation_arrays_decoded', 'optimizer_updates', 'new_physical_steps', 'new_rendered_frames')),
        'Registered training-only diagnostic scope changed')
    require(identity(plan['base_plan']['path']) == plan['base_plan']
        and identity(plan['sequence_summary']['path']) == plan['sequence_summary']
        and all(identity(r['path']) == r for r in plan['last_checkpoints'].values()), 'Saved sources changed')
    prior, guard = read(PRIOR/'result.json'), read(PRIOR/'fit.guard.json')
    require(prior['status'] == 'RETAINED_SEQUENCE_PAIR_TRAINED_PENDING_CLOSED_LOOP'
        and prior['source_integrity'] == 'PASSED' and prior['cost_settlement']['state'] == 'SETTLED'
        and prior['cumulative_gpu_charged_seconds'] == 4790 and prior['actual_optimizer_updates'] == 6000
        and all(r['selected_update'] == 0 and r['selected_adapter_unchanged'] for r in prior['arms'])
        and guard['cleanup']['worker_reaped'] is True and guard['cleanup']['worker_return_code'] == 0
        and not Path('/proc', str(prior['controller']['pid'])).exists()
        and not Path('/proc', str(guard['cleanup']['worker_pid'])).exists(), 'Wait for the completed rejected pair cleanup')
    base = read(plan['base_plan']['path'])
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt, args.manifest,
        PRIOR/'result.json', PRIOR/'fit.guard.json', PRIOR/'authorization.json', PRIOR/'checkpoint_audit.json',
        PRIOR/'fit_result.json', base['initial_checkpoint']['path'], base['initial_fit']['path'])]
    protected = unique(read(PRIOR/'authorization.json')['protected_inputs']+controls+manifest['files']
        +[plan['base_plan'], plan['sequence_summary']]+list(plan['last_checkpoints'].values()))
    require(str(LEDGER) not in {r['path'] for r in protected}
        and all(identity(r['path']) == r for r in protected), 'Old or published protected inputs changed')
    ledger = checked(read(LEDGER))
    by_stage = {s:charged(dict(ledger, runs=[r for r in ledger['runs'] if r['stage'] == s])) for s in STAGES}
    require(charged(ledger) == plan['resource']['expected_cumulative_seconds'] == 4790
        and by_stage == plan['resource']['expected_by_stage']
        and not any(r['state'] == 'RESERVED' for r in ledger['runs'])
        and STAGES['r2']-by_stage['r2'] >= WALL and charged(ledger)+WALL <= LIMIT, 'Original cost ledger must not reset')
    require(allocated_tree(ROOT)['unique_allocated_bytes']+OUTPUT_BYTES < 500_000_000_000, 'Project500GB bound')
    with own_gpu():
        pass
    return sample, protected, plan


class AuditBudget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)['unique_allocated_bytes'] < OUTPUT_BYTES, 'Output64MiB bound')
        return super().check()


def execute(args):
    sample, protected, plan = admission(args)
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    with own_gpu() as lock:
        owner = dict(lock, boot=config['expected_boot'], gpu=config['expected_gpu'])
        args.output.mkdir(exist_ok=False)
        budget = AuditBudget(config, receipt, args.output, owner)
        budget.maximum_wall = WALL
        budget.check()
        previous, requested, reservation = dict(os.environ), False, None
        result = dict(stage='F0_TRAIN_OBJECTIVE_AUDIT', status='RUNNING', controller=owner, started_utc=now(),
            actual_optimizer_updates=0, new_physical_steps=0, new_rendered_frames=0, user_accepted_checkpoint=None)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/'audit.guard.json'
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage='r2', owner=owner,
                gpu=config['expected_gpu'], commit=args.evaluation_commit)
            budget.maximum_wall = min(WALL, reservation['reserved_seconds']-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/'authorization.json', dict(stage=result['stage'], evaluation_commit=args.evaluation_commit,
                protected_inputs=protected, gpu_lock_owner=owner, audit_plan=identity(PLAN),
                maximum_wall_seconds=budget.maximum_wall, maximum_output_bytes=OUTPUT_BYTES,
                maximum_forward_batches=1000, maximum_forward_rows=30000, maximum_autograd_calls=84,
                optimizer_updates=0, validation_arrays_decoded=0, original_training_restarted=False,
                cache_reclaim_attempted=False, final_acceptance=False, user_accepted_checkpoint=None))
            selected = next(r['checkpoint'] for r in read(PRIOR/'checkpoint_audit.json')['records'] if r['robot'] == 'q1')
            original = read(TRAINING/'e1024_train_plan.json')
            values = dict(plan=TRAINING/'e1024_train_plan.json', checkpoint=selected['path'],
                morph_source=original['source'], output=args.output, audit_plan=PLAN,
                expected_gpu=config['expected_gpu'], gpu_owner_pid=owner['pid'], gpu_owner_ticks=owner['start_ticks'],
                deadline_ns=deadline)
            command = [runtime['morph_python'], '-u', str(HERE/'student_objective_worker.py')]
            for key,value in values.items():
                command.extend(('--'+key.replace('_', '-'), str(value)))
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config['expected_gpu'])
            save(args.output/'result.json', result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/'audit.log',
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard['status'] == 'WORKER_COMPLETED' and guard['cleanup']['worker_reaped'] is True
                and guard['cleanup']['worker_return_code'] == 0, 'Bounded objective audit failed')
            summary = read(args.output/'summary.json')
            require(summary['status'] == 'TRAIN_ONLY_OBJECTIVE_AUDIT_PASSED' and summary['optimizer_updates'] == 0
                and len(summary['policies']) == 3 and summary['validation_arrays_decoded'] == 0
                and all(identity(r['path']) == r for r in protected), 'Actual diagnostic terminal gate failed')
            result.update(status=summary['status'], source_integrity='PASSED', counts=summary['counts'],
                summary=identity(args.output/'summary.json'), natural_teleoperation_qualified=False)
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
        print(json.dumps(dict(status='F0_TRAIN_OBJECTIVE_AUDIT_ADMITTED', protected_files=len(protected), resources=sample)))
    else:
        execute(args)
