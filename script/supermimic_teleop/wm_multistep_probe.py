"""Single-use, guarded FP32 WM diagnosis; no optimizer or new physical replay."""

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
from wm_physical_training import precision_evidence

PLAN = HERE/'wm_multistep_plan_20261108.json'
LEDGER = ROOT/'logs/session_records/q1_low_latency_gpu_cost_20261107.json'
FILES = ('wm_multistep_probe.py','wm_multistep_worker.py','wm_multistep_math.py',
    'wm_multistep_plan_20261108.json','test_wm_multistep.py')
WALL, OUTPUT_BYTES = 600, 64*(1 << 20)


def admission(args):
    settings = read(PLAN)
    require(settings['schema'] == 'q1_frozen_fp32_wm_multistep_v1'
        and args.output == Path(settings['output']) and not args.output.exists()
        and settings['horizons'] == [1,2,4,8,16,32,40]
        and settings['common_support_steps'] == [80,51,47,15,80,80,37,2]
        and settings['expected_wm_step_calls'] == 2046 and settings['maximum_wm_step_calls'] == 2048
        and settings['maximum_backward_calls'] == 14 and settings['maximum_wall_seconds'] == WALL
        and settings['maximum_output_bytes'] == OUTPUT_BYTES
        and settings['new_physical_steps'] == settings['new_optimizer_updates'] == settings['new_policy_calls'] == 0
        and settings['test_sources_read'] is False and settings['automatic_training_after_diagnostic'] is False,
        'Use the registered single-use bounded diagnostic')
    sample = resources(True)
    paused_state(read(args.config),read(args.pause_receipt))
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set('0123456789abcdef')
        and identity(args.manifest)['sha256'] == args.manifest_sha256, 'Published exact manifest required')
    manifest = read(args.manifest)
    require(manifest.get('schema') == 'q1_wm_multistep_code_v1'
        and manifest['evaluation_commit'] == args.evaluation_commit and len(manifest['files']) == len(FILES)
        and {r['path'] for r in manifest['files']} == {str(HERE/n) for n in FILES}, 'Execution source scope differs')
    recovery, conditioning = Path(settings['recovery_root']),Path(settings['conditioning_root'])
    require(identity(recovery/'recovery_result.json') == settings['required_recovery_result'], 'Original recovery evidence changed')
    protected = precision_evidence(Path(settings['precision_root']),settings['required_precision_result'])
    controls = [identity(p) for p in (args.config,args.runtime_config,args.pause_receipt,args.manifest,PLAN,
        HERE/'wm_probe_plan_20261107.json')]
    for root,guard_name,status in ((recovery,'recovery.guard.json','COMPLETED'),
            (conditioning,'fit.guard.json','HUMAN_CONDITIONING_PAIR_TRAINED_PENDING_CLOSED_LOOP')):
        result,guard,auth = (read(root/n) for n in ('result.json',guard_name,'authorization.json'))
        require(result['status'] == status and result['source_integrity'] == 'PASSED'
            and result['cost_settlement']['state'] == 'SETTLED' and guard['status'] == 'WORKER_COMPLETED'
            and guard['cleanup']['worker_reaped'] is True and guard['cleanup']['worker_return_code'] == 0
            and not Path('/proc',str(result['controller']['pid'])).exists()
            and not Path('/proc',str(guard['cleanup']['worker_pid'])).exists(), 'Prior task must actually be settled and reaped')
        protected += auth['protected_inputs']
        controls += [identity(root/n) for n in ('result.json',guard_name,'authorization.json','checkpoint_audit.json')]
    forks = read(recovery/'recovery_result.json')
    require(len(forks['cases']) == 16 and all(r['split'] in ('train','validation') for r in forks['cases']),
        'Only the original16 explored forks are admitted')
    controls += [settings['required_recovery_result']]+[r['trace'] for r in forks['cases']]
    controls += read(HERE/'wm_probe_plan_20261107.json')['context_sources']
    require(not {str(HERE/n) for n in FILES}&{r['path'] for r in protected}, 'New code overlaps frozen sources')
    protected = unique(protected+controls+manifest['files'])
    require(str(LEDGER) not in {r['path'] for r in protected}
        and all(identity(r['path']) == r for r in protected), 'Protected source identity changed')
    ledger = checked(read(LEDGER))
    stages = {s:charged(dict(ledger,runs=[r for r in ledger['runs'] if r['stage'] == s])) for s in STAGES}
    require(charged(ledger) == settings['expected_cumulative_seconds'] == 5036
        and stages == settings['expected_by_stage'] == dict(r1=1614,r2=2051,r3=1371)
        and not any(r['state'] == 'RESERVED' for r in ledger['runs'])
        and STAGES['r3']-stages['r3'] >= WALL and charged(ledger)+WALL <= LIMIT, 'Do not reset the actual cost ledger')
    require(allocated_tree(ROOT)['unique_allocated_bytes']+OUTPUT_BYTES < 500_000_000_000, 'Project500GB bound')
    with own_gpu():
        pass
    return sample,protected,settings


class MultistepBudget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)['unique_allocated_bytes'] < OUTPUT_BYTES, 'Output64MiB bound')
        return super().check()


def execute(args):
    sample,protected,settings = admission(args)
    config,runtime,receipt = read(args.config),read(args.runtime_config),read(args.pause_receipt)
    with own_gpu() as lock:
        owner = dict(lock,boot=config['expected_boot'],gpu=config['expected_gpu'])
        args.output.mkdir(exist_ok=False)
        save(args.output/'checkpoint_audit.json',read(Path(settings['recovery_root'])/'checkpoint_audit.json'))
        budget = MultistepBudget(config,receipt,args.output,owner)
        budget.maximum_wall = WALL
        budget.check()
        previous,requested,reservation = dict(os.environ),False,None
        result = dict(stage='FROZEN_FP32_WM_MULTISTEP',status='RUNNING',controller=owner,started_utc=now(),
            new_physical_steps=0,actual_optimizer_updates=0,user_accepted_checkpoint=None)
        def stop(signum,frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        guard_path = args.output/'multistep.guard.json'
        try:
            reservation = reserve(LEDGER,run_id=str(args.output),stage='r3',owner=owner,
                gpu=config['expected_gpu'],commit=args.evaluation_commit)
            budget.maximum_wall = min(WALL,reservation['reserved_seconds']-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/'authorization.json',dict(stage=result['stage'],evaluation_commit=args.evaluation_commit,
                protected_inputs=protected,gpu_lock_owner=owner,maximum_wall_seconds=budget.maximum_wall,
                maximum_output_bytes=OUTPUT_BYTES,maximum_wm_step_calls=2048,maximum_backward_calls=14,
                cache_reclaim_attempted=False,new_physical_steps=0,new_optimizer_updates=0,
                original_training_restarted=False,final_acceptance=False,user_accepted_checkpoint=None))
            command = [runtime['morph_python'],'-u',str(HERE/'wm_multistep_worker.py')]
            values = dict(output=args.output,multistep_plan=PLAN,runtime_config=args.runtime_config,
                config=args.config,plan=TRAINING/'e1024_train_plan.json',expected_gpu=config['expected_gpu'],
                gpu_owner_pid=owner['pid'],gpu_owner_ticks=owner['start_ticks'],deadline_ns=deadline)
            for key,value in values.items():
                command.extend(('--'+key.replace('_','-'),str(value)))
            os.environ.update(environment(runtime),CUDA_VISIBLE_DEVICES=config['expected_gpu'])
            save(args.output/'result.json',result)
            guard = guarded_run(command,ROOT,budget,guard_path,args.output/'multistep.log',
                check_interval_s=.5,grace_s=5,stop_requested=lambda:requested)
            require(guard['status'] == 'WORKER_COMPLETED' and guard['cleanup']['worker_reaped'] is True
                and guard['cleanup']['worker_return_code'] == 0, 'Bounded multi-step diagnostic failed')
            value = read(args.output/'multistep_result.json')
            require(value['status'] == 'WM_MULTISTEP_COMPLETED' and value['source_integrity'] == 'PASSED'
                and value['cuda_wm_step_forward_calls'] == 2046 and value['cuda_backward_calls'] == 14
                and value['actual_optimizer_updates'] == value['new_physical_steps'] == 0
                and all(identity(r['path']) == r for r in protected), 'Incomplete result or changed protected inputs')
            result.update(status=value['status'],source_integrity='PASSED',multistep_result=identity(args.output/'multistep_result.json'),
                diagnostic_screen='PASSED' if value['screen']['all_requested_screens_passed'] else 'FAILED',
                cuda_wm_step_forward_calls=2046,cuda_backward_calls=14,natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status='FAILED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            try:
                if reservation and guard_path.exists() and read(guard_path).get('cleanup',{}).get('worker_reaped') is True:
                    result['cost_settlement'] = settle(LEDGER,run_id=str(args.output),owner=owner,cleanup=read(guard_path)['cleanup'])
                result.update(ended_utc=now(),cumulative_gpu_charged_seconds=charged(read(LEDGER)))
                save(args.output/'result.json',result)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                for signum,handler in handlers.items():
                    signal.signal(signum,handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name,default in dict(config=ROOT/'script/q1_five_stage/conditional_shared_v04_20261104.json',
            runtime_config=ROOT/'data/manifests/primebot_q1_monitored_config_20261103.json',
            pause_receipt=ROOT/'logs/session_records/q1_shared_user_pause_20261106.json').items():
        parser.add_argument('--'+name.replace('_','-'),type=Path,default=default)
    for name in ('output','manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','evaluation-commit'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--admit-only',action='store_true')
    args = parser.parse_args()
    if args.admit_only:
        sample,protected,settings = admission(args)
        print(json.dumps(dict(status='WM_MULTISTEP_ADMITTED',protected_files=len(protected),resources=sample)))
    else:
        execute(args)
