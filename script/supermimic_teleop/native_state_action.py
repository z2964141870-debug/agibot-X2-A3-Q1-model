"""One bounded64-call GPU action-parity gate after234-state CPU equivalence."""

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
from gpu_preview_worker import GpuPolicyProcess
from interim_review import TRAINING, fixed_rows
from native_state_action_math import CALLS_PER_CLIP, action_parity, saved_prefix
from native_state_review import unique
from paused_preview import checkpoint, gpu_command
from paused_runtime import assert_owner, compute_pids, own_gpu
from preview_review import environment
from runtime_guard import allocated_tree, guarded_run
from stream_events import EventSpec, EventTimeline
from student_causal_review import validate_causal_timing
from student_cost import LIMIT, charged, checked, reserve, settle
from student_input_audit import resources
from student_same_state import DiagnosticBudget, profile_summary
from student_same_state_math import CLIPS, SELECTED_SHA, cleanup_confirmed

DATA = ROOT/'data/supermimic_for_q1'
CPU = DATA/'Q1_LOW_LATENCY_CKPT32000_CPU_STATE01_R2_20261108'
SOURCE = DATA/'Q1_LOW_LATENCY_CKPT32000_SAME_STATE01_20261108'
OUTPUT = DATA/'Q1_LOW_LATENCY_CKPT32000_CPU_ACTION01_20261108'
SELECTED = DATA/'Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_20261107/selected.pt'
LEDGER = ROOT/'logs/session_records/q1_low_latency_gpu_cost_20261107.json'
FILES = ('native_state_action.py', 'native_state_action_worker.py', 'native_state_action_math.py',
         'test_native_state_action.py', 'native_state_cpu.py')
ARRAY_SHAS = ('ddc88d4b9a5bafa5e17c10e8b561586f857c611a1edb8f7e4c7231752287f515',
              '83e44f223ddca7213c0465bab9bc02b210983831447ca96c5e814a0198e8e014')


def admission(args):
    require(args.output == OUTPUT and not OUTPUT.exists(), 'Use one fresh independent CPU_ACTION01 root')
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set('0123456789abcdef'),
            'Publish the independent action gate first')
    sample = resources(True)
    cpu_path = CPU/'result.json'
    require(identity(cpu_path)['sha256'] == '9e37e36607bc01ddfdfc5419ebe3fb4b0d7d5e37534d8260ef7a4f7de1b44fd8',
            'CPU equivalence result changed')
    cpu = read(cpu_path)
    require(cpu['status'] == 'CPU_STATE_EQUIVALENCE_PASSED' and cpu['states'] == 234
        and cpu['source_integrity'] == 'PASSED' and cpu['ledger_unchanged'] is True
        and cpu['policy_action_equivalence_checked'] is False, 'Require the completed CPU input gate')
    require(identity(args.manifest)['sha256'] == args.manifest_sha256, 'Published action manifest changed')
    manifest = read(args.manifest)
    require(manifest['schema'] == 'q1_cpu_action_parity_v1' and len(manifest['files']) == len(FILES)
        and {r['path'] for r in manifest['files']} == {str(HERE/n) for n in FILES}, 'Unexpected action source scope')
    ledger = checked(read(LEDGER))
    require(charged(ledger) == 2433 and not any(r['state'] == 'RESERVED' for r in ledger['runs'])
        and charged(ledger)+300 <= LIMIT, 'Resolve the existing ledger; do not reset its2433 seconds')
    prior = cpu['source_inputs']+cpu['runtime_sources']
    require(all(identity(r['path']) == r for r in prior), 'Prior CPU input or runtime source changed')
    immutable = [r for r in prior if r['path'] != str(LEDGER)]
    new_paths = {str(HERE/n) for n in FILES if n != 'native_state_cpu.py'}
    require(not new_paths & {r['path'] for r in immutable}, 'New action code entered protected sources')
    arrays = [identity(CPU/(clip+'.parity.npz')) for clip in CLIPS]
    require([r['sha256'] for r in arrays] == list(ARRAY_SHAS), 'Verified CPU raw arrays changed')
    more = [identity(cpu_path), identity(args.manifest), identity(SOURCE/'authorization.json'), identity(SELECTED)]
    for clip in CLIPS:
        more += [identity(SOURCE/(clip+'_coverage_f0'+suffix))
                 for suffix in ('.summary.json', '.actions.npz', '.frames.jsonl')]
    protected = unique(immutable+arrays+more+manifest['files'])
    require(identity(SELECTED)['sha256'] == SELECTED_SHA and all(identity(r['path']) == r for r in protected),
            'Selected update500 or independent source changed')
    require(allocated_tree(ROOT)['unique_allocated_bytes']+64*(1 << 20) < 500_000_000_000,
            'Private project500GB bound')
    return sample,protected


def run_clip(args):
    import numpy as np
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    auth = read(args.output/'authorization.json')
    require(all(identity(r['path']) == r for r in auth['protected_inputs']), 'Admitted sources changed before child')
    config, runtime = read(args.config), read(args.runtime_config)
    saved = next(r for r in read(SOURCE/'authorization.json')['saved_prefixes'] if r['case']['clip'] == args.clip)
    with np.load(saved['trace']['path'], allow_pickle=False) as trace, \
            np.load(CPU/(args.clip+'.parity.npz'), allow_pickle=False) as cpu:
        actual, expected = saved_prefix(trace['actual'], trace['canonical_action'], cpu['actual'])
        reference_frames = len(trace['reference'])
    with np.load(SOURCE/(args.clip+'_coverage_f0.actions.npz'), allow_pickle=False) as previous:
        require(np.array_equal(previous['canonical_action'][:CALLS_PER_CLIP], expected), 'Saved original actions disagree')
    old = read(SOURCE/(args.clip+'_coverage_f0.summary.json'))
    selected, baseline = checkpoint(args, config)
    plan = read(TRAINING/'e1024_train_plan.json')
    argv = gpu_command(args, config, runtime, plan, selected, baseline, 0, 'masked')
    argv[2] = str(HERE/'native_state_action_worker.py')
    argv += ['--event-kind','nominal','--student-checkpoint',str(SELECTED)]
    item = next(r for r in fixed_rows(read(config['frozen_data'])) if Path(r['q1_primebot']['path']).stem == args.clip)
    start = time.perf_counter_ns()
    actor = GpuPolicyProcess(argv, args.output/(args.clip+'.policy.log'), config['expected_gpu'])
    spawn_ns = time.perf_counter_ns()-start
    rows = []
    maximum = 0.
    try:
        require(actor.ready.get('cpu_converter') == 'isolated_mujoco_kinematics'
            and actor.ready['student_checkpoint'] == identity(SELECTED)
            and actor.ready['checkpoint'] == old['ready']['checkpoint']
            and actor.ready['metadata'] == old['ready']['metadata']
            and actor.ready['canonical_present'] == old['ready']['canonical_present'], 'Wrong policy or converter')
        started = actor.start(item, 0, 'human')
        require(started['paired_reference'] == saved['case']['original_paired_reference'], 'Reference pairing changed')
        timeline = EventTimeline(reference_frames, EventSpec('nominal', 50, 10, 0))
        present = np.asarray(actor.ready['canonical_present']).reshape(29)
        with (args.output/(args.clip+'.frames.jsonl')).open('x') as stream:
            for frame, state in enumerate(actual):
                before = time.perf_counter_ns()
                reply = actor.action(frame, state)
                rpc = time.perf_counter_ns()-before
                error = action_parity(reply['canonical_action'], expected[frame], present)
                require(len(reply['native_action']) == 22 and np.isfinite(reply['native_action']).all(), 'Invalid native action')
                validate_causal_timing(reply['stream'], frame, timeline)
                profile = reply['profile']
                require(sum(profile['stage_wall_ns'].values()) == profile['worker_action_wall_ns']
                    and profile['cpu_converter'] == 'isolated_mujoco_kinematics', 'Timing partition or candidate changed')
                maximum = max(maximum, error)
                row = dict(frame=frame,canonical_action=reply['canonical_action'],saved_action_maximum_difference=error,
                    local_pipe_rpc_ns=rpc,stream=reply['stream'],profile=profile)
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                rows.append(row)
        actor.close()
        require(actor.child.returncode == 0, 'GPU action worker did not exit cleanly')
    finally:
        actor.close(graceful=False)
    save(args.output/(args.clip+'.summary.json'),dict(status='CPU_STATE_ACTION_PARITY_PASSED',
        clip=args.clip,calls=len(rows),maximum_absolute_action_difference=maximum,ready=actor.ready,
        fresh_worker_spawn_to_ready_ns=spawn_ns,timing=profile_summary(rows),new_physical_steps=0,
        new_optimizer_updates=0,hardware_end_to_end_latency_ms=None,
        historical_worker_timing_used_as_contemporaneous_baseline=False))


def execute(args):
    sample, protected = admission(args)
    config,runtime,receipt = read(args.config),read(args.runtime_config),read(args.pause_receipt)
    with own_gpu() as owner:
        require(not compute_pids(), 'GPU must be idle before the independent action check')
        owner = dict(owner,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        args.output.mkdir(exist_ok=False)
        save(args.output/'checkpoint_audit.json', read(SOURCE/'checkpoint_audit.json'))
        save(args.output/'authorization.json',dict(stage='CPU_STATE_ACTION_PARITY',evaluation_commit=args.evaluation_commit,
            protected_inputs=protected,maximum_policy_calls=64,calls_per_clip=32,maximum_wall_seconds=300,
            maximum_output_bytes=64*(1 << 20),new_physical_steps=0,new_optimizer_updates=0,
            hardware_end_to_end_latency_ms=None,user_accepted_checkpoint=None))
        result = dict(stage='CPU_STATE_ACTION_PARITY',status='RUNNING',controller=owner,
            started_utc=now(),admission=sample,phases=[],new_physical_steps=0,new_optimizer_updates=0,
            user_accepted_checkpoint=None)
        budget = DiagnosticBudget(config, receipt, args.output, owner)
        budget.maximum_wall = 300
        requested = False
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        previous,reservation,guards = dict(os.environ),None,[]
        try:
            reservation = reserve(LEDGER,run_id=str(args.output),stage='r3',owner=owner,
                gpu=config['expected_gpu'],commit=args.evaluation_commit)
            result['cost_reservation'] = reservation
            budget.maximum_wall = min(300, reservation['reserved_seconds'])
            os.environ.update(environment(runtime))
            for clip in CLIPS:
                require(all(identity(r['path']) == r for r in protected), 'Protected sources changed before GPU phase')
                result['phase'] = clip
                save(args.output/'result.json', result)
                budget.phase_started = time.monotonic()
                cmd = [runtime['primebot_python'],'-u',str(HERE/'native_state_action.py'),'--child',
                    '--output',str(args.output),'--clip',clip,'--config',str(args.config),
                    '--runtime-config',str(args.runtime_config),'--gpu-owner-pid',str(owner['pid']),
                    '--gpu-owner-ticks',owner['start_ticks']]
                path = args.output/(clip+'.guard.json')
                guards.append(path)
                guard = guarded_run(cmd,ROOT,budget,path,args.output/(clip+'.log'),check_interval_s=.5,
                    grace_s=5,stop_requested=lambda:requested)
                result['phases'].append(dict(clip=clip,guard=identity(path)))
                require(guard['status'] == 'WORKER_COMPLETED' and guard['cleanup']['worker_reaped'] is True
                    and guard['cleanup']['worker_return_code'] == 0, 'Independent action worker failed: '+clip)
            cases = [read(args.output/(clip+'.summary.json')) for clip in CLIPS]
            require(sum(r['calls'] for r in cases) == 64 and all(identity(r['path']) == r for r in protected),
                    'Final action count or source integrity failed')
            result.update(status='CPU_STATE_ACTION_PARITY_PASSED',source_integrity='PASSED',cases=cases,
                new_policy_calls=64,hardware_end_to_end_latency_ms=None,natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status='FAILED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            values = [read(p) for p in guards if p.is_file()]
            if reservation and cleanup_confirmed(values, len(guards)):
                result['cost_settlement'] = settle(LEDGER,run_id=str(args.output),owner=owner,
                    cleanup=dict(worker_reaped=True,worker_return_code=0 if result['status'] == 'CPU_STATE_ACTION_PARITY_PASSED' else -1,
                        worker_started=bool(values),owned_phase_workers_reaped=len(values)))
            result.update(ended_utc=now(),cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/'result.json', result)
            os.environ.clear(); os.environ.update(previous)
            for signum, handler in handlers.items():signal.signal(signum, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name,default in dict(config=ROOT/'script/q1_five_stage/conditional_shared_v04_20261104.json',
            runtime_config=ROOT/'data/manifests/primebot_q1_monitored_config_20261103.json',
            pause_receipt=ROOT/'logs/session_records/q1_shared_user_pause_20261106.json').items():
        p.add_argument('--'+name.replace('_','-'),type=Path,default=default)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--manifest-sha256')
    p.add_argument('--evaluation-commit')
    p.add_argument('--admit-only',action='store_true')
    p.add_argument('--child',action='store_true')
    p.add_argument('--clip',choices=CLIPS)
    p.add_argument('--gpu-owner-pid',type=int)
    p.add_argument('--gpu-owner-ticks')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.child:
        run_clip(args)
    elif args.admit_only:
        sample, records = admission(args)
        with own_gpu():
            require(not compute_pids(), 'Unexpected GPU owner')
        print(json.dumps(dict(status='CPU_ACTION_ADMITTED',protected_files=len(records),resources=sample,new_policy_calls=0)))
    else:
        execute(args)
