"""One bounded CPU-only parity review on the234 existing saved robot states."""

import argparse
from contextlib import ExitStack
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'q1_five_stage'), str(HERE.parent/'primebot_q1')]
from common import ROOT, identity, now, require, save
from delivery import read
from memory_watch import process
from paused_runtime import own_gpu
from runtime_guard import allocated_tree
from student_cost import charged, checked
from student_input_audit import resources

DATA = ROOT/'data/supermimic_for_q1'
SOURCE = DATA/'Q1_LOW_LATENCY_CKPT32000_SAME_STATE01_20261108'
SUMMARY = DATA/'Q1_LOW_LATENCY_CKPT32000_SAME_STATE01_SAVED_ANALYSIS_20261108'
FIRST = DATA/'Q1_LOW_LATENCY_CKPT32000_CPU_STATE01_20261108'
OUTPUT = DATA/'Q1_LOW_LATENCY_CKPT32000_CPU_STATE01_R2_20261108'
LEDGER = ROOT/'logs/session_records/q1_low_latency_gpu_cost_20261107.json'
CLIPS = (('walk_forward_loop_001__A022', 128), ('walk_forward_start_002__A023', 106))
ATOL, RTOL = 2e-5, 2e-5


def unique(records):
    by_path = {}
    for r in records:
        require(r['path'] not in by_path or by_path[r['path']] == r, 'Conflicting protected identity')
        by_path[r['path']] = r
    return list(by_path.values())


def stats(values):
    return dict(samples=len(values), p50_ms=float(np.percentile(values, 50))/1e6,
        p95_ms=float(np.percentile(values, 95))/1e6, p99_ms=float(np.percentile(values, 99))/1e6,
        maximum_ms=float(max(values))/1e6, calls_over_20ms=int(np.sum(np.asarray(values) > 20_000_000)))


def observation(converted, spec, history, frame, default, present):
    import torch
    from morphmimic.model.rewards_torch import analytic_obs
    value = analytic_obs(converted, torch.zeros(29), default, converted[:, 240:], spec.imu_slot)
    obs = torch.cat((torch.where(present, value[:, :29], 0.),
        torch.where(present, value[:, 29:58], 0.), value[:, 58:64],
        torch.where(present, value[:, 64:], 0.)), -1)
    history.reset(None, obs) if frame == 0 else history.push(obs)
    return obs, history.flat()


def measure(robot, spec, traces, output, check):
    import mujoco
    import torch
    from morphmimic.train.obs_history import ObsHistory
    from native_state_cpu import KinematicState
    from q1_native_review import native_state
    candidate = KinematicState(robot, spec)
    default = spec.translator.to_canonical(torch.zeros(22))
    present = spec.translator.to_canonical_delta(torch.ones(22)) != 0
    model_fields = ('qpos0', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass',
                    'body_inertia', 'jnt_axis', 'jnt_pos', 'jnt_qposadr', 'jnt_type')
    model_before = {k:getattr(robot.model, k).copy() for k in model_fields}
    all_rows, results = [], []
    with ExitStack() as stack, torch.inference_mode():
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse'):
            stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No dynamics in CPU parity')))
        for clip, count in CLIPS:
            source = next(s for s in traces if s['case']['clip'] == clip)
            with np.load(source['trace']['path'], allow_pickle=False) as saved:
                actual = saved['actual'][:count].copy()
            require(actual.shape == (count, 79) and np.isfinite(actual).all(), 'Changed saved prefix')
            histories = {v:ObsHistory(1, 10, 93, 'cpu') for v in ('original', 'candidate')}
            rows, arrays = [], {k:[] for k in ('original', 'candidate', 'obs_original', 'obs_candidate',
                                              'history_original', 'history_candidate')}
            with (output/(clip+'.frames.jsonl')).open('x') as stream:
                try:
                    for frame, raw in enumerate(actual):
                        check() if frame % 32 == 0 else None
                        value = torch.from_numpy(raw.copy())[None]
                        saved_value = value.clone()
                        order = ('original', 'candidate') if frame % 2 == 0 else ('candidate', 'original')
                        times, converted = {}, {}
                        for variant in order:
                            begin = time.perf_counter_ns()
                            converted[variant] = native_state(robot, value, spec) if variant == 'original' else candidate(value)
                            times[variant] = time.perf_counter_ns()-begin
                            arrays[variant].append(converted[variant].numpy().copy()[0])
                            begin = time.perf_counter_ns()
                            obs, history = observation(converted[variant], spec, histories[variant], frame, default, present)
                            times[variant+'_observation_and_history'] = time.perf_counter_ns()-begin
                            arrays['obs_'+variant].append(obs.numpy().copy()[0])
                            arrays['history_'+variant].append(history.numpy().copy()[0])
                        differences = {k:float(np.max(np.abs(arrays[k+'original'][-1]-arrays[k+'candidate'][-1])))
                                       for k in ('', 'obs_', 'history_')}
                        row = dict(clip=clip, frame=frame, order=list(order), wall_ns=times,
                                   maximum_absolute_difference=differences)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        stream.flush()
                        rows.append(row); all_rows.append(row)
                        require(torch.equal(saved_value, value) and candidate.data.time == 0., 'Input or clock was mutated')
                        for start in (0, 29, 240):
                            require(torch.count_nonzero(converted['candidate'][:, start:start+29][:, ~present[0]]) == 0,
                                    'Missing canonical joint slot was populated')
                        require(torch.equal(converted['candidate'][:, 240:], spec.translator.to_canonical_delta(value[:, 57:])),
                                'Previous action memory changed')
                        for prefix in ('', 'obs_', 'history_'):
                            require(np.allclose(arrays[prefix+'original'][-1], arrays[prefix+'candidate'][-1],
                                    atol=ATOL, rtol=RTOL), 'Parity failed at '+clip+':'+str(frame)+':'+prefix)
                        require(np.max(np.abs(arrays['original'][-1][58:100]-arrays['candidate'][-1][58:100])) <= 3e-5,
                                'Absolute link-origin FK tolerance failed')
                finally:
                    with (output/(clip+'.parity.npz')).open('xb') as target:
                        np.savez_compressed(target, actual=actual[:len(rows)],
                                            **{k:np.asarray(v) for k, v in arrays.items()})
            stable = rows[2:]
            results.append(dict(clip=clip, states=count, trace=source['trace'],
                maximum_absolute_difference={k:max(r['maximum_absolute_difference'][k] for r in rows)
                                             for k in ('', 'obs_', 'history_')},
                timing={scope:{k:stats([r['wall_ns'][k] for r in subset]) for k in rows[0]['wall_ns']}
                        for scope, subset in (('all', rows), ('without_first_two', stable))}))
    for k, v in model_before.items():
        require(np.array_equal(v, getattr(robot.model, k)), 'Shared model changed: '+k)
    require(len(all_rows) == 234, 'Require exactly234 saved states without extra benchmark loops')
    return results


def execute(args):
    started = time.monotonic()
    require(args.output == OUTPUT and not args.output.exists() and len(args.evaluation_commit) == 40
        and all(c in '0123456789abcdef' for c in args.evaluation_commit), 'Use the new published CPU-only output')
    first = read(FIRST/'result.json')
    require(first.get('status') == 'FAILED' and first.get('error', '').startswith(
        'ImportError: Failed to initialize: Bad git executable.') and 'runtime_sources' not in first
        and not first.get('comparisons') and not Path('/proc', str(first['controller']['pid'])).exists(),
        'Only recover the stopped dependency failure before any state comparisons')
    admitted = resources(True)
    require(identity(args.manifest)['sha256'] == args.manifest_sha256, 'Published CPU manifest changed')
    manifest = read(args.manifest)
    require(manifest.get('schema') == 'q1_cpu_state_equivalence_v1'
        and len(manifest['files']) == 3 and manifest['base_commit'] == 'f466f3c750ca9ab63615a0f0c21b8d916050e354',
        'Unexpected independent CPU source manifest')
    require({r['path'] for r in manifest['files']} == {str(HERE/n) for n in
        ('native_state_cpu.py', 'native_state_review.py', 'test_native_state_cpu.py')}, 'Unexpected CPU file scope')
    require(identity(SUMMARY/'summary.json')['sha256'] == '3ee45edc112ac4a7c3cfcfd5f2d6de2ae2369ecac278a2360814a4eda6569bd7'
        and identity(SUMMARY/'audit.json')['sha256'] == '4213f449b3d063dbfc0f3f58fd191e093b7b1a2ce93f0dda2707c964d2fc7d1e',
        'The verified saved analysis changed')
    summary, audit = read(SUMMARY/'summary.json'), read(SUMMARY/'audit.json')
    require(audit['status'] == 'SAME_STATE_SAVED_ANALYSIS_PASSED' and audit['source_inputs_unchanged'] is True
        and summary['source_integrity'] == 'PASSED' and summary['reused_policy_calls'] == 936,
        'Require the already completed saved analysis')
    auth = read(SOURCE/'authorization.json')
    runtime = read(ROOT/'data/manifests/primebot_q1_monitored_config_20261103.json')
    require(Path(sys.executable) == Path(runtime['morph_python']), 'Use the actual saved policy runtime')
    from preview_review import environment
    os.environ.update(environment(runtime))
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU audit exposed CUDA')
    torch.set_num_threads(1)
    ready = summary['software'][CLIPS[0][0]]['coverage_f0']['ready']
    require({k:importlib.metadata.version(k) for k in ('torch', 'mujoco')} == ready['packages'],
            'Use exact saved policy packages, not the separate CPU physics runtime')
    records = unique(summary['inputs']+manifest['files']+[identity(p) for p in
        (SUMMARY/'summary.json', SUMMARY/'audit.json', FIRST/'result.json', args.manifest, LEDGER)])
    ledger_before = identity(LEDGER)
    ledger = checked(read(LEDGER))
    require(charged(ledger) == 2433 and not any(r['state'] == 'RESERVED' for r in ledger['runs']),
            'Do not reset the cumulative budget or enter an outstanding reservation')
    require(all(identity(r['path']) == r for r in records), 'Protected source changed before CPU review')
    require(allocated_tree(ROOT)['unique_allocated_bytes']+64*(1 << 20) < 500_000_000_000, 'Private storage bound')
    with own_gpu():
        args.output.mkdir(exist_ok=False)
        result = dict(stage='CPU_STATE_EQUIVALENCE', status='RUNNING', started_utc=now(),
            evaluation_commit=args.evaluation_commit, controller=process(Path('/proc'), os.getpid()),
            admission=admitted, maximum_wall_seconds=300, maximum_output_bytes=64*(1 << 20),
            maximum_states=234, atol=ATOL, rtol=RTOL, source_inputs=records,
            new_policy_calls=0, new_physical_steps=0, new_optimizer_updates=0, new_gpu_charged_seconds=0,
            cumulative_gpu_charged_seconds=2433, user_accepted_checkpoint=None)
        save(args.output/'result.json', result)
        def check():
            sample = resources(False)
            owner = process(Path('/proc'), os.getpid())
            require(time.monotonic()-started < 300 and owner['VmRSS_bytes'] < 2*(1 << 30)
                and allocated_tree(args.output)['unique_allocated_bytes'] < 64*(1 << 20), 'CPU time/RSS/output bound')
            require(not torch.cuda.is_initialized(), 'CPU-only review initialized CUDA')
            save(args.output/'resources.json', dict(**sample, owner=owner, cuda_initialized=False))
        try:
            sys.path[:0] = [str(Path(runtime['morph_source'])/'src'), str(Path(runtime['adapted_source'])/'src'),
                str(HERE.parent/'morphmimic_q1'), str(HERE.parent/'primebot_q1_review')]
            from bootstrap import install
            from interim_review import load_physical_robot
            spec = install('q1')['robot']
            require(identity(ready['primebot_checkpoint']['path']) == ready['primebot_checkpoint'],
                    'Physical robot checkpoint changed')
            robot, _ = load_physical_robot(ready['primebot_checkpoint']['path'])
            loaded = [identity(inspect.getfile(type(robot))), ready['primebot_checkpoint']]
            require(loaded[0]['sha256'] == '1d5e8670774ed26b844d3153fbceaa8823d50886bea6a0e8b7a6b460fe477954',
                    'The authoritative Primebot kinematic source changed')
            result['runtime_sources'] = loaded
            result['packages'] = ready['packages']
            result['comparisons'] = measure(robot, spec, auth['saved_prefixes'], args.output, check)
            require(all(identity(r['path']) == r for r in records+loaded)
                    and identity(LEDGER) == ledger_before, 'Source or cost ledger changed during CPU review')
            check()
            result.update(status='CPU_STATE_EQUIVALENCE_PASSED', source_integrity='PASSED', states=234,
                observations_and_consecutive_histories='PASSED', cuda_initialized=False,
                input_and_model_unchanged=True, dynamics_functions_forbidden=True,
                policy_action_equivalence_checked=False, hardware_end_to_end_latency_ms=None,
                natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            result.update(ended_utc=now(), seconds=time.monotonic()-started,
                ledger_unchanged=identity(LEDGER) == ledger_before)
            save(args.output/'result.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--evaluation-commit', required=True)
    execute(p.parse_args())
