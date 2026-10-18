"""Single-use independent expert phase audit without loading checkpoint tensors."""

import argparse
import json
import math
import os

from script.a3 import humanplus_specialist_repaired_trial as trial
from script.a3 import humanplus_recovered_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_kit_log import verify_log
from script.a3.humanplus_specialist_protocol import require_training_receipt
from script.project_tools.humanplus_baseline_audit import verify_trace


def verify_isaac(directory):
    trace = trial.read(directory / 'full_isaac_trace.json')
    metric = trial.read(directory / 'isaac_metrics.json')
    rows = trace['rows']
    metadata = trial.read(directory / 'bridge_metadata.json')
    progress = trial.read(directory / 'replay_progress.json')
    mapping = metadata['mapping']
    if (trace['joint_names'] != trace['reference_joint_names_in_action_order']
            or trace['joint_names'] != mapping['joint_names'] or len(trace['joint_names']) != 29
            or len(set(trace['joint_names'])) != 29
            or mapping['reference_body_indices'] != list(range(len(trace['reference_body_names'])))
            or progress != dict(completed_steps=499, last_reference_tick=498, cuda_synchronized=True)):
        raise ValueError('Repaired bridge identity or actual synchronized steps incomplete')
    if (len(rows) != 499 or [r['tick'] for r in rows] != list(range(499))
            or trace['reference_body_names'] != trace['actual_body_names']):
        raise ValueError('Incomplete or inconsistent Isaac reference/physical trace')
    squared, count, falls = 0., 0, []
    for row in rows:
        if len(row['q']) != 29 or len(row['q_ref']) != 29 or len(row['raw_action']) != 29:
            raise ValueError('Isaac action/joint ABI changed')
        values = row['q'] + row['q_ref'] + row['raw_action'] + [row['height'], row['roll'], row['pitch']]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite Isaac trace')
        squared += sum((q - ref)**2 for q, ref in zip(row['q'], row['q_ref']))
        count += 29
        if row['height'] < .45 or abs(row['roll']) > 60 or abs(row['pitch']) > 60:
            falls.append(row['tick'])
    if (not math.isclose(math.sqrt(squared/count), metric['joint_rmse_rad'], rel_tol=1e-10)
            or metric['first_fall_tick'] != (falls[0] if falls else None)
            or metric['fall'] != bool(falls)
            or metric['trace_sha256'] != sha256(directory / 'full_isaac_trace.json')):
        raise ValueError('Isaac persisted metrics differ from independent full trace')
    return metric


def main(stage):
    plan = trial.checked()
    target = trial.PLAN.with_name(trial.NAME + '_' + stage + '_audit.json')
    if target.exists():
        raise FileExistsError('Expert phase audit cannot repeat')
    result = trial.read(trial.RESULT)
    if result['status'] == 'RUNNING' or result['current_stage'] != stage:
        raise ValueError('Quiescent requested phase required')
    directories = [trial.LOG / r['stage'] for r in result['stages']]
    if trial.LOG / stage not in directories:
        directories.append(trial.LOG / stage)
    states = [trial.read(d / 'state.json') for d in directories]
    rows = [json.loads(line) for d in directories for line in (d / 'health.jsonl').read_text().splitlines()]
    after = guard.collect(trial.read(trial.OUT / 'baseline.json'))
    atomic_json(trial.LOG / (stage + '_after_health.json'), after)
    issues = guard.issues(after, startup=False, monitor_pid=os.getpid(),
        previous=rows[-1] if rows else None, gpu_uuid=rows[-1]['gpu_uuid'] if rows else None)
    survivors = [p for p in after['processes'] if p['pgid'] in [s.get('worker_pid') for s in states]]
    verified = (result['status'] in ('IN_PROGRESS', 'COMPLETE_SPECIALIST_ASSESSMENT')
        and all(s['status'] == 'COMPLETE' and s.get('returncode') == 0 for s in states)
        and not issues and not survivors)
    proof = {}
    if verified and stage in ('train', 'isaac-baseline', 'isaac-final'):
        directory = trial.LOG / stage
        native = trial.read(directory / 'native_exit.json')
        kit = verify_log(directory, native['native_pid'])
        saved = trial.read(directory / 'result.json')
        if native['returncode'] or any(saved[k] != v for k, v in kit.items()):
            raise ValueError('Native exit/Kit receipt mismatch')
        proof = (dict(training=require_training_receipt(trial.read(directory / 'specialist_contract.json')))
                 if stage == 'train' else dict(isaac=verify_isaac(directory)))
    if verified and stage == 'verify-training':
        if not result['stages'][-1]['receipt']['saved_expert_verified']:
            raise ValueError('Saved specialist verification incomplete')
    if verified and stage in ('mujoco', 'effect'):
        row = plan['baseline_motion']
        prefix = trial.OUT / 'mujoco' / row['key']
        metric = trial.read(prefix.with_suffix('.metrics.json'))['motions'][0]
        trace = trial.read(prefix.with_suffix('.timeseries.json'))['motions'][0]
        proof['mujoco'] = verify_trace(metric, trace, 499)
        import numpy as np
        with np.load(prefix.with_suffix('.inputs.npz'), allow_pickle=False) as captured:
            if (captured['observations'].shape != (499, 1570)
                    or captured['actions'].shape != (499, 29)
                    or captured['actions'].dtype != np.float32
                    or not np.isfinite(captured['observations']).all()
                    or not np.isfinite(captured['actions']).all()):
                raise ValueError('Incomplete specialist policy capture')
            np.testing.assert_array_equal(np.clip(captured['actions'].astype(np.float64), -20., 20.),
                                          trace['raw_action_29'])
        if stage == 'effect' and trial.effect(plan) != result['stages'][-1]['receipt']:
            raise ValueError('Expert effect result differs from saved full metrics')
    artifacts = {}
    for directory in (trial.OUT, trial.LOG, trial.RUN):
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_file() and not path.is_symlink():
                    artifacts[str(path.relative_to(trial.ROOT))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    audit = dict(stage=stage, status=result['status'], verified=verified, at_utc=runtime_stamp(),
        samples=len(rows), boot_ids=sorted({r['boot_id'] for r in rows}),
        max_cpu_c=max((r['cpu_celsius'] for r in rows), default=None),
        max_gpu_c=max((r['gpu_celsius'] for r in rows), default=None),
        min_available_bytes=min((r['memory']['available_bytes'] for r in rows), default=None),
        historical_resolved_kernel_matches=sum(len(r['journal']['fatal']) for r in rows),
        new_fatal_matches=sum(len(r['new_kernel_events']) for r in rows),
        oom_matches=sum(len(r['new_oom_events']) for r in rows),
        after_issues=issues, remaining_own_group_processes=survivors,
        proof=proof, artifacts=artifacts, plan_sha256=sha256(trial.PLAN),
        result_sha256=sha256(trial.RESULT), stopped_checkpoint_tensors_loaded=False,
        stage_gaps_are_not_continuously_supervised=True, github_synced=False)
    atomic_json(target, audit)
    print(json.dumps({k: audit[k] for k in ('stage', 'status', 'verified', 'samples',
        'max_cpu_c', 'max_gpu_c', 'after_issues', 'remaining_own_group_processes')}))


def runtime_stamp():
    return trial.runtime.stamp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=trial.STAGES, required=True)
    main(parser.parse_args().stage)
