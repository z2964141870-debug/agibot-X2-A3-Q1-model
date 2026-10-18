"""Independent standard-library HP10 coverage, fall, metric and terminal audit."""

import argparse
import json
import math
import os

from script.a3 import humanplus_baseline as trial
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256


def verify_trace(summary, trace, expected):
    if summary['num_policy_steps'] != expected or trace['policy_tick'] != list(range(expected)):
        raise ValueError('Full replay coverage mismatch')
    errors = []
    for state, reference in zip(trace['q_state_29'], trace['reference_q_29']):
        if len(state) != 29 or len(reference) != 29:
            raise ValueError('Joint width mismatch')
        errors.extend((a - b) ** 2 for a, b in zip(state, reference))
    if len(errors) != expected * 29 or not all(map(math.isfinite, errors)):
        raise ValueError('Missing or nonfinite joint sequence')
    rmse = math.sqrt(sum(errors) / len(errors))
    if not math.isclose(rmse, summary['tracking']['all_29_rmse'], rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError('Independent joint RMSE mismatch')
    fall = None
    for key in ('root_height', 'root_roll_deg', 'root_pitch_deg'):
        if len(trace[key]) != expected or not all(map(math.isfinite, trace[key])):
            raise ValueError('Missing or nonfinite root sequence')
    for i, (height, roll, pitch) in enumerate(zip(trace['root_height'], trace['root_roll_deg'], trace['root_pitch_deg'])):
        if height < .45 or abs(roll) > 60 or abs(pitch) > 60:
            fall = i
            break
    if summary['fall_tick'] != fall or summary['fall'] != (fall is not None):
        raise ValueError('Independent first fall mismatch')
    return dict(policy_steps=expected, joint_rmse_rad=rmse, fall=fall is not None,
                first_fall_s=fall / 50. if fall is not None else None)


def main(stage):
    plan = trial.checked()
    target = trial.PLAN.with_name(trial.NAME + '_' + stage + '_audit.json')
    if target.exists():
        raise FileExistsError('HP10 stage audit already recorded')
    result = trial.read(trial.RESULT)
    if result['status'] == 'RUNNING':
        raise ValueError('Audit requires quiescent stage')
    directories = [trial.LOG / r['stage'] for r in result['stages']]
    failed = trial.LOG / result['current_stage']
    if failed.exists() and failed not in directories:
        directories.append(failed)
    if not directories or result['current_stage'] != stage:
        raise ValueError('Wrong terminal stage')
    rows, groups, states = [], [], []
    for directory in directories:
        state = trial.read(directory / 'state.json')
        states.append(state)
        groups.append(state.get('worker_pid'))
        rows.extend(json.loads(line) for line in (directory / 'health.jsonl').read_text().splitlines())
    after = guard.collect(trial.read(trial.OUT / 'baseline.json'))
    atomic_json(trial.LOG / (stage + '_after_health.json'), after)
    errors = guard.issues(after, startup=False, monitor_pid=os.getpid(), previous=rows[-1],
                          gpu_uuid=rows[-1]['gpu_uuid'])
    survivors = [r for r in after['processes'] if r['pgid'] in groups]
    outcomes = []
    for row in plan['motions']:
        directory = trial.OUT / row['group']
        if row['group'] not in [r['stage'] for r in result['stages']]:
            continue
        summary = trial.read(directory / (row['key'] + '.metrics.json'))['motions'][0]
        trace = trial.read(directory / (row['key'] + '.timeseries.json'))['motions'][0]
        outcomes.append(dict(key=row['key'], group=row['group'],
            **verify_trace(summary, trace, row['full_policy_steps'])))
    verified = (result['status'] in ('IN_PROGRESS', 'COMPLETE_MATCHED_BASELINE')
                and all(s['status'] == 'COMPLETE' and s.get('returncode') == 0 for s in states)
                and not errors and not survivors)
    artifacts = [p for directory in (trial.OUT, trial.LOG) for p in directory.rglob('*') if p.is_file()]
    audit = dict(experiment=plan['experiment'], stage=stage, status=result['status'],
        verified=verified, at_utc=trial.runtime.stamp(), samples=len(rows),
        boot_ids=sorted({r['boot_id'] for r in rows}),
        max_cpu_c=max(r['cpu_celsius'] for r in rows), max_gpu_c=max(r['gpu_celsius'] for r in rows),
        min_available_bytes=min(r['memory']['available_bytes'] for r in rows),
        fatal_matches=sum(len(r['journal']['fatal']) for r in rows),
        oom_matches=sum(len(r['new_oom_events']) for r in rows),
        after_issues=errors, remaining_own_group_processes=survivors,
        max_in_stage_sample_gap_seconds=max((b['monotonic'] - a['monotonic']
            for directory in directories
            for sequence in [[json.loads(line) for line in (directory / 'health.jsonl').read_text().splitlines()]]
            for a, b in zip(sequence, sequence[1:])), default=0),
        stage_gaps_are_not_continuously_supervised=True,
        plan_sha256=sha256(trial.PLAN), result_sha256=sha256(trial.RESULT),
        code_environment_inputs_unchanged=True, outcomes=outcomes,
        artifacts={str(p.relative_to(trial.ROOT)): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in artifacts},
        scope=plan['scope'], optimizer_calls=0, backup_status='LOCAL_ONLY', github_synced=False)
    trial.checked()
    atomic_json(target, audit)
    print(json.dumps({k: audit[k] for k in ('status', 'stage', 'verified', 'samples',
        'max_cpu_c', 'max_gpu_c', 'min_available_bytes', 'after_issues', 'remaining_own_group_processes', 'outcomes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=trial.STAGES, required=True)
    main(parser.parse_args().stage)
