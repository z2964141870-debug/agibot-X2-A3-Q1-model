"""Independent standard-library HP12 phase, transaction, and matched-effect audit."""

import argparse
import json
import math
import os

from script.a3 import humanplus_finite as trial
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.project_tools.humanplus_baseline_audit import verify_trace


def main(stage):
    plan = trial.checked()
    target = trial.PLAN.with_name(trial.NAME + '_' + stage + '_audit.json')
    if target.exists():
        raise FileExistsError('HP12 phase audit must not repeat')
    result = trial.read(trial.RESULT)
    if result['status'] == 'RUNNING' or result['current_stage'] != stage:
        raise ValueError('Requires the requested quiescent phase')
    directories = [trial.LOG / r['stage'] for r in result['stages']]
    failed = trial.LOG / stage
    if failed.exists() and failed not in directories:
        directories.append(failed)
    states = [trial.read(d / 'state.json') for d in directories]
    rows = [json.loads(line) for d in directories for line in (d / 'health.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('No resource evidence')
    after = guard.collect(trial.read(trial.OUT / 'baseline.json'))
    atomic_json(trial.LOG / (stage + '_after_health.json'), after)
    issues = guard.issues(after, startup=False, monitor_pid=os.getpid(), previous=rows[-1], gpu_uuid=rows[-1]['gpu_uuid'])
    survivors = [r for r in after['processes'] if r['pgid'] in [s.get('worker_pid') for s in states]]
    verified = (result['status'] in ('IN_PROGRESS', 'COMPLETE_MATCHED_EFFECT')
        and all(s['status'] == 'COMPLETE' and s.get('returncode') == 0 for s in states)
        and not issues and not survivors)
    outcomes = []
    for row in plan['motions']:
        if row['group'] not in [s['stage'] for s in result['stages']]:
            continue
        prefix = trial.OUT / row['group'] / row['key']
        metric = trial.read(prefix.with_suffix('.metrics.json'))['motions'][0]
        trace = trial.read(prefix.with_suffix('.timeseries.json'))['motions'][0]
        outcomes.append(dict(key=row['key'], group=row['group'],
            **verify_trace(metric, trace, row['full_policy_steps'])))
    if verified and stage == 'train':
        native = trial.read(trial.LOG / 'train/native_training.json')
        if native['global_step'] != 8 or native['optimizer_commits'] != 32 or not native['cuda_synchronized']:
            raise ValueError('Incomplete native training receipt')
        if trial.read(trial.LOG / 'train/native_exit.json')['returncode'] != 0:
            raise ValueError('Native process did not exit zero')
    effect_verified = None
    if verified and stage == 'effect':
        effect = result['stages'][-1]['receipt']
        if len(outcomes) != 10:
            raise ValueError('Ten final complete effect replays required')
        by_key = {r['key']: r for r in outcomes}
        for row in effect['outcomes']:
            actual = by_key[row['key']]
            if (actual['fall'] != row['fall'] or not math.isclose(actual['joint_rmse_rad'],
                    row['after']['joint_rmse_rad'], rel_tol=1e-6, abs_tol=1e-10)):
                raise ValueError('Effect summary differs from independent trace')
        effect_verified = effect['matched_development_gate_passed']
    artifacts = {}
    for directory in (trial.OUT, trial.LOG, trial.RUN):
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_file() and not path.is_symlink():
                    artifacts[str(path.relative_to(trial.ROOT))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    audit = dict(experiment=plan['experiment'], stage=stage, status=result['status'], verified=verified,
        at_utc=trial.runtime.stamp(), samples=len(rows), boot_ids=sorted({r['boot_id'] for r in rows}),
        max_cpu_c=max(r['cpu_celsius'] for r in rows), max_gpu_c=max(r['gpu_celsius'] for r in rows),
        min_available_bytes=min(r['memory']['available_bytes'] for r in rows),
        fatal_matches=sum(len(r['journal']['fatal']) for r in rows),
        oom_matches=sum(len(r['new_oom_events']) for r in rows), after_issues=issues,
        remaining_own_group_processes=survivors, outcomes=outcomes, artifacts=artifacts,
        matched_development_gate_passed=effect_verified,
        plan_sha256=sha256(trial.PLAN), result_sha256=sha256(trial.RESULT),
        stage_gaps_are_not_continuously_supervised=True, backup_status='LOCAL_ONLY', github_synced=False)
    trial.checked()
    atomic_json(target, audit)
    print(json.dumps({k: audit[k] for k in ('stage', 'status', 'verified', 'samples',
        'max_cpu_c', 'max_gpu_c', 'min_available_bytes', 'after_issues', 'remaining_own_group_processes',
        'matched_development_gate_passed')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=trial.STAGES, required=True)
    main(parser.parse_args().stage)
