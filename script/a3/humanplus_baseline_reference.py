"""HP11: correct reference staging, preserving HP10 and skipping model diagnostics."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from script.a3 import humanplus_baseline as base
from script.a3.checkpoint_store import atomic_json, sha256

runtime, guard = base.runtime, base.guard
ROOT, ENV = base.ROOT, base.ENV
NAME = 'a3_humanplus_baseline_reference_20261018_HP11'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
STAGES, read = base.STAGES, base.read
PRIOR = ROOT / 'data/manifests/a3_humanplus_baseline_20261018_HP10'


def stage_references(root, output, selection):
    references = output / 'references'
    sources = [(root / row['source'] / 'reference.csv', references / row['filename']) for row in selection]
    if len(sources) != 6 or any(not source.is_file() for source, _ in sources):
        raise ValueError('Six existing reference sources required')
    if any(target.parent != references for _, target in sources) or len({t.name for _, t in sources}) != 6:
        raise ValueError('Six unique direct-child aliases required')
    references.mkdir(parents=True, exist_ok=False)
    for source, target in sources:
        target.symlink_to(source)
        if not target.is_file() or sha256(target) != sha256(source):
            raise ValueError('Staged reference identity mismatch')
    return references


def configure():
    for name in ('NAME', 'OUT', 'LOG', 'PLAN', 'RESULT'):
        setattr(base, name, globals()[name])


def register():
    runtime.require_root()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP11 already registered')
    old = base.checked()
    result_path = Path(str(PRIOR) + '_result.json')
    audit_path = Path(str(PRIOR) + '_validate_audit.json')
    result, audit = read(result_path), read(audit_path)
    console = ROOT / 'logs/a3_humanplus_baseline_20261018_HP10/validate/console.log'
    if (result['status'] != 'STOPPED' or result['stages'] or audit['verified']
            or audit['after_issues'] or audit['remaining_own_group_processes']
            or 'FileNotFoundError' not in console.read_text()
            or '/validation/references/G_0113_zj_train__183926_zj_walk.csv' not in console.read_text()):
        raise ValueError('Requires preserved HP10 missing-reference failure, before any replay')
    baseline = guard.health.snapshot(read(ROOT / 'data/experiments/a3_humanplus_baseline_20261018_HP10/baseline.json'))
    if guard.health.reasons(baseline, startup=True):
        raise RuntimeError('Registration health rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    plan = dict(old, experiment=NAME.upper(), registered_at_utc=runtime.stamp(),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        main_variable='Stage six explicit reference aliases before existing numerical ABI validator',
        previous_status='HP10_STOPPED_PRESERVED', model_diagnostics_repeated=False,
        missing_hp10_receipts_reconstructed=False, boot_id=baseline['boot_id'],
        code_sha256=dict(old['code_sha256'], **{n: sha256(ROOT / n) for n in (
            'script/a3/humanplus_baseline_reference.py', 'script/a3/test_humanplus_baseline_reference.py',
            'script/project_tools/humanplus_baseline_reference_audit.py')}),
        input_sha256=dict(old['input_sha256'], **{str(p.relative_to(ROOT)): sha256(p)
            for p in (Path(str(PRIOR) + '_plan.json'), result_path, audit_path, console, OUT / 'baseline.json')}),
        budget=dict(old['budget'], model_validation_actor_instances=0,
            model_validation_checkpoint_deserializations=0, validation_forward_calls=0),
        validation_scope='REFERENCE_ABI_ONLY_NO_REPEAT_OF_HP10_MODEL_DIAGNOSTICS')
    cache = ROOT / 'data/experiments/a3_humanplus_baseline_20261018_HP10/solver_bridge'
    bridge = list(cache.rglob('liba3_loop_solver_bridge.so'))
    if len(bridge) != 1:
        raise ValueError('Reuse the already built bridge, without changing stopped HP10')
    plan['input_sha256'].update({str(p.relative_to(ROOT)): sha256(p) for p in cache.rglob('*') if p.is_file()})
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', model_diagnostics_repeated=False,
                          full_policy_rollouts=10, policy_steps=plan['budget']['policy_steps'])))


def checked():
    configure()
    return base.checked()


def reference_only(plan):
    from script.a3 import smpl_frozen_probe as abi
    abi.OUT = OUT / 'validation'
    stage_references(ROOT, abi.OUT, plan['selection'])
    rows = abi.reference_contract(dict(selection=plan['selection'], **plan['reference_contract']))
    if [r['full_policy_steps'] for r in rows] != [r['full_policy_steps'] for r in plan['motions'] if r['group'] == 'cloth']:
        raise ValueError('Reference resampling budget mismatch')
    return dict(status='COMPLETE', stage='validate', reference_contract=rows,
                policy_model_loads=0, policy_forward_calls=0, optimizer_calls=0,
                scope='NEW_HOST_REFERENCE_ABI_ONLY_MODEL_DIAGNOSTICS_NOT_REPEATED')


def worker(stage):
    configure()
    base.validate_model = reference_only
    base.worker(stage)


def run(stage):
    checked()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = read(RESULT) if RESULT.exists() else dict(status='REGISTERED', stages=[], experiment=NAME.upper())
        base.require_next(result, stage)
        result.update(status='RUNNING', current_stage=stage, started_at_utc=runtime.stamp())
        atomic_json(RESULT, result)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=str(ROOT), CONDA_PREFIX=str(ENV), PATH=str(ENV / 'bin') + ':' + os.environ['PATH'],
                A3_LOOP_BRIDGE_CACHE=str(ROOT / 'data/experiments/a3_humanplus_baseline_20261018_HP10/solver_bridge'))
            state = guard.run_stage(LOG / stage, [str(ENV / 'bin/python'), '-B', '-u', '-m',
                'script.a3.humanplus_baseline_reference', 'worker', '--stage', stage], OUT / 'baseline.json', env=env)
            result['stages'].append(dict(stage=stage, state=state, receipt=read(LOG / stage / 'result.json')))
            result['status'] = 'COMPLETE_MATCHED_BASELINE' if stage == STAGES[-1] else 'IN_PROGRESS'
        except BaseException as exc:
            result.update(status='STOPPED', reason=type(exc).__name__ + ':' + str(exc))
            raise
        finally:
            result.update(updated_at_utc=runtime.stamp(), backup_status='LOCAL_ONLY', github_synced=False)
            atomic_json(RESULT, result)
        print(json.dumps(dict(status=result['status'], completed_stage=stage)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=STAGES, default='validate')
    args = parser.parse_args()
    {'register': register, 'run': lambda: run(args.stage), 'worker': lambda: worker(args.stage)}[args.action]()
