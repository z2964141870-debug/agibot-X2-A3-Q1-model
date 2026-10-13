"""E26 tested process launch after two archived pre-policy wrapper errors."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_frozen_probe as shared
from script.a3 import smpl_frozen_followup as prior_helpers
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = shared.ROOT
PRIOR = prior_helpers.OUT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_frozen_20261013_E26'
LOG = ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E26'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E26.json'
CODE = ('script/a3/smpl_frozen_validated.py', 'script/a3/frozen_evaluation_launch.py',
        'script/a3/test_frozen_evaluation_launch.py')


def record_prior():
    job, state = shared.read(PRIOR / 'registration.json'), shared.read(PRIOR / 'state.json')
    console = ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E25/evaluate/console.log'
    contract = shared.read(PRIOR / 'reference_contract.json')
    if (state['status'] != 'STOPPED' or "has no attribute 'STDOUT'" not in console.read_text()
            or list((PRIOR / 'evaluation').glob('*.metrics.json'))):
        raise ValueError('Unexpected E25 failure; do not continue')
    shared.unchanged(job)
    health = prior_helpers.health_summary(ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E25')
    if health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors'] or health['telemetry_errors']:
        raise ValueError('Unexpected machine or telemetry fault')
    receipt = dict(status='ENTRY_ERROR_BEFORE_POLICY', experiment=job['experiment'], state=state,
        reference_abi=contract, reference_contract_reused_from='E24', policy_rollouts=0, optimizer_steps=0,
        error='Subprocess adapter missing STDOUT constant', machine_failure_confirmed=False,
        health=health, code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
        evidence_sha256={str(path.relative_to(ROOT)):sha256(path) for path in
            (PRIOR / 'state.json', PRIOR / 'reference_contract.json', console)}, backup_status='LOCAL_ONLY')
    atomic_json(ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E25.json', receipt)
    print(json.dumps(dict(status=receipt['status'], policy_rollouts=0, health=health)))


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E26')
    value = guard.preflight()
    old = shared.read(PRIOR / 'registration.json')
    previous = shared.read(ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E25.json')
    shared.unchanged(old)
    if previous['status'] != 'ENTRY_ERROR_BEFORE_POLICY' or previous['health']['boot_ids'] != [value['boot_id']]:
        raise ValueError('Requires archived same-boot E25')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    # Exercise the full launch with fake artifacts, without spawning a simulator.
    from script.a3 import test_frozen_evaluation_launch as tests
    import io
    import unittest
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 2:
        raise RuntimeError('Launch contract tests failed: ' + stream.getvalue())
    inputs = dict(old['input_sha256'])
    for path in (PRIOR / 'registration.json', PRIOR / 'state.json', PRIOR / 'reference_contract.json',
            ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E25.json'):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    job = dict(old, experiment='A3_ZHY_SMPL_FROZEN_20261013_E26', registered_at=time.time(),
        parent_experiment=old['experiment'], project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        main_variable='TESTED_TRANSPARENT_SUBPROCESS_LAUNCH_ADAPTER',
        code_sha256={**old['code_sha256'], **{name:sha256(ROOT / name) for name in CODE}}, input_sha256=inputs,
        launch_contract_tests=2, launch_tests_spawn_real_simulator=False,
        kernel_baseline=kernel['stdout'].splitlines(), baseline_oom_kill=value['container_memory']['oom_kill'])
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (OUT / 'references').mkdir()
    for row in job['selection']:
        path = ROOT / row['source'] / 'reference.csv'
        (OUT / 'references' / row['filename']).symlink_to(os.path.relpath(path, OUT / 'references'))
    atomic_json(OUT / 'reference_contract.json', previous['reference_abi'])
    atomic_json(OUT / 'launch_test_receipt.json', dict(tests_run=2, passed=True, simulated_process_launch=True,
        real_policy_loads=0, console=stream.getvalue()))
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], launch_contract_tests=2)))


def configure():
    shared.OUT, shared.LOG = OUT, LOG
    guard.OUT, guard.LOG = OUT, LOG


def evaluate(job):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    if shared.read(OUT / 'reference_contract.json') != shared.read(PRIOR / 'reference_contract.json'):
        raise ValueError('Reused ABI changed')
    original = evaluation.subprocess
    evaluation.preflight = guard.preflight
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    evaluation.subprocess = LaunchProxy(original)
    try:
        code = evaluation.evaluate_explicit(arguments(OUT / 'evaluation', LOG / 'evaluation',
            shared.OFFICIAL, OUT / 'references'))
    finally:
        evaluation.subprocess = original
    if code:
        raise RuntimeError('E26 evaluation failed; no retry')
    shared.unchanged(job)


def worker(mode):
    configure()
    state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
    expected = 'RUNNING' if mode == 'evaluate' else 'VERIFYING'
    if state['status'] != expected or state['monitor_pid'] != os.getppid() or os.environ.get('YUANQI_E26_MONITOR_PID') != str(os.getppid()):
        raise RuntimeError('Worker requires E26 guarded monitor')
    shared.unchanged(job)
    return evaluate(job) if mode == 'evaluate' else shared.verify_worker(job)


def run(mode):
    configure()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
        if state['status'] != ('READY' if mode == 'evaluate' else 'COMPLETE'):
            raise RuntimeError('No repeat or recovery of E26')
        guard.preflight()
        shared.unchanged(job)
        state.update(status='RUNNING' if mode == 'evaluate' else 'VERIFYING', monitor_pid=os.getpid())
        state[mode + '_started_at'] = time.time()
        atomic_json(OUT / 'state.json', state)
        stage = LOG / mode
        stage.mkdir()
        try:
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_frozen_validated', 'worker', '--mode', mode], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E26_MONITOR_PID=str(os.getpid())), stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            state.update(status=('COMPLETE' if mode == 'evaluate' else 'VERIFIED') if code == 0 and reason is None else 'STOPPED',
                returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state[mode + '_ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        if state['status'] == 'VERIFIED':
            health = prior_helpers.health_summary(LOG)
            if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors'] or health['telemetry_errors']
                    or health['oom_counts'] != [job['baseline_oom_kill']]):
                state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
                atomic_json(OUT / 'state.json', state)
                raise ValueError('Health verification failed')
            receipt = dict(shared.read(OUT / 'verified_results.json'), status='VERIFIED_FROZEN_BASELINE',
                experiment=job['experiment'], state=state, health=health, code_sha256=job['code_sha256'],
                input_sha256=job['input_sha256'], reference_contract_reused_from='E24', launch_contract_tests=2)
            atomic_json(OUT / 'independent_verification.json', receipt)
            atomic_json(MANIFEST, receipt)
        print(json.dumps(state))
        return 0 if state['status'] in ('COMPLETE', 'VERIFIED') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('record-prior', 'register', 'run', 'verify', 'worker'))
    parser.add_argument('--mode', choices=('evaluate', 'verify'), default='evaluate')
    args = parser.parse_args()
    result = record_prior() if args.action == 'record-prior' else register() if args.action == 'register' else worker(args.mode) if args.action == 'worker' else run('verify' if args.action == 'verify' else 'evaluate')
    raise SystemExit(result if isinstance(result, int) else 0)
