"""E34 use accepted E33 assembly after its archived pre-step receipt error."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_closed_reference_trial as shared
from script.a3 import smpl_joint_audit as evidence
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.a3.smpl_frozen_followup import health_summary

ROOT = shared.ROOT
PRIOR = shared.OUT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_assembly_20261013_E34'
LOG = ROOT / 'logs/a3_zhy_smpl_assembly_20261013_E34'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_assembly_20261013_E34.json'
FAILURE = ROOT / 'data/manifests/a3_zhy_smpl_assembly_20261013_E33_failure.json'


def read(path):
    return json.loads(path.read_text())


def record_prior():
    if FAILURE.exists():
        raise FileExistsError('Preserve the existing E33 failure archive')
    state, job = read(PRIOR / 'state.json'), read(PRIOR / 'registration.json')
    contract = read(PRIOR / 'assembly_contract.json')
    console = ROOT / 'logs/a3_zhy_smpl_assembly_20261013_E33/evaluation/G_0113_zj_train__183926_zj_walk/attempt_001/console.log'
    if (state['status'] != 'STOPPED' or contract['status'] != 'VERIFIED_PASSIVE_REFERENCE_ASSEMBLY'
            or "FileNotFoundError" not in console.read_text() or '/entry/tmp' not in console.read_text()
            or list((PRIOR / 'evaluation').glob('*.metrics.json'))
            or list((PRIOR / 'evaluation').glob('*.inputs.npz'))):
        raise ValueError('Unexpected E33 failure or actual rollout evidence')
    evidence.unchanged(job)
    health = health_summary(ROOT / 'logs/a3_zhy_smpl_assembly_20261013_E33')
    if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
            or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
        raise RuntimeError('Machine failure; no new branch')
    value = dict(status='ENTRY_RECEIPT_DIRECTORY_ERROR_BEFORE_POLICY_STEPS', experiment=job['experiment'],
        state=state, assembly=contract, health=health, completed_policy_rollouts=0, policy_steps=0,
        optimizer_steps=0, weight_loads_not_separately_counted=True, machine_failure_confirmed=False,
        error='Atomic reference receipt parent directory absent', code_sha256=job['code_sha256'],
        input_sha256=job['input_sha256'], evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in (PRIOR / 'state.json', PRIOR / 'assembly_contract.json', console)}, backup_status='LOCAL_ONLY')
    atomic_json(FAILURE, value)
    print(json.dumps(dict(status=value['status'], policy_steps=0, health=health)))


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E34')
    value, archived = guard.preflight(), read(FAILURE)
    old = read(PRIOR / 'registration.json')
    evidence.unchanged(old)
    if (archived['status'] != 'ENTRY_RECEIPT_DIRECTORY_ERROR_BEFORE_POLICY_STEPS'
            or archived['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires archived same-boot E33 entry error')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    inputs = dict(old['input_sha256'])
    paths = [FAILURE, PRIOR / 'registration.json', PRIOR / 'state.json', PRIOR / 'assembly_contract.json']
    for row in archived['assembly']['motions']:
        path = ROOT / row['artifact']['path']
        if sha256(path) != row['artifact']['sha256']:
            raise ValueError('Accepted E33 assembly changed')
        paths.append(path)
    inputs.update({str(p.relative_to(ROOT)):sha256(p) for p in paths})
    job = dict(old, experiment='A3_ZHY_SMPL_ASSEMBLY_20261013_E34', registered_at=time.time(),
        parent_experiment=old['experiment'], project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        main_variable='ACCEPTED_PASSIVE_REFERENCE_STATE_FROZEN_COMPARISON_WITH_COMPLETE_RECEIPT_DIRECTORY',
        reuse_assembly_from=old['experiment'], assembly_recomputations_allowed=0,
        code_sha256={**old['code_sha256'], 'script/a3/smpl_assembly_recovery.py':sha256(Path(__file__))},
        input_sha256=inputs, kernel_baseline=kernel['stdout'].splitlines(),
        baseline_oom_kill=value['container_memory']['oom_kill'])
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (OUT / 'entry').mkdir()
    (OUT / 'references').mkdir()
    (OUT / 'assembled').symlink_to(os.path.relpath(PRIOR / 'assembled', OUT))
    for row in job['selection']:
        (OUT / 'references' / row['filename']).symlink_to(os.path.relpath(ROOT / row['source'] / 'reference.csv', OUT / 'references'))
    atomic_json(OUT / 'assembly_contract.json', archived['assembly'])
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    # Exercise the exact receipt path and atomic writer before loading policy.
    probe = OUT / 'entry' / 'entry_contract_check.json'
    atomic_json(probe, dict(passed=True, no_policy_or_simulator_steps=True))
    if read(probe)['passed'] is not True or not (OUT / 'assembled').is_dir():
        raise RuntimeError('Receipt/storage contract failed')
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], assembly_recomputations=0)))


def configure():
    shared.OUT, shared.LOG, shared.MANIFEST = OUT, LOG, MANIFEST
    guard.OUT, guard.LOG = OUT, LOG


def worker():
    configure()
    shared.assembly = lambda job:read(OUT / 'assembly_contract.json')
    shared.worker()


def run():
    configure()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E34')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_assembly_recovery', 'worker'], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E33_MONITOR_PID=str(os.getpid()), YUANQI_E33_OUTPUT=str(OUT)),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, LOG / 'worker', job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED', returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        health = health_summary(LOG)
        if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
                or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
            state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
            atomic_json(OUT / 'state.json', state)
        result = read(OUT / 'result.json') if state['status'] == 'COMPLETE' else {}
        receipt = dict(result, status=result.get('status', 'STOPPED'), experiment=job['experiment'], state=state,
            health=health, assembly_recomputations=0, code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
            backup_status='LOCAL_ONLY', evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in (OUT / 'registration.json', LOG / 'console.log')})
        if state['status'] == 'COMPLETE':
            receipt['evidence'][str((OUT / 'result.json').relative_to(ROOT))] = dict(
                bytes=(OUT / 'result.json').stat().st_size, sha256=sha256(OUT / 'result.json'))
            atomic_json(OUT / 'verification.json', receipt)
        atomic_json(MANIFEST, receipt)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('record-prior', 'register', 'run', 'worker'))
    result = {'record-prior':record_prior, 'register':register, 'run':run, 'worker':worker}[parser.parse_args().action]()
    raise SystemExit(result if isinstance(result, int) else 0)
