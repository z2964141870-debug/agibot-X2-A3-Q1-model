"""E25 continue frozen evaluation after E24 stopped before its first rollout."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from script.a3 import smpl_frozen_probe as shared
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = shared.ROOT
PRIOR = shared.OUT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_frozen_20261013_E25'
LOG = ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E25'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E25.json'


def health_summary(directory):
    from script.a3 import zhy_fp32_trial as helper
    before, helper.LOG = helper.LOG, directory
    try:
        return helper.health_summary()
    finally:
        helper.LOG = before


def record_prior():
    job, state = shared.read(PRIOR / 'registration.json'), shared.read(PRIOR / 'state.json')
    contract = shared.read(PRIOR / 'reference_contract.json')
    console = ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E24/evaluate/console.log'
    if (state['status'] != 'STOPPED' or contract['status'] != 'VERIFIED_REFERENCE_ABI'
            or len(contract['motions']) != 6 or "has no attribute 'summarize_only'" not in console.read_text()
            or list((PRIOR / 'evaluation').glob('*.metrics.json'))):
        raise ValueError('E24 is not the expected pre-policy entry failure')
    shared.unchanged(job)
    health = health_summary(ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E24')
    if health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors'] or health['telemetry_errors']:
        raise ValueError('Unexpected E24 machine/telemetry failure')
    receipt = dict(experiment=job['experiment'], status='ENTRY_ERROR_BEFORE_POLICY',
        reference_abi=contract, state=state, health=health, policy_rollouts=0, optimizer_steps=0,
        frozen_official_unchanged=True, error="Missing evaluator argument: summarize_only",
        machine_failure_confirmed=False, code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
        evidence_sha256={str(path.relative_to(ROOT)):sha256(path) for path in
            (PRIOR / 'state.json', PRIOR / 'reference_contract.json', console)}, backup_status='LOCAL_ONLY')
    atomic_json(ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E24.json', receipt)
    print(json.dumps(dict(status=receipt['status'], reference_contract_passed=6, policy_rollouts=0, health=health)))


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E25')
    prior = shared.read(ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E24.json')
    old, value = shared.read(PRIOR / 'registration.json'), guard.preflight()
    shared.unchanged(old)
    if prior['status'] != 'ENTRY_ERROR_BEFORE_POLICY' or prior['health']['boot_ids'] != [value['boot_id']]:
        raise ValueError('Requires archived same-boot pre-policy E24')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    inputs = dict(old['input_sha256'])
    for path in (PRIOR / 'registration.json', PRIOR / 'state.json', PRIOR / 'reference_contract.json',
            ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E24.json'):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    job = dict(old, experiment='A3_ZHY_SMPL_FROZEN_20261013_E25', registered_at=time.time(),
        parent_experiment=old['experiment'], project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        main_variable='COMPLETE_EVALUATOR_ARGUMENTS_AFTER_PRE_POLICY_ENTRY_FAILURE',
        reference_contract_reused=True, summarize_only=False,
        code_sha256={**old['code_sha256'], 'script/a3/smpl_frozen_followup.py':sha256(Path(__file__))},
        input_sha256=inputs, kernel_baseline=kernel['stdout'].splitlines(),
        baseline_oom_kill=value['container_memory']['oom_kill'])
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (OUT / 'references').mkdir()
    for row in job['selection']:
        path = ROOT / row['source'] / 'reference.csv'
        (OUT / 'references' / row['filename']).symlink_to(os.path.relpath(path, OUT / 'references'))
    atomic_json(OUT / 'reference_contract.json', prior['reference_abi'])
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], reference_contract_reused=True)))


def configure():
    shared.OUT, shared.LOG = OUT, LOG
    guard.OUT, guard.LOG = OUT, LOG


def evaluate(job):
    from script.a3 import evaluate_mujoco as evaluation
    contract = shared.read(OUT / 'reference_contract.json')
    if contract != shared.read(PRIOR / 'reference_contract.json'):
        raise ValueError('Previously verified ABI changed')
    evaluation.preflight = guard.preflight
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    def spawn(argv, **kwargs):
        argv = ['script.a3.smpl_frozen_entry' if value == 'script.a3.sim_entry' else value for value in argv]
        kwargs['start_new_session'] = False
        return subprocess.Popen(argv, **kwargs)
    evaluation.subprocess = SimpleNamespace(Popen=spawn)
    args = SimpleNamespace(output=OUT / 'evaluation', logs=LOG / 'evaluation', checkpoint=shared.OFFICIAL,
        motion=OUT / 'references', reference_fps=30., frame_stride=1, max_policy_steps=None,
        action_delay_ms=0., reference_buffer_ms=0., capture_inputs=True, capture_count=100000,
        summarize_only=False, joint_reference_mode=None, joint_predictor=None, joint_noise_std=0.,
        joint_noise_seed=0, causal_orientation=False)
    if evaluation.evaluate_explicit(args):
        raise RuntimeError('E25 evaluation failed; no retry')
    shared.unchanged(job)


def worker(mode):
    configure()
    state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
    expected = 'RUNNING' if mode == 'evaluate' else 'VERIFYING'
    if state['status'] != expected or state['monitor_pid'] != os.getppid() or os.environ.get('YUANQI_E25_MONITOR_PID') != str(os.getppid()):
        raise RuntimeError('Worker requires E25 guarded monitor')
    shared.unchanged(job)
    return evaluate(job) if mode == 'evaluate' else shared.verify_worker(job)


def run(mode):
    configure()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = shared.read(OUT / 'state.json'), shared.read(OUT / 'registration.json')
        if state['status'] != ('READY' if mode == 'evaluate' else 'COMPLETE'):
            raise RuntimeError('No repeat or recovery of E25')
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
                    'script.a3.smpl_frozen_followup', 'worker', '--mode', mode], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E25_MONITOR_PID=str(os.getpid())), stdout=stream, stderr=subprocess.STDOUT,
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
            health = health_summary(LOG)
            if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors'] or health['telemetry_errors']
                    or health['oom_counts'] != [job['baseline_oom_kill']]):
                state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
                atomic_json(OUT / 'state.json', state)
                raise ValueError('Health verification failed')
            receipt = dict(shared.read(OUT / 'verified_results.json'), status='VERIFIED_FROZEN_BASELINE',
                experiment=job['experiment'], state=state, health=health, code_sha256=job['code_sha256'],
                input_sha256=job['input_sha256'], reference_contract_reused_from='E24')
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
