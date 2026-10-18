"""HP09: supervise native shutdown from a standard-library parent process."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from script.a3 import humanplus_physx_callback_validation as source
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256

previous = source.previous
ROOT = source.ROOT
NAME = 'a3_humanplus_physx_native_20261018_HP09'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
SHUTDOWN_MARKERS = ('SimulationApp.close: Stage closed',
                    'SimulationApp.close: shutting down app and releasing framework')


def register():
    old = source.checked()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP09 already registered')
    audit_path = source.PLAN.with_name(source.NAME + '_audit.json')
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'STOPPED' or audit['receipts']['control']['steps'] != 10
            or audit['last_progress']['phase'] != 'context_clear_returned'
            or audit['after_issues'] or audit['remaining_own_group_processes']):
        raise ValueError('Requires preserved HP08 native-shutdown evidence and cleanup')
    baseline = guard.health.snapshot(json.loads((source.OUT / 'baseline.json').read_text()))
    if guard.health.reasons(baseline, startup=True):
        raise RuntimeError('Registration health rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    plan = dict(experiment='A3_HUMANPLUS_PHYSX_NATIVE_20261018_HP09',
        status='REGISTERED', registered_at_utc=previous.prior.stamp(),
        authorization='2026-10-18 quick-fix request includes correcting native shutdown receipt protocol',
        main_variable='Standard-library broker records actual native child exit before final receipt',
        unchanged_physics_and_shutdown_configuration=True,
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        boot_id=baseline['boot_id'], bootstrap_environment=previous.BOOTSTRAP,
        code_sha256=dict(old['code_sha256'], **{
            'script/a3/humanplus_physx_native_validation.py': sha256(Path(__file__))}),
        environment_source_sha256=old['environment_source_sha256'],
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            (source.PLAN, source.RESULT, audit_path, OUT / 'baseline.json')},
        budget=old['budget'], criterion='Ten synchronized CUDA steps, callback cleanup, native shutdown log and exit zero',
        no_automatic_retry=True, repeat_passed_controls=False,
        policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0, backup_status='LOCAL_ONLY')
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', plan=str(PLAN))), flush=True)


def checked():
    previous.prior.require_root()
    plan = json.loads(PLAN.read_text())
    for field in ('code_sha256', 'environment_source_sha256', 'input_sha256'):
        for name, digest in plan[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256(path) != digest:
                raise ValueError('Registered HP09 identity changed: ' + name)
    return plan


def event(phase, **fields):
    row = dict(phase=phase, at_utc=previous.prior.stamp(), monotonic=time.monotonic(),
               pid=os.getpid(), **fields)
    guard.append(LOG / 'physx/progress.jsonl', row)
    print(json.dumps(row), flush=True)


def validate_completion(directory, returncode, native_log):
    if returncode != 0:
        raise RuntimeError('NATIVE_EXIT:' + str(returncode))
    control = json.loads((directory / 'control.json').read_text())
    progress = [json.loads(line) for line in (directory / 'progress.jsonl').read_text().splitlines()]
    phases = [row['phase'] for row in progress]
    if ([row['steps'] for row in progress if row['phase'] == 'step_returned'] != list(range(1, 11))
            or control.get('status') != 'CONTROL_COMPLETE' or control.get('steps') != 10
            or control.get('device') != 'cuda:0' or abs(control.get('dt', 0) - .01) > 1e-12):
        raise ValueError('Incomplete synchronized ten-step control')
    required = ('control_complete', 'app_close_begin', 'context_clear_returned', 'native_close_begin')
    if (any(phases.count(name) != 1 for name in required)
            or [phases.index(name) for name in required] != sorted(phases.index(name) for name in required)
            or not all(marker in native_log for marker in SHUTDOWN_MARKERS)):
        raise ValueError('Incomplete native shutdown evidence')
    return dict(control, status='COMPLETE', native_shutdown_verified=True,
                native_returncode=returncode, shutdown_returned=False)


def handshake(native=False):
    plan = checked()
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    state = json.loads((LOG / 'physx/state.json').read_text())
    leader = state['worker_pid']
    identity = (os.getppid() == leader if native else
                os.getpid() == leader and os.getppid() == state['supervisor_pid'])
    if (state['status'] != 'RUNNING' or not identity or os.getpgrp() != leader
            or any(os.environ.get(k) != v for k, v in plan['bootstrap_environment'].items())):
        raise ValueError('Authorized process group and bootstrap required')


def native():
    handshake(native=True)
    directory = LOG / 'physx'
    faulthandler.enable()
    faulthandler.dump_traceback_later(30)
    event('app_launch_begin')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    event('app_launch_returned')
    context_type = []
    def make_sim():
        from isaaclab.sim import SimulationCfg, SimulationContext
        context_type.append(SimulationContext)
        return SimulationContext(SimulationCfg(dt=.01, device='cuda:0'))
    def clear_context():
        if context_type:
            context_type[0].clear_instance()
    class NativeApp:
        def close(self):
            event('native_close_begin')
            app.close()
    def synchronize():
        import torch
        torch.cuda.synchronize()
    try:
        source.source.run_control(source.ContextClosingApp(NativeApp(), clear_context, event),
            make_sim, synchronize, event, lambda row: atomic_json(directory / 'control.json', row))
    finally:
        faulthandler.cancel_dump_traceback_later()


def worker():
    handshake()
    directory = LOG / 'physx'
    with (directory / 'native_console.log').open('x') as stream:
        child = subprocess.run([str(previous.prior.ENV / 'bin/python'), '-B', '-u', '-m',
            'script.a3.humanplus_physx_native_validation', 'native'], input='READY\n',
            text=True, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    atomic_json(directory / 'native_exit.json', dict(returncode=child.returncode,
                at_utc=previous.prior.stamp(), broker_pid=os.getpid()))
    console = (directory / 'native_console.log').read_text()
    print(console, flush=True)
    matches = re.findall(r'Logging to file: (/[^\s\x1b]+\.log)', console)
    kit_root = previous.prior.ENV / 'lib/python3.11/site-packages/isaacsim/kit/logs'
    kit_logs = [Path(name) for name in matches if Path(name).resolve().is_relative_to(kit_root.resolve())]
    if len(kit_logs) != 1:
        raise ValueError('Exactly one native Kit log required')
    receipt = validate_completion(directory, child.returncode, kit_logs[0].read_text())
    event('native_exit_verified', returncode=child.returncode, steps=10)
    atomic_json(directory / 'result.json', receipt)


def run():
    plan = checked()
    if RESULT.exists():
        raise FileExistsError('HP09 already executed; no retry')
    result = dict(experiment=plan['experiment'], status='RUNNING',
        started_at_utc=previous.prior.stamp(), policy_training_started=False, backup_status='LOCAL_ONLY')
    atomic_json(RESULT, result)
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(ROOT), CONDA_PREFIX=str(previous.prior.ENV),
            PATH=str(previous.prior.ENV / 'bin') + ':' + os.environ['PATH'], **plan['bootstrap_environment'])
        state = guard.run_stage(LOG / 'physx', ['/usr/bin/python3', '-B', '-u', '-m',
            'script.a3.humanplus_physx_native_validation', 'worker'], OUT / 'baseline.json', env=env)
        receipt = json.loads((LOG / 'physx/result.json').read_text())
        result.update(status='COMPLETE_PHYSX_CONTROL', state=state, receipt=receipt)
    except BaseException as exc:
        result.update(status='STOPPED', reason=type(exc).__name__ + ':' + str(exc))
        raise
    finally:
        directory = LOG / 'physx'
        if (directory / 'state.json').exists():
            result['state'] = json.loads((directory / 'state.json').read_text())
        result.update(finished_at_utc=previous.prior.stamp(),
            files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                   for path in directory.iterdir() if path.is_file()} if directory.exists() else {})
        atomic_json(RESULT, result)
    print(json.dumps(dict(status=result['status'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'native'))
    action = parser.parse_args().action
    {'register': register, 'run': run, 'worker': worker, 'native': native}[action]()
