"""HP08: unregister the installed IsaacLab STOP callback before native close."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from script.a3 import humanplus_physx_exit_validation as source
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256

previous = source.previous
ROOT = source.ROOT
NAME = 'a3_humanplus_physx_callback_20261018_HP08'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')


class ContextClosingApp:
    def __init__(self, app, clear_context, event):
        self.app, self.clear_context, self.event = app, clear_context, event

    def close(self):
        self.event('context_clear_begin')
        self.clear_context()
        self.event('context_clear_returned')
        self.app.close()


def register():
    old = source.checked()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP08 already registered')
    audit_path = source.PLAN.with_name(source.NAME + '_audit.json')
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'STOPPED' or not audit['python_timeout_stack_present']
            or audit['receipts']['control']['steps'] != 10
            or audit['last_progress']['phase'] != 'app_close_begin'
            or audit['after_issues'] or audit['remaining_own_group_processes']):
        raise ValueError('Requires HP07 shutdown-loop evidence and cleanup')
    baseline = guard.health.snapshot(json.loads((source.OUT / 'baseline.json').read_text()))
    if guard.health.reasons(baseline, startup=True):
        raise RuntimeError('Registration health rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    plan = dict(experiment='A3_HUMANPLUS_PHYSX_CALLBACK_20261018_HP08',
        status='REGISTERED', registered_at_utc=previous.prior.stamp(),
        authorization='2026-10-18 user quick-fix request; continue confirmed callback lifecycle repair',
        main_variable='Public SimulationContext.clear_instance unsubscribes STOP callback before app.close',
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        boot_id=baseline['boot_id'], bootstrap_environment=previous.BOOTSTRAP,
        code_sha256=dict(old['code_sha256'], **{
            'script/a3/humanplus_physx_callback_validation.py': sha256(Path(__file__))}),
        environment_source_sha256=old['environment_source_sha256'],
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            (source.PLAN, source.RESULT, audit_path, OUT / 'baseline.json')},
        budget=old['budget'],
        criterion='Ten synchronized CUDA steps, public callback cleanup, normal app shutdown and exit',
        no_automatic_retry=True, repeat_passed_controls=False,
        policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0,
        backup_status='LOCAL_ONLY')
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
                raise ValueError('Registered HP08 identity changed: ' + name)
    return plan


def worker():
    plan = checked()
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    directory = LOG / 'physx'
    state = json.loads((directory / 'state.json').read_text())
    if (state['status'] != 'RUNNING' or state['worker_pid'] != os.getpid()
            or state['supervisor_pid'] != os.getppid() or os.getpgrp() != os.getpid()
            or any(os.environ.get(k) != v for k, v in plan['bootstrap_environment'].items())):
        raise ValueError('Fixed authorized supervisor required')
    def event(phase, **fields):
        row = dict(phase=phase, at_utc=previous.prior.stamp(), monotonic=time.monotonic(),
                   pid=os.getpid(), **fields)
        guard.append(directory / 'progress.jsonl', row)
        print(json.dumps(row), flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(30)
    try:
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
        def synchronize():
            import torch
            torch.cuda.synchronize()
        closing_app = ContextClosingApp(app, clear_context, event)
        receipt = source.run_control(closing_app, make_sim, synchronize, event,
            lambda row: atomic_json(directory / 'control.json', row))
        atomic_json(directory / 'result.json', receipt)
        event('complete', steps=10)
    finally:
        faulthandler.cancel_dump_traceback_later()


def run():
    plan = checked()
    if RESULT.exists():
        raise FileExistsError('HP08 already executed; no retry')
    result = dict(experiment=plan['experiment'], status='RUNNING',
        started_at_utc=previous.prior.stamp(), policy_training_started=False,
        backup_status='LOCAL_ONLY')
    atomic_json(RESULT, result)
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(ROOT), CONDA_PREFIX=str(previous.prior.ENV),
            PATH=str(previous.prior.ENV / 'bin') + ':' + os.environ['PATH'],
            **plan['bootstrap_environment'])
        state = guard.run_stage(LOG / 'physx',
            [str(previous.prior.ENV / 'bin/python'), '-B', '-u', '-m',
             'script.a3.humanplus_physx_callback_validation', 'worker'],
            OUT / 'baseline.json', env=env)
        receipt = json.loads((LOG / 'physx/result.json').read_text())
        if receipt.get('steps') != 10 or not receipt.get('shutdown_returned'):
            raise ValueError('Incomplete control or shutdown')
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
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    action = parser.parse_args().action
    {'register': register, 'run': run, 'worker': worker}[action]()
