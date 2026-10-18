"""HP07: finite PhysX control with explicit progress and native app shutdown."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from script.a3 import humanplus_physx_validation as previous
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256

ROOT = previous.ROOT
NAME = 'a3_humanplus_physx_exit_20261018_HP07'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
LAB = Path('/home/humanplus/x2_teleop_final/third_party/IsaacLab-v2.3.2')
SOURCES = (
    LAB / 'scripts/tutorials/00_sim/create_empty.py',
    LAB / 'source/isaaclab/isaaclab/sim/simulation_context.py',
    previous.prior.ENV / 'lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py',
)


def register():
    old = previous.checked()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP07 already registered')
    audit_path = previous.PLAN.with_name(previous.NAME + '_audit.json')
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'STOPPED' or audit['reason'] != 'RuntimeError:STAGE_DEADLINE_900_SECONDS'
            or audit['remaining_own_group_processes'] != [] or audit['after_issues'] != []):
        raise ValueError('Requires preserved HP06 terminal audit and cleanup')
    baseline = guard.health.snapshot(json.loads((previous.OUT / 'baseline.json').read_text()))
    if guard.health.reasons(baseline, startup=True):
        raise RuntimeError('Registration health rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    plan = dict(experiment='A3_HUMANPLUS_PHYSX_EXIT_20261018_HP07',
        status='REGISTERED', registered_at_utc=previous.prior.stamp(),
        authorization='2026-10-18 user said fix quickly after HP06 timeout explanation',
        main_variable='Use installed official example app.close lifecycle; omit explicit sim.stop/clear',
        progress='fsync events for startup/reset/each step/synchronize/close; separate control and exit receipts',
        diagnostic='one Python stack dump after 30 seconds; no additional simulation work',
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        boot_id=baseline['boot_id'], bootstrap_environment=previous.BOOTSTRAP,
        code_sha256=dict(old['code_sha256'], **{
            'script/a3/humanplus_physx_exit_validation.py': sha256(Path(__file__))}),
        environment_source_sha256=dict(old['environment_source_sha256'], **{
            str(path): sha256(path) for path in SOURCES}),
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            (previous.PLAN, previous.RESULT, audit_path, OUT / 'baseline.json')},
        budget=old['budget'],
        criterion='Ten synchronized CUDA PhysX steps AND normal app shutdown and worker exit',
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
                raise ValueError('Registered HP07 identity changed: ' + name)
    return plan


def run_control(app, make_sim, synchronize, event, save_control):
    """Keep step completion observable even when native shutdown fails."""
    try:
        event('context_begin')
        sim = make_sim()
        event('context_ready')
        event('reset_begin')
        sim.reset()
        event('reset_returned')
        for step in range(1, 11):
            sim.step(render=False)
            event('step_returned', steps=step)
        event('synchronize_begin')
        synchronize()
        if sim.device != 'cuda:0' or abs(sim.get_physics_dt() - .01) > 1e-12:
            raise RuntimeError('PhysX device/dt mismatch')
        receipt = dict(status='CONTROL_COMPLETE', at_utc=previous.prior.stamp(),
            steps=10, device=sim.device, dt=sim.get_physics_dt(),
            robot_environments=0, policy_model_loads=0, policy_optimizer_calls=0)
        save_control(receipt)
        event('control_complete', steps=10)
    finally:
        event('app_close_begin')
        app.close()
        event('app_close_returned')
    return dict(receipt, status='COMPLETE', shutdown_returned=True)


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
        def make_sim():
            from isaaclab.sim import SimulationCfg, SimulationContext
            return SimulationContext(SimulationCfg(dt=.01, device='cuda:0'))
        def synchronize():
            import torch
            torch.cuda.synchronize()
        receipt = run_control(app, make_sim, synchronize, event,
                              lambda row: atomic_json(directory / 'control.json', row))
        atomic_json(directory / 'result.json', receipt)
        event('complete', steps=10)
    finally:
        faulthandler.cancel_dump_traceback_later()


def run():
    plan = checked()
    if RESULT.exists():
        raise FileExistsError('HP07 already executed; no retry')
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
             'script.a3.humanplus_physx_exit_validation', 'worker'],
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
