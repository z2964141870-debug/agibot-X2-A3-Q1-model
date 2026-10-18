"""HP06: explicitly reauthorized completion of the missing PhysX control."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from script.a3 import humanplus_runtime as prior
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256

ROOT = guard.ROOT
NAME = 'a3_humanplus_physx_20261018_HP06'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
BOOTSTRAP = {'OMNI_KIT_ACCEPT_EULA': 'YES'}
KIT_SOURCE = prior.ENV / 'lib/python3.11/site-packages/isaacsim/kit/kit_app.py'


def register():
    prior.require_root()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP06 already registered; no overwrite or rerun')
    old = prior.checked_plan()
    audit_path = ROOT / 'data/manifests/a3_humanplus_runtime_20261018_HP05_audit.json'
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'STOPPED_ENVIRONMENT_BOOTSTRAP_EULA_EOF'
            or audit['completed_transaction_tests'] != 4
            or audit['completed_mujoco_steps'] != 10 or audit['physx_steps'] != 0):
        raise ValueError('Requires the actual HP05 partial completion')
    for stage in ('tensor', 'mujoco'):
        directory = prior.LOG / stage
        row = next(item for item in audit['stages'] if item['stage'] == stage)
        for name, identity in row['files'].items():
            if sha256(directory / name) != identity['sha256']:
                raise ValueError('Accepted HP05 receipt changed')
    baseline = guard.health.snapshot(json.loads((prior.OUT / 'baseline.json').read_text()))
    if guard.health.reasons(baseline, startup=True):
        raise RuntimeError('Current health registration rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    plan = dict(experiment='A3_HUMANPLUS_PHYSX_20261018_HP06', status='REGISTERED',
        registered_at_utc=prior.stamp(),
        authorization='2026-10-18 user explicitly allowed correcting bootstrap and validating only incomplete PhysX',
        old_hp05_terminal_preserved=True, repeat_accepted_tensor_mujoco=False,
        main_variable='OMNI_KIT_ACCEPT_EULA=YES supplied before importing Isaac Lab',
        bootstrap_environment=BOOTSTRAP,
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        boot_id=baseline['boot_id'], environment=str(prior.ENV),
        code_sha256=dict(old['code_sha256'], **{
            'script/a3/humanplus_physx_validation.py': sha256(Path(__file__))}),
        environment_source_sha256=dict(old['environment_source_sha256'], **{
            str(KIT_SOURCE): sha256(KIT_SOURCE)}),
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
                      (prior.PLAN, audit_path, OUT / 'baseline.json')},
        budget=dict(stage_seconds=900, startup_samples=3, sample_seconds=5,
                    available_startup_gib=8, memory_reserve_gib=4,
                    cpu_limit_c=90, gpu_limit_c=85, hot_samples=2,
                    physx_steps=10, physx_dt=.01, robot_environments=0,
                    policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0),
        criterion='CUDA PhysX empty context completes 10 steps under unchanged health supervision',
        no_automatic_retry=True, backup_status='LOCAL_ONLY',
        scope='PHYSX_BOOTSTRAP_ONLY_NOT_TRAINING_ASSETS_OR_BASELINE_OR_POLICY_IMPROVEMENT')
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', plan=str(PLAN))), flush=True)


def checked():
    prior.require_root()
    plan = json.loads(PLAN.read_text())
    for field in ('code_sha256', 'environment_source_sha256', 'input_sha256'):
        for name, digest in plan[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256(path) != digest:
                raise ValueError('Registered HP06 identity changed: ' + name)
    return plan


def run():
    plan = checked()
    if RESULT.exists():
        raise FileExistsError('HP06 already executed; terminal states cannot resume')
    result = dict(experiment=plan['experiment'], status='RUNNING',
                  started_at_utc=prior.stamp(), policy_training_started=False,
                  backup_status='LOCAL_ONLY')
    atomic_json(RESULT, result)
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(ROOT), CONDA_PREFIX=str(prior.ENV),
            PATH=str(prior.ENV / 'bin') + ':' + os.environ['PATH'],
            **plan['bootstrap_environment'])
        state = guard.run_stage(LOG / 'physx',
            [str(prior.ENV / 'bin/python'), '-B', '-u', '-m',
             'script.a3.humanplus_physx_validation', 'worker'],
            OUT / 'baseline.json', env=env)
        result.update(status='COMPLETE_PHYSX_CONTROL', state=state,
                      receipt=json.loads((LOG / 'physx/result.json').read_text()))
    except BaseException as exc:
        result.update(status='STOPPED', reason=type(exc).__name__ + ':' + str(exc))
        raise
    finally:
        directory = LOG / 'physx'
        result.update(finished_at_utc=prior.stamp(),
            files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                   for path in directory.iterdir() if path.is_file()} if directory.exists() else {})
        if (directory / 'state.json').exists():
            result['state'] = json.loads((directory / 'state.json').read_text())
        atomic_json(RESULT, result)
    print(json.dumps(dict(status=result['status'])), flush=True)


def worker():
    plan = checked()
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    state = json.loads((LOG / 'physx/state.json').read_text())
    if (state['status'] != 'RUNNING' or state['worker_pid'] != os.getpid()
            or state['supervisor_pid'] != os.getppid() or os.getpgrp() != os.getpid()
            or any(os.environ.get(key) != value for key, value in plan['bootstrap_environment'].items())):
        raise ValueError('Requires authorized supervisor and fixed bootstrap environment')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device='cuda:0')
    app = launcher.app
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext
        sim = SimulationContext(SimulationCfg(dt=.01, device='cuda:0'))
        sim.reset()
        for _ in range(10):
            sim.step(render=False)
        import torch
        torch.cuda.synchronize()
        if sim.device != 'cuda:0':
            raise RuntimeError('PhysX did not select CUDA')
        receipt = dict(status='COMPLETE', at_utc=prior.stamp(), steps=10,
                       device=sim.device, dt=sim.get_physics_dt(),
                       robot_environments=0, policy_model_loads=0, policy_optimizer_calls=0,
                       scope='EMPTY_CUDA_PHYSX_CONTEXT_ONLY')
        sim.stop()
        sim.clear()
    finally:
        app.close()
    atomic_json(LOG / 'physx/result.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    args = parser.parse_args()
    {'register': register, 'run': run, 'worker': worker}[args.action]()
