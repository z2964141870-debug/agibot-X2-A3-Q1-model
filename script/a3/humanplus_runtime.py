"""HP05: independent, single-execution humanplus environment controls."""

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from script.a3 import humanplus_health as health
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.project_tools.humanplus_inventory import package_metadata

ROOT = guard.ROOT
ENV = Path('/home/humanplus/anaconda3/envs/x2-sonic-isaaclab')
NAME = 'a3_humanplus_runtime_20261018_HP05'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
PIN = 'fe6868ba37034f89b912f0fb851bce19f120266d'
MJCF = VENDOR / 'gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml'
STAGES = ('tensor', 'mujoco', 'physx')
CODE = ('script/a3/humanplus_runtime.py', 'script/a3/humanplus_supervisor.py',
        'script/a3/test_humanplus_supervisor.py', 'script/a3/humanplus_health.py',
        'script/a3/finite_adam.py', 'script/a3/test_finite_adam.py',
        'script/a3/checkpoint_store.py', 'script/project_tools/humanplus_inventory.py')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require_root():
    if Path.cwd().resolve() != ROOT or Path.home() != ROOT.parent:
        raise ValueError('Only the authorized humanplus root supported')


def register():
    require_root()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP05 already registered; no reset or rerun')
    vendor_head = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(VENDOR), 'status', '--porcelain'], text=True).strip()
    if vendor_head != PIN or dirty:
        raise ValueError('Pinned clean vendor required')
    verified_path = ROOT / 'data/manifests/a3_humanplus_large_transfer_verification_20261018_HP04.json'
    verified = json.loads(verified_path.read_text())
    if verified['status'] != 'COMPLETE_VERIFIED' or verified['file_count'] != 63:
        raise ValueError('Requires independent HP04 input verification')
    for row in verified['files']:
        if (ROOT / row['path']).stat().st_size != row['bytes']:
            raise ValueError('Migrated input missing or size changed')
    prior_path = ROOT / 'logs/a3_humanplus_20261017/telemetry_readonly_probe_20261018.json'
    if sha256(prior_path) != '7ca40672be0646e8c816d8116e55632619d98017d57b3281fd0fb5ad780aab21':
        raise ValueError('Trusted kernel/memory provenance changed')
    old = json.loads(prior_path.read_text())['samples'][0]
    baseline = health.snapshot(old)
    if health.reasons(baseline, startup=True):
        raise RuntimeError('Health registration rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    assets_path = ROOT / 'data/manifests/a3_humanplus_mjcf_assets_20261017.json'
    assets = json.loads(assets_path.read_text())
    for row in assets['assets']:
        path = VENDOR / row['path']
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Fixed MJCF mesh changed: ' + row['path'])
    lab = Path('/home/humanplus/x2_teleop_final/third_party/IsaacLab-v2.3.2')
    environment_sources = [ENV / 'bin/python3.11',
        lab / 'scripts/tutorials/00_sim/create_empty.py',
        lab / 'source/isaaclab/isaaclab/sim/simulation_cfg.py',
        lab / 'source/isaaclab/isaaclab/sim/simulation_context.py',
        lab / 'source/isaaclab/isaaclab/app/app_launcher.py']
    plan = dict(experiment='A3_HUMANPLUS_RUNTIME_20261018_HP05', status='REGISTERED',
        registered_at_utc=stamp(), host='humanplus@100.101.30.68', root=str(ROOT),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        boot_id=baseline['boot_id'], environment=str(ENV),
        packages=package_metadata(ENV), environment_read_only=True,
        code_sha256={name: sha256(ROOT / name) for name in CODE},
        environment_source_sha256={str(path): sha256(path) for path in environment_sources},
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
                      (verified_path, assets_path, MJCF, OUT / 'baseline.json')},
        mesh_sha256={row['path']: row['sha256'] for row in assets['assets']},
        vendor_commit=PIN, historical_health_reused_as_runtime_approval=False,
        stages=list(STAGES), budget=dict(stage_seconds=900, startup_samples=3,
            interval_seconds=5, telemetry_timeout_seconds=4, startup_available_gib=8,
            runtime_reserve_gib=4, cpu_limit_c=90, gpu_limit_c=85, hot_samples=2,
            transaction_tests=4, native_optimizer_calls_at_most=20,
            mujoco_steps=10, physx_steps=10, physx_environments=0,
            policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0),
        criteria='All 4 transaction tests; finite MJCF 10 steps; CUDA PhysX context 10 steps; reliable supervision',
        no_automatic_retry=True, stopped_must_not_resume=True, backup_status='LOCAL_ONLY',
        scope='ENVIRONMENT_CONTROLS_ONLY_NOT_BASELINE_OR_POLICY_IMPROVEMENT')
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', plan=str(PLAN))), flush=True)


def checked_plan():
    plan = json.loads(PLAN.read_text())
    for name, digest in plan['code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered code changed: ' + name)
    for name, digest in plan['environment_source_sha256'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Registered environment source changed')
    for name, digest in plan['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered runtime input changed')
    for name, digest in plan['mesh_sha256'].items():
        if sha256(VENDOR / name) != digest:
            raise ValueError('Registered mesh changed')
    return plan


def run():
    require_root()
    checked_plan()
    result = dict(experiment='A3_HUMANPLUS_RUNTIME_20261018_HP05',
                  status='RUNNING', started_at_utc=stamp(), stages=[],
                  training_started=False, policy_model_loaded=False, backup_status='LOCAL_ONLY')
    if RESULT.exists():
        raise FileExistsError('HP05 already executed')
    atomic_json(RESULT, result)
    try:
        for stage in STAGES:
            checked_plan()
            directory = LOG / stage
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONPATH=str(ROOT), CONDA_PREFIX=str(ENV),
                       PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
            state = guard.run_stage(directory,
                [str(ENV / 'bin/python'), '-B', '-m', 'script.a3.humanplus_runtime', 'worker',
                 '--stage', stage], OUT / 'baseline.json', env=env)
            result['stages'].append(dict(stage=stage, state=state,
                receipt=json.loads((directory / 'result.json').read_text()),
                health_sha256=sha256(directory / 'health.jsonl'),
                console_sha256=sha256(directory / 'console.log')))
            atomic_json(RESULT, result)
            print(json.dumps(dict(stage=stage, status='COMPLETE')), flush=True)
        result['status'] = 'COMPLETE_RUNTIME_CONTROLS'
    except BaseException as exc:
        result.update(status='STOPPED', reason=type(exc).__name__ + ':' + str(exc))
        raise
    finally:
        result['finished_at_utc'] = stamp()
        atomic_json(RESULT, result)


def worker(stage):
    require_root()
    checked_plan()
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor launch handshake missing')
    directory = LOG / stage
    state = json.loads((directory / 'state.json').read_text())
    if state['status'] != 'RUNNING' or state['worker_pid'] != os.getpid() or os.getpgrp() != os.getpid():
        raise ValueError('Worker must run once under its registered supervisor')
    result = dict(status='COMPLETE', stage=stage, at_utc=stamp(), policy_model_loaded=False,
                  policy_optimizer_calls=0)
    if stage == 'tensor':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        from script.a3 import test_finite_adam
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_finite_adam)
        tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        print(stream.getvalue(), flush=True)
        torch.cuda.synchronize()
        if tested.testsRun != 4 or not tested.wasSuccessful():
            raise RuntimeError('CPU/fused CUDA transaction controls failed')
        result.update(tests_passed=4, torch_version=torch.__version__,
                      cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(),
                      capability=list(torch.cuda.get_device_capability()),
                      allocator_peak_bytes=torch.cuda.max_memory_allocated())
    elif stage == 'mujoco':
        import mujoco
        import numpy as np
        model = mujoco.MjModel.from_xml_path(str(MJCF))
        data = mujoco.MjData(model)
        for _ in range(10):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Nonfinite MuJoCo state')
        result.update(mujoco_version=mujoco.__version__, model_nq=model.nq,
                      model_nv=model.nv, steps=10, simulated_seconds=float(data.time),
                      scope='UNCONTROLLED_10_STEPS_NOT_POLICY_OR_BALANCE_VALIDATION')
    elif stage == 'physx':
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, device='cuda:0')
        app = launcher.app
        try:
            from isaaclab.sim import SimulationCfg, SimulationContext
            sim = SimulationContext(SimulationCfg(dt=0.01, device='cuda:0'))
            sim.reset()
            for _ in range(10):
                sim.step(render=False)
            import torch
            torch.cuda.synchronize()
            if sim.device != 'cuda:0':
                raise RuntimeError('PhysX did not select CUDA')
            result.update(device=sim.device, steps=10, dt=sim.get_physics_dt(),
                          scope='EMPTY_PHYSX_CONTEXT_NOT_ROBOT_TRAINING_OR_ASSET_CLOSURE')
            sim.stop()
            sim.clear()
        finally:
            app.close()
    atomic_json(directory / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'run':
        run()
    else:
        if args.stage is None:
            parser.error('--stage required')
        worker(args.stage)
