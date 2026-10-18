"""HP10: single-use humanplus asset/model validation and matched 035 baseline."""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from script.a3 import humanplus_runtime as runtime
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256

ROOT, ENV, VENDOR = runtime.ROOT, runtime.ENV, runtime.VENDOR
NAME = 'a3_humanplus_baseline_20261018_HP10'
OUT = ROOT / 'data/experiments' / NAME
LOG = ROOT / 'logs' / NAME
PLAN = ROOT / 'data/manifests' / (NAME + '_plan.json')
RESULT = ROOT / 'data/manifests' / (NAME + '_result.json')
ASSET_RECEIPT = ROOT / 'data/manifests/a3_humanplus_policy_assets_20261018.json'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
E26 = ROOT / 'data/experiments/a3_zhy_smpl_frozen_20261013_E26/registration.json'
E02 = ROOT / 'data/experiments/a3_zhy_search_20261012_E02/job.json'
STAGES = ('validate', 'cloth', 'official')
CODE = ('script/a3/humanplus_baseline.py', 'script/a3/test_humanplus_baseline.py',
        'script/project_tools/humanplus_baseline_audit.py',
        'script/a3/humanplus_runtime.py', 'script/a3/humanplus_supervisor.py',
        'script/a3/humanplus_health.py', 'script/a3/checkpoint_store.py',
        'script/a3/smpl_frozen_probe.py', 'script/a3/smpl_frozen_entry.py',
        'script/a3/sim_entry.py', 'script/a3/reference_contract.py',
        'script/a3/fullchain_support.py', 'script/a3/evaluate_mujoco.py',
        'script/a3/evaluation_common.py', 'script/a3/frozen_evaluation_launch.py',
        'script/a3/autoresume.py', 'script/a3/host_health.py', 'script/a3/zhy_trial.py')


def read(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(['git', '-C', str(VENDOR), *args], text=True).strip()


def clean_vendor():
    if git('rev-parse', 'HEAD') != runtime.PIN or git('status', '--porcelain'):
        raise ValueError('Pinned clean vendor required')


def prepare_assets():
    runtime.require_root()
    clean_vendor()
    if ASSET_RECEIPT.exists():
        raise FileExistsError('Asset closure already recorded')
    asset = VENDOR / 'gear_sonic/data/assets/robot_description'
    paths = {runtime.MJCF, asset / 'mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml'}
    urdf = asset / 'urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf'
    paths.add(urdf)
    for xml in tuple(paths):
        tree = ET.parse(xml).getroot()
        if tree.findall('.//include'):
            raise ValueError('Unexpected XML include requires explicit closure review')
        compiler = tree.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        for node in tree.findall('.//mesh'):
            name = node.get('file') or node.get('filename')
            if name:
                paths.add((xml.parent / meshdir / name).resolve())
    solver = asset / 'solver/a3_loop'
    paths.add(solver / 'lib/liba3_ankle_waist_solver.a')
    paths.update((solver / 'include').rglob('*'))
    rows, missing = [], []
    for path in sorted(p for p in paths if not p.is_dir()):
        rel = str(path.relative_to(VENDOR))
        content = subprocess.check_output(['git', '-C', str(VENDOR), 'show', 'HEAD:' + rel])
        if content.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            fields = dict(line.split(' ', 1) for line in content.decode().splitlines())
            size, digest = int(fields['size']), fields['oid'].removeprefix('sha256:')
        else:
            import hashlib
            size, digest = len(content), hashlib.sha256(content).hexdigest()
        ready = path.is_file() and path.stat().st_size == size and sha256(path) == digest
        rows.append(dict(path=rel, bytes=size, sha256=digest, previously_materialized=ready))
        if not ready:
            if not content.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                raise ValueError('Non-LFS tracked asset changed: ' + rel)
            missing.append(rel)
    if missing:
        subprocess.run(['git', '-C', str(VENDOR), 'lfs', 'pull', '--include=' + ','.join(missing),
                        '--exclude='], check=True, timeout=300)
    for row in rows:
        path = VENDOR / row['path']
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Asset closure checksum mismatch: ' + row['path'])
    clean_vendor()
    receipt = dict(status='VERIFIED_STATIC_ASSET_CLOSURE', at_utc=runtime.stamp(),
        vendor_commit=runtime.PIN, assets=rows, newly_materialized=missing,
        bytes=sum(r['bytes'] for r in rows), source='PINNED_GIT_OBJECTS_AND_LFS_OIDS',
        scope='EVALUATION_MJCF_TRAINING_URDF_AND_LOOP_SOLVER_STATIC_FILES_ONLY',
        training_environment_instantiation_verified=False, backup_status='LOCAL_ONLY')
    atomic_json(ASSET_RECEIPT, receipt)
    print(json.dumps(dict(status=receipt['status'], assets=len(rows), downloaded=len(missing))))


def motions_from_history(clothing, official):
    if len(clothing['selection']) != 6 or len(official['diagnostic_motions']) != 4:
        raise ValueError('Exactly six clothing and four official motions required')
    motions = []
    for row in clothing['selection']:
        motions.append(dict(key=Path(row['filename']).stem, group='cloth',
            path=str(Path(row['source']) / 'reference.csv'), sequence=row['sequence'],
            original_partition=row['partition'], reference_fps=30., frame_stride=1))
    for row in official['diagnostic_motions']:
        motions.append(dict(key=Path(row['name']).stem, group='official',
            path='script/vendor/sonic_for_a3/a3_data/agibot_a3/' + row['name'],
            reference_fps=30., frame_stride=4, expected_sha256=row['sha256']))
    if len(motions) != 10 or len({r['key'] for r in motions}) != 10:
        raise ValueError('Exactly six clothing and four official motions required')
    if any(Path(r['path']).is_absolute() or '..' in Path(r['path']).parts for r in motions):
        raise ValueError('Rebuilt project-relative input paths required')
    return motions


def register():
    runtime.require_root()
    if OUT.exists() or PLAN.exists() or RESULT.exists():
        raise FileExistsError('HP10 already registered')
    clean_vendor()
    accepted = ROOT / 'data/manifests/a3_humanplus_physx_native_20261018_HP09_audit.json'
    if not read(accepted)['verified_control_and_exit']:
        raise ValueError('Requires HP09 accepted control')
    assets = read(ASSET_RECEIPT)
    if assets['status'] != 'VERIFIED_STATIC_ASSET_CLOSURE':
        raise ValueError('Static asset closure required')
    prior = read(E26)
    motions = motions_from_history(prior, read(E02))
    migrated = read(ROOT / 'data/manifests/a3_humanplus_large_transfer_verification_20261018_HP04.json')
    inputs = {r['path']: r['sha256'] for r in migrated['files']}
    for name, digest in inputs.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Migrated input changed: ' + name)
    for row in motions:
        path = ROOT / row['path']
        digest = sha256(path)
        if digest != row.get('expected_sha256', inputs.get(row['path'])):
            raise ValueError('Reference SHA does not match provenance')
        with path.open(newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        selected = len(range(0, count, row['frame_stride']))
        row.update(sha256=digest, csv_rows=count,
            full_policy_steps=max(1, math.ceil(((selected - 1) / 30. - 1e-12) * 50)))
        inputs[row['path']] = digest
    for row in assets['assets']:
        path = VENDOR / row['path']
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Static closure changed')
        inputs[str(path.relative_to(ROOT))] = row['sha256']
    if sha256(OFFICIAL) != '9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d':
        raise ValueError('Official035 SHA mismatch')
    baseline = guard.health.snapshot(read(ROOT / 'data/experiments/a3_humanplus_physx_native_20261018_HP09/baseline.json'))
    if guard.health.reasons(baseline, startup=True):
        raise RuntimeError('Registration telemetry rejected')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'baseline.json', baseline)
    files = [ASSET_RECEIPT, accepted, E26, E02, OUT / 'baseline.json']
    inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in files})
    # Pin all tracked vendor source/config files, not unmaterialized unrelated LFS objects.
    vendor_sources = git('ls-files', 'gear_sonic', 'train_a3_035_fromscratch.sh').splitlines()
    for name in vendor_sources:
        path = VENDOR / name
        if path.suffix in ('.py', '.yaml', '.cpp', '.h', '.hpp', '.sh'):
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    plan = dict(experiment=NAME.upper(), status='REGISTERED', registered_at_utc=runtime.stamp(),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        host='humanplus@100.101.30.68', root=str(ROOT), environment=str(ENV),
        boot_id=baseline['boot_id'], vendor_commit=runtime.PIN,
        code_sha256={n: sha256(ROOT / n) for n in CODE}, input_sha256=inputs,
        environment_source_sha256=read(ROOT / 'data/manifests/a3_humanplus_runtime_20261018_HP05_plan.json')['environment_source_sha256'],
        motions=motions, selection=prior['selection'], stages=list(STAGES),
        reference_contract={k: prior[k] for k in ('csv_roundtrip_max_error',
            'model_body_fk_max_m', 'reference_floor_budget_m')},
        protocol=dict(seed=0, encoder='a3_fast', action_delay_ms=0., reference_buffer_ms=0.,
            max_policy_steps=None, capture_count=100000, fall_height_m=.45,
            fall_abs_roll_or_pitch_deg=60., after_fall_frames_included=True),
        budget=dict(stage_seconds=900, startup_samples=3, interval_seconds=5,
            startup_available_gib=8, runtime_reserve_gib=4, cpu_limit_c=90, gpu_limit_c=85,
            hot_samples=2, model_validation_actor_instances=1, validation_forward_calls=0,
            complete_policy_rollouts=10, policy_steps=sum(r['full_policy_steps'] for r in motions),
            optimizer_calls=0, training_updates=0),
        historical_contracts_used_for='INPUT_SELECTION_AND_PROTOCOL_ONLY_NO_OLD_HOST_HEALTH_OR_EFFECTS',
        scope='NEW_HOST_MATCHED_DEVELOPMENT_BASELINE_NOT_POLICY_IMPROVEMENT_OR_UNSEEN_TEST',
        no_automatic_retry=True, stopped_must_not_resume=True, backup_status='LOCAL_ONLY')
    atomic_json(PLAN, plan)
    atomic_json(OUT / 'registration.json', plan)
    print(json.dumps(dict(status='REGISTERED', motions=10, policy_steps=plan['budget']['policy_steps'])))


def checked():
    runtime.require_root()
    plan = read(PLAN)
    clean_vendor()
    for field in ('code_sha256', 'input_sha256', 'environment_source_sha256'):
        for name, digest in plan[field].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256(path) != digest:
                raise ValueError('Registered HP10 identity changed: ' + name)
    return plan


def require_next(result, stage):
    if result['status'] not in ('REGISTERED', 'IN_PROGRESS'):
        raise ValueError('Terminal or running HP10 must never resume')
    completed = [r['stage'] for r in result['stages']]
    if completed != list(STAGES[:STAGES.index(stage)]):
        raise ValueError('Stage must run exactly once in registered order')


class EvaluationLaunch:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, **kwargs):
        if argv[:3] != [str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m', 'script.a3.sim_entry']:
            raise ValueError('Unexpected evaluation launch')
        argv = [str(ENV / 'bin/python'), '-B', '-m', 'script.a3.smpl_frozen_entry', *argv[3:]]
        kwargs['start_new_session'] = False
        return subprocess.Popen(argv, **kwargs)


def validate_model(plan):
    import torch
    from script.a3.reference_contract import load_sim
    from script.a3 import smpl_frozen_probe as abi
    torch.set_num_threads(1)
    torch.manual_seed(0)
    sim = load_sim()
    payload = sim.load_torch_checkpoint_without_trl_imports(OFFICIAL)
    networks = {}
    for name in ('policy_state_dict', 'value_state_dict'):
        state = payload[name]
        if not state or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError('Nonfinite or missing official network tensors')
        networks[name] = dict(tensors=len(state), elements=sum(v.numel() for v in state.values()))
    del payload
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    policy = sim.A3Policy(OFFICIAL, torch.device('cuda:0'), encoder='a3_fast')
    if policy.encoder_input_dim != 640 or policy.actor_obs_dim != 930:
        raise ValueError('Unexpected035 input ABI')
    torch.cuda.synchronize()
    del policy
    abi.OUT = OUT / 'validation'
    abi.OUT.mkdir()
    contract = abi.reference_contract(dict(selection=plan['selection'], **plan['reference_contract']))
    if [r['full_policy_steps'] for r in contract] != [r['full_policy_steps'] for r in plan['motions'] if r['group'] == 'cloth']:
        raise ValueError('Reference resampling budget mismatch')
    return dict(status='COMPLETE', stage='validate', networks=networks,
        strict_actor_load_device='cuda:0', encoder_input_dim=640, actor_obs_dim=930,
        policy_forward_calls=0, optimizer_calls=0, reference_contract=contract,
        scope='STATIC_ASSETS_FINITE_ACTOR_CRITIC_TENSORS_STRICT_INFERENCE_ACTOR_AND_REFERENCE_ABI')


def evaluate(plan, stage):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import arguments
    output = OUT / stage
    if output.exists():
        raise FileExistsError('No motion-level resumption permitted')
    output.mkdir()
    references = OUT / (stage + '_references')
    references.mkdir()
    for row in plan['motions']:
        if row['group'] == stage:
            (references / (row['key'] + '.csv')).symlink_to(ROOT / row['path'])
    args = arguments(output, LOG / stage / 'motions', OFFICIAL, references)
    args.frame_stride = 1 if stage == 'cloth' else 4
    original = evaluation.subprocess
    evaluation.subprocess = EvaluationLaunch()
    evaluation.preflight = lambda: dict(supervision='HUMANPLUS_PARENT_CONTINUOUS_5S')
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    try:
        code = evaluation.evaluate_explicit(args)
    finally:
        evaluation.subprocess = original
    if code:
        raise RuntimeError('Baseline replay failed; no retry')
    summary = evaluation.summarize_explicit(output)
    verify_replays(plan, stage, summary)
    return dict(status='COMPLETE', stage=stage, motions=summary['motions'], optimizer_calls=0)


def verify_replays(plan, stage, summary):
    import numpy as np
    selected = [r for r in plan['motions'] if r['group'] == stage]
    if {r['motion'] for r in summary['motions']} != {r['key'] + '.csv' for r in selected}:
        raise ValueError('Incomplete motion set')
    for row in selected:
        output = OUT / stage
        metric = read(output / (row['key'] + '.metrics.json'))['motions'][0]
        trace = read(output / (row['key'] + '.timeseries.json'))['motions'][0]
        n = row['full_policy_steps']
        if metric['num_policy_steps'] != n or trace['policy_tick'] != list(range(n)):
            raise ValueError('Incomplete full replay')
        height, roll, pitch = (np.asarray(trace[k]) for k in ('root_height', 'root_roll_deg', 'root_pitch_deg'))
        falling = np.flatnonzero((height < .45) | (abs(roll) > 60) | (abs(pitch) > 60))
        if metric['fall_tick'] != (int(falling[0]) if len(falling) else None) or metric['fall'] != bool(len(falling)):
            raise ValueError('Independent fall check failed')
        with np.load(output / (row['key'] + '.inputs.npz'), allow_pickle=False) as captured:
            if captured['observations'].shape != (n, 1570) or captured['actions'].shape != (n, 29):
                raise ValueError('Policy input/action coverage mismatch')
            if not np.isfinite(captured['observations']).all():
                raise ValueError('Nonfinite policy inputs')
            np.testing.assert_array_equal(captured['actions'], trace['raw_action_29'])
        with np.load(output / (row['key'] + '.contact.npz'), allow_pickle=False) as contacts:
            if not np.array_equal(contacts['policy_tick'], np.arange(n)):
                raise ValueError('Contact coverage mismatch')
            for key in ('qpos', 'qvel', 'foot_floor_normal_force_n', 'foot_geometry_min_z_m'):
                if contacts[key].shape[0] != n or not np.isfinite(contacts[key]).all():
                    raise ValueError('Invalid contact/full-state capture')


def worker(stage):
    plan = checked()
    if sys.stdin.readline() != 'READY\n':
        raise ValueError('Supervisor handshake required')
    state = read(LOG / stage / 'state.json')
    if (state['status'] != 'RUNNING' or state['worker_pid'] != os.getpid()
            or state['supervisor_pid'] != os.getppid() or os.getpgrp() != os.getpid()):
        raise ValueError('Registered supervisor and own process group required')
    receipt = validate_model(plan) if stage == 'validate' else evaluate(plan, stage)
    checked()
    atomic_json(LOG / stage / 'result.json', receipt)


def run(stage):
    runtime.require_root()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked()
        result = read(RESULT) if RESULT.exists() else dict(status='REGISTERED', stages=[], experiment=NAME.upper())
        require_next(result, stage)
        result.update(status='RUNNING', current_stage=stage, started_at_utc=runtime.stamp())
        atomic_json(RESULT, result)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=str(ROOT), CONDA_PREFIX=str(ENV),
                PATH=str(ENV / 'bin') + ':' + os.environ['PATH'],
                A3_LOOP_BRIDGE_CACHE=str(OUT / 'solver_bridge'))
            state = guard.run_stage(LOG / stage, [str(ENV / 'bin/python'), '-B', '-u', '-m',
                'script.a3.humanplus_baseline', 'worker', '--stage', stage], OUT / 'baseline.json', env=env)
            receipt = read(LOG / stage / 'result.json')
            result['stages'].append(dict(stage=stage, state=state, receipt=receipt))
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
    parser.add_argument('action', choices=('assets', 'register', 'run', 'worker'))
    parser.add_argument('--stage', choices=STAGES, default='validate')
    args = parser.parse_args()
    {'assets': prepare_assets, 'register': register,
     'run': lambda: run(args.stage), 'worker': lambda: worker(args.stage)}[args.action]()
