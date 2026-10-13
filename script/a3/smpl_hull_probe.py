"""E23 replace empty AABB corners with actual foot hull support vertices."""

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_corpus_qualify as corpus
from script.a3 import smpl_envelope_probe as previous
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = corpus.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_hull_20261013_E23'
LOG = ROOT / 'logs/a3_zhy_smpl_hull_20261013_E23'
PRIOR = previous.OUT
CODE = previous.CODE + ('script/a3/smpl_hull_floor.py', 'script/a3/smpl_hull_probe.py',
                        'script/a3/test_smpl_hull_floor.py')


def configure():
    corpus.configure_helpers()
    guard.OUT, guard.LOG = OUT, LOG


def register():
    configure()
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E23')
    value = guard.preflight()
    before = corpus.read(PRIOR / 'independent_verification.json')
    old = corpus.read(PRIOR / 'registration.json')
    if before['status'] != 'VERIFIED_QUALITY_REJECTION' or before['health']['boot_ids'] != [value['boot_id']]:
        raise ValueError('Requires same-boot verified E22')
    corpus.unchanged(old)
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    from script.a3.smpl_hull_floor import HullA3Retargeter
    from scipy.spatial import _qhull
    retargeter = HullA3Retargeter()
    inputs = dict(old['input_sha256'])
    qhull_path = Path(_qhull.__file__)
    inputs[str(qhull_path.relative_to(ROOT))] = sha256(qhull_path)
    for row in before['sequences']:
        for name in ('kinematic_trace.npz', 'retarget_summary.json'):
            path = PRIOR / 'sequences' / row['sequence'].replace('/', '__') / name
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    job = dict(old, experiment='A3_ZHY_SMPL_HULL_20261013_E23', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        code_sha256={name:sha256(ROOT/name) for name in CODE}, input_sha256=inputs,
        kernel_baseline=kernel['stdout'].splitlines(), baseline_oom_kill=value['container_memory']['oom_kill'],
        parent_experiment=old['experiment'], main_variable='ACTUAL_FOOT_HULL_VERTEX_FLOOR_CONSTRAINT',
        collision_limit='FOOT_GEOM_ACTUAL_HULL_VERTEX_FLOOR',
        floor_constraint='ALL_ACTUAL_FOOT_MESH_CONVEX_HULL_VERTICES',
        envelope_conservative=False, geometry_contract=retargeter.geometry_contract(),
        scipy_version=importlib.metadata.version('scipy'),
        degenerate_hull_fallback='ALL_ACTUAL_VERTICES_NO_JITTER',
        original_quality_gates_retained=True)
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT/'registration.json', job)
    atomic_json(OUT/'state.json', dict(status='READY'))
    append_record(LOG/'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], geometry=job['geometry_contract'])))


def compute_rows(job, verification=False):
    import numpy as np
    from q1_retarget import BodyReference
    from script.a3.mocap_adapter import BODY_MAP
    from script.a3.smpl_base_retarget import base_pair_quality
    from script.a3.smpl_hull_floor import HullA3Retargeter
    retargeter = HullA3Retargeter()
    if retargeter.geometry_contract() != job['geometry_contract']:
        raise ValueError('Registered hull geometry changed')
    prior_rows = corpus.read(PRIOR/'result.json')['sequences']
    recorded = corpus.read(OUT/'result.json')['sequences'] if verification else None
    rows = []
    for number, old in enumerate(prior_rows):
        directory = OUT/'sequences'/old['sequence'].replace('/', '__')
        source_dir = ROOT/old['source_original_clip']
        source = BodyReference.load(source_dir/'body_reference.npz', list(BODY_MAP))
        with np.load(source_dir/'human_source.npz', allow_pickle=False) as human:
            joints = human['joints'].copy()
        retargeter.set_human_source(joints)
        if verification:
            expected = recorded[number]
            if expected['sequence'] != old['sequence']:
                raise ValueError('Source sequence changed')
            for name, identity in expected['artifacts'].items():
                if sha256(directory/name) != identity['sha256']:
                    raise ValueError('Artifact SHA changed')
            summary = corpus.read(directory/'retarget_summary.json')
            target, _ = retargeter.targets(source, old['scale'])
        else:
            directory.mkdir(parents=True)
            summary = retargeter.solve(source, old['scale'], directory)
        with np.load(directory/'kinematic_trace.npz', allow_pickle=False) as trace:
            quality = base_pair_quality(summary, trace, joints, retargeter)
            if verification:
                with np.load(directory/'integration_checks.npz', allow_pickle=False) as steps:
                    checks = steps['records']
                    if (len(checks) != 120+8*(len(trace['qpos'])-1) or np.any(checks[:,2] < 0)
                            or np.any(checks[:,3] < 0) or np.any(checks[:,3] > 1)
                            or not np.isfinite(checks).all()):
                        raise ValueError('Nonlinear integration evidence failed')
                if (not np.array_equal(target, trace['target_position_m'])
                        or not np.array_equal(retargeter.root_offset, trace['root_correction_m'])
                        or not np.array_equal(retargeter.contact_mask, trace['inferred_contact'])):
                    raise ValueError('Independent targets or contact differ')
                support_error = 0.
                for frame, qpos in enumerate(trace['qpos']):
                    retargeter.configuration.update(qpos)
                    actual = retargeter.configuration.data.xpos[retargeter.bodies]
                    if not np.allclose(actual, trace['actual_position_m'][frame], atol=1e-12, rtol=0):
                        raise ValueError('Independent FK differs')
                    if not np.allclose(retargeter.foot_min_z(), trace['foot_geometry_min_z_m'][frame], atol=1e-12, rtol=0):
                        raise ValueError('Independent actual mesh floor differs')
                    support_error = max(support_error, retargeter.verify_floor_support())
                error = np.linalg.norm(trace['actual_position_m']-target, axis=-1)
                if not np.allclose(error, trace['position_error_m'], atol=1e-12, rtol=0):
                    raise ValueError('Independent error differs')
                if quality != expected['pair_quality']:
                    raise ValueError('Independent quality differs')
                with (directory/'reference.csv').open(newline='') as stream:
                    saved = list(csv.DictReader(stream))
                csv_joints = np.deg2rad([[float(value[name]) for name in retargeter.joint_names] for value in saved])
                if not np.allclose(csv_joints, trace['qpos'][:,retargeter.q_indices], atol=1e-12, rtol=0):
                    raise ValueError('CSV joints differ')
                print(json.dumps(dict(sequence=old['sequence'], independent_support_error_max_m=support_error)), flush=True)
        if verification:
            rows.append(expected)
            continue
        row = dict(sequence=old['sequence'], capture_folder=old['capture_folder'], partition=old['partition'],
            scale=old['scale'], source_original_clip=old['source_original_clip'],
            status='PAIR_QUALIFIED_KINEMATIC_ONLY' if quality['passed'] else 'PAIR_REJECTED',
            pair_quality=quality, prior_pair_quality=old['pair_quality'], training_allowed=False,
            artifacts={p.name:dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(directory.iterdir()) if p.is_file()})
        rows.append(row)
        atomic_json(OUT/'progress.json', dict(sequences=rows))
        print(json.dumps(dict(sequence=row['sequence'], status=row['status'], quality=quality)), flush=True)
    return rows


def worker():
    configure()
    import torch
    state, job = corpus.read(OUT/'state.json'), corpus.read(OUT/'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E23_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires guarded monitor')
    corpus.unchanged(job)
    torch.set_num_threads(1)
    rows = compute_rows(job)
    qualified = [row for row in rows if row['pair_quality']['passed']]
    roles = {row['partition'] for row in qualified}
    atomic_json(OUT/'result.json', dict(experiment=job['experiment'], sequences=rows,
        kinematic_pairs_qualified=len(qualified), partitions_qualified=sorted(roles),
        training_ready=roles == {'train', 'validation', 'test'}, optimizer_steps=0, policy_rollouts=0,
        cuda_initialized=torch.cuda.is_initialized(), protected_e14_through_e22_unchanged=True,
        geometry_contract=job['geometry_contract'],
        deploy_allowed=False, physical_balance_verified=False, live_timing_verified=False, backup_status='LOCAL_ONLY'))


def run():
    configure()
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state, job = corpus.read(OUT/'state.json'), corpus.read(OUT/'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E23')
        guard.preflight()
        corpus.unchanged(job)
        state.update(status='RUNNING', started_at=time.time(), monitor_pid=os.getpid())
        atomic_json(OUT/'state.json', state)
        stage = LOG/'worker'
        stage.mkdir()
        try:
            with (stage/'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT/'data/environments/a3-sonic/bin/python'), '-m', 'script.a3.smpl_hull_probe', 'worker'],
                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', YUANQI_E23_MONITOR_PID=str(os.getpid())),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED', returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT/'state.json', state)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


def verify():
    configure()
    import torch
    torch.set_num_threads(1)
    state, job = corpus.read(OUT/'state.json'), corpus.read(OUT/'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('E23 did not complete')
    guard.preflight()
    corpus.unchanged(job)
    compute_rows(job, verification=True)
    append_record(LOG/'health.jsonl', guard.health())
    from script.a3 import zhy_fp32_trial as prior
    before, prior.LOG = prior.LOG, LOG
    try:
        health = prior.health_summary()
    finally:
        prior.LOG = before
    if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
            or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
        raise ValueError('Health verification failed')
    result = corpus.read(OUT/'result.json')
    receipt = dict(result, status='VERIFIED_KINEMATIC_CORPUS' if result['training_ready'] else 'VERIFIED_QUALITY_REJECTION',
        independent_fk_csv_floor_and_contact=True, independent_actual_mesh_hull_support=True,
        quality_gates=job['quality_gates'], source_provenance=job['source_provenance'],
        code_sha256=job['code_sha256'], input_sha256=job['input_sha256'], health=health, state=state)
    atomic_json(OUT/'independent_verification.json', receipt)
    atomic_json(ROOT/'data/manifests/a3_zhy_smpl_hull_20261013_E23.json', receipt)
    state.update(status=receipt['status'], verified_at=time.time())
    atomic_json(OUT/'state.json', state)
    print(json.dumps(dict(status=receipt['status'], pairs_qualified=result['kinematic_pairs_qualified'],
        training_ready=result['training_ready'], health=health), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = {'register':register, 'run':run, 'worker':worker, 'verify':verify}[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
