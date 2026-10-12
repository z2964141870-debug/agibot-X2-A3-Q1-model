"""E15 fixed comparison of segment-length retargeting against saved E14 pairs."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_corpus_qualify as corpus
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = corpus.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_chain_20261012_E15'
LOG = ROOT / 'logs/a3_zhy_smpl_chain_20261012_E15'
PRIOR = corpus.OUT
CODE = corpus.CODE + ('script/a3/smpl_chain_retarget.py', 'script/a3/smpl_chain_probe.py',
                     'script/a3/test_smpl_chain_retarget.py')


def configure():
    corpus.configure_helpers()
    guard.OUT, guard.LOG = OUT, LOG


def read(path):
    return corpus.read(path)


def register():
    if OUT.exists():
        raise FileExistsError('E15 cannot be reset or repeated')
    baseline = guard.preflight()
    previous = read(PRIOR / 'independent_verification.json')
    prior_job = read(PRIOR / 'registration.json')
    if previous['status'] != 'VERIFIED_QUALITY_REJECTION' or previous['health']['boot_ids'] != [baseline['boot_id']]:
        raise ValueError('Requires same-boot independently checked E14 rejection')
    corpus.unchanged(prior_job)
    inputs = dict(prior_job['input_sha256'])
    for row in previous['sequences']:
        directory = PRIOR / 'sequences' / row['sequence'].replace('/', '__')
        for name in ('human_source.npz', 'body_reference.npz', 'kinematic_trace.npz', 'pair_quality.json'):
            path = directory / name
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    job = dict(prior_job, experiment='A3_ZHY_SMPL_CHAIN_20261012_E15',
        registered_at=time.time(), boot_id=baseline['boot_id'],
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        code_sha256={name: sha256(ROOT / name) for name in CODE}, input_sha256=inputs,
        baseline_oom_kill=baseline['container_memory']['oom_kill'], kernel_baseline=kernel['stdout'].splitlines(),
        parent_experiment=prior_job['experiment'],
        main_variable='MORPHOLOGY_MAPPING_FIXED_ROBOT_SEGMENT_LENGTHS_AND_COMPATIBLE_LINK_ROTATIONS',
        same_source_clips=True, same_ik_solver_weights_limits=True, same_quality_gates=True,
        root_floor_adjustment=False, protected_e14_unchanged=True)
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', baseline)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def worker():
    configure()
    import numpy as np
    import torch
    from q1_retarget import BodyReference
    from script.a3.mocap_adapter import BODY_MAP
    from script.a3.smpl_chain_retarget import ChainA3Retargeter
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E15_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires guarded monitor')
    corpus.unchanged(job)
    torch.set_num_threads(1)
    retargeter = ChainA3Retargeter()
    prior = read(PRIOR / 'result.json')
    rows = []
    for old in prior['sequences']:
        name = old['sequence'].replace('/', '__')
        directory = OUT / 'sequences' / name
        directory.mkdir(parents=True)
        source = BodyReference.load(PRIOR / 'sequences' / name / 'body_reference.npz', list(BODY_MAP))
        with np.load(PRIOR / 'sequences' / name / 'human_source.npz', allow_pickle=False) as human:
            joints = human['joints'].copy()
        summary = retargeter.solve(source, old['scale'], directory)
        with np.load(directory / 'kinematic_trace.npz', allow_pickle=False) as trace:
            quality = corpus.retarget_quality(summary, trace, joints, retargeter)
            body_p95 = np.quantile(trace['position_error_m'], .95, axis=0)
        row = dict(sequence=old['sequence'], capture_folder=old['capture_folder'], partition=old['partition'],
            scale=old['scale'], source_original_clip=str((PRIOR / 'sequences' / name).relative_to(ROOT)),
            status='PAIR_QUALIFIED_KINEMATIC_ONLY' if quality['passed'] else 'PAIR_REJECTED',
            pair_quality=quality, prior_pair_quality=old['pair_quality'],
            body_error_p95_m=dict(zip(retargeter.names, body_p95.tolist(), strict=True)),
            training_allowed=False, artifacts={path.name:dict(bytes=path.stat().st_size, sha256=sha256(path))
                for path in sorted(directory.iterdir()) if path.is_file()})
        rows.append(row)
        atomic_json(OUT / 'progress.json', dict(sequences=rows))
        print(json.dumps(dict(sequence=old['sequence'], status=row['status'],
            before_ik_p95=old['pair_quality']['ik_position_p95_m'], after_ik_p95=quality['ik_position_p95_m'],
            checks=quality['checks'], contact=quality['contacts'])), flush=True)
    qualified = [row for row in rows if row['pair_quality']['passed']]
    roles = {row['partition'] for row in qualified}
    result = dict(experiment=job['experiment'], sequences=rows, kinematic_pairs_qualified=len(qualified),
        partitions_qualified=sorted(roles), training_ready=roles == {'train', 'validation', 'test'},
        status='QUALITY_READY_PENDING_INDEPENDENT_CHECK' if roles == {'train', 'validation', 'test'} else 'QUALITY_REJECTED',
        optimizer_steps=0, policy_rollouts=0, cuda_initialized=torch.cuda.is_initialized(),
        protected_e14_unchanged=True, deploy_allowed=False, physical_balance_verified=False,
        live_timing_verified=False, backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'result.json', result)


def run():
    configure()
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E15')
        guard.preflight()
        corpus.unchanged(job)
        state.update(status='RUNNING', started_at=time.time(), monitor_pid=os.getpid())
        atomic_json(OUT / 'state.json', state)
        stage = LOG / 'worker'
        stage.mkdir()
        try:
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.smpl_chain_probe', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', YUANQI_E15_MONITOR_PID=str(os.getpid())),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED', returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


def verify():
    configure()
    import numpy as np
    import torch
    from q1_retarget import BodyReference
    from script.a3.mocap_adapter import BODY_MAP
    from script.a3.smpl_chain_retarget import ChainA3Retargeter
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('E15 did not complete')
    guard.preflight()
    corpus.unchanged(job)
    retargeter = ChainA3Retargeter()
    result = read(OUT / 'result.json')
    for row in result['sequences']:
        directory = OUT / 'sequences' / row['sequence'].replace('/', '__')
        for name, expected in row['artifacts'].items():
            if sha256(directory / name) != expected['sha256']:
                raise ValueError('Artifact SHA mismatch')
        source_directory = ROOT / row['source_original_clip']
        source = BodyReference.load(source_directory / 'body_reference.npz', list(BODY_MAP))
        target, _ = retargeter.targets(source, row['scale'])
        with np.load(source_directory / 'human_source.npz', allow_pickle=False) as human:
            joints = human['joints'].copy()
        with np.load(directory / 'kinematic_trace.npz', allow_pickle=False) as trace:
            if not np.array_equal(target, trace['target_position_m']):
                raise ValueError('Independent target reconstruction differs')
            for frame, qpos in enumerate(trace['qpos']):
                retargeter.configuration.update(qpos)
                actual = retargeter.configuration.data.xpos[retargeter.bodies]
                if not np.allclose(actual, trace['actual_position_m'][frame], atol=1e-12, rtol=0):
                    raise ValueError('Independent A3 FK differs')
                if not np.allclose(retargeter.foot_min_z(), trace['foot_geometry_min_z_m'][frame], atol=1e-12, rtol=0):
                    raise ValueError('Independent foot geometry differs')
            quality = corpus.retarget_quality(read(directory / 'retarget_summary.json'), trace, joints, retargeter)
            error = np.linalg.norm(trace['actual_position_m'] - target, axis=-1)
            if not np.allclose(error, trace['position_error_m'], atol=1e-12, rtol=0) or quality != row['pair_quality']:
                raise ValueError('Independent quality differs')
            with (directory / 'reference.csv').open(newline='') as stream:
                saved = list(csv.DictReader(stream))
            csv_joints = np.deg2rad([[float(value[name]) for name in retargeter.joint_names] for value in saved])
            if not np.allclose(csv_joints, trace['qpos'][:, retargeter.q_indices], atol=1e-12, rtol=0):
                raise ValueError('CSV joints do not match checked FK')
    append_record(LOG / 'health.jsonl', guard.health())
    from script.a3 import zhy_fp32_trial as prior
    before, prior.LOG = prior.LOG, LOG
    try:
        health = prior.health_summary()
    finally:
        prior.LOG = before
    if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
            or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
        raise ValueError('Health verification failed')
    receipt = dict(result, status='VERIFIED_KINEMATIC_CORPUS' if result['training_ready'] else 'VERIFIED_QUALITY_REJECTION',
        independent_source_reconstruction=True, independent_fk_csv_and_quality=True,
        same_quality_gates=True, source_provenance=job['source_provenance'],
        code_sha256=job['code_sha256'], input_sha256=job['input_sha256'], health=health, state=state)
    atomic_json(OUT / 'independent_verification.json', receipt)
    atomic_json(ROOT / 'data/manifests/a3_zhy_smpl_chain_20261012_E15.json', receipt)
    state.update(status=receipt['status'], verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(dict(status=receipt['status'], pairs_qualified=result['kinematic_pairs_qualified'],
        training_ready=result['training_ready'], health=health), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = {'register':register, 'run':run, 'worker':worker, 'verify':verify}[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
