"""E14: qualify fresh original human labels and bounded A3 kinematic pairs."""

import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_smpl_corpus_20261012_E14'
LOG = ROOT / 'logs/a3_zhy_smpl_corpus_20261012_E14'
INPUT = ROOT / 'data/test_inputs/a3_smpl_original_20261012'
HELPERS = ROOT / 'data/references/a3_smpl_q1_helpers_20261012'
CODE = ('script/a3/smpl_corpus_qualify.py', 'script/a3/smpl_corpus_pack.py',
        'script/a3/test_smpl_corpus_qualify.py',
        'script/a3/smpl_encoder.py', 'script/a3/mocap_adapter.py',
        'script/a3/checkpoint_store.py', 'script/a3/zhy_trial.py')
HELPER_NAMES = ('q1_retarget.py', 'q1_sim.py', 'egolocate_input.py', 'inspect_egolocate.py')
GATES = dict(source_fk_max_m=.002, source_bone_range_max_m=.002,
    source_rest_bone_deviation_max_m=.002, source_root_speed_p99_max_m_s=4.,
    human_height_min_m=1.2, human_height_max_m=2.2,
    ik_position_p95_max_m=.05, ik_position_max_m=.15,
    foot_penetration_budget_m=.005, joint_velocity_ratio_max=1.02,
    contact_height_budget_m=.05, contact_source_speed_max_m_s=.25,
    contact_robot_slip_p95_max_m_s=.25, minimum_contact_frames_per_foot=5)


def read(path):
    return json.loads(Path(path).read_text())


def configure_helpers():
    sys.path.insert(0, str(HELPERS))


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Boot changed; stop without recovery')
    for name, identity in {**job['code_sha256'], **job['input_sha256']}.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Registered code or input changed: ' + name)


def register():
    if OUT.exists():
        raise FileExistsError('E14 must not be repeated or reset')
    baseline = guard.preflight()
    selection = read(INPUT / 'selection.json')
    if not selection['source_provenance']['training_use_authorized']:
        raise ValueError('Original label use is not authorized')
    if sha256(INPUT / 'selected_labels.zip') != selection['label_archive']['sha256']:
        raise ValueError('Transferred label archive SHA mismatch')
    partitions = {}
    for row in selection['sequences']:
        role = row['partition']
        folder = row['capture_folder']
        if folder in partitions and partitions[folder] != role:
            raise ValueError('Capture folder overlaps partitions')
        partitions[folder] = role
    if set(partitions.values()) != {'train', 'validation', 'test'}:
        raise ValueError('Requires three disjoint, nonempty partitions')
    prior = read(ROOT / 'data/experiments/a3_zhy_smpl_20261012_E13/independent_verification.json')
    if prior['status'] != 'VERIFIED_INTERFACE_ONLY' or prior['health']['boot_ids'] != [baseline['boot_id']]:
        raise ValueError('Requires same-boot E13 interface verification')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    inputs = [INPUT / 'selection.json', INPUT / 'selected_labels.zip']
    inputs += [HELPERS / name for name in HELPER_NAMES]
    from script.a3.fullchain_support import VENDOR
    assets = VENDOR / 'gear_sonic/data/assets/robot_description'
    inputs += [assets / 'mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml',
        assets / 'urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf']
    job = dict(experiment='A3_ZHY_SMPL_CORPUS_20261012_E14', registered_at=time.time(),
        boot_id=baseline['boot_id'], project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        code_sha256={name: sha256(ROOT / name) for name in CODE},
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        source_provenance=selection['source_provenance'], selection=selection['sequences'],
        source_fps=30, live_timing_verified=False,
        source_fps_evidence=selection['source_fps_evidence'],
        clip_frames=300, clip_rule='CENTER_300_SOURCE_FRAMES_BEFORE_OUTCOME_INSPECTION',
        calibration='CANONICAL_NEUTRAL_FROM_MEDIAN_PARENT_LOCAL_REST_BONES',
        scaling='MEAN_ROBOT_TO_HUMAN_PELVIS_HIP_KNEE_ANKLE_CHAIN_RATIO',
        quality_gates=GATES, optimizer_steps_allowed=0, new_policy_rollouts=0,
        device='cpu', cpu_threads=1, auto_resume=False, repeat_allowed=False,
        baseline_oom_kill=baseline['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), max_cpu_c=90, max_gpu_c=85,
        consecutive_hot_samples=2, sample_seconds=5, memory_reserve_gib=4,
        runtime_limit_seconds=900, backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', baseline)
    print(json.dumps(dict(status='READY', sequences=len(job['selection']), clip_frames=job['clip_frames'])))


def source_metrics(pose, joints, fps=30):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from script.a3.smpl_encoder import PARENTS
    configure_helpers()
    from inspect_egolocate import global_rotations
    pose, joints = np.asarray(pose, dtype=float), np.asarray(joints, dtype=float)
    if (pose.ndim != 3 or pose.shape[1:] != (24, 3) or joints.shape != pose.shape
            or len(pose) < 300 or not np.isfinite(pose).all() or not np.isfinite(joints).all()):
        raise ValueError('Need finite, aligned original [T,24,3] pose/joint labels')
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    world = global_rotations(local)
    bones = joints[:, 1:] - joints[:, PARENTS[1:]]
    lengths = np.linalg.norm(bones, axis=-1)
    parent_local = np.einsum('tbji,tbj->tbi', world[:, PARENTS[1:]], bones)
    rest = np.median(parent_local, axis=0)
    neutral = np.zeros((24, 3))
    reconstructed = np.zeros_like(joints)
    reconstructed[:, 0] = joints[:, 0]
    for joint, parent in enumerate(PARENTS[1:], 1):
        neutral[joint] = neutral[parent] + rest[joint - 1]
        reconstructed[:, joint] = reconstructed[:, parent] + np.einsum('tij,j->ti', world[:, parent], rest[joint - 1])
    fk = np.linalg.norm(reconstructed - joints, axis=-1)
    root_speed = np.linalg.norm(np.diff(joints[:, 0], axis=0) * fps, axis=-1)
    height = float(neutral[:, 1].max() - neutral[:, 1].min())
    summary = dict(frames=len(joints), source_duration_s=(len(joints) - 1) / fps,
        finite=True, joint_units='METRES_CHECKED_AGAINST_REST_SKELETON_SIZE',
        local_axis_angle_pose_verified_by_position_fk=bool(fk.max() <= GATES['source_fk_max_m']),
        source_fk_max_m=float(fk.max()), source_fk_p95_m=float(np.quantile(fk, .95)),
        source_bone_range_max_m=float(np.ptp(lengths, axis=0).max()),
        source_rest_bone_deviation_max_m=float(np.linalg.norm(parent_local - rest, axis=-1).max()),
        human_height_m=height, root_translation_range_smpl_xyz_m=np.ptp(joints[:, 0], axis=0).tolist(),
        source_root_speed_p99_m_s=float(np.quantile(root_speed, .99)),
        sampled_timing='UNIFORM_30HZ_PROJECT_CONTRACT_NO_ORIGINAL_PER_FRAME_TIMESTAMPS',
        physical_pose_truth_independently_measured=False,
        rest_bone_vectors_m=rest.tolist())
    checks = dict(source_fk=summary['source_fk_max_m'] <= GATES['source_fk_max_m'],
        bone_lengths=summary['source_bone_range_max_m'] <= GATES['source_bone_range_max_m'],
        rest_bones=summary['source_rest_bone_deviation_max_m'] <= GATES['source_rest_bone_deviation_max_m'],
        metre_scale=GATES['human_height_min_m'] <= height <= GATES['human_height_max_m'],
        root_speed=summary['source_root_speed_p99_m_s'] <= GATES['source_root_speed_p99_max_m_s'])
    summary.update(checks=checks, structural_quality_passed=all(checks.values()))
    return summary, local, neutral


def contact_metrics(source_joints, actual_positions, foot_minima, names, fps=30):
    import numpy as np
    feet = source_joints[:, [10, 11]]
    floor = float(np.quantile(feet[..., 1], .02))
    speed = np.linalg.norm(np.gradient(feet, 1 / fps, axis=0), axis=-1)
    contact = (feet[..., 1] <= floor + GATES['contact_height_budget_m']) & (speed <= GATES['contact_source_speed_max_m_s'])
    ankles = actual_positions[:, [names.index('left_ankle'), names.index('right_ankle')]]
    robot_speed = np.linalg.norm(np.gradient(ankles[..., :2], 1 / fps, axis=0), axis=-1)
    counts = contact.sum(0)
    slip = [float(np.quantile(robot_speed[:, foot][contact[:, foot]], .95)) if counts[foot] else None for foot in range(2)]
    return dict(contact_method='INFERRED_TOE_HEIGHT_AND_SPEED_NOT_MEASURED_CONTACT',
        source_floor_smpl_y_m=floor, contact_frames_per_foot=counts.astype(int).tolist(),
        robot_contact_ankle_xy_speed_p95_m_s=slip,
        foot_geometry_min_z_m=float(foot_minima.min()),
        penetration_frames=int(np.count_nonzero(np.min(foot_minima, axis=1) < -GATES['foot_penetration_budget_m'])),
        contact_mask=contact)


def retarget_quality(summary, trace, source_joints, retargeter):
    import numpy as np
    import xml.etree.ElementTree as ET
    from script.a3.mocap_adapter import URDF
    contacts = contact_metrics(source_joints, trace['actual_position_m'], trace['foot_geometry_min_z_m'], retargeter.names)
    contacts.pop('contact_mask')
    velocity_limit = {joint.attrib['name']: float(joint.find('limit').attrib['velocity'])
        for joint in ET.parse(URDF).getroot().findall('joint') if joint.find('limit') is not None and 'velocity' in joint.find('limit').attrib}
    joints = trace['qpos'][:, retargeter.q_indices]
    ratio = float(np.max(np.abs(np.diff(joints, axis=0) * 30) / np.asarray([velocity_limit[name] for name in retargeter.joint_names])))
    checks = dict(ik_p95=summary['ik_position_p95_m'] <= GATES['ik_position_p95_max_m'],
        ik_max=summary['ik_position_max_m'] <= GATES['ik_position_max_m'],
        joint_limits=summary['joint_limit_violations'] == 0,
        velocities=ratio <= GATES['joint_velocity_ratio_max'],
        foot_geometry=contacts['penetration_frames'] == 0,
        contact_coverage=min(contacts['contact_frames_per_foot']) >= GATES['minimum_contact_frames_per_foot'],
        contact_slip=all(value is not None and value <= GATES['contact_robot_slip_p95_max_m_s'] for value in contacts['robot_contact_ankle_xy_speed_p95_m_s']))
    return dict(checks=checks, passed=all(checks.values()), joint_velocity_max_limit_ratio=ratio,
        ik_position_p95_m=summary['ik_position_p95_m'], ik_position_max_m=summary['ik_position_max_m'],
        contacts=contacts, scope='BOUNDED_OFFLINE_KINEMATIC_DISTILLATION_NOT_BALANCE_OR_LIVE_ACCEPTANCE',
        physical_balance_verified=False, live_smpl_accuracy_verified=False)


def load_labels(archive, row):
    import hashlib
    import torch
    arrays = []
    for name in ('pose.pt', 'joint.pt'):
        identity = row['labels'][name]
        raw = archive.read(identity['entry'])
        if len(raw) != identity['bytes'] or hashlib.sha256(raw).hexdigest() != identity['sha256']:
            raise ValueError('Selected label identity mismatch: ' + identity['entry'])
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError('Expected an original floating tensor label')
        arrays.append(value.detach().numpy().astype('float64'))
    return arrays


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid() or os.environ.get('YUANQI_E14_MONITOR_PID') != str(os.getppid()):
        raise RuntimeError('Worker requires its guarded monitor')
    unchanged(job)
    configure_helpers()
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from q1_retarget import BodyReference
    from script.a3.mocap_adapter import A3Retargeter, BODY_MAP
    from script.a3.smpl_encoder import PARENTS, SMPL_TO_ROBOT
    torch.set_num_threads(1)
    retargeter = A3Retargeter()
    ids = [0, 9, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    rows = []
    with zipfile.ZipFile(INPUT / 'selected_labels.zip') as archive:
        for row in job['selection']:
            name = row['sequence'].replace('/', '__')
            directory = OUT / 'sequences' / name
            directory.mkdir(parents=True)
            pose, joints = load_labels(archive, row)
            metrics, local, neutral = source_metrics(pose, joints)
            atomic_json(directory / 'source_quality.json', metrics)
            result = dict(sequence=row['sequence'], capture_folder=row['capture_folder'],
                partition=row['partition'], source_quality=metrics, status='SOURCE_REJECTED',
                training_allowed=False)
            rows.append(result)
            if not metrics['structural_quality_passed']:
                atomic_json(OUT / 'progress.json', dict(sequences=rows))
                continue
            start = (len(joints) - job['clip_frames']) // 2
            stop = start + job['clip_frames']
            selected_joints, selected_local = joints[start:stop], local[start:stop]
            times = np.arange(job['clip_frames'], dtype=float) / job['source_fps']
            world = np.empty_like(selected_local)
            world[:, 0] = selected_local[:, 0]
            for joint, parent in enumerate(PARENTS[1:], 1):
                world[:, joint] = world[:, parent] @ selected_local[:, joint]
            positions = selected_joints[:, ids] @ SMPL_TO_ROBOT.T
            rotations = SMPL_TO_ROBOT @ world[:, ids] @ SMPL_TO_ROBOT.T
            calibration = (neutral[ids] + selected_joints[0, 0]) @ SMPL_TO_ROBOT.T
            orientations = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, len(ids), 4)
            source = BodyReference(times, list(BODY_MAP), positions, orientations, calibration,
                np.tile([1., 0., 0., 0.], (len(ids), 1)), list(BODY_MAP),
                'ORIGINAL_SELF_COLLECTED_CLOTHO_LABELS_CANONICAL_NEUTRAL_CALIBRATION')
            lengths = []
            for side in ('left', 'right'):
                chain = [retargeter.names.index(key) for key in ('pelvis', side + '_hip', side + '_knee', side + '_ankle')]
                lengths.append(np.linalg.norm(np.diff(retargeter.rest_position[chain], axis=0), axis=-1).sum() /
                               np.linalg.norm(np.diff(calibration[chain], axis=0), axis=-1).sum())
            scale = float(np.mean(lengths))
            source.save(directory / 'body_reference.npz')
            np.savez_compressed(directory / 'human_source.npz', source_time_s=times,
                joints=selected_joints, pose_matrix=selected_local, parents=PARENTS,
                original_frame_index=np.arange(start, stop), usage=np.asarray('SOURCE_LABEL_QUALIFICATION'),
                training_allowed=np.asarray(False))
            summary = retargeter.solve(source, scale, directory)
            with np.load(directory / 'kinematic_trace.npz', allow_pickle=False) as trace:
                quality = retarget_quality(summary, trace, selected_joints, retargeter)
            atomic_json(directory / 'pair_quality.json', quality)
            result.update(status='PAIR_QUALIFIED_KINEMATIC_ONLY' if quality['passed'] else 'PAIR_REJECTED',
                original_frame_range=[start, stop], reference_frames=job['clip_frames'], scale=scale,
                pair_quality=quality, pair_alignment_verified=True)
            result['artifacts'] = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                for path in sorted(directory.iterdir()) if path.is_file()}
            atomic_json(OUT / 'progress.json', dict(sequences=rows))
            print(json.dumps(dict(sequence=row['sequence'], status=result['status'],
                ik_p95_m=quality['ik_position_p95_m'], contacts=quality['contacts'], checks=quality['checks'])), flush=True)
    qualified = [row for row in rows if row['status'] == 'PAIR_QUALIFIED_KINEMATIC_ONLY']
    roles = {row['partition'] for row in qualified}
    result = dict(experiment=job['experiment'], sequences=rows,
        source_labels_structurally_passed=sum(row['source_quality']['structural_quality_passed'] for row in rows),
        kinematic_pairs_qualified=len(qualified), partitions_qualified=sorted(roles),
        training_ready=roles == {'train', 'validation', 'test'},
        status='QUALITY_READY_PENDING_INDEPENDENT_CHECK' if roles == {'train', 'validation', 'test'} else 'QUALITY_REJECTED',
        optimizer_steps=0, policy_rollouts=0, cuda_initialized=torch.cuda.is_initialized(),
        protected_test_only_artifacts_unchanged=True, deploy_allowed=False,
        physical_balance_verified=False, live_timing_verified=False, backup_status='LOCAL_ONLY')
    if result['cuda_initialized']:
        raise RuntimeError('Unexpected CUDA initialization')
    atomic_json(OUT / 'result.json', result)


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E14')
        guard.preflight()
        unchanged(job)
        guard.OUT, guard.LOG = OUT, LOG
        state.update(status='RUNNING', started_at=time.time(), monitor_pid=os.getpid())
        atomic_json(OUT / 'state.json', state)
        stage = LOG / 'worker'
        stage.mkdir()
        try:
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.smpl_corpus_qualify', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', YUANQI_E14_MONITOR_PID=str(os.getpid())),
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
    import csv
    import mujoco
    import numpy as np
    import torch
    configure_helpers()
    from q1_retarget import BodyReference
    from script.a3.mocap_adapter import A3Retargeter, BODY_MAP
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('E14 computation did not complete')
    guard.preflight()
    unchanged(job)
    retargeter = A3Retargeter()
    result = read(OUT / 'result.json')
    with zipfile.ZipFile(INPUT / 'selected_labels.zip') as archive:
        for row, identity in zip(result['sequences'], job['selection'], strict=True):
            if row['sequence'] != identity['sequence'] or row['partition'] != identity['partition']:
                raise ValueError('Source or partition changed')
            directory = OUT / 'sequences' / row['sequence'].replace('/', '__')
            pose, joints = load_labels(archive, identity)
            metrics, local, _ = source_metrics(pose, joints)
            if metrics != row['source_quality']:
                raise ValueError('Independent original-label structural check differs')
            for name, expected in row.get('artifacts', {}).items():
                if sha256(directory / name) != expected['sha256']:
                    raise ValueError('Pair artifact SHA changed: ' + name)
            if 'pair_quality' not in row:
                continue
            start, stop = row['original_frame_range']
            with np.load(directory / 'human_source.npz', allow_pickle=False) as human:
                if (not np.array_equal(human['joints'], joints[start:stop])
                        or not np.array_equal(human['pose_matrix'], local[start:stop])
                        or not np.array_equal(human['original_frame_index'], np.arange(start, stop))):
                    raise ValueError('Saved human clip differs from original label frames')
            with np.load(directory / 'kinematic_trace.npz', allow_pickle=False) as trace:
                summary = read(directory / 'retarget_summary.json')
                quality = retarget_quality(summary, trace, joints[start:stop], retargeter)
                if quality != row['pair_quality']:
                    raise ValueError('Independent pair quality differs')
                position_error = np.linalg.norm(trace['actual_position_m'] - trace['target_position_m'], axis=-1)
                if not np.allclose(position_error, trace['position_error_m'], atol=1e-12, rtol=0):
                    raise ValueError('IK error trace inconsistent')
                if abs(float(np.quantile(position_error, .95)) - quality['ik_position_p95_m']) > 1e-12:
                    raise ValueError('Independent IK percentile differs')
                reference = BodyReference.load(directory / 'body_reference.npz', list(BODY_MAP))
                target, _ = retargeter.targets(reference, row['scale'])
                if not np.array_equal(target, trace['target_position_m']):
                    raise ValueError('Independent human-to-robot target reconstruction differs')
                for frame, qpos in enumerate(trace['qpos']):
                    retargeter.configuration.update(qpos)
                    actual = retargeter.configuration.data.xpos[retargeter.bodies]
                    if not np.allclose(actual, trace['actual_position_m'][frame], atol=1e-12, rtol=0):
                        raise ValueError('Independent saved-qpos FK differs')
                    if not np.allclose(retargeter.foot_min_z(), trace['foot_geometry_min_z_m'][frame], atol=1e-12, rtol=0):
                        raise ValueError('Independent actual foot geometry differs')
                with (directory / 'reference.csv').open(newline='') as stream:
                    csv_rows = list(csv.DictReader(stream))
                csv_joints = np.deg2rad([[float(value[name]) for name in retargeter.joint_names] for value in csv_rows])
                if not np.allclose(csv_joints, trace['qpos'][:, retargeter.q_indices], atol=1e-12, rtol=0):
                    raise ValueError('CSV label joints differ from checked kinematic poses')
    append_record(LOG / 'health.jsonl', guard.health())
    from script.a3 import zhy_fp32_trial as prior
    old_log, prior.LOG = prior.LOG, LOG
    try:
        health = prior.health_summary()
    finally:
        prior.LOG = old_log
    if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
            or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
        raise ValueError('Health verification failed')
    receipt = dict(result, status='VERIFIED_KINEMATIC_CORPUS' if result['training_ready'] else 'VERIFIED_QUALITY_REJECTION',
        independently_checked_original_labels=True, independently_checked_pair_artifacts=True,
        source_provenance=job['source_provenance'], quality_gates=job['quality_gates'],
        partition_unit='CAPTURE_FOLDER_NOT_VERIFIED_INDEPENDENT_SUBJECT',
        source_fps_evidence=job['source_fps_evidence'], code_sha256=job['code_sha256'],
        input_sha256=job['input_sha256'], health=health, state=state)
    atomic_json(OUT / 'independent_verification.json', receipt)
    atomic_json(ROOT / 'data/manifests/a3_zhy_smpl_corpus_20261012_E14.json', receipt)
    state.update(status=receipt['status'], verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(dict(status=receipt['status'], structurally_passed=result['source_labels_structurally_passed'],
        pairs_qualified=result['kinematic_pairs_qualified'], training_ready=result['training_ready'], health=health), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = {'register': register, 'run': run, 'worker': worker, 'verify': verify}[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
