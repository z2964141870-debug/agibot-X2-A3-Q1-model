"""E13 CPU qualification of SMPL inputs and the frozen official A3 bridge."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_smpl_20261012_E13'
LOG = ROOT / 'logs/a3_zhy_smpl_20261012_E13'
INPUT = ROOT / 'data/test_inputs/a3_smpl_encoder_20261012'
PAIRS = ROOT / 'data/experiments/a3_fullchain_20261010/mocap'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
GOLDEN = ROOT / 'data/experiments/a3_zhy_search_20261012_E02/official_oracle/001_walk_front_slow.inputs.npz'
CASES = ('cloth_walk', 'cloth_oneleg', 'recording_0923', 'recording_0924')
MODES = ('offline_future', 'causal_history')
SOURCES = ('script/a3/smpl_encoder.py', 'script/a3/zhy_smpl_probe.py',
           'script/a3/test_smpl_encoder.py', 'script/a3/zhy_trial.py',
           'script/a3/reference_contract.py', 'script/a3/checkpoint_store.py')


def read(path):
    return json.loads(path.read_text())


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Boot changed; no recovery')
    for name, digest in {**job['code_sha256'], **job['input_sha256']}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered input or code changed: ' + name)


def register():
    if OUT.exists():
        raise FileExistsError('No repeat or reset of E13')
    value = guard.preflight()
    prior = read(ROOT / 'data/experiments/a3_zhy_modules_20261012_E12/independent_verification.json')
    if prior['status'] != 'VERIFIED' or prior['health']['boot_ids'] != [value['boot_id']]:
        raise ValueError('Requires verified same-boot E12')
    if sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise ValueError('Official checkpoint identity mismatch')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    inputs = [OFFICIAL, GOLDEN]
    for name in CASES:
        inputs += [INPUT / name / entry for entry in ('body_only.npz', 'inspection.json')]
        inputs += [PAIRS / name / entry for entry in
                   ('body_reference.npz', 'reference.csv', 'input_audit.json', 'retarget_summary.json')]
    job = dict(experiment='A3_ZHY_SMPL_20261012_E13', registered_at=time.time(),
        boot_id=value['boot_id'], project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(),
        code_sha256={name: sha256(ROOT / name) for name in SOURCES},
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        cases=CASES, modes=MODES, windows_per_case_mode=16, golden_frames=16,
        expected_windows=128, seed=0, device='cpu', cpu_threads=1,
        reference_error_max=1e-6, action_error_max=1e-5,
        optimizer_steps_allowed=0, new_rollouts=0, test_only_unchanged=True,
        scope='INPUT_AND_FROZEN_BRIDGE_CONTRACT_NOT_TRAINING_OR_POLICY_ACCEPTANCE',
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900,
        auto_resume=False, repeat_allowed=False, backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def case_windows(sim, name, job):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from script.a3.smpl_encoder import PARENTS, SMPL_TO_ROBOT, rotations, smpl_window, pack_teacher
    audit = read(PAIRS / name / 'input_audit.json')
    inspection = read(INPUT / name / 'inspection.json')
    path = INPUT / name / 'body_only.npz'
    if audit['sha256'] != sha256(path) or inspection['bundle']['sha256'] != audit['sha256']:
        raise ValueError('Human source differs from the existing retargeted pair')
    with np.load(path, allow_pickle=False) as bundle:
        if bundle['usage'].item() != 'TEST_ONLY' or bundle['training_allowed'].item() is not False:
            raise ValueError('Protected TEST_ONLY marker changed')
        if not np.array_equal(bundle['parents'], PARENTS):
            raise ValueError('Unexpected SMPL joint tree')
        times = bundle['source_time_s'].astype(float)
        times -= times[0]
        joints = bundle['joints'].astype(float)
        pose = rotations(bundle['pose_matrix'].reshape(-1, 24, 3, 3))
        frames = bundle['frame_index'].copy()
    if np.any(np.diff(times) <= 0) or np.any(np.diff(frames) <= 0):
        raise ValueError('Nonincreasing human timing or frame identity')
    ids = [0, 9, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    with np.load(PAIRS / name / 'body_reference.npz', allow_pickle=False) as reference:
        if reference['usage'].item() != 'TEST_ONLY' or reference['training_allowed'].item() is not False:
            raise ValueError('Paired reference is no longer TEST_ONLY')
        reference_times = reference['time_s'].copy()
        recorded = reference['position_m_world'].copy()
    transformed = joints[:, ids] @ SMPL_TO_ROBOT.T
    reconstructed = np.column_stack([np.interp(reference_times, times, column)
                                    for column in transformed.reshape(len(times), -1).T]).reshape(recorded.shape)
    reference_error = float(np.abs(recorded - reconstructed).max())
    if reference_error > job['reference_error_max']:
        raise ValueError('Human source does not reconstruct paired reference positions')
    root, quaternion, q_mj, count = sim.load_a3_flat_csv(
        PAIRS / name / 'reference.csv', source_fps=30, frame_stride=1)
    if count != len(reference_times) or abs((count - 1) / 30 - reference_times[-1]) > 1e-6:
        raise ValueError('Robot and normalized human durations differ')
    root, quaternion, q_mj = sim.resample_csv_motion(root, quaternion, q_mj, source_fps=30)
    names, _ = sim.load_urdf_actuated_joints(sim.DEFAULT_URDF)
    csv_names = [sim.A3_CSV_JOINT_NAMES[index] for index in sim.A3_POLICY_TO_SDK_IDX]
    if len(names) != 29 or set(names) != set(csv_names):
        raise ValueError('Robot reference joint names differ')
    q = q_mj[:, [csv_names.index(joint) for joint in names]]
    v = np.empty_like(q)
    v[:-1] = np.diff(q, axis=0) * 50
    v[-1] = v[-2]
    reference = SimpleNamespace(dof_il=q, dof_vel_il=v, anchor_quat_wxyz=quaternion, num_frames=len(q))
    robot_rotations = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    alignment = robot_rotations[0] @ (SMPL_TO_ROBOT @ pose[0, 0] @ SMPL_TO_ROBOT.T).T
    stop = min(len(q) - 10, int(np.floor((times[-1] - .18) * 50)))
    indices = np.unique(np.linspace(10, stop, job['windows_per_case_mode'], dtype=int))
    if len(indices) != job['windows_per_case_mode']:
        raise ValueError('Insufficient paired duration for registered windows')
    human, teacher, query_times, modes, target_indices, arrivals = [], [], [], [], [], []
    pack_error = 0.
    for mode in MODES:
        for index in indices:
            base = index / 50
            arrival = float(times[np.searchsorted(times, base - 1e-10)])
            features, query = smpl_window(times, joints, pose, robot_rotations[index],
                base, mode=mode, arrival_time=arrival, root_alignment=alignment)
            future = index + np.arange(10)
            ori = sim.root_ori_diff_6d(quaternion[index], quaternion[future])
            packed = pack_teacher(q[future], v[future], ori)
            published = sim.build_encoder_input(reference, index, quaternion[index], frame_skip=1).flatten()
            pack_error = max(pack_error, float(np.max(np.abs(packed - published))))
            human.append(features)
            teacher.append(packed)
            query_times.append(query)
            modes.append(mode)
            target_indices.append(int(index))
            arrivals.append(arrival)
    if pack_error:
        raise ValueError('Custom teacher packing differs from pinned deployment ABI')
    quality = read(PAIRS / name / 'retarget_summary.json')
    summary = dict(case=name, raw_frames=len(times), normalized_frames=count,
        source_duration_s=float(times[-1]), source_max_gap_s=float(np.diff(times).max()),
        windows=len(human), reference_position_max_error_m=reference_error,
        teacher_pack_max_error=pack_error, input_mode=MODES,
        simulated_arrival_rule='NEXT_RECORDED_BRACKETING_SAMPLE',
        simulated_arrival_delay_range_s=[float(min(np.asarray(arrivals) - np.asarray(target_indices) / 50)),
                                         float(max(np.asarray(arrivals) - np.asarray(target_indices) / 50))],
        live_arrival_verified=False,
        source_sha256=audit['sha256'], quality_verified=bool(quality['high_fidelity_verified']),
        usage='TEST_ONLY', training_allowed=False,
        calibration='FIXED_FIRST_FRAME_KINEMATIC_ALIGNMENT_NOT_LIVE_PAIRED_CALIBRATION')
    return (np.stack(human), np.stack(teacher), np.stack(query_times),
            np.asarray(modes), np.asarray(target_indices), summary)


def golden_inputs(job):
    import numpy as np
    with np.load(GOLDEN, allow_pickle=False) as captured:
        indices = np.unique(np.linspace(0, len(captured['actions']) - 1, job['golden_frames'], dtype=int))
        observations = captured['observations'][indices].copy()
        actions = captured['actions'][indices].copy()
    if observations.shape != (job['golden_frames'], 1570) or actions.shape != (job['golden_frames'], 29):
        raise ValueError('Official capture dimensions differ')
    commands = np.concatenate((observations[:, :580].reshape(-1, 10, 58),
                               observations[:, 580:640].reshape(-1, 10, 6)), axis=-1).reshape(-1, 640)
    return indices, commands, observations[:, 640:], actions


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state.get('monitor_pid') != os.getppid()
        or os.environ.get('YUANQI_E13_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires its guarded monitor')
    unchanged(job)
    import numpy as np
    import torch
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_encoder import SMPLStudent, FrozenA3Bridge, require_training_pairs
    from script.a3.controller_reference_probe import tensor_hash
    torch.set_num_threads(1)
    torch.manual_seed(job['seed'])
    if torch.cuda.is_initialized():
        raise ValueError('E13 must not initialize CUDA')
    sim = load_sim()
    bridge = FrozenA3Bridge(sim.A3Policy(OFFICIAL, torch.device('cpu'), encoder='a3_fast'))
    policy_before = tensor_hash(bridge.policy)
    rows = [case_windows(sim, name, job) for name in CASES]
    human, teacher = [np.concatenate([row[i] for row in rows]) for i in range(2)]
    with torch.no_grad():
        latent = bridge.teacher_latent(torch.from_numpy(teacher))
        student = SMPLStudent().eval()
        student_before = tensor_hash(student)
        student_output = student(torch.from_numpy(human))
        replica = SMPLStudent().eval()
        replica.load_state_dict(student.state_dict(), strict=True)
        if not torch.equal(student_output, replica(torch.from_numpy(human))):
            raise ValueError('Strict student reload changed output')
        indices, commands, proprio, captured_actions = golden_inputs(job)
        golden_latent = bridge.teacher_latent(torch.from_numpy(commands.astype(np.float32)))
        tokens, actions = bridge.decode_latent(golden_latent, torch.from_numpy(proprio.astype(np.float32)))
        direct = np.stack([sim.A3Policy.act(bridge.policy, value, feedback, torch.device('cpu'))
                           for value, feedback in zip(commands, proprio, strict=True)])
    captured_error = float(np.max(np.abs(actions.numpy() - captured_actions)))
    direct_error = float(np.max(np.abs(actions.numpy() - direct)))
    if max(captured_error, direct_error) > job['action_error_max']:
        raise ValueError('Bridge fails saved official action or single-frame deployment parity')
    if human.shape != (job['expected_windows'], 10, 90) or teacher.shape != (job['expected_windows'], 640):
        raise ValueError('Registered coverage differs')
    if not torch.isfinite(student_output).all() or not torch.isfinite(latent).all():
        raise ValueError('Nonfinite student or teacher output')
    if tensor_hash(bridge.policy) != policy_before or tensor_hash(student) != student_before:
        raise ValueError('Forward changed policy or untrained student tensors')
    for row in rows:
        try:
            require_training_pairs(row[-1])
        except ValueError:
            continue
        raise ValueError('TEST_ONLY pair unexpectedly passed the training gate')
    captures = OUT / 'contract_inputs_outputs.npz'
    np.savez_compressed(captures, human=human, teacher=teacher, teacher_latent=latent.numpy(),
        student_output=student_output.numpy(), query_times=np.concatenate([row[2] for row in rows]),
        mode=np.concatenate([row[3] for row in rows]),
        target_indices=np.concatenate([row[4] for row in rows]),
        case=np.concatenate([np.repeat(name, len(row[0])) for name, row in zip(CASES, rows, strict=True)]),
        golden_indices=indices, golden_actions=actions.numpy(), golden_tokens=tokens.numpy(),
        usage=np.asarray('TEST_ONLY'), training_allowed=np.asarray(False))
    student_path = OUT / 'student_untrained_contract.pt'
    torch.save(dict(state_dict=student.state_dict(), seed=job['seed'], usage='UNTRAINED_INTERFACE_ONLY',
                    optimizer_steps=0, deploy_allowed=False), student_path)
    result = dict(experiment=job['experiment'], cases=[row[-1] for row in rows],
        windows=len(human), official_teacher_tensor_sha256=policy_before,
        official_tensors_unchanged=True, student_tensors_unchanged=True,
        student_state_sha256=student_before, student_output_shape=list(student_output.shape),
        teacher_latent_shape=list(latent.shape), student_reload_exact=True,
        golden_frames=len(indices), captured_action_max_error=captured_error,
        direct_action_max_error=direct_error, optimizer_steps=0, new_rollouts=0,
        qualified_training_sessions=0, test_only_training_gate_rejected=4,
        policy_effect_acceptance=False, live_smpl_capability=False,
        captures={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                  for path in (captures, student_path)}, scope=job['scope'], backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'result.json', result)
    print(json.dumps(result, indent=2))


def run():
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery')
        unchanged(job)
        guard.preflight()
        guard.OUT, guard.LOG = OUT, LOG
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        stage = LOG / 'worker'
        stage.mkdir()
        try:
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.zhy_smpl_probe', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', YUANQI_E13_MONITOR_PID=str(os.getpid())),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED',
                         returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


def verify():
    import numpy as np
    import torch
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_encoder import SMPLStudent, FrozenA3Bridge
    from script.a3.controller_reference_probe import tensor_hash
    from script.a3 import zhy_fp32_trial as prior
    torch.set_num_threads(1)
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Requires complete E13')
    guard.preflight()
    unchanged(job)
    result = read(OUT / 'result.json')
    for name, identity in result['captures'].items():
        if sha256(OUT / name) != identity['sha256']:
            raise ValueError('Capture identity changed')
    sim = load_sim()
    bridge = FrozenA3Bridge(sim.A3Policy(OFFICIAL, torch.device('cpu'), encoder='a3_fast'))
    if tensor_hash(bridge.policy) != result['official_teacher_tensor_sha256']:
        raise ValueError('Independent official tensor hash differs')
    rows = [case_windows(sim, name, job) for name in CASES]
    stored = np.load(OUT / 'contract_inputs_outputs.npz', allow_pickle=False)
    with stored:
        if stored['usage'].item() != 'TEST_ONLY' or stored['training_allowed'].item() is not False:
            raise ValueError('Derived data no longer test-only')
        for key, index in (('human', 0), ('teacher', 1), ('query_times', 2), ('mode', 3), ('target_indices', 4)):
            if not np.array_equal(stored[key], np.concatenate([row[index] for row in rows])):
                raise ValueError('Independent source reconstruction differs: ' + key)
        with torch.no_grad():
            latent = bridge.teacher_latent(torch.from_numpy(stored['teacher']))
            if not np.array_equal(latent.numpy(), stored['teacher_latent']):
                raise ValueError('Independent teacher recomputation differs')
            payload = torch.load(OUT / 'student_untrained_contract.pt', map_location='cpu', weights_only=True)
            if payload['optimizer_steps'] or payload['deploy_allowed'] or payload['usage'] != 'UNTRAINED_INTERFACE_ONLY':
                raise ValueError('Untrained checkpoint eligibility changed')
            student = SMPLStudent().eval()
            student.load_state_dict(payload['state_dict'], strict=True)
            if tensor_hash(student) != result['student_state_sha256']:
                raise ValueError('Student tensor hash differs')
            if not np.array_equal(student(torch.from_numpy(stored['human'])).numpy(), stored['student_output']):
                raise ValueError('Independent student reload differs')
            indices, commands, proprio, captured_actions = golden_inputs(job)
            direct = np.stack([bridge.policy.act(value, feedback, torch.device('cpu'))
                               for value, feedback in zip(commands, proprio, strict=True)])
            error = float(np.max(np.abs(direct - stored['golden_actions'])))
            if error > job['action_error_max'] or np.max(np.abs(direct - captured_actions)) > job['action_error_max']:
                raise ValueError('Independent direct official action differs')
    append_record(LOG / 'health.jsonl', guard.health())
    previous = prior.LOG
    prior.LOG = LOG
    try:
        health = prior.health_summary()
    finally:
        prior.LOG = previous
    if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
        or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
        raise ValueError('Health coverage failed')
    receipt = dict(result, status='VERIFIED_INTERFACE_ONLY', health=health,
        independent_source_reconstruction=True, independent_cpu_reload=True,
        independent_direct_action_max_error=error, state=dict(state),
        code_sha256=job['code_sha256'], input_sha256=job['input_sha256'])
    atomic_json(OUT / 'independent_verification.json', receipt)
    atomic_json(ROOT / 'data/manifests/a3_zhy_smpl_20261012_E13.json', receipt)
    state.update(status='VERIFIED_INTERFACE_ONLY', verified_at=time.time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps(dict(status=receipt['status'], windows=result['windows'],
                         captured_action_max_error=result['captured_action_max_error'], health=health), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker', 'verify'))
    value = {'register': register, 'run': run, 'worker': worker, 'verify': verify}[parser.parse_args().action]()
    raise SystemExit(value if isinstance(value, int) else 0)
