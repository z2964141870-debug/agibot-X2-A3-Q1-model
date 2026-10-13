"""E24 qualify the reference ABI then evaluate frozen035 once on six E23 clips."""

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
OUT = ROOT / 'data/experiments/a3_zhy_smpl_frozen_20261013_E24'
LOG = ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E24'
PRIOR = ROOT / 'data/experiments/a3_zhy_smpl_hull_20261013_E23'
OFFICIAL = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
ASSETS = VENDOR / 'gear_sonic/data/assets/robot_description'
MJCF = ASSETS / 'mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml'
CODE = ('script/a3/smpl_frozen_probe.py', 'script/a3/smpl_frozen_entry.py',
    'script/a3/evaluate_mujoco.py', 'script/a3/evaluation_common.py',
    'script/a3/sim_entry.py', 'script/a3/reference_contract.py', 'script/a3/verify_zhy_search.py',
    'script/a3/zhy_trial.py', 'script/a3/host_health.py', 'script/a3/checkpoint_store.py')


def read(path):
    return json.loads(path.read_text())


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Confirmed boot change; no recovery')
    for name, identity in {**job['code_sha256'], **job['input_sha256']}.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Registered code or input changed: ' + name)


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E24')
    value = guard.preflight()
    prior = read(PRIOR / 'independent_verification.json')
    if prior['status'] != 'VERIFIED_KINEMATIC_CORPUS' or prior['health']['boot_ids'] != [value['boot_id']]:
        raise ValueError('Requires verified same-boot E23')
    rows = [r for r in prior['sequences'] if r['pair_quality']['passed']]
    if len(rows) != 6 or len(prior['sequences']) != 7 or sha256(OFFICIAL) != guard.OFFICIAL_SHA:
        raise ValueError('Qualified-set or frozen-weight identity mismatch')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = [PRIOR / name for name in ('independent_verification.json', 'registration.json', 'result.json')]
    files += [OFFICIAL, OFFICIAL.with_name('config.yaml'), MJCF,
        ASSETS / 'mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml',
        ASSETS / 'urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf',
        VENDOR / 'gear_sonic/scripts/sim2sim_a3_mujoco.py']
    selection = []
    for row in rows:
        directory = PRIOR / 'sequences' / row['sequence'].replace('/', '__')
        for name in ('reference.csv', 'kinematic_trace.npz', 'retarget_summary.json'):
            path = directory / name
            if sha256(path) != row['artifacts'][name]['sha256']:
                raise ValueError('Verified E23 artifact changed')
            files.append(path)
        selection.append(dict(sequence=row['sequence'], partition=row['partition'],
            source=str(directory.relative_to(ROOT)), filename=row['sequence'].replace('/', '__') + '.csv'))
    if len({r['filename'] for r in selection}) != 6:
        raise ValueError('Duplicate evaluation identity')
    job = dict(experiment='A3_ZHY_SMPL_FROZEN_20261013_E24', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        code_sha256={name:sha256(ROOT / name) for name in CODE},
        input_sha256={str(path.relative_to(ROOT)):sha256(path) for path in files}, selection=selection,
        official_sha256=guard.OFFICIAL_SHA, seed=0, optimizer_steps_allowed=0, policy_rollouts_allowed=6,
        main_variable='E23_QUALIFIED_REFERENCE_FROZEN_OFFICIAL035_CLOSED_LOOP',
        reference_fps=30., frame_stride=1, max_policy_steps=None, action_delay_ms=0.,
        reference_buffer_ms=0., encoder='a3_fast', human_live_timing_verified=False,
        csv_roundtrip_max_error=1e-6, model_body_fk_max_m=.002,
        reference_floor_budget_m=.005, dynamic_gate='COMPLETE_FINITE_AND_ZERO_FALLS_EACH_CLIP',
        fall_height_m=.45, fall_abs_roll_or_pitch_deg=60.,
        fresh_final_test=False, reference_scope='REPEATEDLY_USED_DEVELOPMENT_REFERENCES',
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        baseline_oom_kill=value['container_memory']['oom_kill'], kernel_baseline=kernel['stdout'].splitlines(),
        backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    references = OUT / 'references'
    references.mkdir()
    for row in selection:
        source = ROOT / row['source'] / 'reference.csv'
        (references / row['filename']).symlink_to(os.path.relpath(source, references))
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', clips=6, optimizer_steps=0)))


def reference_contract(job):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_frozen_entry import foot_geometry, foot_minima
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(MJCF))
    runtime = sim.build_loop_runtime(model, sim.DEFAULT_URDF, anchor_body_name='torso_Link')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE,
        sim.DEFAULT_BRIDGE_CACHE, False)
    feet, data = foot_geometry(model, sim.mujoco), sim.mujoco.MjData(model)
    body_names = ['pelvis_link', 'torso_Link', 'left_hip_yaw_Link', 'right_hip_yaw_Link',
        'left_knee_Link', 'right_knee_Link', 'left_ankle_roll_Link', 'right_ankle_roll_Link',
        'left_shoulder_yaw_Link', 'right_shoulder_yaw_Link', 'left_elbow_Link', 'right_elbow_Link',
        'left_wrist_yaw_Link', 'right_wrist_yaw_Link']
    bodies = [sim.mujoco.mj_name2id(model, sim.mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if min(bodies) < 0:
        raise ValueError('Missing reference body in simulator')
    outcomes = []
    try:
        for row in job['selection']:
            directory = ROOT / row['source']
            csv = OUT / 'references' / row['filename']
            root, rotation, joints, count = sim.load_a3_flat_csv(csv, source_fps=30., frame_stride=1)
            if not all(np.isfinite(value).all() for value in (root, rotation, joints)):
                raise ValueError('Nonfinite loaded reference')
            with np.load(directory / 'kinematic_trace.npz', allow_pickle=False) as trace:
                qpos, names = trace['qpos'], trace['joint_names'].tolist()
                times, expected_bodies = trace['time_s'], trace['actual_position_m']
                kinematic = sim.mujoco.MjModel.from_xml_path(str(ASSETS / 'mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml'))
                addresses = [int(kinematic.jnt_qposadr[sim.mujoco.mj_name2id(
                    kinematic, sim.mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
                ordered = [names.index(sim.A3_CSV_JOINT_NAMES[sdk]) for sdk in sim.A3_POLICY_TO_SDK_IDX]
                joint_error = float(np.max(np.abs(joints - qpos[:, addresses][:, ordered])))
                root_error = float(np.max(np.abs(root - qpos[:, :3])))
                rotation_error = float((Rotation.from_quat(rotation, scalar_first=True) *
                    Rotation.from_quat(qpos[:, 3:7], scalar_first=True).inv()).magnitude().max())
                duration_error = abs((count - 1) / 30. - float(times[-1]))
                body_error = 0.
                for frame in range(count):
                    data.qpos[:] = sim.loop_key_qpos(model)
                    data.qpos[:7] = qpos[frame, :7]
                    sim.fill_loop_qpos_motors(data.qpos, runtime, solver, joints[frame])
                    sim.mujoco.mj_forward(model, data)
                    body_error = max(body_error, float(np.linalg.norm(data.xpos[bodies] - expected_bodies[frame], axis=-1).max()))
            reference = sim.load_loop_motion_reference(model, runtime, solver, csv,
                csv_source_fps=30., csv_frame_stride=1, announce=False)
            minimum = float('inf')
            for q in reference.qpos:
                data.qpos[:] = q
                sim.mujoco.mj_forward(model, data)
                minimum = min(minimum, min(foot_minima(data, feet)))
            item = dict(sequence=row['sequence'], csv_rows=count, full_policy_steps=reference.num_frames,
                duration_s=(count - 1) / 30., joint_max_error_rad=joint_error, root_max_error_m=root_error,
                rotation_max_error_rad=rotation_error, duration_error_s=duration_error,
                simulator_body_fk_max_error_m=body_error, resampled_actual_foot_min_z_m=minimum)
            outcomes.append(item)
            atomic_json(OUT / 'reference_contract.json', dict(status='CHECKING', motions=outcomes))
            if (count != 300 or max(joint_error, root_error, rotation_error, duration_error) > job['csv_roundtrip_max_error']
                    or body_error > job['model_body_fk_max_m'] or minimum < -job['reference_floor_budget_m']):
                raise ValueError('Reference ABI or simulator geometry gate rejected: ' + row['sequence'])
    finally:
        solver.close()
    atomic_json(OUT / 'reference_contract.json', dict(status='VERIFIED_REFERENCE_ABI', motions=outcomes))
    return outcomes


def evaluation_worker(job):
    from script.a3 import evaluate_mujoco as evaluation
    reference_contract(job)
    unchanged(job)
    evaluation.preflight = guard.preflight
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    def spawn(argv, **kwargs):
        argv = ['script.a3.smpl_frozen_entry' if word == 'script.a3.sim_entry' else word for word in argv]
        # Keep descendants in the guarded worker group so a stop reaches them all.
        kwargs['start_new_session'] = False
        return subprocess.Popen(argv, **kwargs)
    evaluation.subprocess = SimpleNamespace(Popen=spawn)
    args = SimpleNamespace(output=OUT / 'evaluation', logs=LOG / 'evaluation', checkpoint=OFFICIAL,
        motion=OUT / 'references', reference_fps=30., frame_stride=1, max_policy_steps=None,
        action_delay_ms=0., reference_buffer_ms=0., capture_inputs=True, capture_count=100000,
        joint_reference_mode=None, joint_predictor=None, joint_noise_std=0., joint_noise_seed=0,
        causal_orientation=False)
    result = evaluation.evaluate_explicit(args)
    if result:
        raise RuntimeError('Frozen evaluation failed; no retry')
    unchanged(job)


def verify_worker(job):
    import numpy as np
    from script.a3.verify_zhy_search import verify_metrics
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_frozen_entry import foot_geometry, foot_minima
    contract = read(OUT / 'reference_contract.json')
    if contract['status'] != 'VERIFIED_REFERENCE_ABI':
        raise ValueError('Reference contract incomplete')
    records = verify_metrics(OUT / 'evaluation')
    manifest = read(OUT / 'evaluation/explicit_manifest.json')
    summary = read(OUT / 'evaluation/explicit_summary.json')
    protocol = manifest['protocol']
    if (len(records) != 6 or protocol['checkpoint_sha256'] != job['official_sha256']
            or protocol['frame_stride'] != 1 or protocol['reference_fps_after_stride'] != 30.
            or protocol['action_delay_ms'] != 0. or protocol['max_policy_steps'] is not None):
        raise ValueError('Frozen evaluation protocol mismatch')
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(MJCF))
    data, feet = sim.mujoco.MjData(model), foot_geometry(model, sim.mujoco)
    outcomes = []
    for row in job['selection']:
        name = Path(row['filename']).stem
        trace = read(OUT / 'evaluation' / f'{name}.timeseries.json')['motions'][0]
        metric = read(OUT / 'evaluation' / f'{name}.metrics.json')['motions'][0]
        n = metric['num_policy_steps']
        expected = next(r['full_policy_steps'] for r in contract['motions'] if r['sequence'] == row['sequence'])
        height, roll, pitch = [np.asarray(trace[key]) for key in ('root_height', 'root_roll_deg', 'root_pitch_deg')]
        falling = np.flatnonzero((height < job['fall_height_m']) | (np.abs(roll) > job['fall_abs_roll_or_pitch_deg']) |
            (np.abs(pitch) > job['fall_abs_roll_or_pitch_deg']))
        fall_tick = int(falling[0]) if len(falling) else None
        if n != expected or metric['fall_tick'] != fall_tick or metric['fall'] != bool(len(falling)):
            raise ValueError('Independent coverage or fall check failed')
        with np.load(OUT / 'evaluation' / f'{name}.inputs.npz', allow_pickle=False) as captured:
            if len(captured['observations']) != n or captured['actions'].shape != (n, 29):
                raise ValueError('Missing policy input/action capture')
            np.testing.assert_allclose(captured['actions'], trace['raw_action_29'], atol=0, rtol=0)
            if not np.isfinite(captured['observations']).all():
                raise ValueError('Nonfinite policy observations')
        with np.load(OUT / 'evaluation' / f'{name}.contact.npz', allow_pickle=False) as contacts:
            if not np.array_equal(contacts['policy_tick'], np.arange(n)) or contacts['qpos'].shape != (n, model.nq):
                raise ValueError('Missing contact/full-state coverage')
            for key in ('qpos', 'qvel', 'foot_floor_normal_force_n', 'foot_geometry_min_z_m'):
                if not np.isfinite(contacts[key]).all():
                    raise ValueError('Nonfinite contact/full-state capture')
            minima = []
            for q in contacts['qpos']:
                data.qpos[:] = q
                sim.mujoco.mj_forward(model, data)
                minima.append(foot_minima(data, feet))
            np.testing.assert_allclose(minima, contacts['foot_geometry_min_z_m'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(contacts['qpos'][:, :3], trace['sim_root_pos_w'], atol=0, rtol=0)
            contact_frames = np.sum(contacts['foot_floor_contact_count'] > 0, axis=0).tolist()
            force_frames = np.sum(contacts['foot_floor_normal_force_n'] > 1., axis=0).tolist()
        action = np.asarray(trace['raw_action_29'])
        public = next(r for r in summary['motions'] if r['motion'] == row['filename'])
        outcomes.append(dict(sequence=row['sequence'], motion=row['filename'], full_policy_steps=n,
            full_coverage=True, fall=metric['fall'], first_fall_s=metric['fall_time_s'],
            windows=public['windows'], contact_frames_left_right=contact_frames,
            force_over_1n_frames_left_right=force_frames, foot_actual_min_z_m=float(np.min(minima)),
            action_rms=float(np.sqrt(np.mean(action ** 2))), action_abs_max=float(np.abs(action).max()),
            action_delta_rms=float(np.sqrt(np.mean(np.diff(action, axis=0) ** 2))),
            dynamic_gate_passed=not metric['fall'], artifacts={suffix:dict(bytes=path.stat().st_size, sha256=sha256(path))
                for suffix in ('metrics.json', 'timeseries.json', 'inputs.npz', 'contact.npz')
                for path in [OUT / 'evaluation' / f'{name}.{suffix}']}))
    unchanged(job)
    atomic_json(OUT / 'verified_results.json', dict(motions=outcomes, optimizer_steps=0,
        policy_rollouts=6, frozen_official_unchanged=True, reference_abi=contract,
        dynamic_passed=sum(row['dynamic_gate_passed'] for row in outcomes),
        fresh_final_test=False, policy_improvement_proven=False, deploy_allowed=False,
        backup_status='LOCAL_ONLY'))


def worker(mode):
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    expected = 'RUNNING' if mode == 'evaluate' else 'VERIFYING'
    if state['status'] != expected or state['monitor_pid'] != os.getppid() or os.environ.get('YUANQI_E24_MONITOR_PID') != str(os.getppid()):
        raise RuntimeError('Worker requires guarded monitor')
    unchanged(job)
    return evaluation_worker(job) if mode == 'evaluate' else verify_worker(job)


def run(mode='evaluate'):
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        required = 'READY' if mode == 'evaluate' else 'COMPLETE'
        if state['status'] != required:
            raise RuntimeError('No repeat or recovery of E24')
        guard.preflight()
        unchanged(job)
        state.update(status='RUNNING' if mode == 'evaluate' else 'VERIFYING', monitor_pid=os.getpid())
        state[mode + '_started_at'] = time.time()
        atomic_json(OUT / 'state.json', state)
        stage = LOG / mode
        stage.mkdir()
        try:
            with (stage / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_frozen_probe', 'worker', '--mode', mode], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E24_MONITOR_PID=str(os.getpid())), stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                code, reason = guard.guarded_monitor(child, stage, job)
            state.update(status=('COMPLETE' if mode == 'evaluate' else 'VERIFIED') if code == 0 and reason is None else 'STOPPED',
                returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state[mode + '_ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        if state['status'] == 'VERIFIED':
            from script.a3 import zhy_fp32_trial as prior
            before, prior.LOG = prior.LOG, LOG
            try:
                health = prior.health_summary()
            finally:
                prior.LOG = before
            if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors'] or health['telemetry_errors']
                    or health['oom_counts'] != [job['baseline_oom_kill']]):
                state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
                atomic_json(OUT / 'state.json', state)
                raise ValueError('Health verification failed')
            result = read(OUT / 'verified_results.json')
            receipt = dict(result, status='VERIFIED_FROZEN_BASELINE', state=state, health=health,
                code_sha256=job['code_sha256'], input_sha256=job['input_sha256'])
            atomic_json(OUT / 'independent_verification.json', receipt)
            atomic_json(ROOT / 'data/manifests/a3_zhy_smpl_frozen_20261013_E24.json', receipt)
        print(json.dumps(state))
        return 0 if state['status'] in ('COMPLETE', 'VERIFIED') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'verify', 'worker'))
    parser.add_argument('--mode', choices=('evaluate', 'verify'), default='evaluate')
    args = parser.parse_args()
    result = register() if args.action == 'register' else worker(args.mode) if args.action == 'worker' else run('verify' if args.action == 'verify' else 'evaluate')
    raise SystemExit(result if isinstance(result, int) else 0)
