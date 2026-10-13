"""E27 root-command distinguishability and passive E26 failure chronology."""

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'data/experiments/a3_zhy_smpl_frozen_20261013_E26'
OUT = ROOT / 'data/experiments/a3_zhy_smpl_root_audit_20261013_E27'
LOG = ROOT / 'logs/a3_zhy_smpl_root_audit_20261013_E27'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_root_audit_20261013_E27.json'
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
SIM = VENDOR / 'gear_sonic/scripts/sim2sim_a3_mujoco.py'
MJCF = VENDOR / 'gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml'
CONFIG = ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/config.yaml'
CODE = ('script/a3/smpl_root_audit.py', 'script/a3/reference_contract.py',
        'script/a3/zhy_trial.py', 'script/a3/host_health.py',
        'script/a3/smpl_frozen_followup.py', 'script/a3/zhy_fp32_trial.py',
        'script/diagnostics/host_loadcheck.py', 'script/a3/checkpoint_store.py')


def read(path):
    return json.loads(path.read_text())


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Confirmed new boot; do not recover')
    for name, identity in {**job['code_sha256'], **job['input_sha256']}.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Registered code or evidence changed: ' + name)


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E27')
    value = guard.preflight()
    prior = read(PRIOR / 'independent_verification.json')
    old = read(PRIOR / 'registration.json')
    if (prior['status'] != 'VERIFIED_FROZEN_BASELINE'
            or read(PRIOR / 'state.json')['status'] != 'VERIFIED'
            or prior['health']['boot_ids'] != [value['boot_id']]
            or len(prior['motions']) != 6):
        raise ValueError('Requires verified same-boot E26 evidence')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = [PRIOR / name for name in ('registration.json', 'state.json',
                                     'independent_verification.json')]
    files += [SIM, MJCF, CONFIG,
        VENDOR / 'gear_sonic/envs/manager_env/mdp/observations.py',
        VENDOR / 'gear_sonic/envs/manager_env/mdp/rewards.py']
    for row in prior['motions']:
        name = Path(row['motion']).stem
        for suffix, identity in row['artifacts'].items():
            path = PRIOR / 'evaluation' / (name + '.' + suffix)
            if sha256(path) != identity['sha256']:
                raise ValueError('E26 captured evidence changed')
            files.append(path)
    selection = next(r for r in old['selection'] if r['sequence'].endswith('/zzx_111505_walk'))
    csv = ROOT / selection['source'] / 'reference.csv'
    console = ROOT / 'logs/a3_zhy_smpl_frozen_20261013_E26/evaluation' / Path(selection['filename']).stem / 'attempt_001/console.log'
    if 'anchor body: pelvis_link (id=1), root body: pelvis_link (id=1)' not in console.read_text():
        raise ValueError('Actual rollout anchor not confirmed as pelvis')
    files += [csv, console]
    job = dict(experiment='A3_ZHY_SMPL_ROOT_AUDIT_20261013_E27', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=old['experiment'], official_sha256=old['official_sha256'],
        selection=selection, saved_motions=prior['motions'],
        main_variable='REFERENCE_ROOT_X_RAMP_WITH_FIXED_JOINTS_ORIENTATIONS_AND_ROBOT_STATE',
        root_x_added_velocity_m_s=.2, frame_skip=1, history_frames=0,
        root_encoder_difference_expected=0., capture_rebuild_atol=4e-6,
        positive_control_joint_offset_rad=.01, positive_control_yaw_offset_rad=.01,
        positive_control_min_difference=1e-5, expected_frames=499,
        state_time_convention='E26_LABEL_TICK_DT_IS_POST_STEP_STATE_AT_(TICK+1)_DT',
        saved_trace_audit='DESCRIPTIVE_ONLY_NO_CAUSAL_INTERVENTION',
        negligible_foot_force_n=1., negligible_force_run_frames=5,
        chronology_tilt_thresholds_deg=[15., 30., 60.],
        chronology_joint_rmse_threshold_rad=.2, fall_height_m=.45,
        snapshot_start_s=4.6, snapshot_interval_frames=5,
        policy_rollouts_allowed=0, policy_weight_loads_allowed=0, optimizer_steps_allowed=0,
        cuda_allowed=False, max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2,
        sample_seconds=5, memory_reserve_gib=4, runtime_limit_seconds=900,
        auto_resume=False, repeat_allowed=False, backup_status='LOCAL_ONLY',
        code_sha256={name:sha256(ROOT / name) for name in CODE},
        input_sha256={str(path.relative_to(ROOT)):sha256(path) for path in files},
        baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines())
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], new_rollouts=0)))


def chronology(job, row):
    import numpy as np
    name = Path(row['motion']).stem
    trace = read(PRIOR / 'evaluation' / (name + '.timeseries.json'))['motions'][0]
    metric = read(PRIOR / 'evaluation' / (name + '.metrics.json'))['motions'][0]
    with np.load(PRIOR / 'evaluation' / (name + '.contact.npz'), allow_pickle=False) as saved:
        ticks, qpos = saved['policy_tick'], saved['qpos']
        force, count = saved['foot_floor_normal_force_n'], saved['foot_floor_contact_count']
    n = job['expected_frames']
    if not np.array_equal(ticks, np.arange(n)) or not np.array_equal(ticks, trace['policy_tick']):
        raise ValueError('Incomplete saved chronology')
    np.testing.assert_allclose(qpos[:, :3], trace['sim_root_pos_w'], atol=1e-12, rtol=0)
    if (force.shape != (n, 2) or count.shape != force.shape or not np.isfinite(force).all()
            or np.any(force < -1e-8) or np.any(count < 0)):
        raise ValueError('Invalid saved contact evidence')
    if np.sum(force > 1., axis=0).tolist() != row['force_over_1n_frames_left_right']:
        raise ValueError('Contact evidence differs from E26 verification')
    height, roll, pitch = [np.asarray(trace[k]) for k in ('root_height', 'root_roll_deg', 'root_pitch_deg')]
    times = np.asarray(trace['time_s'])
    np.testing.assert_allclose(times, ticks / 50., atol=1e-12, rtol=0)
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    falling = (height < job['fall_height_m']) | (tilt > 60.)
    falls = np.flatnonzero(falling)
    fall = int(falls[0]) if len(falls) else None
    if fall != metric['fall_tick']:
        raise ValueError('Independent fall chronology differs')
    end = fall + 1 if fall is not None else n
    state, reference, action, target = [np.asarray(trace[k]) for k in
        ('q_state_29', 'reference_q_29', 'raw_action_29', 'q_des_29')]
    for array in (state, reference, action, target):
        if array.shape != (n, 29) or not np.isfinite(array).all():
            raise ValueError('Incomplete control chronology')
    joint_error = np.sqrt(np.mean((state - reference) ** 2, axis=1))
    no_force = np.all(force <= job['negligible_foot_force_n'], axis=1)
    episodes, start = [], None
    for tick in range(end + 1):
        if tick < end and no_force[tick]:
            if start is None:
                start = tick
        elif start is not None:
            if tick - start >= job['negligible_force_run_frames']:
                episodes.append(dict(start_tick=start, end_tick_exclusive=tick,
                    start_label_s=start / 50., duration_s=(tick - start) / 50.))
            start = None
    def first(mask):
        indices = np.flatnonzero(mask[:end])
        return dict(tick=int(indices[0]), label_s=float(times[indices[0]]),
            post_step_state_time_s=float(times[indices[0]] + .02)) if len(indices) else None
    def changes(values):
        return np.r_[0., np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))]
    dq, da, dt = changes(reference), changes(action), changes(target)
    snapshots = []
    if fall is not None:
        indices = list(range(int(round(job['snapshot_start_s'] * 50)), end,
                             job['snapshot_interval_frames']))
        indices = sorted(set(indices + [fall - 2, fall - 1, fall]))
        for tick in indices:
            snapshots.append(dict(tick=tick, label_s=float(times[tick]),
                post_step_state_time_s=float(times[tick] + .02), root_height_m=float(height[tick]),
                roll_deg=float(roll[tick]), pitch_deg=float(pitch[tick]),
                floor_force_left_right_n=force[tick].tolist(), contact_count_left_right=count[tick].tolist(),
                joint_rmse_rad=float(joint_error[tick]), reference_delta_rms_rad=float(dq[tick]),
                action_delta_rms=float(da[tick]), target_delta_rms_rad=float(dt[tick]),
                action_abs_max=float(np.abs(action[tick]).max())))
    return dict(sequence=row['sequence'], complete_steps=n, first_fall=first(falling),
        first_tilt={str(v):first(tilt > v) for v in job['chronology_tilt_thresholds_deg']},
        first_joint_rmse_over_threshold=first(joint_error > job['chronology_joint_rmse_threshold_rad']),
        consecutive_negligible_foot_force_episodes_through_first_fall=episodes,
        negligible_force_frames_through_first_fall=int(no_force[:end].sum()),
        reference_delta_rms_max_before_fall=float(dq[:end].max()),
        action_delta_rms_max_before_fall=float(da[:end].max()),
        target_delta_rms_max_before_fall_rad=float(dt[:end].max()), snapshots=snapshots)


def distinguishability(job):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(MJCF))
    runtime = sim.build_loop_runtime(model, sim.DEFAULT_URDF, anchor_body_name='pelvis_link')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE,
                                  sim.DEFAULT_BRIDGE_CACHE, False)
    try:
        reference = sim.load_loop_motion_reference(model, runtime, solver,
            ROOT / job['selection']['source'] / 'reference.csv',
            csv_source_fps=30., csv_frame_stride=1, announce=False)
    finally:
        solver.close()
    name = Path(job['selection']['filename']).stem
    trace = read(PRIOR / 'evaluation' / (name + '.timeseries.json'))['motions'][0]
    with np.load(PRIOR / 'evaluation' / (name + '.inputs.npz'), allow_pickle=False) as saved:
        actual = saved['observations']
    n = job['expected_frames']
    if reference.num_frames != n or actual.shape[0] != n or actual.shape[1] <= 640:
        raise ValueError('Incomplete input audit')
    np.testing.assert_allclose(reference.dof_il, trace['reference_q_29'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(reference.root_pos, trace['ref_root_pos_w'], atol=1e-12, rtol=0)
    before_quat = np.vstack([reference.anchor_quat_wxyz[0],
                            np.asarray(trace['sim_root_quat_wxyz'])[:-1]])
    ramp = np.arange(n) / 50. * job['root_x_added_velocity_m_s']
    positions, qpos, qvel = reference.root_pos.copy(), reference.qpos.copy(), reference.qvel.copy()
    positions[:, 0] += ramp
    qpos[:, 0] += ramp
    qvel[:, 0] += job['root_x_added_velocity_m_s']
    altered = replace(reference, root_pos=positions, qpos=qpos, qvel=qvel)
    def inputs(ref):
        enc = np.asarray([sim.build_encoder_input(ref, tick, before_quat[tick],
            on_end='hold_last', frame_skip=1, history_frames=0) for tick in range(n)])
        return np.concatenate([enc[:, :, :58].reshape(n, 580), enc[:, :, 58:].reshape(n, 60)], axis=1)
    original, shifted = inputs(reference), inputs(altered)
    if not np.isfinite(original).all() or not np.isfinite(shifted).all():
        raise ValueError('Nonfinite rebuilt input')
    capture_error = float(np.max(np.abs(original - actual[:, :640])))
    delta = float(np.max(np.abs(original - shifted)))
    if capture_error > job['capture_rebuild_atol'] or delta != job['root_encoder_difference_expected']:
        raise ValueError('Input reconstruction or root distinguishability hypothesis rejected')
    joints = reference.dof_il.copy()
    joints[:, 0] += job['positive_control_joint_offset_rad']
    joint_delta = float(np.max(np.abs(inputs(replace(reference, dof_il=joints)) - original)))
    yaw = Rotation.from_euler('z', job['positive_control_yaw_offset_rad'])
    orientations = (yaw * Rotation.from_quat(reference.anchor_quat_wxyz, scalar_first=True)).as_quat(scalar_first=True)
    yaw_delta = float(np.max(np.abs(inputs(replace(reference, anchor_quat_wxyz=orientations)) - original)))
    if min(joint_delta, yaw_delta) <= job['positive_control_min_difference']:
        raise ValueError('Positive controls did not change input')
    return dict(status='VERIFIED_ROOT_TRAJECTORY_INPUT_ALIAS', frames=n,
        anchor_body=runtime.mapping.anchor_body_name, encoder_dimension=640,
        actual_full_observation_dimension=int(actual.shape[1]),
        capture_rebuild_max_abs_error=capture_error, root_encoder_max_abs_difference=delta,
        root_endpoint_x_difference_m=float(ramp[-1]), root_initial_position_difference_m=float(ramp[0]),
        root_added_velocity_m_s=job['root_x_added_velocity_m_s'],
        joint_positive_control_max_abs_difference=joint_delta,
        yaw_positive_control_max_abs_difference=yaw_delta,
        actor_proprioception_condition='FIXED_CAPTURED_ROBOT_STATE_HISTORY',
        full_input_unchanged=True, policy_weights_loaded=0, new_rollouts=0, optimizer_steps=0,
        limitation='COUNTERFACTUAL_INPUT_OBSERVABILITY_NOT_A_FEASIBLE_GAIT_OR_FALL_CAUSE_PROOF')


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E27_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E27 guarded monitor')
    unchanged(job)
    root_result = distinguishability(job)
    traces = [chronology(job, row) for row in job['saved_motions']]
    unchanged(job)
    atomic_json(OUT / 'result.json', dict(root_input_audit=root_result, saved_trace_chronology=traces,
        note='Events are sampled post-step evidence; ordering alone is not causation'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E27')
        guard.preflight()
        unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),
                    '-m', 'script.a3.smpl_root_audit', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E27_MONITOR_PID=str(os.getpid())), stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, LOG / 'worker', job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED',
                         returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        if state['status'] == 'COMPLETE':
            from script.a3.smpl_frozen_followup import health_summary
            health = health_summary(LOG)
            if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
                    or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
                state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
                atomic_json(OUT / 'state.json', state)
                raise ValueError('Health verification failed')
            receipt = dict(read(OUT / 'result.json'), experiment=job['experiment'],
                status='VERIFIED_SAVED_TRACE_AND_ROOT_INPUT_AUDIT', state=state, health=health,
                code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
                evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
                    for p in (OUT / 'registration.json', OUT / 'result.json', LOG / 'console.log')},
                backup_status='LOCAL_ONLY')
            atomic_json(OUT / 'verification.json', receipt)
            atomic_json(MANIFEST, receipt)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    result = {'register':register, 'run':run, 'worker':worker}[parser.parse_args().action]()
    raise SystemExit(result if isinstance(result, int) else 0)
