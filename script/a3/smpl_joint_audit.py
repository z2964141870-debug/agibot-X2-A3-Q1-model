"""E28 audit saved joint targets, feedback, limits and closed-loop mappings."""

import argparse
import ast
import fcntl
import hashlib
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
PARENT = ROOT / 'data/experiments/a3_zhy_smpl_root_audit_20261013_E27'
OUT = ROOT / 'data/experiments/a3_zhy_smpl_joint_audit_20261013_E28'
LOG = ROOT / 'logs/a3_zhy_smpl_joint_audit_20261013_E28'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_joint_audit_20261013_E28.json'
VENDOR = ROOT / 'script/vendor/sonic_for_a3'
ASSETS = VENDOR / 'gear_sonic/data/assets/robot_description'
SIM = VENDOR / 'gear_sonic/scripts/sim2sim_a3_mujoco.py'
MJCF = ASSETS / 'mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml'
URDF = ASSETS / 'urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf'
SOLVER = ASSETS / 'solver/a3_loop/lib/liba3_ankle_waist_solver.a'
INCLUDE = ASSETS / 'solver/a3_loop/include'
CODE = ('script/a3/smpl_joint_audit.py', 'script/a3/reference_contract.py',
        'script/a3/zhy_trial.py', 'script/a3/host_health.py',
        'script/a3/smpl_frozen_followup.py', 'script/a3/zhy_fp32_trial.py',
        'script/diagnostics/host_loadcheck.py', 'script/a3/checkpoint_store.py')


def read(path):
    return json.loads(path.read_text())


def cached_bridge():
    tree = ast.parse(SIM.read_text())
    cpp = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == 'BRIDGE_CPP' for name in node.targets))
    identity = hashlib.sha256((cpp + '\n' + str(SOLVER.resolve()) + '\n' + str(INCLUDE.resolve())).encode()).hexdigest()[:16]
    path = Path(os.environ.get('A3_LOOP_BRIDGE_CACHE', '/tmp/a3_loop_solver_bridge')) / identity / 'liba3_loop_solver_bridge.so'
    if not path.is_file():
        raise RuntimeError('Requires the existing bridge; no rebuild in this audit')
    return path


def unchanged(job):
    if guard.health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Confirmed new boot; no recovery')
    for name, identity in {**job['code_sha256'], **job['input_sha256']}.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Registered code or evidence changed: ' + name)
    if sha256(Path(job['bridge_path'])) != job['bridge_sha256']:
        raise ValueError('Registered solver bridge changed')


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E28')
    value = guard.preflight()
    parent, prior = read(PARENT / 'verification.json'), read(PRIOR / 'independent_verification.json')
    if (parent['status'] != 'VERIFIED_SAVED_TRACE_AND_ROOT_INPUT_AUDIT'
            or parent['health']['boot_ids'] != [value['boot_id']]
            or prior['status'] != 'VERIFIED_FROZEN_BASELINE'):
        raise ValueError('Requires verified same-boot E27 and E26')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = [PARENT / 'verification.json', PARENT / 'state.json',
             PRIOR / 'independent_verification.json', PRIOR / 'registration.json', SIM, MJCF, URDF, SOLVER]
    files += sorted(INCLUDE.rglob('*.h'))
    for row in prior['motions']:
        for suffix, identity in row['artifacts'].items():
            path = PRIOR / 'evaluation' / (Path(row['motion']).stem + '.' + suffix)
            if sha256(path) != identity['sha256']:
                raise ValueError('E26 evidence changed')
            files.append(path)
    bridge = cached_bridge()
    job = dict(experiment='A3_ZHY_SMPL_JOINT_AUDIT_20261013_E28', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=parent['experiment'], saved_motions=prior['motions'],
        main_variable='NAMED_JOINT_BREAKDOWN_AND_INDEPENDENT_SAVED_FEEDBACK_RECONSTRUCTION',
        causal_intervention=False, expected_frames=499, focus_window_s=[4.4, 4.9],
        joint_error_threshold_rad=.35, target_limit_margin_rad=.05,
        nominal_sustained_frames=5, feedback_rebuild_atol=1e-8,
        captured_position_atol=4e-6, captured_velocity_atol=2e-4,
        target_rebuild_atol=1e-10, physical_loop_discrepancy_rad=.02,
        policy_weight_loads_allowed=0, policy_rollouts_allowed=0, optimizer_steps_allowed=0,
        simulator_steps_allowed=0, cuda_allowed=False, bridge_rebuild_allowed=False,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        state_time_convention='POST_STEP_STATE_TIME=(POLICY_TICK+1)/50',
        bridge_path=str(bridge), bridge_sha256=sha256(bridge),
        code_sha256={name:sha256(ROOT / name) for name in CODE},
        input_sha256={str(path.relative_to(ROOT)):sha256(path) for path in files},
        baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], new_rollouts=0)))


def latest_terms(observations, sim):
    n, offset, terms = len(observations), 640, {}
    for name in sim.OBS_TERM_ORDER:
        dimension = sim.OBS_TERM_DIMS[name]
        stop = offset + sim.NUM_FUTURE_FRAMES * dimension
        terms[name] = observations[:, offset:stop].reshape(n, sim.NUM_FUTURE_FRAMES, dimension)[:, -1]
        offset = stop
    if offset != observations.shape[1]:
        raise ValueError('Unexpected captured observation ABI')
    return terms


def first_run(mask, length):
    start = None
    for tick, present in enumerate(mask):
        start = tick if present and start is None else start if present else None
        if start is not None and tick - start + 1 >= length:
            return dict(start_tick=int(start), label_s=float(start / 50.),
                        post_step_state_s=float((start + 1) / 50.),
                        first_confirmed_tick=int(tick), nominal_frames=length)
    return None


def audit_motion(job, row, sim, model, runtime, solver):
    import numpy as np
    mapping = runtime.mapping
    name = Path(row['motion']).stem
    trace = read(PRIOR / 'evaluation' / (name + '.timeseries.json'))['motions'][0]
    metric = read(PRIOR / 'evaluation' / (name + '.metrics.json'))['motions'][0]
    with np.load(PRIOR / 'evaluation' / (name + '.contact.npz'), allow_pickle=False) as saved:
        qpos, qvel, ticks = saved['qpos'], saved['qvel'], saved['policy_tick']
        force = saved['foot_floor_normal_force_n']
    with np.load(PRIOR / 'evaluation' / (name + '.inputs.npz'), allow_pickle=False) as saved:
        observations, captured_actions = saved['observations'], saved['actions']
    n = job['expected_frames']
    if (qpos.shape != (n, model.nq) or qvel.shape != (n, model.nv)
            or not np.array_equal(ticks, np.arange(n)) or len(observations) != n):
        raise ValueError('Incomplete saved state')
    reference, state, target, actions = [np.asarray(trace[key]) for key in
        ('reference_q_29', 'q_state_29', 'q_des_29', 'raw_action_29')]
    if any(value.shape != (n, 29) for value in (reference, state, target, actions)):
        raise ValueError('Incomplete saved joint arrays')
    if any(not np.isfinite(value).all() for value in
           (qpos, qvel, observations, captured_actions, reference, state, target, actions, force)):
        raise ValueError('Nonfinite saved evidence')
    data, reconstructed, velocities = sim.mujoco.MjData(model), [], []
    for tick in range(n):
        data.qpos[:], data.qvel[:] = qpos[tick], qvel[tick]
        q, dq = sim.loop_state_overrides(data, runtime, solver)
        reconstructed.append([q.get(joint, data.qpos[mapping.qpos_addr_by_joint[joint]])
                              for joint in mapping.il_joint_names])
        velocities.append([dq.get(joint, data.qvel[mapping.qvel_addr_by_joint[joint]])
                           for joint in mapping.il_joint_names])
    reconstructed, velocities = np.asarray(reconstructed), np.asarray(velocities)
    physical = qpos[:, [mapping.qpos_addr_by_joint[joint] for joint in mapping.il_joint_names]]
    physical_velocity = qvel[:, [mapping.qvel_addr_by_joint[joint] for joint in mapping.il_joint_names]]
    latest = latest_terms(observations, sim)
    feedback_error = float(np.max(np.abs(reconstructed - state)))
    prop_q_error = float(np.max(np.abs(latest['joint_pos'][1:] - (reconstructed[:-1] - mapping.default_q_il))))
    prop_v_error = float(np.max(np.abs(latest['joint_vel'][1:] - velocities[:-1])))
    prop_action_error = float(np.max(np.abs(latest['actions'][1:] - actions[:-1])))
    targets = mapping.default_q_il + np.clip(captured_actions, -sim.ACTION_CLIP, sim.ACTION_CLIP) * mapping.action_scale_il
    target_error = float(np.max(np.abs(targets - target)))
    if (feedback_error > job['feedback_rebuild_atol'] or prop_q_error > job['captured_position_atol']
            or prop_v_error > job['captured_velocity_atol'] or prop_action_error > 1e-7
            or target_error > job['target_rebuild_atol']):
        raise ValueError('Feedback, history alignment or target reconstruction rejected')
    fall = metric['fall_tick']
    end = fall if fall is not None else n
    joint_ids = [sim.mujoco.mj_name2id(model, sim.mujoco.mjtObj.mjOBJ_JOINT, joint)
                 for joint in mapping.il_joint_names]
    if min(joint_ids) < 0 or not np.all(model.jnt_limited[joint_ids]):
        raise ValueError('Missing logical joint limits')
    limits = model.jnt_range[joint_ids]
    error, physical_error = state - reference, reconstructed - physical
    excess = np.maximum(np.maximum(limits[:, 0] - target, target - limits[:, 1]), 0.)
    reference_excess = np.maximum(np.maximum(limits[:, 0] - reference, reference - limits[:, 1]), 0.)
    if np.max(reference_excess) > 1e-6:
        raise ValueError('Qualified reference violates the captured simulator joint limits')
    focus = (ticks / 50. >= job['focus_window_s'][0]) & (ticks / 50. <= job['focus_window_s'][1])
    rows = []
    for index, joint in enumerate(mapping.il_joint_names):
        mask = np.abs(error[:end, index]) > job['joint_error_threshold_rad']
        out = excess[:end, index] > job['target_limit_margin_rad']
        discrepancy = np.abs(physical_error[:end, index]) > job['physical_loop_discrepancy_rad']
        group = 'legs' if any(token in joint for token in ('hip_', 'knee_', 'ankle_')) else 'waist' if joint.startswith('waist_') else 'arms'
        rows.append(dict(joint=joint, index=index, group=group, limit_rad=limits[index].tolist(),
            action_scale_rad=float(mapping.action_scale_il[index]),
            focus_joint_rmse_rad=float(np.sqrt(np.mean(error[focus, index] ** 2))),
            focus_target_minus_reference_rms_rad=float(np.sqrt(np.mean((target[focus, index] - reference[focus, index]) ** 2))),
            first_sustained_tracking_error=first_run(mask, job['nominal_sustained_frames']),
            first_sustained_target_outside_limit=first_run(out, job['nominal_sustained_frames']),
            first_sustained_logical_physical_q_discrepancy=first_run(discrepancy, job['nominal_sustained_frames']),
            logical_physical_q_discrepancy_frames_before_fall=int(discrepancy.sum()),
            target_outside_limit_frames_before_fall=int(out.sum()),
            target_limit_excess_max_before_fall_rad=float(excess[:end, index].max()),
            reference_min_max_before_fall_rad=[float(reference[:end, index].min()), float(reference[:end, index].max())],
            state_min_max_before_fall_rad=[float(state[:end, index].min()), float(state[:end, index].max())],
            target_min_max_before_fall_rad=[float(target[:end, index].min()), float(target[:end, index].max())],
            logical_physical_q_difference_max_before_fall_rad=float(np.abs(physical_error[:end, index]).max()),
            logical_physical_velocity_difference_max_before_fall_rad_s=float(np.abs(velocities[:end, index] - physical_velocity[:end, index]).max())))
    rows.sort(key=lambda r:r['focus_joint_rmse_rad'], reverse=True)
    snapshots = []
    if fall is not None:
        selected = [r['index'] for r in rows[:5]]
        for tick in range(220, 246, 5):
            snapshots.append(dict(tick=tick, label_s=tick / 50., floor_force_left_right_n=force[tick].tolist(),
                joints=[dict(joint=mapping.il_joint_names[index], reference_rad=float(reference[tick, index]),
                    target_rad=float(target[tick, index]), feedback_rad=float(state[tick, index]),
                    feedback_velocity_rad_s=float(velocities[tick, index]),
                    physical_rad=float(physical[tick, index]),
                    target_limit_excess_rad=float(excess[tick, index])) for index in selected]))
    groups = {group:float(np.sqrt(np.mean(error[focus][:, [r['index'] for r in rows if r['group'] == group]] ** 2)))
              for group in ('legs', 'waist', 'arms')}
    return dict(sequence=row['sequence'], actual_steps=n, strictly_before_fall_steps=end,
        first_fall_label_s=metric['fall_time_s'], focus_group_rmse_rad=groups,
        validation=dict(feedback_rebuild_max_rad=feedback_error, previous_state_capture_max_rad=prop_q_error,
            previous_velocity_capture_max_rad_s=prop_v_error, previous_action_capture_max=prop_action_error,
            target_rebuild_max_rad=target_error, reference_limit_excess_max_rad=float(reference_excess.max())),
        named_joints=rows, focus_snapshots=snapshots)


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E28_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E28 guarded monitor')
    unchanged(job)
    import numpy as np
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(MJCF))
    runtime = sim.build_loop_runtime(model, URDF, anchor_body_name='pelvis_link')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE, sim.DEFAULT_BRIDGE_CACHE, False)
    mapping = runtime.mapping
    probe = np.zeros((1, 1570), dtype=np.float32)
    offset = 640
    for ordinal, name in enumerate(sim.OBS_TERM_ORDER):
        dimension = sim.OBS_TERM_DIMS[name]
        stop = offset + sim.NUM_FUTURE_FRAMES * dimension
        probe[:, offset:stop] = -100 - ordinal
        probe[:, stop - dimension:stop] = 100 + ordinal
        offset = stop
    for ordinal, name in enumerate(sim.OBS_TERM_ORDER):
        np.testing.assert_array_equal(latest_terms(probe, sim)[name], np.full((1, sim.OBS_TERM_DIMS[name]), 100 + ordinal))
    try:
        motions = [audit_motion(job, row, sim, model, runtime, solver) for row in job['saved_motions']]
    finally:
        solver.close()
    actuator_mappings = []
    for joint in mapping.il_joint_names:
        actuator = mapping.actuator_id_by_joint[joint]
        driven = int(model.actuator_trnid[actuator, 0])
        actuator_mappings.append(dict(policy_joint=joint, driven_joint=sim.mujoco.mj_id2name(
            model, sim.mujoco.mjtObj.mjOBJ_JOINT, driven), closed_loop=joint in sim.LOOP_PR_JOINTS))
    unchanged(job)
    atomic_json(OUT / 'result.json', dict(motions=motions, actuator_mappings=actuator_mappings,
        latest_history_decoder_positive_controls=5, new_policy_rollouts=0, optimizer_steps=0,
        simulator_steps=0, policy_weight_loads=0,
        limit_scope='MODEL_POSITION_LIMIT_COMPARISON_NOT_MEASURED_SUBSTEP_TORQUE_SATURATION',
        conclusion_scope='SAVED_STATE_DESCRIPTIVE_LOCALIZATION_NOT_FALL_CAUSE_OR_POLICY_IMPROVEMENT'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E28')
        guard.preflight()
        unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_joint_audit', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', YUANQI_E28_MONITOR_PID=str(os.getpid())),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, LOG / 'worker', job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED', returncode=code, reason=reason)
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
            result = read(OUT / 'result.json')
            receipt = dict(result, status='VERIFIED_SAVED_JOINT_AND_MAPPING_AUDIT', experiment=job['experiment'],
                state=state, health=health, code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
                bridge_path=job['bridge_path'], bridge_sha256=job['bridge_sha256'],
                evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
                    for p in (OUT / 'registration.json', OUT / 'result.json', LOG / 'console.log')}, backup_status='LOCAL_ONLY')
            atomic_json(OUT / 'verification.json', receipt)
            atomic_json(MANIFEST, receipt)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    result = {'register':register, 'run':run, 'worker':worker}[parser.parse_args().action]()
    raise SystemExit(result if isinstance(result, int) else 0)
