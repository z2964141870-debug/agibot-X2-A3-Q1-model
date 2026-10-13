"""E32 centroidal support demand from qualified references and saved states."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_joint_audit as evidence
from script.a3 import smpl_motor_domain_trace as parent_stage
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = evidence.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_support_20261013_E32'
LOG = ROOT / 'logs/a3_zhy_smpl_support_20261013_E32'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_support_20261013_E32.json'
CODE = ('script/a3/smpl_support_demand.py', 'script/a3/smpl_frozen_entry.py') + parent_stage.CODE


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E32')
    value = guard.preflight()
    parent = read(parent_stage.OUT / 'verification.json')
    parent_job = read(parent_stage.OUT / 'registration.json')
    if (parent['status'] != 'VERIFIED_SAVED_MOTOR_RANGE_TRACE'
            or parent['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot E31')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = dict(parent['input_sha256'])
    paths = [parent_stage.OUT / 'verification.json', parent_stage.OUT / 'registration.json',
             evidence.PRIOR / 'registration.json']
    selection = read(evidence.PRIOR / 'registration.json')['selection']
    paths += [ROOT / row['source'] / name for row in selection
              for name in ('reference.csv', 'kinematic_trace.npz')]
    files.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    for name, identity in files.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Parent evidence changed: ' + name)
    job = dict(experiment='A3_ZHY_SMPL_SUPPORT_20261013_E32', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=parent['experiment'], selection=selection,
        saved_motions=read(evidence.PRIOR / 'independent_verification.json')['motions'],
        main_variable='CENTROIDAL_DEMAND_VS_PERMISSIVE_FOOT_SUPPORT_ENVELOPE',
        expected_frames=499, dt=.02, derivative_windows=[5, 11], derivative_polynomial=3,
        excluded_endpoint_frames=6, focus_window_s=[4.4, 4.9],
        minimum_vertical_force_bodyweights=.1, descriptive_margin_m=.01,
        nominal_sustained_frames=5, momentum_crosscheck_atol=1e-7,
        reference_velocity='CENTRAL_MJ_DIFFERENTIATEPOS_QUATERNION_AWARE',
        actual_velocity='CAPTURED_POST_STEP_QVEL',
        force_demand='TOTAL_MASS_TIMES_COM_SECOND_DERIVATIVE_MINUS_GRAVITY',
        moment_demand='DERIVATIVE_OF_TOTAL_ANGULAR_MOMENTUM_ABOUT_COM',
        support_assumption='ALL_COLLIDABLE_FOOT_MESH_XY_VERTICES_PLANAR_UPPER_ENVELOPE',
        reference_contact='NEAREST_SOURCE_INFERRED_CONTACT_NOT_GROUND_TRUTH',
        actual_contact='SAVED_FOOT_NORMAL_FORCE_GT_1N_NOT_CONTINUOUS_CONTACT_RECORD',
        internal_forces='ALL_BODY_MASSES_AND_INERTIAS_INCLUDED_INTERNAL_LOOP_FORCES_CANCEL',
        causal_intervention=False, policy_weight_loads_allowed=0, policy_rollouts_allowed=0,
        optimizer_steps_allowed=0, simulator_steps_allowed=0, cuda_allowed=False,
        bridge_rebuild_allowed=False, max_cpu_c=90, max_gpu_c=85,
        consecutive_hot_samples=2, sample_seconds=5, memory_reserve_gib=4,
        runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        state_time_convention='ACTUAL=(TICK+1)/50_REFERENCE=TICK/50',
        bridge_path=parent_job['bridge_path'], bridge_sha256=parent_job['bridge_sha256'],
        code_sha256={name:sha256(ROOT / name) for name in dict.fromkeys(CODE)},
        input_sha256=files, baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def polygon_margin(points, point):
    import numpy as np
    from scipy.spatial import ConvexHull
    points = np.unique(np.asarray(points), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
        return None
    hull = ConvexHull(points)
    if np.max(hull.equations[:, :2] @ point + hull.equations[:, 2]) <= 0:
        return float(np.min(-(hull.equations[:, :2] @ point + hull.equations[:, 2])))
    vertices = points[hull.vertices]
    edges = np.roll(vertices, -1, axis=0) - vertices
    weights = np.clip(np.sum((point - vertices) * edges, axis=1) / np.sum(edges ** 2, axis=1), 0, 1)
    return -float(np.linalg.norm(point - vertices - weights[:, None] * edges, axis=1).min())


def required_cop(com, force, moment):
    import numpy as np
    about_ground = moment + np.cross(com, force)
    return np.column_stack((-about_ground[:, 1] / force[:, 2],
                             about_ground[:, 0] / force[:, 2]))


def central_velocity(mujoco, model, qpos, dt):
    import numpy as np
    velocity = np.zeros((len(qpos), model.nv))
    for i in range(1, len(qpos) - 1):
        mujoco.mj_differentiatePos(model, velocity[i], 2 * dt, qpos[i - 1], qpos[i + 1])
    mujoco.mj_differentiatePos(model, velocity[0], dt, qpos[0], qpos[1])
    mujoco.mj_differentiatePos(model, velocity[-1], dt, qpos[-2], qpos[-1])
    return velocity


def reconstruct(job, mujoco, model, qpos, qvel, feet):
    import numpy as np
    from scipy.spatial import ConvexHull
    data = mujoco.MjData(model)
    mass = float(model.body_mass.sum())
    bodies = np.flatnonzero(model.body_mass > 0)
    com, linear, angular, projected, loop_residual = [], [], [], [], []
    max_error = np.zeros(3)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for q, v in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        mujoco.mj_subtreeVel(model, data)
        equality = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        loop_residual.append(float(np.max(np.abs(data.efc_pos[:data.nefc][equality]))) if equality.any() else 0.)
        center = np.sum(model.body_mass[:, None] * data.xipos, axis=0) / mass
        momentum, spin = np.zeros(3), np.zeros(3)
        for body in bodies:
            mujoco.mj_jacBodyCom(model, data, jacp, jacr, int(body))
            velocity, omega = jacp @ v, jacr @ v
            rotation = data.ximat[body].reshape(3, 3)
            inertia_omega = rotation @ (model.body_inertia[body] * (rotation.T @ omega))
            momentum += model.body_mass[body] * velocity
            spin += inertia_omega + model.body_mass[body] * np.cross(data.xipos[body] - center, velocity)
        errors = np.array([np.max(np.abs(center - data.subtree_com[0])),
                           np.max(np.abs(momentum / mass - data.subtree_linvel[0])),
                           np.max(np.abs(spin - data.subtree_angmom[0]))])
        max_error = np.maximum(max_error, errors)
        if np.max(errors) > job['momentum_crosscheck_atol']:
            raise ValueError('Independent all-body mass/momentum crosscheck rejected')
        outlines = []
        for selected in feet:
            vertices = np.concatenate([points @ data.geom_xmat[g].reshape(3, 3).T
                                       + data.geom_xpos[g] for g, points in selected])[:, :2]
            outlines.append(vertices[ConvexHull(vertices).vertices])
        com.append(center.copy())
        linear.append(momentum / mass)
        angular.append(spin)
        projected.append(outlines)
    return dict(com=np.asarray(com), velocity=np.asarray(linear), momentum=np.asarray(angular),
                outlines=projected, mass=mass, crosscheck=max_error.tolist(), loop_residual=np.asarray(loop_residual))


def summarize_mask(mask, valid, job, source):
    events = [evidence.first_run(mask & valid, length) for length in (1, job['nominal_sustained_frames'])]
    for event in events:
        if event:
            event.pop('post_step_state_s')
            event['sample_time_s'] = event['label_s'] + (job['dt'] if source == 'actual' else 0.)
    return dict(valid_frames=int(valid.sum()), flagged_frames=int((mask & valid).sum()),
        first=events[0], first_sustained=events[1])


def demand(job, physical, contacts, end, name):
    import numpy as np
    from scipy.signal import savgol_filter
    n, dt, mass = len(physical['com']), job['dt'], physical['mass']
    base_valid = (np.arange(n) >= job['excluded_endpoint_frames']) & (
        np.arange(n) < min(end - job['excluded_endpoint_frames'], n - job['excluded_endpoint_frames']))
    focus = (np.arange(n) * dt >= job['focus_window_s'][0]) & (
        np.arange(n) * dt <= job['focus_window_s'][1]) & base_valid
    arrays, methods = {}, []
    for window in job['derivative_windows']:
        acc = savgol_filter(physical['com'], window, job['derivative_polynomial'], deriv=2, delta=dt, axis=0)
        moment = savgol_filter(physical['momentum'], window, job['derivative_polynomial'], deriv=1, delta=dt, axis=0)
        force = mass * (acc - np.array([0., 0., -9.81]))
        eligible = force[:, 2] > mass * 9.81 * job['minimum_vertical_force_bodyweights']
        cop = np.zeros((n, 2))
        cop[eligible] = required_cop(physical['com'][eligible], force[eligible], moment[eligible])
        both, selected = np.full(n, np.nan), np.full(n, np.nan)
        for i in np.flatnonzero(eligible):
            both[i] = polygon_margin(np.concatenate(physical['outlines'][i]), cop[i])
            points = [p for p, active in zip(physical['outlines'][i], contacts[i]) if active]
            if points:
                selected[i] = polygon_margin(np.concatenate(points), cop[i])
        valid = base_valid & eligible & np.isfinite(both)
        selected_valid = base_valid & eligible & np.isfinite(selected)
        outside = both < -job['descriptive_margin_m']
        selected_outside = selected < -job['descriptive_margin_m']
        if not all(np.isfinite(x).all() for x in (acc, moment, force, cop)):
            raise ValueError('Nonfinite centroidal reconstruction')
        derivative_velocity = savgol_filter(physical['com'], window, job['derivative_polynomial'], deriv=1, delta=dt, axis=0)
        arrays.update({f'w{window}_{key}':value for key, value in dict(
            cop_xy=cop, force=force, moment=moment, margin_both=both,
            margin_selected=selected, valid=valid, selected_valid=selected_valid).items()})
        methods.append(dict(window_frames=window, source=name,
            support_both=summarize_mask(outside, valid, job, name),
            support_selected=summarize_mask(selected_outside, selected_valid, job, name),
            nonpositive_or_small_fz_frames=int((base_valid & ~eligible).sum()),
            both_margin_min_m=float(both[valid].min()) if valid.any() else None,
            selected_margin_min_m=float(selected[selected_valid].min()) if selected_valid.any() else None,
            both_outside_focus_frames=int((focus & valid & outside).sum()),
            selected_outside_focus_frames=int((focus & selected_valid & selected_outside).sum()),
            force_horizontal_vertical_ratio_p95=float(np.percentile(np.linalg.norm(force[valid, :2], axis=1) / force[valid, 2], 95)),
            com_velocity_consistency_p95_m_s=float(np.percentile(np.linalg.norm(
                derivative_velocity[base_valid] - physical['velocity'][base_valid], axis=1), 95))))
    w1, w2 = job['derivative_windows']
    robust_valid = arrays[f'w{w1}_valid'] & arrays[f'w{w2}_valid']
    robust = (arrays[f'w{w1}_margin_both'] < -job['descriptive_margin_m']) & (
              arrays[f'w{w2}_margin_both'] < -job['descriptive_margin_m'])
    result = dict(methods=methods, crosscheck_max_com_velocity_momentum_error=physical['crosscheck'],
        loop_equality_position_residual_max_m=float(physical['loop_residual'][:end].max()),
        sample_time_s='TICK/50' if name == 'reference' else '(TICK+1)/50',
        robust_both_support_event=summarize_mask(robust, robust_valid, job, name),
        robust_both_outside_focus_frames=int((focus & robust & robust_valid).sum()),
        derivative_scope='5_AND_11_FRAME_LOCAL_POLYNOMIAL_ESTIMATES_NOT_TRUE_SUBSTEP_ACCELERATIONS')
    return result, dict(com=physical['com'], velocity=physical['velocity'],
        angular_momentum=physical['momentum'], contact_mask=contacts,
        loop_equality_residual=physical['loop_residual'],
        **arrays)


def controls(job, mujoco):
    import numpy as np
    from scipy.signal import savgol_filter
    t = np.arange(61) * job['dt']
    polynomial = np.column_stack((.7 * t ** 2, -.4 * t ** 2, np.ones(len(t))))
    for window in job['derivative_windows']:
        acceleration = savgol_filter(polynomial, window, 3, deriv=2, delta=job['dt'], axis=0)
        np.testing.assert_allclose(acceleration[6:-6], np.tile([1.4, -.8, 0.], (49, 1)), atol=1e-10)
    center, force, moment = np.array([[.2, -.1, 1.]]), np.array([[0., 0., 100.]]), np.zeros((1, 3))
    np.testing.assert_allclose(required_cop(center, force, moment), center[:, :2], atol=1e-12)
    np.testing.assert_allclose(required_cop(center, np.array([[10., 0., 100.]]), moment), [[.1, -.1]], atol=1e-12)
    np.testing.assert_allclose(required_cop(center, force, np.array([[5., 10., 0.]])), [[.1, -.05]], atol=1e-12)
    square = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    np.testing.assert_allclose([polygon_margin(square, np.array(p)) for p in [(0., 0.), (2., 0.), (2., 2.)]],
                               [1., -1., -np.sqrt(2)], atol=1e-12)
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="box" size=".1 .2 .3" mass="2"/></body></worldbody></mujoco>')
    qpos = np.tile(model.qpos0, (len(t), 1))
    qpos[:, 0] = .3 * t
    qpos[:, 3], qpos[:, 6] = np.cos(.35 * t), np.sin(.35 * t)
    qvel = central_velocity(mujoco, model, qpos, job['dt'])
    np.testing.assert_allclose(qvel[:, 0], .3, atol=1e-12)
    np.testing.assert_allclose(qvel[:, 3:6], np.tile([0., 0., .7], (len(t), 1)), atol=1e-12)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_subtreeVel(model, data)
    np.testing.assert_allclose(data.subtree_angmom[0], [0., 0., 2 / 3 * (.1 ** 2 + .2 ** 2) * .7], atol=1e-12)
    return dict(polynomial_windows=2, wrench_cases=3, support_cases=3,
                free_body_velocity_quaternion_and_spin=3, passed=True)


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E32_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E32 guarded monitor')
    evidence.unchanged(job)
    import numpy as np
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_frozen_entry import foot_geometry
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(evidence.MJCF))
    if (not np.allclose(model.opt.gravity, [0., 0., -9.81])
            or model.opt.density or model.opt.viscosity or np.any(model.body_gravcomp)):
        raise ValueError('Centroidal assumption requires gravity-only external load in this model')
    tests = controls(job, sim.mujoco)
    runtime = sim.build_loop_runtime(model, evidence.URDF, anchor_body_name='pelvis_link')
    feet = [[(g, p) for g, p in side if model.geom_contype[g] or model.geom_conaffinity[g]]
            for side in foot_geometry(model, sim.mujoco)]
    if not all(feet):
        raise ValueError('Missing collidable support mesh')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE,
                                   sim.DEFAULT_BRIDGE_CACHE, False)
    rows = []
    try:
        for row in job['saved_motions']:
            stem = Path(row['motion']).stem
            selection = next(s for s in job['selection'] if s['sequence'] == row['sequence'])
            reference = sim.load_loop_motion_reference(model, runtime, solver,
                ROOT / selection['source'] / 'reference.csv', csv_source_fps=30., csv_frame_stride=1, announce=False)
            with np.load(ROOT / selection['source'] / 'kinematic_trace.npz', allow_pickle=False) as saved:
                source_time, inferred = saved['time_s'], saved['inferred_contact']
            indices = np.minimum(np.floor(np.arange(499) * job['dt'] * 30 + .5).astype(int), len(source_time) - 1)
            reference_contact = inferred[indices].astype(bool)
            with np.load(evidence.PRIOR / 'evaluation' / (stem + '.contact.npz'), allow_pickle=False) as saved:
                actual_q, actual_v = saved['qpos'], saved['qvel']
                ticks, actual_contact = saved['policy_tick'], saved['foot_floor_normal_force_n'] > 1.
            trace = read(evidence.PRIOR / 'evaluation' / (stem + '.timeseries.json'))['motions'][0]
            metric = read(evidence.PRIOR / 'evaluation' / (stem + '.metrics.json'))['motions'][0]
            end = metric['fall_tick'] if metric['fall_tick'] is not None else job['expected_frames']
            if (reference.qpos.shape != actual_q.shape or actual_q.shape != (499, model.nq)
                    or actual_v.shape != (499, model.nv) or reference_contact.shape != (499, 2)
                    or not np.array_equal(ticks, np.arange(499))):
                raise ValueError('Reference or captured-state ABI rejected')
            np.testing.assert_allclose(reference.dof_il, trace['reference_q_29'], atol=1e-12, rtol=0)
            reference_v = central_velocity(sim.mujoco, model, reference.qpos, job['dt'])
            result = dict(sequence=row['sequence'], strictly_before_fall_frames=end,
                first_fall_label_s=metric['fall_time_s'], methods={}, artifacts={})
            for name, qpos, qvel, contacts in (
                    ('reference', reference.qpos, reference_v, reference_contact),
                    ('actual', actual_q[:end], actual_v[:end], actual_contact[:end])):
                if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                    raise ValueError('Nonfinite state')
                physical = reconstruct(job, sim.mujoco, model, qpos, qvel, feet)
                output, arrays = demand(job, physical, contacts, end, name)
                result['methods'][name] = output
                path = OUT / (stem + '.' + name + '.npz')
                np.savez_compressed(path, **arrays)
                result['artifacts'][str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            rows.append(result)
            atomic_json(OUT / 'partial.json', dict(motions=rows))
    finally:
        solver.close()
    evidence.unchanged(job)
    atomic_json(OUT / 'result.json', dict(motions=rows, controls=tests, body_mass_kg=float(model.body_mass.sum()),
        collidable_foot_geometries=[len(f) for f in feet], model_equality_constraints=int(model.neq),
        policy_weight_loads=0, policy_rollouts=0, optimizer_steps=0, simulator_steps=0,
        actual_torque_measurements=0, actual_substep_accelerations=0,
        interpretation='NECESSARY_PLANAR_SUPPORT_DEMAND_ESTIMATE_NOT_SUFFICIENT_FEASIBILITY_OR_CAUSAL_PROOF',
        internal_loop_force_scope='CENTROIDAL_SUM_CANCELS_INTERNAL_FORCES_NO_UNCONSTRAINED_INVERSE_TORQUE_CLAIM',
        derivative_scope='SMOOTHED_50HZ_ESTIMATES_WITH_ENDPOINTS_AND_POST_FALL_EXCLUDED', backup_status='LOCAL_ONLY'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E32')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_support_demand', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', YUANQI_E32_MONITOR_PID=str(os.getpid())),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, LOG / 'worker', job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED', returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        from script.a3.smpl_frozen_followup import health_summary
        health = health_summary(LOG)
        if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
                or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
            state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
            atomic_json(OUT / 'state.json', state)
        result = read(OUT / 'result.json') if state['status'] == 'COMPLETE' else {}
        receipt = dict(result, status='VERIFIED_CENTROIDAL_DEMAND_ESTIMATES' if state['status'] == 'COMPLETE' else 'STOPPED',
            experiment=job['experiment'], state=state, health=health, code_sha256=job['code_sha256'],
            input_sha256=job['input_sha256'], evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in (OUT / 'registration.json', LOG / 'console.log')}, backup_status='LOCAL_ONLY')
        if state['status'] == 'COMPLETE':
            receipt['evidence'][str((OUT / 'result.json').relative_to(ROOT))] = dict(
                bytes=(OUT / 'result.json').stat().st_size, sha256=sha256(OUT / 'result.json'))
            atomic_json(OUT / 'verification.json', receipt)
        atomic_json(MANIFEST, receipt)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    result = {'register':register, 'run':run, 'worker':worker}[parser.parse_args().action]()
    raise SystemExit(result if isinstance(result, int) else 0)
