"""E29 independently compare saved-state FK finite differences with DFK."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_joint_audit as evidence
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = evidence.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_differential_20261013_E29'
LOG = ROOT / 'logs/a3_zhy_smpl_differential_20261013_E29'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_differential_20261013_E29.json'
CODE = ('script/a3/smpl_differential_audit.py',) + evidence.CODE


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E29')
    value = guard.preflight()
    parent = read(evidence.OUT / 'verification.json')
    if (parent['status'] != 'VERIFIED_SAVED_JOINT_AND_MAPPING_AUDIT'
            or parent['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot E28')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = {**parent['input_sha256'],
             str((evidence.OUT / 'verification.json').relative_to(ROOT)):sha256(evidence.OUT / 'verification.json')}
    files.update({str(p.relative_to(ROOT)):sha256(p)
                  for p in evidence.INCLUDE.rglob('*') if p.is_file()})
    for name, identity in files.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Parent evidence changed: ' + name)
    bridge = evidence.cached_bridge()
    job = dict(experiment='A3_ZHY_SMPL_DIFFERENTIAL_20261013_E29', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=parent['experiment'],
        saved_motions=read(evidence.PRIOR / 'independent_verification.json')['motions'],
        main_variable='INDEPENDENT_LOCAL_FK_DIFFERENTIAL_VERSUS_DFK_ON_SAVED_MOTOR_STATES',
        causal_intervention=False, expected_frames=499, focus_window_s=[4.4, 4.9],
        motor_position_eps_rad=[1e-3, 1e-4, 1e-5], chosen_derivative_eps_rad=1e-5,
        jacobian_atol=.005, jacobian_rtol=.01,
        velocity_atol_rad_s=.03, velocity_rtol=.01,
        fk_revisit_atol_rad=1e-8, captured_velocity_atol_rad_s=2e-4,
        policy_weight_loads_allowed=0, policy_rollouts_allowed=0, optimizer_steps_allowed=0,
        simulator_steps_allowed=0, cuda_allowed=False, bridge_rebuild_allowed=False,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        state_time_convention='POST_STEP_STATE_TIME=(POLICY_TICK+1)/50',
        bridge_path=str(bridge), bridge_sha256=sha256(bridge),
        code_sha256={name:sha256(ROOT / name) for name in CODE}, input_sha256=files,
        baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], new_rollouts=0)))


def exceeds(actual, expected, absolute, relative):
    import numpy as np
    return np.any(np.abs(actual - expected) > absolute + relative * np.abs(expected), axis=(-2, -1))


def local_derivatives(fk, dfk, motor_q, motor_v, job):
    import numpy as np
    base = fk(motor_q)
    analytic_v = dfk(motor_v)
    analytic_columns = []
    for axis in np.eye(2):
        fk(motor_q)
        analytic_columns.append(dfk(axis))
    analytic = np.column_stack(analytic_columns)
    numerical = []
    revisits = []
    for epsilon in job['motor_position_eps_rad']:
        columns = []
        for axis in np.eye(2):
            revisits.append(fk(motor_q))
            positive = fk(motor_q + epsilon * axis)
            revisits.append(fk(motor_q))
            negative = fk(motor_q - epsilon * axis)
            columns.append((positive - negative) / (2 * epsilon))
        numerical.append(np.column_stack(columns))
    revisits.append(fk(motor_q))
    return analytic, analytic_v, np.asarray(numerical), float(np.max(np.abs(np.asarray(revisits) - base)))


def audit_motion(job, row, sim, runtime, solver):
    import numpy as np
    name = Path(row['motion']).stem
    with np.load(evidence.PRIOR / 'evaluation' / (name + '.contact.npz'), allow_pickle=False) as saved:
        qpos, qvel, ticks = saved['qpos'], saved['qvel'], saved['policy_tick']
    with np.load(evidence.PRIOR / 'evaluation' / (name + '.inputs.npz'), allow_pickle=False) as saved:
        captured_vel = evidence.latest_terms(saved['observations'], sim)['joint_vel']
    metric = read(evidence.PRIOR / 'evaluation' / (name + '.metrics.json'))['motions'][0]
    n = job['expected_frames']
    if (len(qpos) != n or len(qvel) != n or not np.array_equal(ticks, np.arange(n))
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()):
        raise ValueError('Invalid saved state')
    end = metric['fall_tick'] if metric['fall_tick'] is not None else n
    devices = (
        ('left_ankle', sim.LEFT_ANKLE_MOTORS, sim.LEFT_ANKLE_PR,
            lambda q:solver.ankle_fk(0, q), lambda v:solver.ankle_dfk(0, v)),
        ('right_ankle', sim.RIGHT_ANKLE_MOTORS, sim.RIGHT_ANKLE_PR,
            lambda q:solver.ankle_fk(1, q), lambda v:solver.ankle_dfk(1, v)),
        ('waist', sim.WAIST_MOTORS, ('waist_pitch_joint', 'waist_roll_joint'),
            solver.waist_fk, solver.waist_dfk))
    summaries, arrays = [], {}
    for device, motors, logical, fk, dfk in devices:
        motor_q = qpos[:, [runtime.motor_qpos_addr[m] for m in motors]]
        motor_v = qvel[:, [runtime.motor_qvel_addr[m] for m in motors]]
        values = [local_derivatives(fk, dfk, q, v, job) for q,v in zip(motor_q, motor_v)]
        analytic, analytic_v, numerical, revisit = [np.asarray([value[i] for value in values]) for i in range(4)]
        if any(not np.isfinite(a).all() for a in (analytic, analytic_v, numerical, revisit)):
            raise ValueError('Nonfinite differential computation')
        indices = [runtime.mapping.il_joint_names.index(j) for j in logical]
        capture_error = float(np.abs(analytic_v[:-1] - captured_vel[1:, indices]).max())
        if capture_error > job['captured_velocity_atol_rad_s']:
            raise ValueError('DFK audit does not match captured actor velocity')
        derivative_index = job['motor_position_eps_rad'].index(job['chosen_derivative_eps_rad'])
        small = numerical[:, derivative_index]
        stable = (np.all(np.abs(numerical - small[:, None]) <=
                  job['jacobian_atol'] + job['jacobian_rtol'] * np.abs(small[:, None]), axis=(1,2,3))
                  & (revisit <= job['fk_revisit_atol_rad']))
        matrix_bad = exceeds(analytic, small, job['jacobian_atol'], job['jacobian_rtol'])
        numerical_v = np.einsum('nij,nj->ni', small, motor_v)
        velocity_bad = np.any(np.abs(analytic_v - numerical_v) >
            job['velocity_atol_rad_s'] + job['velocity_rtol'] * np.abs(numerical_v), axis=1)
        valid = stable & (ticks < end)
        focus = valid & (ticks / 50. >= job['focus_window_s'][0]) & (ticks / 50. <= job['focus_window_s'][1])
        difference = np.abs(analytic_v - numerical_v)
        matrix_difference = np.abs(analytic - small)
        if not bool(exceeds(-small[0], small[0], job['jacobian_atol'], job['jacobian_rtol'])):
            raise ValueError('Sign-error positive control did not trigger')
        counts = dict(pre_fall_states=end, stable_pre_fall_states=int(valid.sum()),
            unstable_pre_fall_states=int(end - valid.sum()),
            matrix_mismatch_stable_states=int((valid & matrix_bad).sum()),
            velocity_mismatch_stable_states=int((valid & velocity_bad).sum()))
        selected = np.flatnonzero(valid)
        worst = selected[np.argsort(difference[selected].max(axis=1))[-5:][::-1]]
        summaries.append(dict(device=device, logical_joint_order=list(logical), motor_order=list(motors),
            counts=counts, captured_velocity_max_error_rad_s=capture_error,
            fk_revisit_max_before_fall_rad=float(revisit[:end].max()),
            matrix_error_max_stable=float(matrix_difference[valid].max()) if valid.any() else None,
            velocity_error_max_stable_rad_s=float(difference[valid].max()) if valid.any() else None,
            focus_stable_states=int(focus.sum()),
            focus_velocity_error_max_rad_s=float(difference[focus].max()) if focus.any() else None,
            first_stable_matrix_mismatch=evidence.first_run(valid & matrix_bad, 1),
            first_stable_velocity_mismatch=evidence.first_run(valid & velocity_bad, 1),
            worst_stable_states=[dict(tick=int(t), label_s=float(t/50.), post_step_state_s=float((t+1)/50.),
                motor_q_rad=motor_q[t].tolist(), motor_v_rad_s=motor_v[t].tolist(),
                analytic_velocity_rad_s=analytic_v[t].tolist(), numerical_velocity_rad_s=numerical_v[t].tolist(),
                analytic_jacobian=analytic[t].tolist(), numerical_jacobian=small[t].tolist()) for t in worst]))
        arrays.update({device + '_' + key:value for key,value in
            dict(analytic=analytic, analytic_v=analytic_v, numerical=numerical,
                 numerical_v=numerical_v, fk_revisit=revisit, stable=stable).items()})
    path = OUT / (name + '.differential.npz')
    np.savez_compressed(path, policy_tick=ticks, **arrays)
    return dict(sequence=row['sequence'], actual_saved_frames=n, strictly_before_fall_steps=end,
        devices=summaries, sign_error_positive_controls=3,
        artifact=dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path)))


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E29_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E29 guarded monitor')
    evidence.unchanged(job)
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(evidence.MJCF))
    runtime = sim.build_loop_runtime(model, evidence.URDF, anchor_body_name='pelvis_link')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE, sim.DEFAULT_BRIDGE_CACHE, False)
    motions = []
    try:
        for row in job['saved_motions']:
            motions.append(audit_motion(job, row, sim, runtime, solver))
            atomic_json(OUT / 'progress.json', dict(completed_motions=len(motions), total=6, at_unix=time.time()))
            print(json.dumps(dict(sequence=row['sequence'], completed_motions=len(motions))), flush=True)
    finally:
        solver.close()
    evidence.unchanged(job)
    counts = {key:sum(device['counts'][key] for motion in motions for device in motion['devices'])
              for key in motions[0]['devices'][0]['counts']}
    status = ('REJECTED' if counts['matrix_mismatch_stable_states'] or counts['velocity_mismatch_stable_states']
              else 'INCOMPLETE_NUMERICAL_STABILITY' if counts['unstable_pre_fall_states'] else 'PASS')
    atomic_json(OUT / 'result.json', dict(motions=motions, counts=counts, differential_contract=status,
        sign_error_positive_controls=18, new_policy_rollouts=0, optimizer_steps=0,
        simulator_steps=0, policy_weight_loads=0,
        conclusion_scope='LOCAL_SAVED_STATE_CONVERSION_CONTRACT_NOT_ACTUAL_TORQUE_OR_FALL_CAUSALITY'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E29')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_differential_audit', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', YUANQI_E29_MONITOR_PID=str(os.getpid())),
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
        receipt = dict(result, status='VERIFIED_SAVED_DIFFERENTIAL_AUDIT' if state['status'] == 'COMPLETE' else 'STOPPED',
            experiment=job['experiment'], state=state, health=health,
            code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
            bridge_path=job['bridge_path'], bridge_sha256=job['bridge_sha256'],
            evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
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
