"""E30 characterize right-ankle FK at the saved motor limit boundary."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_differential_audit as prior
from script.a3 import smpl_joint_audit as evidence
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = evidence.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_solver_domain_20261013_E30'
LOG = ROOT / 'logs/a3_zhy_smpl_solver_domain_20261013_E30'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_solver_domain_20261013_E30.json'
CODE = ('script/a3/smpl_solver_domain.py',) + prior.CODE


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E30')
    value = guard.preflight()
    parent = read(prior.OUT / 'verification.json')
    if (parent['status'] != 'VERIFIED_SAVED_DIFFERENTIAL_AUDIT'
            or parent['differential_contract'] != 'REJECTED'
            or parent['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot E29 rejection')
    motion = next(m for m in parent['motions'] if m['sequence'] == 'G_20251229_zzx/zzx_111505_walk')
    right = next(d for d in motion['devices'] if d['device'] == 'right_ankle')
    cases = sorted([r for r in right['worst_stable_states'] if r['tick'] in (272, 281)], key=lambda r:r['tick'])
    if [r['tick'] for r in cases] != [272, 281]:
        raise ValueError('Required boundary evidence missing')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = {**parent['input_sha256'],
        str((prior.OUT / 'verification.json').relative_to(ROOT)):sha256(prior.OUT / 'verification.json')}
    for name, identity in files.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Parent evidence changed: ' + name)
    bridge = evidence.cached_bridge()
    job = dict(experiment='A3_ZHY_SMPL_SOLVER_DOMAIN_20261013_E30', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=parent['experiment'], cases=cases,
        main_variable='RIGHT_ANKLE_UP_MOTOR_INPUT_AROUND_FIXED_MODEL_LIMIT',
        causal_intervention=False, upper_motor_limit_rad=1.55,
        upper_offsets_rad=[-.02, -.005, -.001, -.0001, 0., .0001, .001, .005, .02, .08],
        platform_atol_rad=1e-9, below_limit_response_min_rad=1e-4,
        in_domain_roundtrip_atol_rad=.002, fk_revisit_atol_rad=1e-8,
        policy_weight_loads_allowed=0, policy_rollouts_allowed=0, optimizer_steps_allowed=0,
        simulator_steps_allowed=0, cuda_allowed=False, bridge_rebuild_allowed=False,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        bridge_path=str(bridge), bridge_sha256=sha256(bridge),
        code_sha256={name:sha256(ROOT / name) for name in CODE}, input_sha256=files,
        baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'], saved_contexts=2)))


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E30_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E30 guarded monitor')
    evidence.unchanged(job)
    import numpy as np
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(evidence.MJCF))
    motors = sim.RIGHT_ANKLE_MOTORS
    ids = [sim.mujoco.mj_name2id(model, sim.mujoco.mjtObj.mjOBJ_JOINT, m) for m in motors]
    if min(ids) < 0 or not np.all(model.jnt_limited[ids]):
        raise ValueError('Missing limited motor joints')
    limits = model.jnt_range[ids]
    upper = job['upper_motor_limit_rad']
    if not np.array_equal(limits, np.array([[-1.55, upper], [-1.55, upper]])):
        raise ValueError('Registered motor limits changed')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE, sim.DEFAULT_BRIDGE_CACHE, False)
    results = []
    try:
        for case in job['cases']:
            original = np.asarray(case['motor_q_rad'])
            anchor_motor = np.array([upper, original[1]])
            anchor = solver.ankle_fk(1, anchor_motor)
            rows = []
            for up in [upper + delta for delta in job['upper_offsets_rad']] + [float(original[0])]:
                motor = np.array([up, original[1]])
                logical = solver.ankle_fk(1, motor)
                inverse = solver.ankle_ik(1, logical)
                if not np.isfinite(logical).all() or not np.isfinite(inverse).all():
                    raise ValueError('Nonfinite solver boundary output')
                rows.append(dict(up_motor_rad=float(up), inside_model_limit=bool(up <= upper),
                    offset_from_upper_rad=float(up - upper), fk_pitch_roll_rad=logical.tolist(),
                    fk_max_difference_from_upper_rad=float(np.abs(logical - anchor).max()),
                    ik_of_fk_motor_rad=inverse.tolist(),
                    inverse_minus_input_rad=(inverse - motor).tolist()))
            outside = [r for r in rows if not r['inside_model_limit']]
            inside = [r for r in rows if r['inside_model_limit']]
            plateau_error = max(r['fk_max_difference_from_upper_rad'] for r in outside)
            below_response = rows[0]['fk_max_difference_from_upper_rad']
            revisit = float(np.abs(solver.ankle_fk(1, anchor_motor) - anchor).max())
            in_domain_error = max(max(abs(v) for v in r['inverse_minus_input_rad']) for r in inside)
            results.append(dict(source_tick=case['tick'], source_label_s=case['label_s'],
                original_motor_q_rad=original.tolist(), original_dfk_velocity_rad_s=case['analytic_velocity_rad_s'],
                original_fd_velocity_rad_s=case['numerical_velocity_rad_s'], anchor_motor_rad=anchor_motor.tolist(),
                anchor_fk_pitch_roll_rad=anchor.tolist(), scanned_motor_inputs=len(rows), rows=rows,
                outside_platform_error_max_rad=plateau_error, below_limit_response_rad=below_response,
                fk_revisit_error_rad=revisit, in_domain_roundtrip_error_max_rad=in_domain_error,
                platform_matches_model_upper_limit=bool(plateau_error <= job['platform_atol_rad']
                    and below_response > job['below_limit_response_min_rad']
                    and revisit <= job['fk_revisit_atol_rad']),
                in_domain_roundtrip_pass=bool(in_domain_error <= job['in_domain_roundtrip_atol_rad'])))
    finally:
        solver.close()
    evidence.unchanged(job)
    atomic_json(OUT / 'result.json', dict(cases=results, motor_order=list(motors), model_limits_rad=limits.tolist(),
        scan_inputs=sum(c['scanned_motor_inputs'] for c in results),
        boundary_behavior='CONFIRMED_MODEL_LIMIT_FK_PLATFORM' if all(c['platform_matches_model_upper_limit'] for c in results)
            else 'NO_CONFIRMED_MODEL_LIMIT_PLATFORM',
        in_domain_roundtrip_contract='PASS' if all(c['in_domain_roundtrip_pass'] for c in results) else 'REJECTED',
        new_policy_rollouts=0, optimizer_steps=0, simulator_steps=0, policy_weight_loads=0,
        conclusion_scope='SPECIFIED_MOTOR_INPUT_DOMAIN_ONLY_NOT_INITIAL_FALL_CAUSE_OR_POLICY_IMPROVEMENT'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E30')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_solver_domain', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', YUANQI_E30_MONITOR_PID=str(os.getpid())),
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
        receipt = dict(result, status='VERIFIED_SOLVER_BOUNDARY_SCAN' if state['status'] == 'COMPLETE' else 'STOPPED',
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
