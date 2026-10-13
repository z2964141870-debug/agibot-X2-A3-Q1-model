"""E33 passive reference assembly, then the matched frozen035 comparison."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_support_demand as parent_stage
from script.a3 import smpl_frozen_probe as shared
from script.a3 import smpl_joint_audit as evidence
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = evidence.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_assembly_20261013_E33'
LOG = ROOT / 'logs/a3_zhy_smpl_assembly_20261013_E33'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_assembly_20261013_E33.json'
CODE = ('script/a3/smpl_closed_reference.py', 'script/a3/smpl_closed_reference_entry.py',
    'script/a3/smpl_closed_reference_trial.py', 'script/a3/frozen_evaluation_launch.py',
    'script/a3/smpl_frozen_probe.py', 'script/a3/verify_zhy_search.py', 'script/a3/evaluate_mujoco.py',
    'script/a3/sim_entry.py', 'script/a3/evaluation_common.py') + parent_stage.CODE


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E33')
    value = guard.preflight()
    parent = read(parent_stage.OUT / 'verification.json')
    previous = read(parent_stage.OUT / 'registration.json')
    if (parent['status'] != 'VERIFIED_CENTROIDAL_DEMAND_ESTIMATES'
            or parent['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot E32')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    old = read(evidence.PRIOR / 'registration.json')
    files = dict(parent['input_sha256'])
    files.update({str(p.relative_to(ROOT)):sha256(p) for p in
        (parent_stage.OUT / 'registration.json', parent_stage.OUT / 'verification.json',
         evidence.PRIOR / 'reference_contract.json')})
    job = dict(old, experiment='A3_ZHY_SMPL_ASSEMBLY_20261013_E33', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=parent['experiment'], main_variable='COMPLETE_SIX_PASSIVE_ROD_BALL_STATES_ONLY',
        endpoint_norm_gate_m=.002, external_qpos_atol=0., controlled_qvel_atol=0.,
        foot_geometry_unchanged_atol_m=1e-12, policy_initial_input_atol=0.,
        reference_foot_budget_m=.005, retention_rmse_ratio_max=1.02,
        improvement_gate='ALL_6_ZERO_FALLS_NO_NEW_FALLS_EACH_OLD_NONFALL_RMSE_WITHIN_2PCT_MEAN_RMSE_LOWER',
        policy_rollouts_allowed=6, optimizer_steps_allowed=0, bridge_rebuild_allowed=False,
        runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        bridge_path=previous['bridge_path'], bridge_sha256=previous['bridge_sha256'],
        code_sha256={name:sha256(ROOT / name) for name in dict.fromkeys(CODE)}, input_sha256=files,
        kernel_baseline=kernel['stdout'].splitlines(), baseline_oom_kill=value['container_memory']['oom_kill'],
        backup_status='LOCAL_ONLY')
    evidence.unchanged(job)
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (OUT / 'assembled').mkdir()
    (OUT / 'references').mkdir()
    for row in job['selection']:
        path = ROOT / row['source'] / 'reference.csv'
        (OUT / 'references' / row['filename']).symlink_to(os.path.relpath(path, OUT / 'references'))
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def assembly(job):
    import numpy as np
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_frozen_entry import foot_geometry, foot_minima
    from script.a3.smpl_closed_reference import controls, close_reference
    sim = load_sim()
    tests = controls(sim.mujoco)
    model = sim.mujoco.MjModel.from_xml_path(str(evidence.MJCF))
    runtime = sim.build_loop_runtime(model, evidence.URDF, anchor_body_name='pelvis_link')
    solver = sim.A3LoopSolverBridge(sim.DEFAULT_SOLVER_LIB, sim.DEFAULT_SOLVER_INCLUDE, sim.DEFAULT_BRIDGE_CACHE, False)
    data, feet = sim.mujoco.MjData(model), foot_geometry(model, sim.mujoco)
    collision = np.flatnonzero((model.geom_contype | model.geom_conaffinity) > 0)
    active_v = list(range(6)) + [runtime.mapping.qvel_addr_by_joint[name]
        for name in runtime.mapping.mj_full_joint_names] + list(runtime.motor_qvel_addr.values())
    rows = []
    try:
        for row in job['selection']:
            stem = Path(row['filename']).stem
            reference = sim.load_loop_motion_reference(model, runtime, solver,
                ROOT / row['source'] / 'reference.csv', csv_source_fps=30., csv_frame_stride=1, announce=False)
            qpos, qvel, receipt = close_reference(model, sim.mujoco, reference.qpos, .02)
            if receipt['ball_joint_count'] != 6:
                raise ValueError('Expected exactly six passive rods')
            controlled_error = float(np.max(np.abs(qvel[:, active_v] - reference.qvel[:, active_v])))
            geometry_error, floor = 0., float('inf')
            for original, completed in zip(reference.qpos, qpos):
                data.qpos[:] = original
                sim.mujoco.mj_fwdPosition(model, data)
                positions, rotations = data.geom_xpos[collision].copy(), data.geom_xmat[collision].copy()
                data.qpos[:] = completed
                sim.mujoco.mj_fwdPosition(model, data)
                geometry_error = max(geometry_error, float(np.max(np.abs(data.geom_xpos[collision] - positions))),
                                     float(np.max(np.abs(data.geom_xmat[collision] - rotations))))
                floor = min(floor, min(foot_minima(data, feet)))
            maximum = float(receipt['after_endpoint_error_m'].max())
            path = OUT / 'assembled' / (stem + '.npz')
            np.savez_compressed(path, original_qpos=reference.qpos, qpos=qpos, qvel=qvel,
                before_endpoint_error_m=receipt['before_endpoint_error_m'],
                after_endpoint_error_m=receipt['after_endpoint_error_m'])
            item = dict(sequence=row['sequence'], stem=stem, reference_frames=len(qpos),
                endpoint_before_max_m=float(receipt['before_endpoint_error_m'].max()),
                endpoint_after_max_m=maximum, endpoint_after_p95_m=float(np.percentile(receipt['after_endpoint_error_m'], 95)),
                controlled_qvel_max_difference=controlled_error, collision_geometry_max_difference=geometry_error,
                actual_foot_min_z_m=floor, passive_ball_joints=receipt['ball_joint_count'],
                twist_scope=receipt['twist_scope'], artifact=dict(path=str(path.relative_to(ROOT)),
                    bytes=path.stat().st_size, sha256=sha256(path)),
                passed=bool(maximum <= job['endpoint_norm_gate_m']
                    and controlled_error <= job['controlled_qvel_atol']
                    and geometry_error <= job['foot_geometry_unchanged_atol_m']
                    and floor >= -job['reference_foot_budget_m']))
            rows.append(item)
            atomic_json(OUT / 'assembly_contract.json', dict(status='CHECKING', motions=rows, controls=tests))
    finally:
        solver.close()
    value = dict(status='VERIFIED_PASSIVE_REFERENCE_ASSEMBLY' if all(r['passed'] for r in rows) else 'ASSEMBLY_QUALITY_REJECTED',
                 motions=rows, controls=tests, external_qpos_preserved_exact=True)
    atomic_json(OUT / 'assembly_contract.json', value)
    return value


def evaluate(job):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    original = evaluation.subprocess

    def spawn(argv, **kwargs):
        argv = ['script.a3.smpl_closed_reference_entry' if item == 'script.a3.smpl_frozen_entry' else item for item in argv]
        return original.Popen(argv, **kwargs)

    evaluation.preflight = guard.preflight
    evaluation.monitor_child = lambda child, directory, config:(child.wait(), None)
    evaluation.subprocess = LaunchProxy(original, spawn)
    try:
        code = evaluation.evaluate_explicit(arguments(OUT / 'evaluation', LOG / 'evaluation', shared.OFFICIAL, OUT / 'references'))
    finally:
        evaluation.subprocess = original
    if code:
        raise RuntimeError('New passive-assembly evaluation failed; no retry')
    shared.OUT = OUT
    atomic_json(OUT / 'reference_contract.json', read(evidence.PRIOR / 'reference_contract.json'))
    shared.verify_worker(job)


def compare(job):
    import numpy as np
    rows = []
    for row in job['selection']:
        stem = Path(row['filename']).stem
        original = read(evidence.PRIOR / 'evaluation' / (stem + '.timeseries.json'))['motions'][0]
        candidate = read(OUT / 'evaluation' / (stem + '.timeseries.json'))['motions'][0]
        old_metric = read(evidence.PRIOR / 'evaluation' / (stem + '.metrics.json'))['motions'][0]
        new_metric = read(OUT / 'evaluation' / (stem + '.metrics.json'))['motions'][0]
        with np.load(evidence.PRIOR / 'evaluation' / (stem + '.inputs.npz'), allow_pickle=False) as old:
            original_input = old['observations'][0].copy()
        with np.load(OUT / 'evaluation' / (stem + '.inputs.npz'), allow_pickle=False) as new:
            initial_error = float(np.max(np.abs(new['observations'][0] - original_input)))
        if initial_error > job['policy_initial_input_atol']:
            raise ValueError('Initial policy observation changed beyond the passive-state intervention')
        errors = [np.asarray(trace['q_state_29']) - np.asarray(trace['reference_q_29']) for trace in (original, candidate)]
        rms = [float(np.sqrt(np.mean(error ** 2))) for error in errors]
        entry = read(OUT / 'entry' / (stem + '.json'))
        if not entry['passive_reference_loaded'] or entry['reference_frames'] != 499:
            raise ValueError('No passive-reference entry receipt')
        rows.append(dict(sequence=row['sequence'], baseline_rmse_rad=rms[0], candidate_rmse_rad=rms[1],
            rmse_ratio=rms[1] / rms[0], baseline_fall=old_metric['fall'], candidate_fall=new_metric['fall'],
            baseline_fall_label_s=old_metric['fall_time_s'], candidate_fall_label_s=new_metric['fall_time_s'],
            initial_policy_input_max_difference=initial_error,
            retention_passed=bool(not new_metric['fall'] and rms[1] <= rms[0] * job['retention_rmse_ratio_max']) if not old_metric['fall'] else None))
    candidate_mean, baseline_mean = [sum(r[key] for r in rows) / len(rows)
        for key in ('candidate_rmse_rad', 'baseline_rmse_rad')]
    return dict(motions=rows, baseline_mean_rmse_rad=baseline_mean, candidate_mean_rmse_rad=candidate_mean,
        candidate_zero_fall_clips=sum(not r['candidate_fall'] for r in rows),
        development_gate_passed=bool(all(not r['candidate_fall'] for r in rows)
            and all(r['retention_passed'] for r in rows if not r['baseline_fall'])
            and candidate_mean < baseline_mean), optimizer_steps=0, policy_rollouts=6,
        fresh_final_test=False, policy_weight_improvement_proven=False, deploy_allowed=False)


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E33_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E33 guarded monitor')
    evidence.unchanged(job)
    contract = assembly(job)
    if contract['status'] == 'ASSEMBLY_QUALITY_REJECTED':
        atomic_json(OUT / 'result.json', dict(assembly=contract, policy_rollouts=0, optimizer_steps=0,
            status='VERIFIED_ASSEMBLY_QUALITY_REJECTION'))
        return
    evaluate(job)
    result = compare(job)
    evidence.unchanged(job)
    atomic_json(OUT / 'result.json', dict(assembly=contract, comparison=result,
        status='VERIFIED_PASSIVE_ASSEMBLY_FROZEN_COMPARISON'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E33')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_closed_reference_trial', 'worker'], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E33_MONITOR_PID=str(os.getpid()), YUANQI_E33_OUTPUT=str(OUT)),
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
        receipt = dict(result, status=result.get('status', 'STOPPED'), experiment=job['experiment'], state=state,
            health=health, code_sha256=job['code_sha256'], input_sha256=job['input_sha256'], backup_status='LOCAL_ONLY',
            evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in (OUT / 'registration.json', LOG / 'console.log')})
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
