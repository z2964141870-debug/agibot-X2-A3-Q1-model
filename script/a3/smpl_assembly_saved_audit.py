"""E35 audit saved E34 action clipping and matched dynamic rejection."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_assembly_recovery as parent_stage
from script.a3 import smpl_joint_audit as evidence
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.a3.smpl_frozen_followup import health_summary

ROOT, PRIOR = evidence.ROOT, parent_stage.OUT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_assembly_audit_20261013_E35'
LOG = ROOT / 'logs/a3_zhy_smpl_assembly_audit_20261013_E35'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_assembly_audit_20261013_E35.json'


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E35')
    value, parent = guard.preflight(), read(PRIOR / 'registration.json')
    state = read(PRIOR / 'state.json')
    console = ROOT / 'logs/a3_zhy_smpl_assembly_20261013_E34/console.log'
    if (state['status'] != 'STOPPED' or 'AssertionError' not in console.read_text()
            or "captured['actions']" not in console.read_text()):
        raise ValueError('Unexpected E34 failure')
    previous_health = health_summary(ROOT / 'logs/a3_zhy_smpl_assembly_20261013_E34')
    if (previous_health['boot_ids'] != [value['boot_id']] or previous_health['critical_kernel_errors']
            or previous_health['telemetry_errors'] or previous_health['oom_counts'] != [parent['baseline_oom_kill']]):
        raise RuntimeError('Requires healthy same-boot saved E34')
    evidence.unchanged(parent)
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    paths = [PRIOR / 'registration.json', PRIOR / 'state.json', PRIOR / 'assembly_contract.json', console]
    for row in parent['selection']:
        stem = Path(row['filename']).stem
        paths += [PRIOR / 'evaluation' / (stem + '.' + suffix) for suffix in
                  ('metrics.json', 'timeseries.json', 'inputs.npz', 'contact.npz')]
        paths.append(PRIOR / 'entry' / (stem + '.json'))
    inputs = dict(parent['input_sha256'])
    inputs.update({str(p.relative_to(ROOT)):sha256(p) for p in paths})
    job = dict(parent, experiment='A3_ZHY_SMPL_ASSEMBLY_AUDIT_20261013_E35', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], parent_experiment=parent['experiment'],
        main_variable='PRECLIP_CAPTURE_VS_POSTCLIP_RECORD_SEMANTICS_AND_SAVED_COMPARISON',
        legacy_e34_exact_raw_comparison_preserved='FAILED_NOT_REWRITTEN', action_clip=20.,
        clipped_action_atol=0., target_atol=1e-10, foot_reconstruction_atol_m=1e-10,
        policy_rollouts_allowed=0, policy_weight_loads_allowed=0, optimizer_steps_allowed=0,
        simulator_steps_allowed=0, assembly_recomputations_allowed=0, cuda_allowed=False,
        code_sha256={**parent['code_sha256'], 'script/a3/smpl_assembly_saved_audit.py':sha256(Path(__file__))},
        input_sha256=inputs, kernel_baseline=kernel['stdout'].splitlines(),
        baseline_oom_kill=value['container_memory']['oom_kill'])
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E35_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E35 guarded monitor')
    evidence.unchanged(job)
    import numpy as np
    from script.a3.reference_contract import load_sim
    from script.a3.smpl_frozen_entry import foot_geometry, foot_minima
    sim = load_sim()
    if sim.ACTION_CLIP != job['action_clip']:
        raise ValueError('Official clipping constant changed')
    np.testing.assert_array_equal(np.clip([-21., -20., 0., 20., 21.], -20., 20.), [-20., -20., 0., 20., 20.])
    model = sim.mujoco.MjModel.from_xml_path(str(evidence.MJCF))
    runtime = sim.build_loop_runtime(model, evidence.URDF, anchor_body_name='pelvis_link')
    data, feet = sim.mujoco.MjData(model), foot_geometry(model, sim.mujoco)
    rows = []
    for row in job['selection']:
        stem = Path(row['filename']).stem
        trace = read(PRIOR / 'evaluation' / (stem + '.timeseries.json'))['motions'][0]
        old_trace = read(evidence.PRIOR / 'evaluation' / (stem + '.timeseries.json'))['motions'][0]
        metric = read(PRIOR / 'evaluation' / (stem + '.metrics.json'))['motions'][0]
        old_metric = read(evidence.PRIOR / 'evaluation' / (stem + '.metrics.json'))['motions'][0]
        with np.load(PRIOR / 'evaluation' / (stem + '.inputs.npz'), allow_pickle=False) as captured:
            observations, raw = captured['observations'], captured['actions']
        with np.load(evidence.PRIOR / 'evaluation' / (stem + '.inputs.npz'), allow_pickle=False) as old:
            initial_error = float(np.max(np.abs(observations[0] - old['observations'][0])))
        recorded = np.asarray(trace['raw_action_29'])
        clipped = np.clip(raw.astype(np.float64), -sim.ACTION_CLIP, sim.ACTION_CLIP)
        target = runtime.mapping.default_q_il + clipped * runtime.mapping.action_scale_il
        if raw.shape != (499, 29) or observations.shape != (499, 1570):
            raise ValueError('Saved policy ABI rejected')
        np.testing.assert_allclose(clipped, recorded, atol=job['clipped_action_atol'], rtol=0)
        np.testing.assert_allclose(target, trace['q_des_29'], atol=job['target_atol'], rtol=0)
        with np.load(PRIOR / 'evaluation' / (stem + '.contact.npz'), allow_pickle=False) as captured:
            qpos, qvel, ticks = captured['qpos'], captured['qvel'], captured['policy_tick']
            minima = captured['foot_geometry_min_z_m']
        if qpos.shape != (499, model.nq) or qvel.shape != (499, model.nv) or not np.array_equal(ticks, np.arange(499)):
            raise ValueError('Saved full-state coverage rejected')
        if not all(np.isfinite(x).all() for x in (qpos, qvel, raw, observations, recorded, target, minima)):
            raise ValueError('Nonfinite saved data')
        reconstructed = []
        for q in qpos:
            data.qpos[:] = q
            sim.mujoco.mj_fwdPosition(model, data)
            reconstructed.append(foot_minima(data, feet))
        foot_error = float(np.max(np.abs(np.asarray(reconstructed) - minima)))
        np.testing.assert_allclose(reconstructed, minima, atol=job['foot_reconstruction_atol_m'], rtol=0)
        np.testing.assert_array_equal(qpos[:, :3], trace['sim_root_pos_w'])
        height, roll, pitch = [np.asarray(trace[key]) for key in ('root_height', 'root_roll_deg', 'root_pitch_deg')]
        falling = np.flatnonzero((height < job['fall_height_m']) | (np.abs(roll) > job['fall_abs_roll_or_pitch_deg']) |
                                 (np.abs(pitch) > job['fall_abs_roll_or_pitch_deg']))
        fall = int(falling[0]) if len(falling) else None
        if metric['num_policy_steps'] != 499 or metric['fall_tick'] != fall or metric['fall'] != bool(len(falling)):
            raise ValueError('Independent dynamic outcome mismatch')
        errors = [np.asarray(t['q_state_29']) - np.asarray(t['reference_q_29']) for t in (old_trace, trace)]
        rms = [float(np.sqrt(np.mean(error ** 2))) for error in errors]
        coordinates = np.argwhere(np.abs(raw) > sim.ACTION_CLIP)
        entry = read(PRIOR / 'entry' / (stem + '.json'))
        if not entry['passive_reference_loaded'] or entry['reference_frames'] != 499:
            raise ValueError('Missing passive-reference entry evidence')
        item = dict(sequence=row['sequence'], steps=499, initial_policy_input_max_difference=initial_error,
            initial_policy_input_gate_passed=initial_error <= job['policy_initial_input_atol'],
            clipping_elements=int(len(coordinates)), clipping_frames=int(len(np.unique(coordinates[:, 0]))) if len(coordinates) else 0,
            clipping_examples=[dict(tick=int(t), joint=runtime.mapping.il_joint_names[j],
                captured_preclip=float(raw[t, j]), recorded_postclip=float(recorded[t, j])) for t, j in coordinates[:8]],
            clipped_action_max_difference=float(np.max(np.abs(clipped - recorded))),
            target_max_difference=float(np.max(np.abs(target - trace['q_des_29']))), foot_reconstruction_max_m=foot_error,
            baseline_rmse_rad=rms[0], candidate_rmse_rad=rms[1], rmse_ratio=rms[1] / rms[0],
            baseline_fall=old_metric['fall'], candidate_fall=metric['fall'],
            baseline_fall_label_s=old_metric['fall_time_s'], candidate_fall_label_s=metric['fall_time_s'],
            retention_passed=bool(not metric['fall'] and rms[1] <= rms[0] * job['retention_rmse_ratio_max']) if not old_metric['fall'] else None)
        rows.append(item)
        atomic_json(OUT / 'partial.json', dict(motions=rows))
    evidence.unchanged(job)
    means = {key:sum(r[key] for r in rows) / len(rows) for key in ('baseline_rmse_rad', 'candidate_rmse_rad')}
    gate = all(not r['candidate_fall'] for r in rows) and all(r['retention_passed'] for r in rows if not r['baseline_fall'])
    gate = bool(gate and means['candidate_rmse_rad'] < means['baseline_rmse_rad']
                and all(r['initial_policy_input_gate_passed'] for r in rows))
    atomic_json(OUT / 'result.json', dict(motions=rows, **means, development_gate_passed=gate,
        capture_contract_verified=True, captured_steps=2994, clipping_boundary_controls=5,
        new_policy_rollouts=0, policy_weight_loads=0, optimizer_steps=0, simulator_steps=0,
        assembly_recomputations=0, candidate_zero_fall_clips=sum(not r['candidate_fall'] for r in rows),
        e34_legacy_gate_status='STOPPED_EXACT_UNCLIPPED_EQUALS_CLIPPED_ASSERTION_PRESERVED',
        fresh_final_test=False, policy_weight_improvement_proven=False, deploy_allowed=False))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E35')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING', monitor_pid=os.getpid(), started_at=time.time())
        atomic_json(OUT / 'state.json', state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'), '-m',
                    'script.a3.smpl_assembly_saved_audit', 'worker'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        YUANQI_E35_MONITOR_PID=str(os.getpid())), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code, reason = guard.guarded_monitor(child, LOG / 'worker', job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED', returncode=code, reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json', state)
        health = health_summary(LOG)
        if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
                or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
            state.update(status='STOPPED', reason='HEALTH_VERIFICATION_FAILED')
            atomic_json(OUT / 'state.json', state)
        result = read(OUT / 'result.json') if state['status'] == 'COMPLETE' else {}
        receipt = dict(result, status='VERIFIED_CAPTURE_SEMANTICS_AND_DYNAMIC_REJECTION' if state['status'] == 'COMPLETE' else 'STOPPED',
            experiment=job['experiment'], state=state, health=health, code_sha256=job['code_sha256'], input_sha256=job['input_sha256'],
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
