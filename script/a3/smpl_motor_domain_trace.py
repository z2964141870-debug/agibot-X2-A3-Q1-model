"""E31 order saved parallel-motor range excursions against tracking and tilt."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import smpl_joint_audit as evidence
from script.a3 import smpl_solver_domain as parent_stage
from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command

ROOT = evidence.ROOT
OUT = ROOT / 'data/experiments/a3_zhy_smpl_motor_trace_20261013_E31'
LOG = ROOT / 'logs/a3_zhy_smpl_motor_trace_20261013_E31'
MANIFEST = ROOT / 'data/manifests/a3_zhy_smpl_motor_trace_20261013_E31.json'
CODE = ('script/a3/smpl_motor_domain_trace.py',) + parent_stage.CODE


def read(path):
    return json.loads(path.read_text())


def register():
    if OUT.exists():
        raise FileExistsError('Do not reset or repeat E31')
    value = guard.preflight()
    parent = read(parent_stage.OUT / 'verification.json')
    if (parent['status'] != 'VERIFIED_SOLVER_BOUNDARY_SCAN'
            or parent['health']['boot_ids'] != [value['boot_id']]):
        raise ValueError('Requires verified same-boot E30')
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    files = {**parent['input_sha256'],
        str((parent_stage.OUT / 'verification.json').relative_to(ROOT)):sha256(parent_stage.OUT / 'verification.json')}
    for name, identity in files.items():
        if sha256(ROOT / name) != identity:
            raise ValueError('Parent evidence changed: ' + name)
    job = dict(experiment='A3_ZHY_SMPL_MOTOR_TRACE_20261013_E31', registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'], boot_id=value['boot_id'],
        parent_experiment=parent['experiment'],
        saved_motions=read(evidence.PRIOR / 'independent_verification.json')['motions'],
        main_variable='SIX_SAVED_MOTOR_POSITION_RANGES_EVENT_ORDERING',
        causal_intervention=False, expected_frames=499, focus_window_s=[4.4, 4.9],
        range_epsilon_rad=1e-6, nominal_sustained_frames=5, tilt_degrees=[15,30,60],
        joint_frame_rmse_rad=.2, left_knee_error_rad=.35,
        policy_weight_loads_allowed=0, policy_rollouts_allowed=0, optimizer_steps_allowed=0,
        simulator_steps_allowed=0, cuda_allowed=False, bridge_rebuild_allowed=False,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False, repeat_allowed=False,
        state_time_convention='POST_STEP_STATE_TIME=(POLICY_TICK+1)/50',
        bridge_path=parent['bridge_path'], bridge_sha256=parent['bridge_sha256'],
        code_sha256={name:sha256(ROOT / name) for name in CODE}, input_sha256=files,
        baseline_oom_kill=value['container_memory']['oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(), backup_status='LOCAL_ONLY')
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    atomic_json(OUT / 'registration.json', job)
    atomic_json(OUT / 'state.json', dict(status='READY'))
    append_record(LOG / 'health.jsonl', value)
    print(json.dumps(dict(status='READY', experiment=job['experiment'])))


def audit_motion(job, row, sim, model, runtime):
    import numpy as np
    name = Path(row['motion']).stem
    with np.load(evidence.PRIOR / 'evaluation' / (name + '.contact.npz'), allow_pickle=False) as saved:
        qpos, ticks = saved['qpos'], saved['policy_tick']
    trace = read(evidence.PRIOR / 'evaluation' / (name + '.timeseries.json'))['motions'][0]
    metric = read(evidence.PRIOR / 'evaluation' / (name + '.metrics.json'))['motions'][0]
    n = job['expected_frames']
    if qpos.shape != (n,model.nq) or not np.array_equal(ticks,np.arange(n)) or not np.isfinite(qpos).all():
        raise ValueError('Invalid saved states')
    end = metric['fall_tick'] if metric['fall_tick'] is not None else n
    motors = sim.LEFT_ANKLE_MOTORS + sim.RIGHT_ANKLE_MOTORS + sim.WAIST_MOTORS
    ids = [sim.mujoco.mj_name2id(model,sim.mujoco.mjtObj.mjOBJ_JOINT,m) for m in motors]
    if min(ids) < 0 or not np.all(model.jnt_limited[ids]):
        raise ValueError('Missing motor ranges')
    limits = model.jnt_range[ids]
    values = qpos[:,[runtime.motor_qpos_addr[m] for m in motors]]
    excess = np.maximum(np.maximum(limits[:,0] - values,values - limits[:,1]),0.)
    outside = excess > job['range_epsilon_rad']
    focus = (ticks / 50. >= job['focus_window_s'][0]) & (ticks / 50. <= job['focus_window_s'][1]) & (ticks < end)
    tracking = np.asarray(trace['q_state_29']) - np.asarray(trace['reference_q_29'])
    rmse = np.sqrt(np.mean(tracking ** 2,axis=1))
    tilt = np.maximum(np.abs(trace['root_roll_deg']),np.abs(trace['root_pitch_deg']))
    left_knee = runtime.mapping.il_joint_names.index('left_knee_joint')
    for array in (values,excess,tracking,rmse,tilt):
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite motor trace evidence')
    first_joint = evidence.first_run(rmse[:end] > job['joint_frame_rmse_rad'],1)
    event_tick = first_joint['start_tick'] if first_joint else None
    rows = []
    for i,motor in enumerate(motors):
        rows.append(dict(motor=motor,limit_rad=limits[i].tolist(),
            first_outside=evidence.first_run(outside[:end,i],1),
            first_sustained_outside=evidence.first_run(outside[:end,i],job['nominal_sustained_frames']),
            outside_samples_before_fall=int(outside[:end,i].sum()),
            outside_samples_focus=int(outside[focus,i].sum()),
            outside_samples_before_first_joint_rmse_event=int(outside[:event_tick,i].sum()) if event_tick is not None else None,
            max_excess_before_fall_rad=float(excess[:end,i].max()),
            actual_min_max_before_fall_rad=[float(values[:end,i].min()),float(values[:end,i].max())]))
    any_outside = np.any(outside,axis=1)
    event_ticks = {t for r in rows if r['first_outside'] for t in (r['first_outside']['start_tick'],)}
    event_ticks.update(e['start_tick'] for e in [first_joint]+[
        evidence.first_run(tilt[:end] > angle,1) for angle in job['tilt_degrees']] if e)
    event_ticks.update((233,234,251,262,270,272,281) if metric['fall_tick'] is not None else ())
    return dict(sequence=row['sequence'],actual_saved_frames=n,strictly_before_fall_steps=end,
        first_fall_label_s=metric['fall_time_s'],motors=rows,
        first_any_motor_outside=evidence.first_run(any_outside[:end],1),
        first_joint_rmse_event=first_joint,
        first_left_knee_sustained_error=evidence.first_run(np.abs(tracking[:end,left_knee]) > job['left_knee_error_rad'],5),
        first_tilt_events={str(angle):evidence.first_run(tilt > angle,1) for angle in job['tilt_degrees']},
        event_snapshots=[dict(tick=t,label_s=t/50.,post_step_state_s=(t+1)/50.,
            max_abs_root_tilt_deg=float(tilt[t]),joint_rmse_rad=float(rmse[t]),
            motor_values_rad=values[t].tolist(),motor_limit_excess_rad=excess[t].tolist())
            for t in sorted(event_ticks) if t < end])


def worker():
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if (state['status'] != 'RUNNING' or state['monitor_pid'] != os.getppid()
            or os.environ.get('YUANQI_E31_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Worker requires E31 guarded monitor')
    evidence.unchanged(job)
    import numpy as np
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    model = sim.mujoco.MjModel.from_xml_path(str(evidence.MJCF))
    runtime = sim.build_loop_runtime(model,evidence.URDF,anchor_body_name='pelvis_link')
    # Verify range direction and tolerance with independent scalar boundary cases.
    probe = np.array([-1.6,-1.55,0.,1.55,1.6])
    outside = np.maximum(np.maximum(-1.55-probe,probe-1.55),0.) > job['range_epsilon_rad']
    np.testing.assert_array_equal(outside,np.array([True,False,False,False,True]))
    motions = [audit_motion(job,row,sim,model,runtime) for row in job['saved_motions']]
    evidence.unchanged(job)
    atomic_json(OUT / 'result.json',dict(motions=motions,boundary_positive_controls=5,
        new_policy_rollouts=0,optimizer_steps=0,simulator_steps=0,policy_weight_loads=0,
        model_range_scope='COMPILED_MODEL_MOTOR_LIMITS_NOT_ALL_SOLVER_DOMAINS_OR_HARD_LIMIT_ENFORCEMENT',
        sample_scope='50HZ_POST_STEP_SAMPLES_NOT_FULL_1MS_SUBSTEP_COVERAGE',
        conclusion_scope='EVENT_ORDERING_ONLY_NOT_CAUSALITY_OR_POLICY_IMPROVEMENT'))


def run():
    guard.OUT, guard.LOG = OUT, LOG
    with (OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        state,job = read(OUT / 'state.json'),read(OUT / 'registration.json')
        if state['status'] != 'READY':
            raise RuntimeError('No repeat or recovery of E31')
        guard.preflight()
        evidence.unchanged(job)
        state.update(status='RUNNING',monitor_pid=os.getpid(),started_at=time.time())
        atomic_json(OUT / 'state.json',state)
        try:
            with (LOG / 'console.log').open('w') as stream:
                child = subprocess.Popen([str(ROOT / 'data/environments/a3-sonic/bin/python'),'-m',
                    'script.a3.smpl_motor_domain_trace','worker'],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1',YUANQI_E31_MONITOR_PID=str(os.getpid())),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                code,reason = guard.guarded_monitor(child,LOG / 'worker',job)
            state.update(status='COMPLETE' if code == 0 and reason is None else 'STOPPED',returncode=code,reason=reason)
        except BaseException as exc:
            state.update(status='STOPPED',reason=repr(exc))
            raise
        finally:
            state['ended_at'] = time.time()
            atomic_json(OUT / 'state.json',state)
        from script.a3.smpl_frozen_followup import health_summary
        health = health_summary(LOG)
        if (health['boot_ids'] != [job['boot_id']] or health['critical_kernel_errors']
                or health['telemetry_errors'] or health['oom_counts'] != [job['baseline_oom_kill']]):
            state.update(status='STOPPED',reason='HEALTH_VERIFICATION_FAILED')
            atomic_json(OUT / 'state.json',state)
        result = read(OUT / 'result.json') if state['status'] == 'COMPLETE' else {}
        receipt = dict(result,status='VERIFIED_SAVED_MOTOR_RANGE_TRACE' if state['status'] == 'COMPLETE' else 'STOPPED',
            experiment=job['experiment'],state=state,health=health,code_sha256=job['code_sha256'],
            input_sha256=job['input_sha256'],evidence={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha256(p))
                for p in (OUT / 'registration.json',LOG / 'console.log')},backup_status='LOCAL_ONLY')
        if state['status'] == 'COMPLETE':
            receipt['evidence'][str((OUT / 'result.json').relative_to(ROOT))] = dict(
                bytes=(OUT / 'result.json').stat().st_size,sha256=sha256(OUT / 'result.json'))
            atomic_json(OUT / 'verification.json',receipt)
        atomic_json(MANIFEST,receipt)
        print(json.dumps(state))
        return 0 if state['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','run','worker'))
    result = {'register':register,'run':run,'worker':worker}[parser.parse_args().action]()
    raise SystemExit(result if isinstance(result,int) else 0)
