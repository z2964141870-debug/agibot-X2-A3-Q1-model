"""Audit completed HP16 arrays at the actual pre/post-action-clip boundary."""

import ast
import json
import os
from pathlib import Path

from script.a3 import humanplus_head_effect as trial
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256
from script.project_tools.humanplus_baseline_audit import verify_trace

TARGET = trial.ROOT / 'data/manifests/a3_humanplus_saved_cloth_20261018_HP16_audit.json'


def action_clip(source):
    tree = ast.parse(source)
    assignment = [n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'ACTION_CLIP' for t in n.targets)]
    if len(assignment) != 1 or ast.literal_eval(assignment[0].value) != 20.0:
        raise ValueError('Pinned action clipping constant changed')
    step = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_step_policy']
    expected = ast.dump(ast.parse('raw_action_il = np.clip(raw_action_il, -ACTION_CLIP, ACTION_CLIP)').body[0],
                        include_attributes=False)
    if len(step) != 1 or sum(ast.dump(n, include_attributes=False) == expected for n in step[0].body) != 1:
        raise ValueError('Pinned control action clipping changed')
    return 20.0


def main():
    if TARGET.exists():
        raise FileExistsError('Saved-cloth artifact audit is single-use')
    plan = trial.checked()
    terminal = trial.read(trial.RESULT)
    console = (trial.LOG / 'cloth/console.log').read_text()
    if (terminal['status'] != 'STOPPED' or terminal['current_stage'] != 'cloth'
            or terminal['stages'] or 'np.testing.assert_array_equal' not in console
            or 'Arrays are not equal' not in console):
        raise ValueError('Only exact post-replay action-capture mismatch is eligible')
    previous = trial.read(trial.PLAN.with_name(trial.NAME + '_cloth_audit.json'))
    if previous['after_issues'] or previous['remaining_own_group_processes']:
        raise ValueError('Machine anomaly is not a data-boundary correction')
    health = guard.collect(trial.read(trial.OUT / 'baseline.json'))
    issues = guard.issues(health, startup=True, monitor_pid=os.getpid())
    if issues:
        raise RuntimeError('Saved-data audit resources unavailable: ' + ';'.join(issues))
    import numpy as np
    vendor = trial.VENDOR / 'gear_sonic/scripts/sim2sim_a3_mujoco.py'
    limit = action_clip(vendor.read_text())
    directory = trial.OUT / 'cloth'
    manifest = trial.read(directory / 'explicit_manifest.json')
    summary = trial.read(directory / 'explicit_summary.json')
    selected = [r for r in plan['motions'] if r['group'] == 'cloth']
    if (len(selected) != 6 or set(manifest['runs']) != {r['key'] for r in selected}
            or any(r['status'] != 'passed' or r['attempt'] != 1 or r['returncode'] != 0
                   for r in manifest['runs'].values())):
        raise ValueError('Exactly six original successful full replays required')
    outcomes, artifacts = [], {}
    for row in selected:
        key, n = row['key'], row['full_policy_steps']
        metric = trial.read(directory / (key + '.metrics.json'))['motions'][0]
        trace = trial.read(directory / (key + '.timeseries.json'))['motions'][0]
        checked = verify_trace(metric, trace, n)
        with np.load(directory / (key + '.inputs.npz'), allow_pickle=False) as captured:
            actions, obs = captured['actions'], captured['observations']
            if (actions.shape != (n, 29) or obs.shape != (n, 1570)
                    or not np.isfinite(actions).all() or not np.isfinite(obs).all()):
                raise ValueError('Incomplete actual policy capture')
            applied = np.clip(actions.astype(np.float64), -limit, limit)
            np.testing.assert_array_equal(applied, trace['raw_action_29'])
            clipped = int((actions.astype(np.float64) != applied).sum())
        with np.load(directory / (key + '.contact.npz'), allow_pickle=False) as contacts:
            np.testing.assert_array_equal(contacts['policy_tick'], np.arange(n))
            for field in ('qpos', 'qvel', 'foot_floor_normal_force_n', 'foot_geometry_min_z_m'):
                if contacts[field].shape[0] != n or not np.isfinite(contacts[field]).all():
                    raise ValueError('Invalid full-state/contact capture')
        outcomes.append(dict(key=key, group='cloth', clipped_scalar_actions=clipped, **checked))
        for suffix in ('metrics.json', 'timeseries.json', 'inputs.npz', 'contact.npz'):
            path = directory / (key + '.' + suffix)
            artifacts[str(path.relative_to(trial.ROOT))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    mean = sum(r['joint_rmse_rad'] for r in outcomes) / 6
    baseline = trial.read(trial.ROOT / plan['matched_baseline_contract'])
    result = dict(status='VERIFIED_SAVED_SIX_CLOTH_TRACES_NOT_FINAL_TEN_EFFECT', verified=True,
        at_utc=trial.runtime.stamp(), source_trial_remains_stopped=True,
        source_plan_sha256=sha256(trial.PLAN), source_result_sha256=sha256(trial.RESULT),
        code_sha256=sha256(Path(__file__)), vendor_source_sha256=sha256(vendor),
        action_boundary='CAPTURE_PRECLIP_FLOAT32_TO_FLOAT64_THEN_EXACT_VENDOR_CLIP_TRACE_POSTCLIP',
        action_clip=limit, outcomes=outcomes, artifacts=artifacts,
        accepted_cloth_summary=summary, source_cloth_directory=str(directory.relative_to(trial.ROOT)),
        full_policy_steps=sum(r['policy_steps'] for r in outcomes),
        no_fall_count=sum(not r['fall'] for r in outcomes), clothing_all_no_fall=all(not r['fall'] for r in outcomes),
        clothing_mean_rmse_rad=mean,
        relative_mean_improvement=1-mean/baseline['baseline_clothing_arithmetic_mean_rmse_rad'],
        new_tensor_checkpoint_loads=0, new_policy_forwards=0, new_simulation_steps=0, new_optimizer_steps=0,
        after_issues=issues, backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(TARGET, result)
    print(json.dumps({k: result[k] for k in ('status', 'verified', 'full_policy_steps', 'no_fall_count',
        'clothing_mean_rmse_rad', 'relative_mean_improvement', 'outcomes')}))


if __name__ == '__main__':
    main()
