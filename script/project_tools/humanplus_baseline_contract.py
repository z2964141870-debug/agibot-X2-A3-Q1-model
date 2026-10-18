"""Publish a compact matched-baseline contract from HP11's existing final audit."""

from script.a3 import humanplus_baseline_reference as trial
from script.a3.checkpoint_store import atomic_json, sha256


def main():
    plan = trial.checked()
    result = trial.read(trial.RESULT)
    audit_path = trial.PLAN.with_name(trial.NAME + '_official_audit.json')
    audit = trial.read(audit_path)
    target = trial.PLAN.with_name(trial.NAME + '_contract.json')
    if target.exists():
        raise FileExistsError('HP11 contract already published')
    if (result['status'] != 'COMPLETE_MATCHED_BASELINE' or not audit['verified']
            or audit['plan_sha256'] != sha256(trial.PLAN)
            or audit['result_sha256'] != sha256(trial.RESULT)
            or audit['after_issues'] or audit['remaining_own_group_processes']
            or audit['fatal_matches'] or audit['oom_matches']):
        raise ValueError('Requires unchanged completed baseline and final independent audit')
    outcomes = []
    for motion in plan['motions']:
        stage = next(r for r in result['stages'] if r['stage'] == motion['group'])
        row = next(r for r in stage['receipt']['motions'] if r['motion'] == motion['key'] + '.csv')
        if (not row['full_replay_completed']
                or row['expected_full_policy_steps'] != motion['full_policy_steps']):
            raise ValueError('Baseline coverage mismatch')
        outcomes.append(dict(motion, fall=row['fall'], first_fall_s=row['first_fall_s'],
            windows=row['windows'], timeseries_sha256=row['timeseries_sha256']))
    value = dict(status='VERIFIED_MATCHED_DEVELOPMENT_BASELINE', at_utc=trial.runtime.stamp(),
        experiment=plan['experiment'], checkpoint=str(trial.base.OFFICIAL.relative_to(trial.ROOT)),
        checkpoint_sha256=plan['input_sha256'][str(trial.base.OFFICIAL.relative_to(trial.ROOT))],
        vendor_commit=plan['vendor_commit'], environment=plan['environment'],
        environment_source_sha256=plan['environment_source_sha256'],
        protocol=plan['protocol'], motions=outcomes,
        total_policy_steps=sum(r['full_policy_steps'] for r in outcomes),
        baseline_clothing_arithmetic_mean_rmse_rad=sum(r['windows']['full_rollout']['joint_rmse_rad']
            for r in outcomes if r['group'] == 'cloth') / 6,
        no_fall_counts={group: sum(not r['fall'] for r in outcomes if r['group'] == group)
                        for group in ('cloth', 'official')},
        effect_gate=dict(all_six_clothing_no_fall=True, mean_clothing_rmse_improvement_min_percent=1.,
            previous_nonfall_clothing_joint_regression_max_percent=2.,
            official_each_joint_regression_max_percent=2., no_new_official_falls=True,
            root_and_wrist_world_regression_max_percent=5.),
        compare_only_matching_protocol_inputs_environment_and_full_horizons=True,
        final_candidate_step=8, early_checkpoint_effect_evaluation_allowed=False,
        truly_unseen_final_test=False, policy_improvement_proven=False,
        policy_optimizer_calls=0, training_started=False,
        plan_sha256=sha256(trial.PLAN), result_sha256=sha256(trial.RESULT),
        final_audit_sha256=sha256(audit_path),
        publisher_sha256=sha256(trial.ROOT / 'script/project_tools/humanplus_baseline_contract.py'),
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(target, value)
    print(value['status'], value['total_policy_steps'], value['no_fall_counts'])


if __name__ == '__main__':
    main()
