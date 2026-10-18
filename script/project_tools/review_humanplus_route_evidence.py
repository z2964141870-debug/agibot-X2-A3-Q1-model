"""One saved-JSON difference review; never loads weights or runs a policy."""

import json
import math
from pathlib import Path
import subprocess

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3 import humanplus_runtime as runtime


def main():
    runtime.require_root()
    root = runtime.ROOT
    target = root / 'data/manifests/a3_humanplus_route_design_20261018_D03.json'
    if target.exists():
        raise FileExistsError('Saved difference review is single-use')
    sources = {}

    def read(relative):
        path = root / relative
        sources[str(relative)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        return json.loads(path.read_text())

    base = 'data/experiments/a3_humanplus_baseline_reference_20261018_HP11'
    trials = {
        'D01': ('data/experiments/a3_humanplus_head_effect_20261018_HP16/cloth',
                'data/experiments/a3_humanplus_head_completion_20261018_HP17/official'),
        'D02': ('data/experiments/a3_humanplus_replay_loss_20261018_HP18/cloth',
                'data/experiments/a3_humanplus_replay_loss_20261018_HP18/official'),
    }
    contract = read('data/manifests/a3_humanplus_baseline_reference_20261018_HP11_contract.json')
    outcomes = []
    for row in contract['motions']:
        group, key, n = row['group'], row['key'], row['full_policy_steps']
        baseline = read(f'{base}/{group}/{key}.timeseries.json')['motions'][0]
        bm = read(f'{base}/{group}/{key}.metrics.json')['motions'][0]
        for name, directories in trials.items():
            directory = directories[group == 'official']
            after = read(f'{directory}/{key}.timeseries.json')['motions'][0]
            am = read(f'{directory}/{key}.metrics.json')['motions'][0]
            if any(len(t['q_state_29']) != n for t in (baseline, after)):
                raise ValueError('Expected complete persisted trace')
            common = min(v for v in (bm['fall_tick'], am['fall_tick'], n) if v is not None)

            def rmse(trace, start, stop):
                return math.sqrt(math.fsum((a - b) ** 2
                    for q, ref in zip(trace['q_state_29'][start:stop],
                                      trace['reference_q_29'][start:stop])
                    for a, b in zip(q, ref)) / ((stop - start) * 29))

            windows = {}
            for label, start, stop in [('common_upright', 0, common),
                    *[(f'quarter_{i}', i * n // 4, (i + 1) * n // 4) for i in range(4)]]:
                before, current = rmse(baseline, start, stop), rmse(after, start, stop)
                windows[label] = dict(start=start, stop_exclusive=stop,
                    baseline_rmse=before, candidate_rmse=current, relative_change=current / before - 1)
            outcomes.append(dict(candidate=name, key=key, group=group, steps=n,
                baseline_fall_tick=bm['fall_tick'], candidate_fall_tick=am['fall_tick'],
                first_action_difference_tick=next((i for i, (a, b) in enumerate(zip(
                    baseline['raw_action_29'], after['raw_action_29'])) if a != b), None),
                full_joint_relative_change=am['tracking']['all_29_rmse'] / bm['tracking']['all_29_rmse'] - 1,
                windows=windows))
    route_evidence = {}
    for name, trial in [('D01', 'a3_humanplus_head_logged_20261018_HP14'),
                        ('D02', 'a3_humanplus_replay_loss_20261018_HP18')]:
        routes = read(f'logs/{trial}/train/reference_route_coverage.json')['rollouts']
        total = {k: sum(r['routes'].get(k, 0) for r in routes) for k in ('g1', 'a3_fast')}
        route_evidence[name] = dict(counts=total, transitions=sum(total.values()),
            a3_fast_fraction=total['a3_fast'] / sum(total.values()), rollouts=routes)
    directory = 'logs/a3_humanplus_replay_loss_20261018_HP18/train'
    p = root / directory / 'finite_candidates.jsonl'
    sources[str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    commits = [json.loads(s) for s in p.read_text().splitlines() if json.loads(s)['status'] == 'COMMITTED']
    replay = read(f'{directory}/replay_loss_contract.json')
    cloth = read(f'{directory}/cloth_contract.json')
    for path in ('script/vendor/sonic_for_a3/gear_sonic/envs/manager_env/mdp/commands.py',
                 'script/a3/cloth_curve_trainer.py', 'script/a3/humanplus_replay_trainer.py',
                 'script/a3/humanplus_replay_loss.py', 'reports/README_A3_OUTCOME_CURVE_20261013.md',
                 'reports/research/README_IDEAS.md'):
        file = root / path
        sources[path] = dict(bytes=file.stat().st_size, sha256=sha256(file))
    record = dict(status='SAVED_DIFFERENCE_REVIEW_AND_SINGLE_VARIABLE_NOT_EXECUTION_CONTRACT',
        at_utc=runtime.stamp(), parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        read_only_sources=sources, outcomes=outcomes, routes=route_evidence,
        loss_evidence=dict(commits=len(commits), all_first_scale_one=all(
            len(r['candidates']) == 1 and r['candidates'][0]['scale'] == 1 for r in commits),
            objective_deltas=[r['candidates'][-1]['objective'] - r['before']['objective'] for r in commits],
            replay_mse_before_last_batch=replay['batches'][-1]['loss'],
            max_hard_teacher_rmse=max(r['teacher_action_rmse'] for r in cloth['optimizer_steps'])),
        single_variable='NATIVE_TRAINING_REFERENCE_ROUTE_SAMPLING_G1_1_TO_0_A3_FAST_1_UNCHANGED',
        preserve=['active_encoders_g1_and_a3_fast', 'native_g1_pair_coactivation',
            'D02_replay_loss_and_legacy_hard_guard', 'fresh035_empty_Adam', 'head_two_keys_and_Critic',
            'cached_actual_tokens', 'reset_reward_LR_clip', '8_updates_32_commits_3072_interactions',
            '2440_outer_call_budget', 'final8_only_ten_full_matched_effects', 'HP11_effect_gates'],
        hypothesis='Align training action-token route with deployed a3_fast; causal benefit unproven',
        inference_limits=['No claim mixed routing caused observed regression',
            'Changing route changes state distribution and sampled tokens; counters are not a paired causal estimate',
            'G1 pairing remains native auxiliary computation; not removal of G1 network',
            'Official motions are DEVELOPMENT_REGULARIZATION, not held-out tests',
            'All retired checkpoint PTs remain deleted; this review uses only persisted JSON'],
        activity=dict(tensor_loads=0, policy_forwards=0, simulation_steps=0, optimizer_steps=0),
        github_status='LOCAL_COMMITTED_UNPUSHED', backup_status='LOCAL_ONLY')
    atomic_json(target, record)
    print(json.dumps(dict(status=record['status'], sources=len(sources),
        routes={k: {x: v[x] for x in ('counts', 'a3_fast_fraction')} for k, v in route_evidence.items()},
        common_upright={name: {r['key']: r['windows']['common_upright']['relative_change']
            for r in outcomes if r['candidate'] == name} for name in trials})))


if __name__ == '__main__':
    main()
