"""Read completed numeric records and deliver both conditioning training curves."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from deliver_retained_sequence import identity

ARMS = ('zero_human', 'real_human')
RISKS = ('full_old_validation_risk', 'fixed_anchor_r1_risk', 'fixed_anchor_mixed_risk')
METRICS = ('loss', 'point_loss', 'retention_loss', 'gradient_norm')
RAW_NAMES = ('result.json', 'fit_result.json', 'fit.guard.json', 'checkpoint_cpu_audit.json',
    'authorization.json', *(arm+'/training_trace.json' for arm in ARMS))
SERVER_OUTPUT = '/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_HUMAN_CONDITIONING_PAIR01_20261108'
PLAN = Path(__file__).with_name('student_conditioning_plan_20261108.json')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate(bundle, plan):
    campaign, fit, guard, proof, auth = [bundle[n] for n in RAW_NAMES[:5]]
    status = 'HUMAN_CONDITIONING_PAIR_TRAINED_PENDING_CLOSED_LOOP'
    cleanup = guard.get('cleanup', {})
    cost = campaign.get('cost_settlement', {})
    if (campaign.get('status') != status or fit.get('status') != status
            or campaign.get('source_integrity') != 'PASSED'
            or guard.get('status') != 'WORKER_COMPLETED' or cleanup.get('worker_reaped') is not True
            or type(cleanup.get('worker_return_code')) is not int or cleanup['worker_return_code'] != 0
            or cost.get('state') != 'SETTLED' or cost.get('cleanup') != cleanup
            or cost.get('run_id') != SERVER_OUTPUT or cost.get('stage') != 'r2'
            or type(cost.get('charged_seconds')) is not int or cost['charged_seconds'] <= 0
            or cost.get('over_reserved_budget') is not False
            or campaign.get('actual_optimizer_updates') != 6000 or fit.get('actual_optimizer_updates') != 6000
            or fit.get('matched_initialization_capacity_and_source_slots') is not True
            or fit.get('initial_action_audit', {}).get('status') != 'CONDITIONING_INITIAL_ACTION_PARITY_PASSED'
            or fit.get('human_real_future_frames') != 0 or fit.get('test_sources_read') is not False
            or fit.get('retention_weight') != plan['fit']['retention_weight']
            or proof.get('status') != 'PASSED' or proof.get('result') != campaign or proof.get('guard') != guard
            or proof.get('task_and_original_training_processes_exited') is not True
            or proof.get('gpu_lock_released') is not True or proof.get('cuda_initialized') is not False
            or auth.get('gpu_lock_owner') != campaign.get('controller')
            or auth.get('maximum_human_future_frames') != 0
            or auth.get('original_training_restarted') is not False or auth.get('final_acceptance') is not False):
        raise ValueError('Require linked original training, causal inputs, cleanup, settlement and CPU proof')
    expected_cost = plan['resource']['expected_cumulative_seconds']+cost['charged_seconds']
    expected_stages = dict(plan['resource']['expected_by_stage'])
    expected_stages['r2'] += cost['charged_seconds']
    ledger = proof.get('cost_ledger', {})
    if (campaign.get('cumulative_gpu_charged_seconds') != expected_cost
            or ledger.get('charged_seconds') != expected_cost or ledger.get('by_stage') != expected_stages
            or ledger.get('outstanding_reservations') != 0
            or any(proof.get(k) != 0 for k in ('new_model_calls', 'new_optimizer_updates', 'new_physical_steps', 'new_rendered_frames'))):
        raise ValueError('CPU proof or settled cumulative cost differs')
    if any([r.get('variant') for r in value.get('arms', [])] != list(ARMS)
            for value in (campaign, fit, proof)):
        raise ValueError('Both ordered arms must occur exactly once in each original record')
    runs = {r['variant']:r for r in fit['arms']}
    traces = {arm:bundle[arm+'/training_trace.json'] for arm in ARMS}
    matched = ('initial_adapter_sha256', 'initial_human_adapter_sha256', 'initial_validation_risk',
        'initial_fixed_anchor_risks', 'schedule_sha256', 'source_sample_counts', 'trainable_parameters', 'trainable_tensors')
    if any(runs[ARMS[0]][k] != runs[ARMS[1]][k] for k in matched):
        raise ValueError('Matched initialization, capacity or source exposure differs')
    selected = {}
    for arm, summary, cpu in zip(ARMS, campaign['arms'], proof['arms']):
        rows, item = traces[arm], runs[arm]
        if ([r.get('update') for r in rows] != list(range(0, 3001, 500))
                or item.get('actual_updates') != 3000 or item.get('actual_adam_steps') != [3000]*8
                or item.get('trainable_tensors') != 8 or cpu.get('cpu_finite') is not True
                or cpu.get('current_statistics_exact') is not True or cpu.get('frozen_teacher_and_decoder_exact') is not True
                or item['provenance']['initial_student'] != proof.get('initial_checkpoint')):
            raise ValueError('Original exposure, frozen metadata or initial checkpoint differs')
        if not all(finite(rows[0][k]) and rows[0][k] > 0 for k in ('validation_risk', 'pure_validation_risk', *RISKS)):
            raise ValueError('Percentage changes require positive original denominators')
        source_ids = None
        for row in rows:
            numbers = [row[k] for k in ('validation_risk', 'old_matched_validation_risk', 'pure_validation_risk', *RISKS, 'elapsed_seconds')]
            if (not all(finite(v) for v in numbers)
                    or not math.isclose(row['validation_risk'], .5*(row['old_matched_validation_risk']+row['pure_validation_risk']), rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError('Nonfinite risk or wrong matched validation weighting')
            allowed = all(row[k] <= rows[0][k]+1e-12 for k in RISKS)
            if row.get('fixed_anchor_non_regression') is not allowed:
                raise ValueError('Validation eligibility differs from all three original anchors')
            metrics = row.get('metrics')
            if row['update'] == 0:
                if metrics is not None:
                    raise ValueError('Initial validation must not invent a training loss')
            elif (not isinstance(metrics, dict) or not all(finite(metrics.get(k)) for k in METRICS)
                    or not math.isclose(metrics['loss'], metrics['point_loss']+20.*metrics['retention_loss'], rel_tol=0., abs_tol=1e-6)):
                raise ValueError('Missing, nonfinite or mismatched recorded objective')
            values = row.get('pure_validation', [])
            names = [r['source_clip_id'] for r in values]
            if (not names or len(names) != len(set(names)) or not all(finite(r.get('action_mse')) for r in values)
                    or source_ids is not None and set(names) != source_ids
                    or row['update'] == 0 and any(r['action_mse'] <= 0 for r in values)):
                raise ValueError('Missing, duplicate or changed per-source validation coverage')
            source_ids = set(names)
            peer = traces[ARMS[1] if arm == ARMS[0] else ARMS[0]][row['update']//500]
            if any(row[k] != peer[k] for k in ('schedule_sha256', 'source_sample_counts')):
                raise ValueError('Paired source exposure differs at a recorded point')
        chosen = min((r for r in rows if r['fixed_anchor_non_regression']), key=lambda r:r['validation_risk'])
        selected[arm] = chosen
        if (item['selected_update'] != chosen['update'] or cpu.get('selected_update') != chosen['update']
                or any(summary.get(k) != item[k] for k in ('selected_update', 'selected_policy_unchanged', 'initial_validation_risk', 'selected_validation_risk'))
                or item['initial_validation_risk'] != rows[0]['validation_risk']
                or item['selected_validation_risk'] != chosen['validation_risk']
                or item['selected_fixed_anchor_risks'] != [chosen[k] for k in RISKS]
                or item['schedule_sha256'] != rows[-1]['schedule_sha256']
                or item['source_sample_counts'] != rows[-1]['source_sample_counts']):
            raise ValueError('Selection or final exposure differs from the original curves')
        checkpoints = cpu.get('checkpoints', [])
        if ([r.get('name') for r in checkpoints] != ['selected.pt', 'last.pt']
                or [r.get('checkpoint_update') for r in checkpoints] != [chosen['update'], 3000]
                or any(r.get('cpu_finite_tensor_count', 0) <= 0 for r in checkpoints)
                or any(r['checkpoint']['path'] != SERVER_OUTPUT+'/'+arm+'/'+r['name'] for r in checkpoints)):
            raise ValueError('CPU proof belongs to different selected or last checkpoints')
        unchanged = checkpoints[0].get('robot_adapter_initial_equal') is True and checkpoints[0].get('human_output_layer_zero') is True
        if (cpu.get('initial_policy_equal') is not unchanged or item['selected_policy_unchanged'] is not unchanged
                or cpu.get('new_closed_loop_required') is not (not unchanged)
                or cpu.get('prior_closed_loop_cases_reused') != (7 if unchanged else None)
                or cpu.get('prior_closed_loop_complete_cases_reused') != (0 if unchanged else None)):
            raise ValueError('Prior closed-loop reuse requires the selected tensor identity proof')
    return runs, selected


def load_evidence(source, plan_path=PLAN):
    bundle = {name:json.loads((source/name).read_text()) for name in RAW_NAMES}
    plan = json.loads(plan_path.read_text())
    expected = ((bundle['result.json']['fit_result'], source/'fit_result.json', SERVER_OUTPUT+'/fit_result.json'),
        (bundle['checkpoint_cpu_audit.json']['protected_authorization'], source/'authorization.json', SERVER_OUTPUT+'/authorization.json'))
    for record, local, original in expected:
        value = identity(local)
        if record != dict(value, path=original):
            raise ValueError('Original JSON bytes or server identity differ: '+local.name)
    local_plan = identity(plan_path)
    server_plan = '/opt/data/private/ZHY/script/supermimic_teleop/'+plan_path.name
    if bundle['authorization.json'].get('conditioning_plan') != dict(local_plan, path=server_plan):
        raise ValueError('Published conditioning plan does not match authorization')
    runs, selected = validate(bundle, plan)
    return bundle, runs, selected


def deliver(source, output, archive):
    raw_files = sorted(p for p in source.rglob('*') if p.is_file())
    if any(p.suffix in ('.pt', '.npz') for p in raw_files):
        raise ValueError('Use compact numeric evidence without tensors')
    inputs = [identity(p) for p in raw_files]
    bundle, runs, selections = load_evidence(source)
    campaign, proof = bundle['result.json'], bundle['checkpoint_cpu_audit.json']
    traces = {arm:bundle[arm+'/training_trace.json'] for arm in ARMS}
    if output.exists() or archive.exists():
        raise FileExistsError('Preserve existing deliveries; use a fresh root')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    stats = []
    for arm in ARMS:
        rows, item = traces[arm], runs[arm]
        selected = selections[arm]
        x = [r['update'] for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for name, color in zip(ARMS, ('#286b9a', '#19826b')):
            axes[0, 0].plot(x, [r['validation_risk'] for r in traces[name]], label=name, color=color)
        axes[0, 0].axvline(selected['update'], color='#b95148', linestyle='--')
        axes[0, 0].set(title='Matched validation (old/pure 50:50)', ylabel='Group-equal action MSE')
        axes[0, 0].legend(fontsize=8)
        for key, label in (('old_matched_validation_risk', 'Matched old'), ('pure_validation_risk', 'Pure student')):
            axes[0, 1].plot(x, [r[key] for r in rows], label=label)
        axes[0, 1].set(title='Separate validation distributions', ylabel='Action MSE')
        axes[0, 1].legend(fontsize=8)
        for key, label in zip(RISKS, ('All old', 'R1', 'Mixed')):
            line, = axes[0, 2].plot(x, [r[key] for r in rows], label=label)
            axes[0, 2].axhline(rows[0][key], color=line.get_color(), linestyle=':', linewidth=.8)
        axes[0, 2].set(title='Each old anchor must not regress', ylabel='Action MSE')
        axes[0, 2].legend(fontsize=8)
        observed = [r for r in rows if r['metrics'] is not None]
        ox = [r['update'] for r in observed]
        axes[1, 0].plot(ox, [r['metrics']['point_loss'] for r in observed], label='Point')
        axes[1, 0].plot(ox, [20.*r['metrics']['retention_loss'] for r in observed], label='20 x retention')
        axes[1, 0].set(title='Recorded objective terms (every 500)', ylabel='Weighted action MSE')
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(ox, [r['metrics']['gradient_norm'] for r in observed], color='#9b6841')
        axes[1, 1].axhline(1., color='#555555', linestyle=':')
        axes[1, 1].set(title='Eight-tensor gradient before clipping', ylabel='L2 norm')
        initial, chosen, last = [{r['source_clip_id']:r['action_mse'] for r in p['pure_validation']}
            for p in (rows[0], selected, rows[-1])]
        if not set(initial) == set(chosen) == set(last) or any(v <= 0 for v in initial.values()):
            raise ValueError('Validation source coverage or denominators changed')
        names = sorted(initial)
        changes = [100.*(chosen[k]/initial[k]-1.) for k in names]
        last_changes = [100.*(last[k]/initial[k]-1.) for k in names]
        axes[1, 2].bar(range(len(names)), last_changes, label='Last',
            color=['#19826b' if v < 0 else '#b95148' for v in last_changes])
        axes[1, 2].plot(range(len(names)), changes, 'o', color='#286b9a', label='Selected '+str(selected['update']))
        axes[1, 2].axhline(0., color='#555555', linewidth=.8)
        axes[1, 2].set_xticks(range(len(names)), [str(k) for k in names])
        axes[1, 2].set(title='Pure validation by registered source', xlabel='Source index', ylabel='MSE change (%)')
        axes[1, 2].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
            if ax != axes[1, 2]:
                ax.set_xlabel('Additional Adam updates')
        charge = campaign['cost_settlement']['charged_seconds']
        fig.suptitle(f'Q1 f0 conditioning | {arm} | selected {selected["update"]} | pair charge {charge}s | saved-state fitting', fontsize=13)
        for suffix in ('png', 'pdf'):
            fig.savefig(output/(arm+'_six_panel.'+suffix), dpi=160)
        plt.close(fig)
        fields = ('update', 'validation_risk', 'old_matched_validation_risk', 'pure_validation_risk',
            *RISKS, 'fixed_anchor_non_regression', *METRICS, 'elapsed_seconds')
        with (output/(arm+'_curves.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:row.get(k) if k not in METRICS else (row['metrics'] or {}).get(k) for k in fields})
        cpu = next(r for r in proof['arms'] if r['variant'] == arm)
        stats.append(dict(variant=arm, selected_update=selected['update'], initial_policy_equal=cpu['initial_policy_equal'],
            eligible_nonzero_updates=[r['update'] for r in rows if r['update'] and r['fixed_anchor_non_regression']],
            ineligible_nonzero_updates=[r['update'] for r in rows if r['update'] and not r['fixed_anchor_non_regression']],
            selected_validation_change_percent=100.*(selected['validation_risk']/rows[0]['validation_risk']-1.),
            last_validation_change_percent=100.*(rows[-1]['validation_risk']/rows[0]['validation_risk']-1.),
            last_pure_validation_change_percent=100.*(rows[-1]['pure_validation_risk']/rows[0]['pure_validation_risk']-1.),
            last_old_anchor_change_percent=[100.*(rows[-1][k]/rows[0][k]-1.) for k in RISKS],
            selected_source_change_percent=dict(zip(map(str, names), changes)), actual_adam_steps=item['actual_adam_steps']))
    if any(identity(Path(r['path'])) != r for r in inputs):
        raise ValueError('Numeric evidence changed while plotting')
    result = dict(status='PASSED', stage='HUMAN_CONDITIONING_TRAINING_DELIVERY', arms=stats, source_inputs=inputs,
        original_pair_gpu_seconds=campaign['cost_settlement']['charged_seconds'],
        cumulative_gpu_charged_seconds=campaign['cumulative_gpu_charged_seconds'],
        loss_recorded_every_updates=500, raw_per_update_loss_available=False,
        new_model_calls=0, new_optimizer_updates=0, new_physical_steps=0, new_rendered_frames=0,
        hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        original_shared_training_resumed=False, user_accepted_checkpoint=None)
    (output/'delivery.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Q1 Causal Human Conditioning Pair', '',
        'Only the extra branch input differs: real frozen causal human latent versus zero. Both arms have the same initial weights, nominal capacity, source windows and point-plus-output-retention objective. Zero input makes some weight gradients zero; effective information capacity is not identical.', '',
        'Each arm completed 3000 additional Adam updates. Additional update0 is the already-trained causal_history3000 policy. Curves were recorded every500 updates, without episode returns or raw per-update losses. Saved-state MSE is not closed-loop or hardware quality.', '']
    for item in stats:
        arm = item['variant']
        status = 'Initial policy equality was proven; the prior0/7 result is reused.' if item['initial_policy_equal'] else 'Changed candidate; independent closed-loop evaluation is still required.'
        lines += [f'## {arm}', '', f'Selected additional update {item["selected_update"]}. {status}', '',
            f'![Six-panel]({arm}_six_panel.png)', '', f'[PDF]({arm}_six_panel.pdf) | [CSV]({arm}_curves.csv)', '']
    lines += ['[Delivery audit](delivery.json)', '', '## Original Records', '']
    lines += [f'- [{p.relative_to(source)}](raw/{p.relative_to(source)})' for p in raw_files]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
        for p in sorted(output.iterdir()):
            package.write(p, p.name)
        for p in raw_files:
            package.write(p, 'raw/'+str(p.relative_to(source)))
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or len(package.namelist()) != len(set(package.namelist())):
            raise ValueError('Evidence CRC or duplicate member failure')
        members = []
        for name in package.namelist():
            b = package.read(name)
            original = source/name[4:] if name.startswith('raw/') else output/name
            if b != original.read_bytes():
                raise ValueError('Archive member differs: '+name)
            members.append(dict(relative=name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
        links = re.findall(r'\]\(([^)]+)\)', package.read('README.md').decode())
        if any(link not in package.namelist() for link in links):
            raise ValueError('Unresolved evidence link')
    return dict(result, archive=identity(archive), members=members, readme_links_checked=len(links))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('source', 'output', 'archive'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output, args.archive), indent=2, allow_nan=False))
