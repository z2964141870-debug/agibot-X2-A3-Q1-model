"""Render the actual retained-sequence pair without policy or physics calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

ARMS = ('retained_point', 'retained_delta')
RISKS = ('full_old_validation_risk', 'fixed_anchor_r1_risk', 'fixed_anchor_mixed_risk')
METRICS = ('loss', 'point_loss', 'retention_loss', 'increment_loss', 'gradient_norm')


def identity(path):
    content = path.read_bytes()
    return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def deliver(source, output, archive):
    def read(name):
        return json.loads((source/name).read_text())
    campaign, fit, guard = [read(n) for n in ('result.json', 'fit_result.json', 'fit.guard.json')]
    proof = read('checkpoint_cpu_audit.json')
    if (campaign.get('status') != 'RETAINED_SEQUENCE_PAIR_TRAINED_PENDING_CLOSED_LOOP'
            or campaign.get('source_integrity') != 'PASSED'
            or campaign.get('cost_settlement', {}).get('state') != 'SETTLED'
            or guard.get('status') != 'WORKER_COMPLETED'
            or guard.get('cleanup', {}).get('worker_reaped') is not True
            or guard['cleanup'].get('worker_return_code') != 0
            or fit.get('actual_optimizer_updates') != 6000
            or fit.get('matched_initialization_capacity_and_source_slots') is not True
            or fit.get('human_real_future_frames') != 0
            or proof.get('status') != 'PASSED' or proof.get('result') != campaign
            or [r['variant'] for r in proof.get('arms', [])] != list(ARMS)
            or not all(r.get('cpu_finite') is True for r in proof['arms'])):
        raise ValueError('Require actual training, source gate, settlement, cleanup and CPU proof')
    runs = {r['variant']: r for r in fit['arms']}
    traces = {name: read(name+'/training_trace.json') for name in ARMS}
    for key in ('initial_adapter_sha256', 'initial_validation_risk', 'initial_fixed_anchor_risks',
            'schedule_sha256', 'source_sample_counts', 'trainable_parameters'):
        if runs[ARMS[0]][key] != runs[ARMS[1]][key]:
            raise ValueError('Paired control differs: '+key)
    raw_files = sorted(p for p in source.rglob('*') if p.is_file())
    if any(p.suffix in ('.pt', '.npz') for p in raw_files):
        raise ValueError('Use the compact numeric evidence, without large tensors')
    inputs = [identity(p) for p in raw_files]
    if output.exists() or archive.exists():
        raise FileExistsError('Preserve earlier deliveries; use a fresh root')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    stats = []
    for arm in ARMS:
        rows, item = traces[arm], runs[arm]
        if [r['update'] for r in rows] != list(range(0, 3001, 500)) or item['actual_adam_steps'] != [3000]*4:
            raise ValueError('Missing matched actual optimizer exposure')
        for row in rows:
            allowed = all(row[k] <= rows[0][k]+1e-12 for k in RISKS)
            if allowed != row['fixed_anchor_non_regression']:
                raise ValueError('A reported validation gate differs from its raw values')
        selected = min((r for r in rows if r['fixed_anchor_non_regression']), key=lambda r: r['validation_risk'])
        if selected['update'] != item['selected_update']:
            raise ValueError('Registered selection differs from actual checkpoint')
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
        axes[1, 0].plot(ox, [20*r['metrics']['retention_loss'] for r in observed], label='20 x retention')
        if arm == 'retained_delta':
            axes[1, 0].plot(ox, [.5*r['metrics']['increment_loss'] for r in observed], label='0.5 x increment')
        axes[1, 0].set(title='Recorded objective terms (every 500)', ylabel='Weighted action MSE')
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(ox, [r['metrics']['gradient_norm'] for r in observed], color='#9b6841')
        axes[1, 1].axhline(1., color='#555555', linestyle=':')
        axes[1, 1].set(title='Adapter gradient before clipping', ylabel='L2 norm')
        by_source = [{r['source_clip_id']: r['action_mse'] for r in point['pure_validation']}
            for point in (rows[0], selected, rows[-1])]
        initial, chosen, last = by_source
        if not set(initial) == set(chosen) == set(last) or any(v <= 0 for v in initial.values()):
            raise ValueError('Actual validation sources or denominators differ')
        names = sorted(initial)
        changes = [100*(chosen[k]/initial[k]-1) for k in names]
        last_changes = [100*(last[k]/initial[k]-1) for k in names]
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
        fig.suptitle(f'Q1 f0 | {arm} | selected {selected["update"]} | pair charge {charge}s | saved-state fitting', fontsize=13)
        for suffix in ('png', 'pdf'):
            fig.savefig(output/(arm+'_six_panel.'+suffix), dpi=160)
        plt.close(fig)
        fields = ('update', 'validation_risk', 'old_matched_validation_risk', 'pure_validation_risk',
            *RISKS, 'fixed_anchor_non_regression', *METRICS, 'elapsed_seconds')
        with (output/(arm+'_curves.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) if k not in METRICS else (row['metrics'] or {}).get(k) for k in fields})
        cpu = next(r for r in proof['arms'] if r['variant'] == arm)
        stats.append(dict(variant=arm, selected_update=selected['update'],
            selected_initial_policy_equal=cpu['initial_policy_equal'],
            selected_validation_change_percent=100*(selected['validation_risk']/rows[0]['validation_risk']-1),
            last_validation_change_percent=100*(rows[-1]['validation_risk']/rows[0]['validation_risk']-1),
            selected_source_change_percent=dict(zip(map(str, names), changes)),
            selected_fixed_anchor_risks=item['selected_fixed_anchor_risks'], actual_adam_steps=item['actual_adam_steps']))
    if any(identity(Path(r['path'])) != r for r in inputs):
        raise ValueError('Saved numeric evidence changed while plotting')
    result = dict(status='PASSED', stage='RETAINED_SEQUENCE_TRAINING_DELIVERY', arms=stats,
        source_inputs=inputs, original_pair_gpu_seconds=campaign['cost_settlement']['charged_seconds'],
        cumulative_gpu_charged_seconds=campaign['cumulative_gpu_charged_seconds'],
        loss_recorded_every_updates=500, raw_per_update_loss_available=False,
        new_model_calls=0, new_optimizer_updates=0, new_physical_steps=0, new_rendered_frames=0,
        hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        original_shared_training_resumed=False, user_accepted_checkpoint=None)
    (output/'delivery.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Q1 Retained Sequence Pair', '',
        'Both arms use the same initial policy, source windows and output retention. Only retained_delta adds teacher action-increment matching at 40ms label spacing. These are saved-state losses, without physical rollout.', '',
        'Each arm completed 3000 additional Adam updates. Metrics were saved every 500 updates. Episode returns and raw per-update losses were not recorded.', '']
    for item in stats:
        arm = item['variant']
        status = 'Exact initial policy; the prior 0/7 outcome is reused.' if item['selected_initial_policy_equal'] else 'Changed candidate; independent closed-loop evaluation is still required.'
        lines += [f'## {arm}', '', f'Selected update {item["selected_update"]}. {status}', '',
            f'![Six-panel]({arm}_six_panel.png)', '', f'[PDF]({arm}_six_panel.pdf) | [CSV]({arm}_curves.csv)', '']
    lines += ['[Delivery audit](delivery.json)', '', '## Original Records', '']
    lines += [f'- [{p.relative_to(source)}](raw/{p.relative_to(source)})' for p in raw_files]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output.iterdir()):
            package.write(path, path.name)
        for path in raw_files:
            package.write(path, 'raw/'+str(path.relative_to(source)))
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or len(package.namelist()) != len(set(package.namelist())):
            raise ValueError('Evidence CRC or member identities failed')
        members = []
        for name in package.namelist():
            content = package.read(name)
            original = source/name[4:] if name.startswith('raw/') else output/name
            if content != original.read_bytes():
                raise ValueError('Archive differs: '+name)
            members.append(dict(relative=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        links = re.findall(r'\]\(([^)]+)\)', package.read('README.md').decode())
        if any(link not in package.namelist() for link in links):
            raise ValueError('An evidence link does not resolve')
    return dict(result, archive=identity(archive), members=members, readme_links_checked=len(links))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('source', 'output', 'archive'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output, args.archive), indent=2, allow_nan=False))
