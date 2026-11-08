"""Plot saved train-only objective diagnostics alongside prior validation values."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

POLICIES = ('initial', 'retained_point_last', 'retained_delta_last')
ARMS = ('retained_point', 'retained_delta')
TRAIN_KEYS = ('full_old_group_mse', 'fixed_r1_source_mse', 'fixed_mixed_source_mse',
    'matched_old_group_mse', 'pure_group_mse', 'combined_matched_pure_mse')
VAL_KEYS = ('full_old_validation_risk', 'fixed_anchor_r1_risk', 'fixed_anchor_mixed_risk',
    'old_matched_validation_risk', 'pure_validation_risk', 'validation_risk')
LABELS = ('All old', 'R1', 'Mixed', 'Matched old', 'Pure', 'Combined')


def identity(path):
    content = path.read_bytes()
    return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def deliver(source, prior, output, archive):
    def read(path):
        return json.loads(path.read_text())
    campaign, summary, guard = [read(source/n) for n in ('result.json', 'summary.json', 'audit.guard.json')]
    if (campaign.get('status') != 'TRAIN_ONLY_OBJECTIVE_AUDIT_PASSED'
            or campaign.get('source_integrity') != 'PASSED'
            or campaign.get('cost_settlement', {}).get('state') != 'SETTLED'
            or guard.get('status') != 'WORKER_COMPLETED'
            or guard.get('cleanup', {}).get('worker_reaped') is not True
            or guard['cleanup'].get('worker_return_code') != 0
            or summary.get('validation_arrays_decoded') != 0 or summary.get('optimizer_updates') != 0
            or summary.get('new_physical_steps') != 0 or summary.get('new_rendered_frames') != 0
            or [r['policy'] for r in summary['policies']] != list(POLICIES)
            or not all(r['policy_weights_unchanged'] for r in summary['policies'])):
        raise ValueError('Require actual non-mutating diagnostics, source gate, cleanup and settlement')
    if output.exists() or archive.exists():
        raise FileExistsError('Use one fresh diagnostic delivery root')
    traces = {arm:read(prior/arm/'training_trace.json') for arm in ARMS}
    if any(rows[-1]['update'] != 3000 for rows in traces.values()):
        raise ValueError('Prior validation values must describe the actual last3000 policies')
    raw = sorted(p for p in source.rglob('*') if p.is_file())
    prior_files = [prior/arm/'training_trace.json' for arm in ARMS]
    inputs = [identity(p) for p in raw+prior_files]
    reports = summary['policies']
    train_change, val_change, table = {}, {}, []
    for index, arm in enumerate(ARMS, 1):
        train_change[arm] = [100*(reports[index]['train_metrics'][k]/reports[0]['train_metrics'][k]-1) for k in TRAIN_KEYS]
        rows = traces[arm]
        val_change[arm] = [100*(rows[-1][k]/rows[0][k]-1) for k in VAL_KEYS]
        for label,tk,vk,a,b in zip(LABELS, TRAIN_KEYS, VAL_KEYS, train_change[arm], val_change[arm]):
            table.append(dict(arm=arm, metric=label, train_key=tk, validation_key=vk,
                initial_train_mse=reports[0]['train_metrics'][tk], last_train_mse=reports[index]['train_metrics'][tk],
                train_change_percent=a, initial_validation_mse=rows[0][vk], last_validation_mse=rows[-1][vk],
                validation_change_percent=b))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    colors = ('#286b9a', '#19826b')
    for ax,values,title in ((axes[0,0], train_change, 'Full training data: last vs initial'),
            (axes[0,1], val_change, 'Prior validation values: reused, no new forwards')):
        for i,(arm,color) in enumerate(zip(ARMS,colors)):
            ax.bar(np.arange(6)+(i-.5)*.36, values[arm], width=.36, label=arm.removeprefix('retained_'), color=color)
        ax.axhline(0., color='#555555', linewidth=.8)
        ax.set_xticks(range(6), LABELS, rotation=25, ha='right')
        ax.set(title=title, ylabel='Action MSE change (%)')
        ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.2)
    probes = reports[0]['gradient_probes']
    names = probes[0]['objectives']
    matrix = np.asarray([[np.mean(values) if values else np.nan for values in
        [[r['cosine'][i][j] for r in probes if r['cosine'][i][j] is not None] for j in range(len(names))]]
        for i in range(len(names))])
    ax=axes[0,2]
    image=ax.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap='RdYlGn')
    ax.set_facecolor('#eeeeee')
    short=('Old','R1','Mix','Cover','Pure','Retain','Delta','Total P','Total D')
    ax.set_xticks(range(9), short, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(9), short, fontsize=8)
    for i in range(9):
        for j in range(9):
            ax.text(j,i,'NA' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=6)
    ax.set_title('Initial gradients: mean of 4 fixed batches; NA=undefined',fontsize=10)
    fig.colorbar(image,ax=ax,fraction=.046,pad=.04,label='Cosine')
    pairs=(('old_point','pure_point'),('point_total','old_point'),('delta_total','old_point'),
        ('retention','pure_point'),('increment','pure_point'))
    ax=axes[1,0]
    for index,(policy,color) in enumerate(zip(reports,('#777777',*colors))):
        for j,(a,b) in enumerate(pairs):
            values=[]
            for row in policy['gradient_probes']:
                i,k=row['objectives'].index(a),row['objectives'].index(b)
                if row['cosine'][i][k] is not None:
                    values.append(row['cosine'][i][k])
            if values:
                x=j+(index-1)*.2
                ax.plot([x,x],[min(values),max(values)],color=color,linewidth=2)
                ax.plot(x,np.mean(values),'o',color=color,label=policy['policy'] if j==0 else None,markersize=4)
    ax.axhline(0.,color='#555555',linewidth=.8)
    ax.set_xticks(range(5),('Old/Pure','Total P/Old','Total D/Old','Retain/Pure','Delta/Pure'),rotation=25,ha='right')
    ax.set(title='Gradient direction: 4-batch mean and range, not CI',ylabel='Cosine',ylim=(-1.08,1.08))
    ax.legend(fontsize=7)
    ax.grid(axis='y',alpha=.2)
    ax=axes[1,1]
    selected=('old_point','pure_point','increment','point_total','delta_total')
    for index,(arm,color) in enumerate(zip(ARMS,colors)):
        values=[[r['dot_actual_final_displacement'][arm+'_last'][k] for r in probes] for k in selected]
        means=np.mean(values,axis=1)
        ax.bar(np.arange(5)+(index-.5)*.36,means,width=.36,color=color,label=arm.removeprefix('retained_'))
    ax.axhline(0.,color='#555555',linewidth=.8)
    ax.set_xticks(range(5),('Old','Pure','Increment','Total P','Total D'),rotation=25,ha='right')
    ax.set(title='Initial gradient dot actual final displacement',ylabel='Local first-order approximation; not measured loss')
    ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.2)
    ax=axes[1,2]
    ax.axis('off')
    counts=summary['counts']
    data=[('Fixed policies','initial + 2 actual last3000'),('Training rows','6725 old + 406 pure'),
        ('Forward batches / rows',f'{counts["student_forward_batches"]} / {counts["student_forward_rows"]}'),
        ('Autograd calls',str(counts['autograd_calls'])),('Optimizer / physics / media','0 / 0 / 0'),
        ('Validation arrays decoded','0 (saved scalar values reused)'),('All weights / source gate','unchanged / PASSED'),
        ('GPU seconds charged',str(campaign['cost_settlement']['charged_seconds'])),
        ('Cumulative / original cap',f'{campaign["cumulative_gpu_charged_seconds"]} / 14400s')]
    cell=ax.table(cellText=data,colWidths=(.45,.55),loc='center',cellLoc='left')
    cell.auto_set_font_size(False)
    cell.set_fontsize(8)
    cell.scale(1,1.6)
    ax.set_title('Read-only scope; no natural-teleoperation claim',fontsize=10)
    fig.suptitle('Q1 f0 objective audit | training and validation kept separate | no policy update',fontsize=13)
    for suffix in ('png','pdf'):
        fig.savefig(output/('objective_six_panel.'+suffix),dpi=160)
    plt.close(fig)
    with (output/'train_validation_comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    result=dict(status='PASSED',stage='F0_TRAIN_OBJECTIVE_AUDIT_DELIVERY',source_inputs=inputs,
        comparisons=table,train_change_percent=train_change,validation_change_percent=val_change,
        counts=counts,new_model_calls=0,new_optimizer_updates=0,new_physical_steps=0,new_rendered_frames=0,
        natural_teleoperation_qualified=False,causal_mechanism_proven=False,user_accepted_checkpoint=None)
    (output/'delivery.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Q1 Train-Only Objective Audit','','No optimizer updates or new physics. This audit does not prove a causal failure mechanism. Validation scalar values below are reused from the completed pair, without validation gradients.','',
        '![Six-panel](objective_six_panel.png)','','[PDF](objective_six_panel.pdf) | [CSV](train_validation_comparison.csv) | [Audit](delivery.json)','','## Original Records','']
    lines += [f'- [{p.relative_to(source)}](raw/{p.relative_to(source)})' for p in raw]
    lines += [f'- [Prior {arm} validation](prior/{arm}_training_trace.json)' for arm in ARMS]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    if any(identity(Path(r['path']))!=r for r in inputs):
        raise ValueError('Saved evidence changed while plotting')
    members={p.name:p for p in output.iterdir()}
    members.update({'raw/'+str(p.relative_to(source)):p for p in raw})
    members.update({'prior/'+arm+'_training_trace.json':prior/arm/'training_trace.json' for arm in ARMS})
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as package:
        for name,path in members.items():package.write(path,name)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or set(package.namelist())!=set(members):
            raise ValueError('ZIP CRC or member scope failed')
        records=[]
        for name,path in members.items():
            content=package.read(name)
            if content!=path.read_bytes():raise ValueError('ZIP bytes differ')
            records.append(dict(relative=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        links=re.findall(r'\]\(([^)]+)\)',package.read('README.md').decode())
        if any(link not in members for link in links):raise ValueError('Broken evidence link')
    return dict(result,archive=identity(archive),members=records,readme_links_checked=len(links))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('source','prior','output','archive','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=deliver(args.source,args.prior,args.output,args.archive)
    with args.receipt.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],archive=result['archive'],members=len(result['members']))))
