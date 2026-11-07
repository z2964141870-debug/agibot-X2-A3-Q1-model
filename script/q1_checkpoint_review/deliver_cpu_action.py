"""Plot and package the saved64-call action gate without executing a policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from deliver_cpu_state import identity, read


def deliver(source, output, bundle, receipt):
    if not all(p.is_absolute() for p in (source,output,bundle,receipt)) or any(p.exists() for p in (output,bundle,receipt)):
        raise ValueError('Require absolute saved originals and fresh output paths')
    result, done, manifest = read(source/'result.json'),read(source/'q1_cpu_action_completed_20261108.json'),read(source/'originals.json')
    if (result['status'] != 'CPU_STATE_ACTION_PARITY_PASSED' or result['source_integrity'] != 'PASSED'
            or result['new_policy_calls'] != 64 or done['actual_cuda_forward_calls'] != 64
            or result['new_physical_steps'] != 0 or result['new_optimizer_updates'] != 0
            or done['cumulative_gpu_charged_seconds'] != 2492):
        raise ValueError('Require the actual64-call completed gate and unchanged control scope')
    for record in manifest['files']:
        actual = identity(source/record['name'])
        if any(actual[k] != record[k] for k in ('bytes','sha256')):
            raise ValueError('Original evidence identity changed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2,3,figsize=(16,9.5),constrained_layout=True)
    colors = ('#147c82','#ac4c69')
    table, inputs = [],[identity(source/'result.json'),identity(source/'q1_cpu_action_completed_20261108.json')]
    totals = []
    for case,color in zip(result['cases'],colors):
        clip = case['clip']; label = clip.rsplit('__',1)[-1]
        path = source/(clip+'.frames.jsonl')
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if len(rows) != 32 or [r['frame'] for r in rows] != list(range(32)):
            raise ValueError('Require both continuous frame-zero32-call prefixes')
        inputs.append(identity(path))
        converted = np.asarray([r['profile']['stage_wall_ns']['cpu_native_state'] for r in rows])/1e6
        worker = np.asarray([r['profile']['converter_dispatch_worker_wall_ns'] for r in rows])/1e6
        rpc = np.asarray([r['local_pipe_rpc_ns'] for r in rows])/1e6
        errors = np.asarray([r['saved_action_maximum_difference'] for r in rows])
        axes[0,0].plot(range(32),worker,label=label,color=color,linewidth=1)
        axes[0,1].plot(range(32),converted,label=label,color=color,linewidth=1)
        axes[0,2].plot(range(32),errors,label=label,color=color,linewidth=1)
        for name,values,style in (('Worker',worker,'-'),('RPC',rpc,'--')):
            stable = np.sort(values[2:])
            axes[1,0].plot(stable,np.arange(1,31)/30,label=label+' '+name,color=color,linestyle=style)
            totals.append((label+' '+name,np.median(stable),np.percentile(stable,95),stable.max(),color))
        groups = (('State',('cpu_native_state',)),('Obs/history',('cpu_analytic_observation','cpu_mask_and_history')),
            ('Current target',('cpu_causal_receiver','cpu_current_history_features')),
            ('Input host',('input_transfer_and_history_normalization_host',)),
            ('Forward host',('student_forward_host',)),('Output host',('output_transfer_or_explicit_hold_host',)))
        medians = [np.median([sum(r['profile']['stage_wall_ns'][k] for k in keys)/1e6 for r in rows[2:]])
                   for _,keys in groups]
        offset = -.18 if label == 'A022' else .18
        axes[1,1].bar(np.arange(len(groups))+offset,medians,width=.34,color=color,label=label)
        for scope,subset in (('all',slice(None)),('without_first_two',slice(2,None))):
            for name,values in (('conversion',converted),('complete_worker',worker),('local_pipe_rpc',rpc)):
                a = values[subset]
                table.append(dict(clip=clip,scope=scope,metric=name,samples=len(a),p50_ms=float(np.median(a)),
                    p95_ms=float(np.percentile(a,95)),maximum_ms=float(a.max()),calls_over20ms=int(np.sum(a>20))))
    axes[0,0].set(title='Complete worker: all calls, including initialization',xlabel='Frame',ylabel='ms (log scale)',yscale='log')
    axes[0,1].set(title='Candidate CPU conversion inside actual worker',xlabel='Frame',ylabel='ms')
    axes[0,2].set(title='Maximum saved-action difference per call',xlabel='Frame',ylabel='Absolute action difference')
    axes[0,2].axhline(2e-5,color='#555555',linestyle='--',linewidth=.7,label='Absolute tolerance')
    axes[0,2].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axes[1,0].set(title='Worker / local RPC CDF, excluding first2',xlabel='ms',ylabel='Fraction')
    axes[1,1].set(title='Selected host stages, excluding first2',xticks=range(len(groups)),
        xticklabels=[label.replace(' ','\n') for label,_ in groups],ylabel='Median ms')
    x = np.arange(4)
    for offset,column,label in ((-.23,1,'p50'),(0,2,'p95'),(.23,3,'Max')):
        axes[1,2].bar(x+offset,[r[column] for r in totals],width=.22,label=label)
    axes[1,2].set(title='Complete software call, excluding first2',xticks=x,xticklabels=[r[0].replace(' ','\n') for r in totals],ylabel='ms')
    axes[1,2].axhline(20,color='#555555',linestyle='--',linewidth=.8)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2);ax.tick_params(labelsize=8);ax.legend(fontsize=7)
    fig.suptitle('CPU_ACTION01:64 actual CUDA forwards; saved actions within original2e-5 tolerance\n'
        'Software measurements only. No paired original-worker rerun, new physics/training, hardware latency or natural-teleoperation claim.',fontsize=10)
    for suffix in ('png','pdf'):fig.savefig(output/('cpu_action_chain.'+suffix),dpi=150)
    plt.close(fig)
    with (output/'cpu_action_chain.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (output/'sources.json').write_text(json.dumps(dict(inputs=inputs,new_policy_calls=0,new_physical_steps=0,
        new_optimizer_updates=0,hardware_end_to_end_latency_ms=None),indent=2)+'\n')
    payload = {'originals/'+p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    payload.update({'figures/'+p.name:p.read_bytes() for p in output.iterdir() if p.is_file()})
    payload['README.md'] = '''# CPU加速接入策略：真实动作与软件链路结果

2026-11-08，北京时间。前走、起走各32保存状态，共64次真实CUDA前向；动作最大差2.74e-6/3.34e-6，原atol=rtol=2e-5门通过。906保护输入同，两个worker干净回收，原共享训练继续暂停。

| 去每条前两次调用，ms | 前走A022 | 起走A023 |
| --- | ---: | ---: |
| 新CPU转换中位 | 0.83 | 0.98 |
| 完整worker中位 / p95 | 9.88 / 34.72 | 11.05 / 15.05 |
| 本机pipe RPC中位 / p95 | 11.43 / 36.85 | 12.09 / 16.26 |

前走去前两次后worker仍4/30超20ms、RPC6/30超20ms；起走两者0/30超20ms。全部调用保留worker约299–311ms、RPC约300–318ms初始化尖峰。不声称20ms期限已解决，不把CPU单项约17倍加速外推成全链路同比改善。本轮没有同场旧worker配对复算；主机与CUDA流区间不可相加。

- [软件链六宫图](figures/cpu_action_chain.png)
- [PDF](figures/cpu_action_chain.pdf)
- [全部/稳态计时CSV](figures/cpu_action_chain.csv)
- [真实结果](originals/result.json)
- [回收、资源及计时回执](originals/q1_cpu_action_completed_20261108.json)
- [原件来源清单](originals/originals.json)
- [逐件SHA清单](MANIFEST.json)

本阶段GPU占用59秒，累计2492/14400秒且已结算。零新物理、渲染、训练和视频；此分析绘图也未调用策略。CPU输入234状态等价的两张六宫图证据包仍单独保留，没有改写。

todo：回到H4时间mask/闭环目标，独立评估未来帧依赖与真实控制质量。H4/H10仍是特权未来参考，本轮是f0保存动作等价，不能据此称f0七失败已修复或自然遥操合格。采纳保持null。
'''.encode('utf-8')
    records=[dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(payload.items())]
    payload['MANIFEST.json']=(json.dumps(dict(schema='q1_cpu_action_evidence_v1',members=records),indent=2)+'\n').encode()
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as target:
        for name,raw in sorted(payload.items()):target.writestr(name,raw)
    with zipfile.ZipFile(bundle) as target:
        if target.testzip() is not None or any(target.read(n)!=b for n,b in payload.items()):
            raise ValueError('Evidence archive CRC/member identity failed')
        links=re.findall(r'\]\(([^)]+)\)',target.read('README.md').decode())
        if any(n not in target.namelist() for n in links):raise ValueError('Broken bundle README link')
    record=dict(stage='CPU_ACTION_EVIDENCE_DELIVERY',status='PASSED',bundle=identity(bundle),members=len(payload),
        readme_links=len(links),crc_and_member_hashes='PASSED',figures=[identity(p) for p in sorted(output.iterdir())],
        original_files=len(manifest['files'])+1,actual_saved_cuda_forwards=64,new_policy_calls=0,
        new_physical_steps=0,new_optimizer_updates=0,cumulative_gpu_charged_seconds=2492,
        hardware_end_to_end_latency_ms=None,user_accepted_checkpoint=None)
    with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(bundle=record['bundle'],members=len(payload),links=len(links))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('source','output','bundle','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();deliver(a.source,a.output,a.bundle,a.receipt)
