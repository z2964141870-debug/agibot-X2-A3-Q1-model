"""Draw two six-panel CPU parity figures from saved results, without inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np


def read(path):
    return json.loads(path.read_text())


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def bundle(source, figures, output, receipt):
    if not all(p.is_absolute() for p in (source, figures, output, receipt)) or output.exists() or receipt.exists():
        raise ValueError('Require absolute existing evidence and fresh bundle/receipt paths')
    result, originals = read(source/'result.json'), read(source/'originals.json')
    if (result['status'] != 'CPU_STATE_EQUIVALENCE_PASSED' or result['states'] != 234
            or result['policy_action_equivalence_checked'] is not False or not result['ledger_unchanged']
            or originals['arrays_in_archive'] is not False):
        raise ValueError('Require CPU-only234-state evidence with server-only raw arrays')
    payload = {}
    for item in originals['files']:
        path = source/item['name']
        actual = identity(path)
        if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:
            raise ValueError('Original evidence identity changed: '+str(path))
        payload['originals/'+path.name] = path.read_bytes()
    payload['originals/originals.json'] = (source/'originals.json').read_bytes()
    for item in read(figures/'sources.json')['inputs']:
        if identity(Path(item['path'])) != item:
            raise ValueError('Figure source identity changed')
    figure_files = sorted(figures.iterdir())
    if len(figure_files) != 6 or {p.suffix for p in figure_files} != {'.png', '.pdf', '.csv', '.json'}:
        raise ValueError('Expected exactly two PNGs, two PDFs, a CSV and sources.json')
    for path in figure_files:
        payload['figures/'+path.name] = path.read_bytes()
    payload['README.md'] = '''# CPU状态转换等价加速证据

2026-11-08，北京时间。低延迟SuperMimic的软件处理实验，原共享训练保持暂停。

done：在原策略Torch2.11.0+cu128/MuJoCo3.11.0运行时，以前走128、起走106个保存状态交替比较旧Torch运动学与新隔离MuJoCo运动学。269状态、93观测和930连续历史通过原atol=rtol=2e-5；最大状态差5.72e-6，观测/历史1.91e-6。五个坐标与不可变性合同全部通过。

| 去各片前两调用的CPU转换 | 前走 | 起走 |
| --- | ---: | ---: |
| 旧→新中位，ms | 6.199→0.368 | 7.160→0.402 |
| 旧→新p95，ms | 13.015→0.760 | 16.291→0.977 |

此配对试验中转换约快17倍。初始观测/历史89.38ms尖峰保留在全部调用记录；不能用稳态数据冒充全链路或硬件延迟。首次启动在导入时失败、未比较状态，失败原件一并保留；修复既有环境后独立R2于28.252531秒完成。

本轮零策略前向、物理积分、渲染或训练，GPU账本2433/14400秒原字节不变。没有新视频；完整269维等价数组仍在服务器，路径与SHA见原件清单。

- [前走六宫图](figures/walk_forward_loop_001__A022.png)
- [起走六宫图](figures/walk_forward_start_002__A023.png)
- [前走PDF](figures/walk_forward_loop_001__A022.pdf)
- [起走PDF](figures/walk_forward_start_002__A023.pdf)
- [计时CSV](figures/cpu_input_timing.csv)
- [全部状态计时与结果](originals/result.json)
- [首次导入失败原件](originals/first_failed_result.json)
- [服务器完成与资源回执](originals/q1_cpu_state_completed_20261108.json)
- [原件及服务器数组清单](originals/originals.json)
- [包内逐件大小与SHA](MANIFEST.json)

todo：独立小样本GPU动作等价和完整worker/RPC计时，再研究H4时间mask与闭环目标。此项不减少人体未来需求、不修复已有f0七例失败、不构成自然遥操或最终验收，采纳仍为null。
'''.encode('utf-8')
    members = [dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
               for name, raw in sorted(payload.items())]
    manifest = dict(schema='q1_cpu_state_evidence_v1', members=members,
        server_only_arrays=originals['server_array_artifacts'], new_policy_calls=0,
        new_physical_steps=0, new_optimizer_updates=0, policy_action_equivalence_checked=False)
    payload['MANIFEST.json'] = (json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payload.items()):
            archive.writestr(name, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload):
            raise ValueError('Bundle CRC or membership failed')
        for name, raw in payload.items():
            if archive.read(name) != raw:
                raise ValueError('Bundle member mismatch: '+name)
        links = re.findall(r'\]\(([^)]+)\)', archive.read('README.md').decode('utf-8'))
        if not links or any(link not in archive.namelist() for link in links):
            raise ValueError('Broken evidence README link')
    record = dict(stage='CPU_STATE_RESULT_DELIVERY', status='EVIDENCE_VERIFIED',
        source_result=identity(source/'result.json'), numeric_originals=identity(source/'originals.json'),
        figures=[identity(p) for p in figure_files], bundle=identity(output), members=len(payload),
        crc='PASSED', member_bytes_and_sha256='PASSED', relative_readme_links=len(links),
        figure_visual_inspection='Two PNGs inspected by agent before packaging',
        states=234, conversion_only_approximate_speedup=17,
        new_policy_calls=0, new_physical_steps=0, new_optimizer_updates=0,
        cumulative_gpu_seconds=2433, policy_action_equivalence_checked=False,
        hardware_end_to_end_latency_ms=None, raw_arrays_in_archive=False, new_videos=0,
        user_accepted_checkpoint=None, old_document_mirror='PENDING')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False); stream.write('\n')
    print(json.dumps(dict(bundle=record['bundle'], members=len(payload), readme_links=len(links))))


def deliver(source, output):
    if not source.is_absolute() or not output.is_absolute() or output.exists():
        raise ValueError('Require saved originals and a fresh figure directory')
    result = read(source/'result.json')
    if (result.get('status') != 'CPU_STATE_EQUIVALENCE_PASSED' or result.get('states') != 234
            or result.get('source_integrity') != 'PASSED' or result.get('ledger_unchanged') is not True
            or result.get('cuda_initialized') is not False or result.get('new_policy_calls') != 0
            or result.get('new_physical_steps') != 0 or result.get('new_optimizer_updates') != 0
            or result.get('policy_action_equivalence_checked') is not False):
        raise ValueError('Require real CPU-only234-state parity, without claiming policy equivalence')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    files, table = [source/'result.json'], []
    labels = ('Original Torch FK', 'Isolated MuJoCo FK')
    colors = ('#a45a35', '#197d85')
    for case in result['comparisons']:
        clip = case['clip']
        path = source/(clip+'.frames.jsonl')
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if len(rows) != case['states'] or [r['frame'] for r in rows] != list(range(len(rows))):
            raise ValueError('Missing or nonsequential saved timing rows')
        files.append(path)
        n = len(rows)
        timing = {k:np.asarray([r['wall_ns'][k] for r in rows])/1e6 for k in rows[0]['wall_ns']}
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), constrained_layout=True)
        for variant, label, color in zip(('original', 'candidate'), labels, colors):
            axes[0, 0].plot(range(n), timing[variant], label=label, color=color, linewidth=1)
        axes[0, 0].set(title='Conversion time: every recorded call', xlabel='Recorded state index', ylabel='ms')
        axes[0, 0].legend(fontsize=8)
        for shift, variant, color in ((-.18, 'original', colors[0]), (.18, 'candidate', colors[1])):
            values = [case['timing']['without_first_two'][variant][k] for k in ('p50_ms', 'p95_ms', 'p99_ms', 'maximum_ms')]
            axes[0, 1].bar(np.arange(4)+shift, values, width=.34, color=color)
        axes[0, 1].set(title='Conversion, excluding first2', xticks=range(4), xticklabels=('p50', 'p95', 'p99', 'Max'), ylabel='ms')
        for k, label, color in (('', 'State269', '#924462'), ('obs_', 'Observation93', '#197d85'), ('history_', 'History930', '#a45a35')):
            axes[0, 2].plot(range(n), [r['maximum_absolute_difference'][k] for r in rows], label=label, color=color, linewidth=1)
        axes[0, 2].set(title='Maximum absolute component difference', xlabel='Recorded state index')
        axes[0, 2].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        axes[0, 2].legend(fontsize=8)
        for variant, label, color in zip(('original', 'candidate'), labels, colors):
            values = np.sort(timing[variant][2:])
            axes[1, 0].plot(values, np.arange(1, len(values)+1)/len(values), label=label, color=color)
        axes[1, 0].set(title='Conversion empirical CDF, excluding first2', xlabel='ms', ylabel='Fraction')
        axes[1, 0].legend(fontsize=8)
        for variant, label, color in zip(('original', 'candidate'), labels, colors):
            axes[1, 1].plot(np.arange(2, n), timing[variant+'_observation_and_history'][2:], label=label, color=color, linewidth=1)
        axes[1, 1].set(title='Analytic observation + history, excluding first2', xlabel='Recorded state index', ylabel='ms')
        axes[1, 1].legend(fontsize=8)
        for shift, variant, color in ((-.18, 'original', colors[0]), (.18, 'candidate', colors[1])):
            values = timing[variant]+timing[variant+'_observation_and_history']
            stable = values[2:]
            axes[1, 2].bar(np.arange(4)+shift, [np.median(stable), np.percentile(stable, 95), np.percentile(stable, 99), stable.max()], width=.34, color=color)
            for scope, a in (('all', values), ('without_first_two', stable)):
                table.append(dict(clip=clip, variant=variant, scope=scope, samples=len(a),
                    conversion_median_ms=float(np.median(timing[variant] if scope == 'all' else timing[variant][2:])),
                    cpu_subtotal_median_ms=float(np.median(a)), cpu_subtotal_p95_ms=float(np.percentile(a, 95)),
                    cpu_subtotal_maximum_ms=float(a.max()), cpu_subtotal_over20ms=int(np.sum(a > 20))))
        axes[1, 2].set(title='CPU conversion + obs/history subtotal', xticks=range(4), xticklabels=('p50', 'p95', 'p99', 'Max'), ylabel='ms')
        for ax in axes.flat:
            ax.grid(axis='y', alpha=.2)
            ax.tick_params(labelsize=8)
        fig.suptitle(f'{clip}: {n} saved states, alternating paired order, one call per converter per state\n'
            'CPU input equivalence only; no policy, physics, training, RPC or hardware end-to-end latency claim. First calls remain in raw evidence.', fontsize=10)
        for suffix in ('png', 'pdf'):
            fig.savefig(output/(clip+'.'+suffix), dpi=150)
        plt.close(fig)
    with (output/'cpu_input_timing.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    records = []
    for path in files:
        raw = path.read_bytes()
        records.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    (output/'sources.json').write_text(json.dumps(dict(inputs=records, new_inference=0, new_physics=0,
        new_training=0, policy_equivalence_checked=False, hardware_end_to_end_latency_ms=None), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--receipt', type=Path)
    a = parser.parse_args()
    if a.bundle is not None:
        if a.receipt is None:
            parser.error('--bundle requires --receipt')
        bundle(a.source, a.output, a.bundle, a.receipt)
    else:
        if a.receipt is not None:
            parser.error('--receipt requires --bundle')
        deliver(a.source, a.output)
