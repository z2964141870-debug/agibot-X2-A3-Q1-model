"""Read saved load labels without inventing contact truth or response latency."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time
import zipfile

import numpy as np

from deliver_teacher_recovery import ARMS, identity, read, require, saved_identity
from saved_velocity_review import matched_support


def load_modes(load, threshold):
    load = np.asarray(load)
    require(load.ndim == 2 and load.shape[1] == 2 and len(load) >= 2
        and np.isfinite(load).all() and np.isfinite(threshold) and threshold > 0,
        'Finite paired current-load samples and positive threshold required')
    return (load[:,0] >= threshold).astype(int)+2*(load[:,1] >= threshold).astype(int)


def confirm_modes(modes, dwell):
    modes = np.asarray(modes)
    require(modes.ndim == 1 and len(modes) >= 2 and np.issubdtype(modes.dtype,np.integer)
        and np.isin(modes,[0,1,2,3]).all() and type(dwell) is int and dwell in (1,2),
        'Require ordered finite support-mode labels and registered dwell')
    accepted, candidate, first, count = int(modes[0]), None, None, 0
    visible, events = [accepted], []
    for i,value in enumerate(modes[1:],1):
        value = int(value)
        if value == accepted:
            candidate, first, count = None, None, 0
        else:
            if value != candidate:
                candidate, first, count = value, i, 1
            else:
                count += 1
            if count == dwell:
                events.append(dict(from_mode=accepted,to_mode=value,emission_index=i,
                    retrospective_candidate_first_index=first))
                accepted, candidate, first, count = value, None, None, 0
        visible.append(accepted)
    pending = None if candidate is None else dict(mode=candidate,observed_samples=count,first_index=first,
        confirmed=False,right_censored_at_common_prefix=True)
    return dict(visible_modes=visible,events=events,pending_final_candidate=pending)


def compare_loads(loads, threshold, dwell):
    require(len(loads) == 2 and loads[0].shape == loads[1].shape
        and np.array_equal(loads[0][0],loads[1][0]), 'Require identical anchor loads and common support')
    raw = [load_modes(load,threshold) for load in loads]
    streams = [confirm_modes(modes,dwell) for modes in raw]
    for item,modes in zip(streams,raw):
        item['raw_modes'] = modes.tolist()
        item['retrospective_single_sample_excursions'] = int(np.sum((modes[1:-1] != modes[:-2]) & (modes[:-2] == modes[2:])))
    different = np.flatnonzero(np.asarray(streams[0]['visible_modes'])[1:] != np.asarray(streams[1]['visible_modes'])[1:])+1
    return dict(threshold_n=threshold,confirmation_samples=dwell,streams=streams,
        differing_samples=len(different),observed_samples=len(raw[0])-1,
        different_fraction=float(len(different)/(len(raw[0])-1)),
        first_difference_observed_ms=int(different[0])*20 if len(different) else None,
        no_difference_right_censored=not bool(len(different)),response_latency_measured=False)


def draw(output, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    labels = [('T:' if p['split'] == 'train' else 'V:')+p['clip'].split('__')[-1]+'/'+str(p['anchor']) for p in pairs]
    fig,axes = plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    colors = ListedColormap(['#b9bdc5','#3977a6','#cf9948','#3e9176'])
    colors.set_bad('#f4f4f4')
    width = max(p['common_valid_steps'] for p in pairs)+1
    for ax,dwell in zip(axes[0,:2],(1,2)):
        image = np.full((16,width),np.nan)
        for j,pair in enumerate(pairs):
            item = next(r for r in pair['settings'] if r['threshold_n'] == 5. and r['confirmation_samples'] == dwell)
            for i,stream in enumerate(item['streams']): image[2*j+i,:pair['common_valid_steps']+1] = stream['visible_modes']
        plot = ax.imshow(np.ma.masked_invalid(image),cmap=colors,vmin=-.5,vmax=3.5,interpolation='nearest',aspect='auto')
        ax.set_xticks([0,20,40,60,80],[0,400,800,1200,1600])
        ax.set_yticks(np.arange(8)*2+.5,labels,fontsize=7)
        ax.set(title=f'5N current labels; confirmation {dwell} sample(s)\nEach pair: student above teacher; gray = no support',xlabel='Saved time from identical anchor (ms)')
        fig.colorbar(plot,ax=ax,ticks=[0,1,2,3],shrink=.65,label='0 none / 1 left / 2 right / 3 both')
    ax = axes[0,2]
    for dwell,color,marker in ((1,'#9f5050','o'),(2,'#247d78','s')):
        for j,pair in enumerate(pairs):
            item = next(r for r in pair['settings'] if r['threshold_n'] == 5. and r['confirmation_samples'] == dwell)
            value = item['first_difference_observed_ms']
            ax.scatter(pair['common_valid_steps']*20 if value is None else value,j+(dwell-1.5)*.17,
                marker='>' if value is None else marker,facecolors='none' if value is None else color,edgecolors=color)
        ax.scatter([],[],color=color,marker=marker,label=f'{dwell} sample(s)')
    ax.set_yticks(range(8),labels,fontsize=7)
    ax.set(title='First different label; not response lag\nHollow > = none observed before support ends',xlabel='Sample observation time (ms)')
    ax.legend(fontsize=7)
    delta = np.array([[len(next(r for r in p['settings'] if r['threshold_n'] == threshold and r['confirmation_samples'] == 1)['streams'][1]['events'])
        -len(next(r for r in p['settings'] if r['threshold_n'] == threshold and r['confirmation_samples'] == 1)['streams'][0]['events'])
        for threshold in (1.,5.,10.)] for p in pairs])
    span = max(1.,float(abs(delta).max()))
    ax = axes[1,0]
    plot = ax.imshow(delta,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-span,vcenter=0,vmax=span))
    for i,j in np.ndindex(delta.shape): ax.text(j,i,f'{delta[i,j]:+d}',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(3),['1N','5N','10N']); ax.set_yticks(range(8),labels,fontsize=7)
    ax.set(title='Raw mode-change count: teacher minus student\nMore changes is not a quality score')
    fig.colorbar(plot,ax=ax,shrink=.65)
    defaults = [next(r for r in p['settings'] if r['threshold_n'] == 5. and r['confirmation_samples'] == 2) for p in pairs]
    ax = axes[1,1]
    for i,color in enumerate(('#9f5050','#247d78')):
        ax.bar(np.arange(8)+(i-.5)*.35,[r['streams'][i]['retrospective_single_sample_excursions'] for r in defaults],.35,label=ARMS[i],color=color)
    ax.set_xticks(range(8),labels,rotation=45,ha='right',fontsize=7)
    ax.set(title='Retrospective isolated A-B-A labels at5N\nNeeds both neighbors; never an online input',ylabel='Observed single-sample excursions')
    ax.legend(fontsize=7)
    pending = np.array([[int(s['pending_final_candidate'] is not None) for s in r['streams']] for r in defaults])
    ax = axes[1,2]
    ax.imshow(pending,cmap=ListedColormap(['#e9ecef','#b67b45']),vmin=0,vmax=1,aspect='auto')
    for i,j in np.ndindex(pending.shape): ax.text(j,i,str(pending[i,j]),ha='center',va='center',fontsize=9)
    ax.set_xticks([0,1],['student','teacher']); ax.set_yticks(range(8),labels,fontsize=7)
    ax.set(title='Unconfirmed final candidate at5N / 2 samples\n1 = censored; no future confirmation added')
    for ax in axes.flat: ax.title.set_fontsize(9)
    fig.suptitle('Q1 saved load-label timing | eight dependent anchors / four explored clips\n'
        'Last-substep force labels, not reference contact truth or phase latency; no new policy/physics/training',fontsize=12)
    for suffix in ('png','pdf'): fig.savefig(output/('load_timing_six_panel.'+suffix),dpi=155)
    plt.close(fig)


def review(packet, prior_summary, plan_path, output, archive):
    started = time.monotonic()
    plan,prior = read(plan_path),read(prior_summary)
    require(plan['schema'] == 'q1_saved_load_timing_v1' and plan['step_seconds'] == .02
        and plan['load_thresholds_n'] == [1.,5.,10.] and plan['confirmation_samples'] == [1,2]
        and plan['common_valid_steps'] == [80,51,47,15,80,80,37,2]
        and plan['new_policy_calls'] == plan['new_physical_steps'] == plan['new_optimizer_updates'] == 0,
        'Registered saved-only analysis changed')
    require(prior['stage'] == 'TEACHER_RECOVERY_MATCHED_SAVED_REVIEW' and prior['status'] == 'PASSED', 'Require original matched saved analysis')
    paths = sorted(p for p in packet.rglob('*') if p.is_file())
    require(len(paths) == plan['source_files'] == 62 and sum(p.stat().st_size for p in paths) <= plan['maximum_packet_bytes'], 'Source packet scope differs')
    before = {str(p.relative_to(packet)):identity(p) for p in paths}
    require(before == prior['source_files'], 'Original packet SHA differs')
    receipt = read(packet/'logs/session_records/q1_teacher_recovery_completed_20261108.json')
    result = read(saved_identity(packet,receipt['result']))
    forks = read(saved_identity(packet,receipt['fork_result']))
    require(result['status'] == 'COMPLETED' and result['source_integrity'] == 'PASSED'
        and result['cost_settlement']['state'] == 'SETTLED' and len(forks['cases']) == 16, 'Require completed original forks')
    require(output.is_absolute() and archive.is_absolute() and not output.exists() and not archive.exists(), 'Use fresh absolute saved-analysis paths')
    pairs,events,samples = [],[],[]
    for index,expected in enumerate(plan['common_valid_steps']):
        cases = [next(c for c in forks['cases'] if (c['job_index'],c['arm']) == (index,arm)) for arm in ARMS]
        arrays = []
        for case in cases:
            with np.load(saved_identity(packet,case['trace']),allow_pickle=False) as value: arrays.append(dict(value))
        n = matched_support(cases,arrays,expected)
        loads = [a['load'][:n+1] for a in arrays]
        settings = [compare_loads(loads,threshold,dwell) for threshold in plan['load_thresholds_n'] for dwell in plan['confirmation_samples']]
        pair = dict(job_index=index,clip=cases[0]['clip'],split=cases[0]['split'],anchor=cases[0]['anchor'],
            common_valid_steps=n,shorter_than80ms=n < 4,settings=settings)
        pairs.append(pair)
        for item in settings:
            for arm,stream,load in zip(ARMS,item['streams'],loads):
                common = dict(job_index=index,clip=pair['clip'],split=pair['split'],anchor=pair['anchor'],arm=arm,
                    threshold_n=item['threshold_n'],confirmation_samples=item['confirmation_samples'])
                for event in stream['events']:
                    events.append(dict(**common,**event,emission_ms=event['emission_index']*20,
                        retrospective_candidate_first_ms=event['retrospective_candidate_first_index']*20))
                for i,mode in enumerate(stream['visible_modes']):
                    samples.append(dict(**common,offset_index=i,observation_ms=i*20,left_load_n=float(load[i,0]),right_load_n=float(load[i,1]),
                        raw_mode=stream['raw_modes'][i],causally_emitted_mode=mode))
    require(before == {str(p.relative_to(packet)):identity(p) for p in paths}, 'Saved input changed during analysis')
    output.mkdir(parents=True)
    for name,rows in (('mode_events.csv',events),('per_sample.csv',samples)):
        with (output/name).open('x',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    draw(output,pairs)
    value = dict(stage='SAVED_FORK_LOAD_TIMING_REVIEW',status='PASSED',pairs=pairs,event_rows=len(events),sample_rows=len(samples),
        source_files=before,original_analysis=dict(path=str(prior_summary),**identity(prior_summary)),
        plan=dict(path=str(plan_path),**identity(plan_path)),code_identity=identity(Path(__file__)),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        reference_contact_truth_available=False,physical_contact_gradient_validated=False,phase_latency_measured=False,
        reference_preprocessing_causality='offline_unverified',hardware_end_to_end_latency_ms=None,
        new_policy_calls=0,new_physical_steps=0,new_optimizer_updates=0,new_rendered_frames=0,new_gpu_charged_seconds=0,
        original_shared_training_resumed=False,natural_teleoperation_qualified=False,user_accepted_checkpoint=None)
    require(value['elapsed_seconds'] <= plan['maximum_seconds'] and value['peak_rss_bytes'] <= plan['maximum_rss_bytes'], 'Local resource bound')
    (output/'summary.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    text = ['# Q1 Saved Load-Label Timing','',
        'A new read-only diagnostic of16 existing physical forks. Eight anchors from four explored train/validation clips are dependent. No new model, physics, training or rendered simulation is run.','',
        'Force samples describe the last simulator substep. Thresholds1/5/10N are label sensitivity probes, not ground-truth planned contacts. Teacher labels are observed privileged behavior, not a correct contact schedule. Neither fewer transitions nor closer labels alone proves better control.','',
        'Each pair uses identical initial loads and the exact common nonterminal prefix. A changed label is emitted only when1/2 consecutive samples have already been observed. Candidate onset is retrospective metadata, never the online emission time. An unconfirmed tail is censored; no future or terminating frame completes it.','',
        'Single-sample A-B-A excursions require both neighboring samples and are retrospective statistics. First label disagreement is a sampled observation time, not policy, phase or hardware latency. The40ms anchor remains too short for a recovery conclusion.','',
        '![Six-panel](load_timing_six_panel.png)','',
        '[PDF](load_timing_six_panel.pdf) | [Summary](summary.json) | [Events](mode_events.csv) | [Samples](per_sample.csv)','',
        'Exact per-file source identities remain in summary.json. Raw originals and prior video evidence are preserved separately.']
    (output/'README.md').write_text('\n'.join(text)+'\n')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as package:
        for p in sorted(output.iterdir()): package.write(p,p.name)
    with zipfile.ZipFile(archive) as package:
        require(package.testzip() is None and len(package.namelist()) == len(set(package.namelist())), 'ZIP CRC or duplicate failure')
        require(all(package.read(p.name) == p.read_bytes() for p in output.iterdir()), 'ZIP members differ')
        links = re.findall(r'\]\(([^)]+)\)',package.read('README.md').decode())
        require(all(link in package.namelist() for link in links), 'Evidence link missing')
        members = [dict(relative=n,bytes=len(package.read(n)),sha256=hashlib.sha256(package.read(n)).hexdigest()) for n in package.namelist()]
    require(before == {str(p.relative_to(packet)):identity(p) for p in paths}, 'Saved input changed while plotting')
    return dict(status=value['status'],pairs=8,event_rows=len(events),sample_rows=len(samples),
        numerical_and_plot_seconds=value['elapsed_seconds'],peak_rss_bytes=value['peak_rss_bytes'],
        archive=dict(path=str(archive),**identity(archive)),members=members,readme_links_checked=len(links),
        source_files_unchanged=len(before),new_model_calls=0,new_physical_steps=0,new_optimizer_updates=0,new_gpu_charged_seconds=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('packet','prior-summary','plan','output','archive'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.packet,args.prior_summary,args.plan,args.output,args.archive),indent=2,allow_nan=False))
