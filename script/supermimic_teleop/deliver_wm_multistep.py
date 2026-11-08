"""Validate and plot settled multistep WM evidence, including failed screens."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import zipfile

from deliver_wm_probe import METRICS

HERE = Path(__file__).resolve().parent
ARMS = ('student_f0', 'privileged_teacher_h10')
METHODS = ('open_loop', 'true_state_one_step', 'hold_initial')
HORIZONS = (1, 2, 4, 8, 16, 32, 40)
PHYSICAL = tuple(item[0] for item in METRICS)
ERRORS = (*PHYSICAL, 'command_memory_rms')
RAW_NAMES = ('result.json', 'multistep.guard.json', 'multistep_result.json', 'authorization.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
        and (not nonnegative or value >= 0), 'Require finite numeric evidence')
    return value


def matches(raw, record):
    return len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return type(a) is type(b) and a == b


def grouped(rows, method, metric, summary='prefix_rms'):
    by_clip = {}
    for row in rows:
        by_clip.setdefault(row['clip'], []).append(row['methods'][method][summary][metric])
    require(by_clip, 'Do not invent an empty cohort')
    return mean(mean(values) for values in by_clip.values())


def validation_screen(rows, gradients, plan):
    screens = []
    for steps in HORIZONS:
        numerical = all(g['numerical_directional_gradient_passed'] for g in gradients if g['steps'] == steps)
        for arm in ARMS:
            group = [r for r in rows if r['steps'] == steps and r['arm'] == arm and r['split'] == 'validation']
            metrics = []
            for metric in plan['validation_screen']['metrics']:
                wm, hold = [grouped(group, method, metric) for method in ('open_loop', 'hold_initial')]
                ratio = wm/hold if hold > 1e-12 else None
                individual = all(r['methods']['open_loop']['prefix_rms'][metric]
                    <= plan['validation_screen']['individual_identity_ratio_max']
                    * max(r['methods']['hold_initial']['prefix_rms'][metric], 1e-12) for r in group)
                metrics.append(dict(metric=metric, open_loop_prefix_rms_equal_clip_mean=wm,
                    identity_prefix_rms_equal_clip_mean=hold, ratio=ratio,
                    passed=ratio is not None and ratio <= plan['validation_screen']['mean_identity_ratio_max'] and individual))
            screens.append(dict(steps=steps, arm=arm, anchors=len(group),
                independent_clip_groups=len({r['clip'] for r in group}), metrics=metrics,
                numerical_directional_screen_passed=numerical,
                exploratory_screen_passed=numerical and all(m['passed'] for m in metrics)))
    return dict(per_horizon=screens, all_requested_screens_passed=all(r['exploratory_screen_passed'] for r in screens),
        physical_action_gradient_validated=False, automatic_training_authorized=False,
        limitation='Same explored four clips; even a pass does not validate physical action gradients, contacts or teleoperation')


def validate(bundle, plan, recovery):
    campaign, guard, result, auth = [bundle[name] for name in RAW_NAMES]
    charge = campaign.get('cost_settlement', {})
    cleanup = guard.get('cleanup', {})
    require(campaign.get('status') == result.get('status') == 'WM_MULTISTEP_COMPLETED'
        and campaign.get('source_integrity') == result.get('source_integrity') == 'PASSED'
        and guard.get('status') == 'WORKER_COMPLETED' and cleanup.get('worker_reaped') is True
        and type(cleanup.get('worker_return_code')) is int and cleanup['worker_return_code'] == 0
        and charge.get('state') == 'SETTLED' and charge.get('cleanup') == cleanup
        and charge.get('run_id') == plan['output'] and charge.get('stage') == 'r3'
        and type(charge.get('charged_seconds')) is int and 0 <= charge['charged_seconds'] <= 600
        and campaign.get('cumulative_gpu_charged_seconds') == plan['expected_cumulative_seconds']+charge['charged_seconds'],
        'Require completed execution, clean reaping and settled original cost')
    require(plan['horizons'] == list(HORIZONS) and plan['arms'] == list(ARMS)
        and plan['common_support_steps'] == [80,51,47,15,80,80,37,2]
        and result.get('common_support_steps') == plan['common_support_steps']
        and result.get('horizon_cohorts') == [dict(steps=h, pairs=sum(n >= h for n in plan['common_support_steps'])) for h in HORIZONS]
        and all(value.get('cuda_wm_step_forward_calls') == 2046 and value.get('cuda_backward_calls') == 14
            and value.get('actual_optimizer_updates') == value.get('new_physical_steps') == 0
            and value.get('user_accepted_checkpoint') is None for value in (campaign, result))
        and result.get('new_policy_calls') == 0 and result.get('test_sources_read') is False
        and result.get('terminal_transitions_excluded') is True
        and result.get('numerical_gradient_is_not_physical_gradient') is True
        and result.get('natural_teleoperation_qualified') is False
        and result.get('hardware_end_to_end_latency_ms') is None,
        'Registered support, scope or call counts differ')
    require(auth.get('new_optimizer_updates') == auth.get('new_physical_steps') == 0
        and auth.get('original_training_restarted') is False and auth.get('final_acceptance') is False,
        'Do not use a training or final-acceptance authorization')
    cases = recovery['cases']
    require(len(cases) == 16, 'Require the original sixteen forks')
    original = {(c['job_index'], c['arm']):c for c in cases}
    require(set(original) == {(j, a) for j in range(8) for a in ARMS}, 'Missing or duplicate original fork')
    require(all(min(original[j,a]['valid_prefix_steps'] for a in ARMS) == n
        for j,n in enumerate(plan['common_support_steps'])), 'Original common nonterminal support differs')
    rows = result['per_prefix']
    expected = {(j, a, h) for j, n in enumerate(plan['common_support_steps']) for a in ARMS for h in HORIZONS if h <= n}
    require(len(rows) == len(expected) == 94
        and {(r['job_index'], r['arm'], r['steps']) for r in rows} == expected, 'Missing, duplicated or padded horizon')
    for row in rows:
        case = original[row['job_index'], row['arm']]
        require(all(row[k] == case[k] for k in ('clip', 'anchor', 'split')) and row['split'] in ('train', 'validation')
            and row['common_valid_steps'] == plan['common_support_steps'][row['job_index']]
            and set(row['methods']) == set(METHODS), 'Original anchor identity differs')
        for method in METHODS:
            for summary in ('endpoint', 'prefix_rms'):
                values = row['methods'][method][summary]
                require(set(values) == set(ERRORS), 'Physical or separate command-memory metric missing')
                for value in values.values():
                    finite(value, True)
            for key, metric in (('translation_mse_decomposition','body_position_m'),
                    ('linear_velocity_mse_decomposition','body_linear_velocity_m_s')):
                terms = row[key][method]
                require(set(terms) == {'world','root','relative','signed_cross'}, 'Incomplete signed decomposition')
                for term, value in terms.items():
                    finite(value, term != 'signed_cross')
                require(math.isclose(terms['world'], terms['root']+terms['relative']+terms['signed_cross'], rel_tol=1e-5, abs_tol=1e-8)
                    and math.isclose(terms['world'], row['methods'][method]['prefix_rms'][metric]**2, rel_tol=1e-5, abs_tol=1e-8),
                    'Translation decomposition or RMS does not close')
        modes = row['last_substep_load_modes']
        require(len(modes) == row['steps']+1 and all(type(m) is int and 0 <= m <= 3 for m in modes)
            and row['observed_load_mode_changes'] == sum(a != b for a,b in zip(modes, modes[1:])),
            'Saved load-mode support differs')
        longest = max((r for r in rows if (r['job_index'],r['arm']) == (row['job_index'],row['arm'])), key=lambda r:r['steps'])
        require(modes == longest['last_substep_load_modes'][:len(modes)], 'Load-mode prefixes disagree')
    reconstruction = result['native_state_reconstruction']
    require(len(reconstruction) == 16 and {(r['job_index'],r['arm']) for r in reconstruction} == set(original)
        and all(0 <= finite(r['maximum_link_position_difference_m']) <= 3e-5 for r in reconstruction), 'Native FK reconstruction missing')
    gradients = result['gradients']
    require(len(gradients) == 14 and {(g['job_index'], g['steps']) for g in gradients} == {(j,h) for j in (0,4) for h in HORIZONS},
        'Both fixed gradient anchors must exist at every horizon')
    for check in gradients:
        require(check['split'] == original[check['job_index'], 'student_f0']['split']
            and check['differentiated_action_index'] == 0 and check['later_actions_fixed'] is True
            and check['command_memory_excluded'] is True and check['physical_action_gradient_validated'] is False
            and type(check['absent_joint_gradient_exact_zero']) is bool and type(check['nonzero_response']) is bool,
            'Numerical derivative scope differs')
        finite(check['gradient_norm'], True)
        analytic = finite(check['autograd_directional_derivative'])
        checks = check['finite_differences']
        require([c['epsilon'] for c in checks] == plan['finite_difference_epsilons'], 'Missing registered difference scale')
        for item in checks:
            numeric = finite(item['finite_difference'])
            error = abs(numeric-analytic)
            passed = error <= plan['gradient_absolute_tolerance']+plan['gradient_relative_tolerance']*max(abs(analytic),abs(numeric))
            require(equivalent(error,item['absolute_error']) and passed is item['passed'], 'Saved numerical derivative flag differs')
        passed = check['absent_joint_gradient_exact_zero'] and check['nonzero_response'] and any(c['passed'] for c in checks)
        require(passed is check['numerical_directional_gradient_passed'], 'Numerical screen differs')
    screen = validation_screen(rows, gradients, plan)
    require(equivalent(result['screen'], screen) and campaign['diagnostic_screen'] == ('PASSED' if screen['all_requested_screens_passed'] else 'FAILED'),
        'Reported scientific screen differs from raw evidence')
    return screen


def load_evidence(source, plan_path, manifest_path, recovery_path):
    paths = {name:source/name for name in RAW_NAMES}
    paths.update(registered_plan=plan_path, execution_manifest=manifest_path, original_recovery=recovery_path)
    raw = {name:path.read_bytes() for name,path in paths.items()}
    bundle = {name:json.loads(value) for name,value in raw.items()}
    plan, manifest, recovery = [bundle[n] for n in ('registered_plan','execution_manifest','original_recovery')]
    campaign, auth = bundle['result.json'], bundle['authorization.json']
    record = campaign['multistep_result']
    require(record['path'] == plan['output']+'/multistep_result.json' and matches(raw['multistep_result.json'],record)
        and matches(raw['original_recovery'],plan['required_recovery_result']), 'Transferred original result bytes/SHA differ')
    protected = {r['path']:r for r in auth['protected_inputs']}
    require(len(protected) == len(auth['protected_inputs']) and manifest['schema'] == 'q1_wm_multistep_code_v1'
        and auth['evaluation_commit'] == manifest['evaluation_commit']
        and len(auth['evaluation_commit']) == 40 and set(auth['evaluation_commit']) <= set('0123456789abcdef')
        and len(manifest['files']) == 5 and all(protected.get(r['path']) == r for r in manifest['files'])
        and any(matches(raw['execution_manifest'], r) for r in protected.values())
        and any(Path(r['path']).name == plan_path.name and matches(raw['registered_plan'],r) for r in manifest['files'])
        and protected.get(plan['required_recovery_result']['path']) == plan['required_recovery_result']
        and protected.get(plan['required_precision_result']['path']) == plan['required_precision_result'],
        'Published source manifest, plan or parent evidence identity differs')
    screen = validate(bundle, plan, recovery)
    return paths, raw, bundle, screen


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deliver(source, output, archive, plan_path, manifest_path, recovery_path):
    paths, raw, bundle, screen = load_evidence(source,plan_path,manifest_path,recovery_path)
    require(output.is_absolute() and archive.is_absolute() and output != archive and not output.exists() and not archive.exists()
        and not any(output == p or output in p.parents or archive == p for p in paths.values()), 'Use fresh independent absolute delivery paths')
    result, campaign = bundle['multistep_result.json'], bundle['result.json']
    rows = result['per_prefix']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    for arm in ARMS:
        fig, axes = plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,(metric,title,units,scale) in zip(axes.flat,METRICS):
            for method,color in zip(METHODS,('#247d78','#306b9b','#b56749')):
                groups = [[r for r in rows if r['steps'] == h and r['arm'] == arm and r['split'] == 'validation'] for h in HORIZONS]
                ax.plot([20*h for h in HORIZONS],[scale*grouped(g,method,metric) for g in groups], 'o-', label=method, color=color)
            for row in rows:
                if row['arm'] == arm and row['split'] == 'validation':
                    ax.scatter(20*row['steps'],scale*row['methods']['open_loop']['prefix_rms'][metric],s=12,color='#247d78',alpha=.3)
            ax.set(title=title,xlabel='Prediction horizon (ms)',ylabel=units,xscale='log',ylim=(0,None))
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Frozen32k FP32 WM | '+arm+' | explored validation clips, equal clip weight\n'
            'Prefix RMS; dots retain anchors; no new physics or training',fontsize=12)
        for suffix in ('png','pdf'):
            fig.savefig(output/(arm+'_six_panel.'+suffix),dpi=160)
        plt.close(fig)
    numeric = [dict(job_index=r['job_index'],clip=r['clip'],anchor=r['anchor'],split=r['split'],arm=r['arm'],
        steps=r['steps'],method=method,summary=summary,metric=metric,error=r['methods'][method][summary][metric],
        common_valid_steps=r['common_valid_steps'],observed_load_mode_changes=r['observed_load_mode_changes'])
        for r in rows for method in METHODS for summary in ('endpoint','prefix_rms') for metric in ERRORS]
    decomposition = [dict(job_index=r['job_index'],clip=r['clip'],anchor=r['anchor'],split=r['split'],arm=r['arm'],
        steps=r['steps'],quantity=quantity,method=method,**r[key][method]) for r in rows for method in METHODS
        for key,quantity in (('translation_mse_decomposition','position'),('linear_velocity_mse_decomposition','linear_velocity'))]
    gradient_rows = [dict(job_index=g['job_index'],split=g['split'],steps=g['steps'],
        autograd=g['autograd_directional_derivative'],gradient_norm=g['gradient_norm'],**fd,
        numerical_screen=g['numerical_directional_gradient_passed'],physical_gradient_validated=False)
        for g in result['gradients'] for fd in g['finite_differences']]
    write_csv(output/'per_prefix.csv',numeric)
    write_csv(output/'signed_decomposition.csv',decomposition)
    write_csv(output/'directional_derivatives.csv',gradient_rows)
    require(all(path.read_bytes() == raw[name] for name,path in paths.items()), 'Evidence changed while plotting')
    audit = dict(status='WM_MULTISTEP_SAVED_DELIVERY_PASSED',diagnostic_screen=campaign['diagnostic_screen'],
        screen=screen,per_prefix_rows=len(rows),gradient_checks=len(result['gradients']),
        horizon_cohorts=result['horizon_cohorts'],source_inputs=[dict(path=str(p),bytes=len(raw[n]),
            sha256=hashlib.sha256(raw[n]).hexdigest()) for n,p in paths.items()],
        original_gpu_charged_seconds=campaign['cost_settlement']['charged_seconds'],
        cumulative_gpu_charged_seconds=campaign['cumulative_gpu_charged_seconds'],
        new_model_calls=0,new_optimizer_updates=0,new_physical_steps=0,new_rendered_frames=0,
        physical_action_gradient_validated=False,hardware_end_to_end_latency_ms=None,
        natural_teleoperation_qualified=False,original_shared_training_resumed=False,user_accepted_checkpoint=None)
    (output/'delivery.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    lines = ['# Frozen Q1 WM Multistep Diagnostic','',
        'Execution completed and cost settled. Scientific screen: **'+campaign['diagnostic_screen']+'**. A failed screen remains a valid diagnostic result.','',
        'Known saved future actions condition a dynamics probe, not human forecasting. The79D observation/FK and nominal context are not a full hidden simulator snapshot. Validation clips were already explored; no held-out generalization is claimed.','',
        'Curves use equal clip weighting, then equal anchor weighting. Each horizon has its own cohort; missing or terminal support is never padded. Faint dots show individual anchors. Command memory is recorded separately and excluded from scientific screening.','',
        '| Steps | Milliseconds | Paired anchors |','| --- | --- | --- |']
    lines += [f'| {c["steps"]} | {20*c["steps"]} | {c["pairs"]} |' for c in result['horizon_cohorts']]
    for arm in ARMS:
        lines += ['', '## '+arm,'',f'![Six-panel]({arm}_six_panel.png)','',f'[PDF]({arm}_six_panel.pdf)']
    lines += ['','[Per-prefix endpoint/RMS](per_prefix.csv) | [Signed root/relative/cross MSE](signed_decomposition.csv) | [Numerical derivatives](directional_derivatives.csv) | [Delivery audit](delivery.json)','',
        'Signed cross terms may be negative and are preserved. Load modes are saved labels, not predicted contact force. Directional derivative agreement is not evidence of a correct physical action gradient. No new policy, physics, rendering or training is performed by this delivery. Hardware latency and natural teleoperation remain unverified.','','## Original Evidence','']
    names = {name:name if name.endswith('.json') else name+'.json' for name in paths}
    lines += [f'- [{name}](raw/{names[name]})' for name in paths]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    (output/'raw').mkdir()
    for name,value in raw.items():
        (output/'raw'/names[name]).write_bytes(value)
    archive.parent.mkdir(parents=True,exist_ok=True)
    expected = {str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as package:
        for name,value in expected.items():
            package.writestr(name,value)
    with zipfile.ZipFile(archive) as package:
        require(package.testzip() is None and len(package.namelist()) == len(set(package.namelist()))
            and set(package.namelist()) == set(expected) and all(package.read(n) == v for n,v in expected.items()),
            'Archive members or CRC differ')
        links = re.findall(r'\]\(([^)]+)\)',package.read('README.md').decode())
        require(all(n in expected for n in links), 'Unresolved evidence link')
    encoded = archive.read_bytes()
    return dict(audit,archive=dict(path=str(archive),bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest()),
        members=len(expected),readme_links_checked=len(links))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('source','output','archive','plan','manifest','recovery-result'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source,args.output,args.archive,args.plan,args.manifest,args.recovery_result),indent=2,allow_nan=False))
