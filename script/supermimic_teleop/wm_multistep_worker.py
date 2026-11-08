"""Calibrate saved common-support physical forks with a frozen FP32 WM."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'q1_five_stage'), str(HERE.parent/'primebot_q1')]
from common import identity, require, save
from delivery import read
from paused_runtime import assert_owner
from wm_multistep_math import HORIZONS, common_support, directional_gradient, prefix_errors, rollout, translation_decomposition
from wm_probe_data import nominal_context, physical_errors
from wm_probe_worker import load_world_model, tensor_state_sha


def screen(rows, gradients, settings):
    results = []
    for horizon in HORIZONS:
        checks = [g for g in gradients if g['steps'] == horizon]
        require(len(checks) == 2 and {g['job_index'] for g in checks} == {0,4},
            'Both fixed numerical-gradient anchors are required at every horizon')
        for arm in settings['arms']:
            group = [r for r in rows if r['split'] == 'validation' and r['arm'] == arm and r['steps'] == horizon]
            require(group, 'Every registered horizon must retain validation support')
            metrics = []
            for metric in settings['validation_screen']['metrics']:
                by_clip = {}
                individual = []
                for row in group:
                    a = row['methods']['open_loop']['prefix_rms'][metric]
                    b = row['methods']['hold_initial']['prefix_rms'][metric]
                    by_clip.setdefault(row['clip'], []).append((a,b))
                    individual.append(a <= settings['validation_screen']['individual_identity_ratio_max']*max(b, 1e-12))
                wm = float(np.mean([np.mean([a for a,b in pairs]) for pairs in by_clip.values()]))
                hold = float(np.mean([np.mean([b for a,b in pairs]) for pairs in by_clip.values()]))
                ratio = wm/hold if hold > 1e-12 else None
                metrics.append(dict(metric=metric, open_loop_prefix_rms_equal_clip_mean=wm,
                    identity_prefix_rms_equal_clip_mean=hold, ratio=ratio,
                    passed=ratio is not None and ratio <= settings['validation_screen']['mean_identity_ratio_max']
                        and all(individual)))
            numerical = all(g['numerical_directional_gradient_passed'] for g in checks)
            results.append(dict(steps=horizon, arm=arm, anchors=len(group), independent_clip_groups=len(by_clip),
                metrics=metrics, numerical_directional_screen_passed=numerical,
                exploratory_screen_passed=all(m['passed'] for m in metrics) and numerical))
    return dict(per_horizon=results, all_requested_screens_passed=all(r['exploratory_screen_passed'] for r in results),
        physical_action_gradient_validated=False, automatic_training_authorized=False,
        limitation='Same explored four clips; even a pass does not validate physical action gradients, contacts or teleoperation')


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    auth = read(args.output/'authorization.json')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == args.expected_gpu
        and all(identity(r['path']) == r for r in auth['protected_inputs']), 'Admitted inputs or GPU changed')
    settings, runtime, config, plan = (read(p) for p in
        (args.multistep_plan, args.runtime_config, args.config, args.plan))
    sys.path[:0] = [str(Path(plan['source'])/'src'), str(Path(runtime['adapted_source'])/'src'),
        str(HERE.parent/'morphmimic_q1')]
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One admitted CUDA device required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from bootstrap import install
    spec = install('q1')['robot']
    from interim_review import load_physical_robot
    from q1_native_review import native_state
    from morphmimic.model.state import canonicalize
    from wm_precision_math import fp32_shadow
    from wm_probe_math import StepCounter

    audit = read(args.output/'checkpoint_audit.json')
    checkpoint = next(r['checkpoint'] for r in audit['records'] if r['robot'] == 'q1')
    baseline = read(Path(config['primebot_review'])/'checkpoint_audit.json')['final']['checkpoint']
    require(all(identity(r['path']) == r for r in (checkpoint,baseline)), 'Native or common physical model changed')
    native, metadata, frozen, payload = load_world_model(checkpoint['path'], spec, plan)
    precision = read(settings['required_precision_result']['path'])
    require(frozen == precision['frozen_tensor_sha256'], 'Model differs from the audited one-step FP32 shadow')
    model = fp32_shadow(native).to('cuda:0')
    present = model.joint_presence
    robot, physical_settings = load_physical_robot(baseline['path'])
    context, provenance = nominal_context(robot, spec, physical_settings)
    original_probe = read(HERE/'wm_probe_plan_20261107.json')
    require(all(identity(r['path']) == r for r in original_probe['context_sources']), 'Nominal context source changed')
    indices = [robot.model.body(name).id for name in spec.tracked_bodies]
    require(indices.count(1) == 1, 'The tracked link set must contain the free-root body exactly once')
    root_slot = indices.index(1)
    ctx = torch.tensor(context, device='cuda:0')[None]
    step = StepCounter(model, args.deadline_ns, settings['maximum_wm_step_calls'])
    forks = read(settings['required_recovery_result']['path'])
    require(len(forks['cases']) == 16, 'Use the preserved16 forks')
    rows, gradients, new_sources, reconstruction = [], [], [], []
    for index, expected in enumerate(settings['common_support_steps']):
        cases = [next(r for r in forks['cases'] if (r['job_index'],r['arm']) == (index,arm)) for arm in settings['arms']]
        arrays = []
        for case in cases:
            require(identity(case['trace']['path']) == case['trace'], 'Original recovery trace changed')
            with np.load(case['trace']['path'], allow_pickle=False) as value:
                arrays.append(dict(value))
            new_sources.append(case['trace'])
        n = common_support(cases, arrays)
        require(n == expected, 'Registered common nonterminal support changed')
        count = min(n, max(HORIZONS))
        for case,data in zip(cases,arrays):
            true_cpu = native_state(robot, torch.tensor(data['actual'][:count+1], dtype=torch.float32), spec).detach()
            true = true_cpu.to('cuda:0')
            action = torch.tensor(data['canonical_action'][:count], device='cuda:0', dtype=torch.float32)[:, None]
            require(torch.equal(action[:, :, ~present], torch.zeros_like(action[:, :, ~present]))
                and torch.allclose(true[1:, 240:], action[:, 0], atol=3e-5, rtol=0.), 'No-delay command memory differs')
            body = true_cpu[:,58:100].numpy().reshape(-1,14,3)
            difference = float(np.max(abs(body-data['body_actual'][:count+1, indices])))
            require(difference <= 3e-5, 'Frozen native FK does not reproduce saved link positions')
            reconstruction.append(dict(job_index=index,arm=case['arm'],maximum_link_position_difference_m=difference))
            with torch.no_grad():
                predicted = rollout(step, true[:1], action, ctx)
                oracle = torch.cat([step(true[t:t+1], action[t], ctx) for t in range(count)])
            require(torch.equal(predicted[:1],oracle[:1]), 'The two methods must reproduce the same initial one-step call')
            target = true[1:].cpu().numpy()
            values = {'open_loop':predicted.cpu().numpy(), 'true_state_one_step':oracle.cpu().numpy(),
                'hold_initial':np.repeat(true_cpu[:1].numpy(), count, axis=0)}
            errors = {k:physical_errors(v, target, present.cpu().numpy()) for k,v in values.items()}
            decomposed = {k:translation_decomposition(v[:,58:100].reshape(-1,14,3),
                target[:,58:100].reshape(-1,14,3), root_slot) for k,v in values.items()}
            velocity = {k:translation_decomposition(v[:,156:198].reshape(-1,14,3),
                target[:,156:198].reshape(-1,14,3), root_slot) for k,v in values.items()}
            modes = (data['load'][:count+1,0] >= 5).astype(int)+2*(data['load'][:count+1,1] >= 5).astype(int)
            for horizon in HORIZONS:
                if horizon > count:
                    continue
                rows.append(dict(job_index=index, clip=case['clip'], anchor=case['anchor'], split=case['split'],
                    arm=case['arm'], steps=horizon, common_valid_steps=n,
                    methods={k:prefix_errors(v,horizon) for k,v in errors.items()},
                    translation_mse_decomposition={k:{m:float(a[:horizon].mean()) for m,a in v.items()}
                        for k,v in decomposed.items()},
                    linear_velocity_mse_decomposition={k:{m:float(a[:horizon].mean()) for m,a in v.items()}
                        for k,v in velocity.items()},
                    last_substep_load_modes=modes[:horizon+1].tolist(),
                    observed_load_mode_changes=int(np.sum(modes[1:horizon+1] != modes[:horizon]))))
            if index in settings['gradient_job_indices'] and case['arm'] == settings['gradient_arm']:
                anchor = model.frame(true[:1])
                anchor = anchor._replace(pos_xy=anchor.pos_xy.detach(), quat=anchor.quat.detach())
                for horizon in HORIZONS:
                    require(horizon <= count, 'Fixed numerical-gradient anchor lacks support')
                    def features(value):
                        return (canonicalize(value[..., :240], anchor)/model.normalizer.std)[...,model.feature_presence]
                    endpoint = features(true[horizon:horizon+1]).detach()
                    def scalar(value):
                        return (features(value)-endpoint).square().mean()
                    check = directional_gradient(step,true[:1],action[:horizon],ctx,present,scalar,
                        epsilons=settings['finite_difference_epsilons'],
                        relative=settings['gradient_relative_tolerance'], absolute=settings['gradient_absolute_tolerance'])
                    gradients.append(dict(job_index=index,split=case['split'],**check))
            save(args.output/'partial_progress.json', dict(status='RUNNING',completed_arm_prefixes=len(reconstruction),
                wm_step_calls=step.calls,backward_calls=len(gradients),new_physical_steps=0,new_optimizer_updates=0))
    require(step.calls == settings['expected_wm_step_calls'] == 2046 and len(gradients) == 14,
        'Exact forward/backward diagnostic budget differs')
    require(tensor_state_sha(model.state_dict()) == frozen['wm'] and tensor_state_sha(native.state_dict()) == frozen['wm']
        and all(p.grad is None and not p.requires_grad for p in model.parameters())
        and all(tensor_state_sha(payload[k]) == digest for k,digest in frozen.items())
        and all(identity(r['path']) == r for r in auth['protected_inputs']+new_sources), 'Frozen sources or parameters changed')
    torch.cuda.synchronize()
    result = dict(status='WM_MULTISTEP_COMPLETED',source_integrity='PASSED',screen=screen(rows,gradients,settings),
        per_prefix=rows,gradients=gradients,common_support_steps=settings['common_support_steps'],
        horizon_cohorts=[dict(steps=h,pairs=sum(n >= h for n in settings['common_support_steps'])) for h in HORIZONS],
        native_state_reconstruction=reconstruction,root_tracked_slot=root_slot,context_values=context.tolist(),
        context_provenance=provenance,model_metadata=metadata,frozen_tensor_sha256=frozen,
        cuda_wm_step_forward_calls=step.calls,cuda_backward_calls=len(gradients),cuda_primitive_kernel_count=None,
        initial_state_kind=settings['state'],prediction_kind=settings['prediction'],oracle_kind=settings['oracle'],
        terminal_transitions_excluded=True,numerical_gradient_is_not_physical_gradient=True,
        test_sources_read=False,new_policy_calls=0,new_physical_steps=0,actual_optimizer_updates=0,
        natural_teleoperation_qualified=False,hardware_end_to_end_latency_ms=None,user_accepted_checkpoint=None)
    save(args.output/'multistep_result.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('output','multistep-plan','runtime-config','config','plan'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-gpu',required=True)
    parser.add_argument('--gpu-owner-pid',type=int,required=True)
    parser.add_argument('--gpu-owner-ticks',required=True)
    parser.add_argument('--deadline-ns',type=int,required=True)
    run(parser.parse_args())
