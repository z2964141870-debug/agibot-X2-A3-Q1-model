"""E49: one tangent projection of a saved-batch Adam displacement; no simulation."""

import copy
import json
import math
import time

import torch
from omegaconf import OmegaConf
from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper
from gear_sonic.trl.utils.common import custom_instantiate

from script.a3.adam_direction_protocol import candidate, clip_factor
from script.a3.candidate_retention_protocol import Budget, projection_scale, acceptance, match_control
from script.a3.fresh_batch_collect import install_call_guards
from script.a3.adam_direction_probe import FullLossHarness, mapped_optimizer, norm, dot
from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import cached_distribution
from script.a3.cloth_constrained_trainer import initial_probability, restrict_policy
from script.a3.retention_direction_probe import ActorLossHarness, tree_digest, cuda_tree
from script.a3.retention_direction_protocol import direction


class CandidatePolicy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, batch):
        self.actor.forward(batch['mb_obs_dict'], is_training=True,
                           episode_attnmask=batch['episode_attnmask'])
        aux, coefs = self.actor.aux_losses, self.actor.aux_loss_coef
        result = cached_distribution(self.actor, batch['mb_obs_dict'],
                                     batch['mb_sampled_tokens'], batch['mb_actions'])
        result.update(aux_losses=aux, aux_loss_coef=coefs)
        return result


def execute(job, root, out, metadata_path, budget):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    def forbidden(*args, **kwargs):
        raise RuntimeError('E49 prohibits optimizer construction/step and tensor saves')
    torch.optim.Optimizer.__init__ = forbidden
    torch.optim.AdamW.step = forbidden
    torch.save = forbidden
    meta = OmegaConf.create(json.loads(metadata_path.read_text()))
    run = root / 'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0'
    config = OmegaConf.load(run / 'config.yaml')
    if OmegaConf.to_container(config.algo.config, resolve=True) != OmegaConf.to_container(meta.algo_config, resolve=True):
        raise ValueError('E45 configuration differs from resolved runtime metadata')
    budget.take('policy_states')
    payload = torch.load(run / 'milestone_4/model_step_000004.pt', map_location='cpu', weights_only=False)
    if int(payload['state'].global_step) != 4 or payload['args'].max_grad_norm != .1:
        raise ValueError('Saved policy step or full gradient clipping contract differs')
    saved_before = tree_digest({k: payload[k] for k in ('policy_state_dict', 'value_state_dict', 'optimizer_state_dict')})
    actor = custom_instantiate(meta.algo_config.actor, env_config=meta.env_config,
                              algo_config=meta.algo_config, _resolve=False)
    critic = custom_instantiate(meta.algo_config.critic, env_config=meta.env_config,
                               algo_config=meta.algo_config, _resolve=False)
    actor.load_state_dict(payload['policy_state_dict'], strict=True)
    critic.load_state_dict(payload['value_state_dict'], strict=True)
    restrict_policy(actor)
    wrapper = PolicyAndValueWrapper(actor, critic).to('cuda').train()
    wrapper.set_mode('train')
    wrapper.transform_train()
    calls = install_call_guards(actor, critic, budget)
    if any(isinstance(m, (torch.nn.LazyLinear, torch.nn.LazyConv2d)) for m in wrapper.modules()):
        raise ValueError('Unexpected lazy model initialization; no extra forward authorized')
    if actor.running_mean_std is not None or critic.running_mean_std is None:
        raise ValueError('Unexpected normalization scope')
    if any(p.dtype != torch.float32 for p in wrapper.parameters()):
        raise ValueError('Requires original FP32 parameters')
    if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p != 0 for m in wrapper.modules()):
        raise ValueError('Stochastic dropout invalidates fixed-input probe')
    named, mapped, mapping, group_metadata = mapped_optimizer(wrapper, payload, job)
    parameters = [p for _, p in named]
    actor_indices = [i for i, (n, _) in enumerate(named) if n.startswith('policy.')]
    actor_parameters = [parameters[i] for i in actor_indices]
    actor_before = tree_digest(actor.state_dict())
    critic_parameters_before = tree_digest(dict(critic.named_parameters()))
    critic_buffers_before = {n: v.detach().cpu().clone() for n, v in critic.named_buffers()}
    source = root / 'data/experiments/a3_zhy_fresh_batch_20261013_E48'
    captured = torch.load(source / 'fresh_rollout.pt', map_location='cpu', weights_only=False)
    batch = cuda_tree(captured['batch'])
    capture = json.loads((source / 'capture.json').read_text())
    if captured['permutation'][:4].tolist() != [7, 10, 9, 12] or capture['environment_indices'] != [7, 10, 9, 12]:
        raise ValueError('Unique E48 first batch identity changed')
    del captured
    rollout = torch.load(root / 'data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt',
                         map_location='cpu', weights_only=False)
    teacher_obs = cuda_tree(rollout['all_obs_dict'])
    del rollout
    targets = torch.load(root / 'data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt',
                         map_location='cpu', weights_only=False)['initial']['mean'].to('cuda')
    if targets.shape != (16, 24, 29) or batch['mb_actions'].shape != (4, 24, 29):
        raise ValueError('Fixed batch/teacher coverage differs')
    input_before = tree_digest((batch, teacher_obs, targets))
    previous = json.loads((source / 'result.json').read_text())
    for label, actual in dict(actor_before_sha256=actor_before, inputs_before_sha256=input_before,
            saved_state_before_sha256=saved_before, critic_parameters_before_sha256=critic_parameters_before,
            critic_buffers_before_sha256=tree_digest(critic_buffers_before)).items():
        if actual != previous[label]:
            raise ValueError('Saved E48 state/input identity changed: ' + label)
    audit = [json.loads(line) for line in (root / 'logs/a3_zhy_cloth_failure_20261013_E45/training/actual_optimizer_steps.jsonl').read_text().splitlines()]
    sixteenth = next(row for row in audit if row['attempt_optimizer_step'] == 16)
    if [g['lr'] for g in group_metadata] != sixteenth['applied_group_lrs']:
        raise ValueError('Saved optimizer LR does not match actual step16 receipt')
    atomic_json(out / 'mapping.json', dict(parameters=mapping, groups=group_metadata,
        optimizer=str(payload['args'].optim), saved_source_sha256=saved_before,
        source_actor_sha256=actor_before, fixed_input_sha256=input_before))

    actor.forward(batch['mb_obs_dict'], is_training=True, episode_attnmask=batch['episode_attnmask'])
    aux, coefs = actor.aux_losses, actor.aux_loss_coef
    policy = cached_distribution(actor, batch['mb_obs_dict'], batch['mb_sampled_tokens'], batch['mb_actions'])
    policy.update(aux_losses=aux, aux_loss_coef=coefs)
    probability = initial_probability(policy, batch)
    atomic_json(out / 'initial_probability.json', probability)
    value = critic.evaluate(batch['mb_obs_dict'], episode_attnmask=batch['episode_attnmask'])
    if value.shape != batch['mb_values'].shape:
        raise ValueError('Critic value ABI differs from saved batch')
    full = FullLossHarness(payload['args'], meta.algo_config, group_metadata)
    losses = full._compute_loss(dict(policy_results=policy, value_results=value), batch)
    reference = ActorLossHarness(copy.deepcopy(payload['args']), meta.algo_config)
    reference_loss = reference._compute_loss(dict(policy_results=policy,
                                                 value_results=batch['mb_values'].detach()), batch)
    teacher_losses, teacher_maxima = [], []
    for start in range(0, 16, 4):
        means = actor.forward({n: v[start:start+4] for n, v in teacher_obs.items()}, is_training=False)
        error = means - targets[start:start+4]
        teacher_losses.append(error.square().mean())
        teacher_maxima.append(float(error.detach().abs().max()))
    teacher = torch.stack(teacher_losses).mean()
    total = losses['loss'] + teacher_losses[0]
    constant_total = reference_loss['loss'] + teacher_losses[0]
    if not all(bool(torch.isfinite(t)) for t in (total, teacher, constant_total)):
        raise ValueError('Nonfinite full objective')
    controls = {}
    def control(label, actual, expected, directional=False):
        controls[label] = match_control(actual, expected, job['tolerances'], directional)
    control('before_teacher_mse', float(teacher.detach()), previous['before_teacher_mse'])
    control('full_loss', float(total.detach()), previous['full_loss'])
    control('constant_value_actor_loss', float(constant_total.detach()), previous['constant_value_actor_control_loss'])
    budget.take('autograd_grad_calls')
    gradients = torch.autograd.grad(total, parameters, retain_graph=True, allow_unused=True)
    budget.take('autograd_grad_calls')
    teacher_gradient = torch.autograd.grad(teacher, actor_parameters, retain_graph=True, allow_unused=True)
    budget.take('autograd_grad_calls')
    reference_gradient = torch.autograd.grad(constant_total, actor_parameters, allow_unused=True)
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Missing/nonfinite full parameter gradient')
    if any(g is None or not bool(torch.isfinite(g).all()) for g in teacher_gradient + reference_gradient):
        raise ValueError('Missing/nonfinite Actor control gradient')
    actor_gradients = [gradients[i] for i in actor_indices]
    if any(not torch.allclose(a, b, atol=job['gradient_comparison_atol'], rtol=job['gradient_comparison_rtol'])
           for a, b in zip(actor_gradients, reference_gradient)):
        raise ValueError('Full Critic objective changes Actor partial gradient')
    actor_norm = norm(actor_gradients)
    # Match the original float32 L2-of-L2 global clipping expression, without .grad writes.
    full_norm_tensor = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in gradients]), 2)
    factor_tensor = torch.clamp(.1 / (full_norm_tensor + 1e-6), max=1.)
    if not math.isclose(float(factor_tensor), clip_factor(float(full_norm_tensor)), rel_tol=1e-6):
        raise ValueError('Clip formula control failed')
    budget.take('original_candidates')
    candidate_parameters, displacements = {}, []
    with torch.no_grad():
        for i in actor_indices:
            name, parameter = named[i]
            moments, group = mapped[name]
            group_index = next(r['group'] for r in mapping if r['name'] == name)
            lr = full.optimizer.param_groups[group_index]['lr']
            value, _, _ = candidate(parameter, gradients[i] * factor_tensor,
                moments['exp_avg'].to('cuda'), moments['exp_avg_sq'].to('cuda'),
                int(moments['step']), lr, *group['betas'], group['eps'], group['weight_decay'], sqrt=torch.sqrt)
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite mathematical candidate')
            candidate_parameters[name.removeprefix('policy.')] = value
            displacements.append(value - parameter)
    directional = dot(teacher_gradient, displacements)
    delta_norm, teacher_norm = norm(displacements), norm(teacher_gradient)
    raw = direction(dot(teacher_gradient, actor_gradients), teacher_norm, actor_norm)
    atomic_json(out / 'pre_candidate.json', dict(raw_total_direction=raw,
        candidate_teacher_directional_change=directional,
        candidate_displacement_norm=delta_norm, full_gradient_norm=float(full_norm_tensor),
        full_clip_factor=float(factor_tensor), candidate_lrs=[g['lr'] for g in full.optimizer.param_groups]))
    candidate_losses, candidate_maxima = [], []
    with torch.no_grad():
        for start in range(0, 16, 4):
            obs = {n: v[start:start+4] for n, v in teacher_obs.items()}
            means = torch.func.functional_call(actor, candidate_parameters, (obs,),
                                                dict(is_training=False), strict=False)
            error = means - targets[start:start+4]
            candidate_losses.append(float(error.square().mean()))
            candidate_maxima.append(float(error.abs().max()))
    torch.cuda.synchronize()
    candidate_mse = sum(candidate_losses) / len(candidate_losses)
    if not all(math.isfinite(v) for v in candidate_losses + candidate_maxima):
        raise ValueError('Nonfinite candidate teacher output')
    # Verify the original candidate before creating the unique projected one.
    expected_mapping = json.loads((source / 'mapping.json').read_text())
    if mapping != expected_mapping['parameters'] or group_metadata != expected_mapping['groups']:
        raise ValueError('Original optimizer mapping or flags changed')
    for name, actual in dict(full_gradient_norm_float32=float(full_norm_tensor),
            full_clip_factor=float(factor_tensor), actor_gradient_norm=actor_norm,
            candidate_teacher_mse=candidate_mse, candidate_teacher_max_abs=max(candidate_maxima)).items():
        control(name, actual, previous[name])
    control('adam_first_order', directional, previous['candidate_teacher_directional_change'], True)
    control('raw_descent', raw['teacher_change_along_descent'], previous['raw_total_direction']['teacher_change_along_descent'], True)
    control('fresh_batch_kl', full.recorded_kl[0], previous['fresh_batch_kl'][0])
    if ([g['lr'] for g in full.optimizer.param_groups] != previous['candidate_lrs'] or
            len(full.recorded_kl) != 1 or not (directional > 0 and raw['teacher_change_along_descent'] < 0)):
        raise ValueError('Original E48 candidate controls did not reproduce')
    atomic_json(out / 'original_controls.json', controls)
    budget.take('projected_candidates')
    teacher_squared_norm = dot(teacher_gradient, teacher_gradient)
    alpha = projection_scale(directional, teacher_squared_norm)
    projected_parameters, projected_displacements, ideal_displacements = {}, [], []
    with torch.no_grad():
        for i, g, delta in zip(actor_indices, teacher_gradient, displacements):
            name, parameter = named[i]
            ideal = delta.double() - alpha * g.double()
            value = (parameter.double() + ideal).to(parameter.dtype)
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite projected candidate')
            projected_parameters[name.removeprefix('policy.')] = value
            ideal_displacements.append(ideal)
            projected_displacements.append(value - parameter)
    projected_direction = dot(teacher_gradient, projected_displacements)
    ideal_direction = dot(teacher_gradient, ideal_displacements)
    atomic_json(out / 'projection.json', dict(alpha=alpha, teacher_squared_norm=teacher_squared_norm,
        original_direction=directional, ideal_direction=ideal_direction,
        rounded_direction=projected_direction, projected_delta_norm=norm(projected_displacements),
        candidates_created=1, precision='FP64_PROJECT_AND_ADD_THEN_FP32_PARAMETERS',
        backtracking=False))
    projected_losses, projected_maxima = [], []
    with torch.no_grad():
        for start in range(0, 16, 4):
            obs = {n: v[start:start+4] for n, v in teacher_obs.items()}
            means = torch.func.functional_call(actor, projected_parameters, (obs,),
                                               dict(is_training=False), strict=False)
            error = means - targets[start:start+4]
            projected_losses.append(float(error.square().mean()))
            projected_maxima.append(float(error.abs().max()))
        # One wrapper keeps both the raw auxiliary forward and cached-token
        # distribution inside the same functional parameter substitution.
        projected_policy = torch.func.functional_call(CandidatePolicy(actor),
            {'actor.' + n: v for n, v in projected_parameters.items()}, (batch,), strict=False)
        projected_reference = ActorLossHarness(copy.deepcopy(payload['args']), meta.algo_config)
        projected_cloth = projected_reference._compute_loss(dict(policy_results=projected_policy,
            value_results=batch['mb_values'].detach()), batch)['loss']
        projected_objective = float(projected_cloth) + projected_losses[0]
    torch.cuda.synchronize()
    if not all(math.isfinite(v) for v in projected_losses + projected_maxima):
        raise ValueError('Nonfinite projected teacher outputs')
    projected_mse = sum(projected_losses) / len(projected_losses)
    before_mse = float(teacher.detach())
    verdict = acceptance(projected_direction, projected_mse - before_mse,
        projected_objective - float(constant_total.detach()), math.sqrt(projected_mse),
        max(projected_maxima), job['tolerances'])
    actor_after = tree_digest(actor.state_dict())
    input_after = tree_digest((batch, teacher_obs, targets))
    saved_after = tree_digest({k: payload[k] for k in ('policy_state_dict', 'value_state_dict', 'optimizer_state_dict')})
    critic_parameters_after = tree_digest(dict(critic.named_parameters()))
    critic_buffers_after = {n: v.detach().cpu() for n, v in critic.named_buffers()}
    changed = [n for n in critic_buffers_before if not torch.equal(critic_buffers_before[n], critic_buffers_after[n])]
    if any(not n.startswith('running_mean_std.') for n in changed):
        raise ValueError('Unexpected temporary Critic buffer mutation')
    if (actor_before != actor_after or input_before != input_after or saved_before != saved_after or
            critic_parameters_before != critic_parameters_after or any(p.grad is not None for p in wrapper.parameters())):
        raise ValueError('Probe changed protected state or accumulated .grad')
    before_mse = float(teacher.detach())
    classification = verdict['classification']
    if (budget.counts != Budget.LIMITS or calls['actor'] != budget.counts['actor_forwards'] or calls['critic'] != budget.counts['critic_forwards'] or
            budget.counts['actor_forwards'] != 16 or budget.counts['critic_forwards'] != 1):
        raise ValueError('Audited forward coverage differs')
    result = dict(status='COMPLETE_SINGLE_PROJECTION_COUNTERFACTUAL', main_answer=classification,
        experiment=job['experiment'], budget=budget.LIMITS, counters=dict(budget.counts),
        full_gradient_norm_float32=float(full_norm_tensor), full_clip_factor=float(factor_tensor),
        actor_gradient_norm=actor_norm, critic_gradient_norm=norm([g for i, g in enumerate(gradients) if i not in actor_indices]),
        actor_vs_constant_value_gradient_max_abs=max(float((a-b).abs().max()) for a,b in zip(actor_gradients, reference_gradient)),
        raw_total_direction=raw, candidate_teacher_directional_change=directional,
        candidate_delta_norm=delta_norm, candidate_teacher_cosine=(directional / (teacher_norm * delta_norm) if teacher_norm * delta_norm else None),
        before_teacher_mse=before_mse, before_teacher_rmse=math.sqrt(before_mse),
        candidate_teacher_mse=candidate_mse, candidate_teacher_rmse=math.sqrt(candidate_mse),
        candidate_teacher_max_abs=max(candidate_maxima), finite_teacher_mse_change=candidate_mse-before_mse,
        candidate_guard_passed=math.sqrt(candidate_mse) <= .05 and max(candidate_maxima) <= .5,
        source_group_metadata=group_metadata, candidate_lrs=[g['lr'] for g in full.optimizer.param_groups],
        fresh_batch_kl=full.recorded_kl, initial_probability=probability,
        capture=capture, actual_call_guards=dict(calls), optimizer_step_source=16, mathematical_candidate_step=17,
        full_loss=float(total.detach()), constant_value_actor_control_loss=float(constant_total.detach()),
        critic_buffer_changes=changed, critic_buffers_before_sha256=tree_digest(critic_buffers_before),
        critic_buffers_after_sha256=tree_digest(critic_buffers_after),
        actor_before_sha256=actor_before, actor_after_sha256=actor_after,
        inputs_before_sha256=input_before, inputs_after_sha256=input_after,
        saved_state_before_sha256=saved_before, saved_state_after_sha256=saved_after,
        critic_parameters_before_sha256=critic_parameters_before, critic_parameters_after_sha256=critic_parameters_after,
        optimizer_steps=0, new_rollouts=0, transitions=0, new_deployable_checkpoints=0,
        optimizer_constructed=False, candidate_is_fused_bitwise_replay=False,
        historical_batch_17_19_reconstructed=False, baseline_improvement_proven=False, backup_status='LOCAL_ONLY')
    result.update(projected_teacher_mse=projected_mse, projected_teacher_rmse=math.sqrt(projected_mse),
        projected_teacher_max_abs=max(projected_maxima), projected_mse_change=projected_mse-before_mse,
        projection_alpha=alpha, ideal_projected_direction=ideal_direction,
        rounded_projected_direction=projected_direction, projected_delta_norm=norm(projected_displacements),
        before_fixed_actor_objective=float(constant_total.detach()), projected_fixed_actor_objective=projected_objective,
        fixed_actor_objective_change=projected_objective-float(constant_total.detach()),
        before_clothing_loss=float(reference_loss['loss'].detach()), projected_clothing_loss=float(projected_cloth),
        acceptance=verdict, original_controls=controls, tolerances=job['tolerances'],
        original_candidates=1, projected_candidates=1, simulation_created=False)
    atomic_json(out / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'main_answer', 'counters', 'candidate_teacher_rmse', 'candidate_guard_passed')}), flush=True)


def probe(job, root, out, metadata_path):
    budget = Budget()
    try:
        execute(job, root, out, metadata_path, budget)
    finally:
        atomic_json(out / 'progress.json', dict(counters=budget.counts, at=time.time(),
                                               result_exists=(out / 'result.json').exists()))
