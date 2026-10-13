"""E48: one fresh rollout at saved step4, followed by one no-update candidate."""

import copy
import json
import math
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper
from gear_sonic.trl.utils.common import custom_instantiate

from script.a3.adam_direction_protocol import candidate, clip_factor
from script.a3.fresh_batch_protocol import Budget, classify
from script.a3.fresh_batch_collect import launch_environment, collect, install_call_guards
from script.a3.adam_direction_probe import FullLossHarness, mapped_optimizer, norm, dot
from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import cached_distribution
from script.a3.cloth_constrained_trainer import initial_probability, restrict_policy
from script.a3.retention_direction_probe import ActorLossHarness, tree_digest, cuda_tree
from script.a3.retention_direction_protocol import direction


def execute(job, root, out, metadata_path, budget):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    def forbidden(*args, **kwargs):
        raise RuntimeError('E48 prohibits optimizer construction/step')
    torch.optim.Optimizer.__init__ = forbidden
    torch.optim.AdamW.step = forbidden
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
    # The original launcher resolves robot assets from the pinned repository.
    os.chdir(root / 'script/vendor/sonic_for_a3')
    sys.argv = [sys.argv[0]]
    app, env = launch_environment(config, out)
    for field in ('robot.algo_obs_dim_dict', 'obs.group_obs_dims', 'obs.group_obs_names'):
        if OmegaConf.to_container(OmegaConf.select(env.config, field), resolve=True) != OmegaConf.to_container(OmegaConf.select(meta.env_config, field), resolve=True):
            raise ValueError('Current environment observation contract differs: ' + field)
    actor = custom_instantiate(meta.algo_config.actor, env_config=meta.env_config,
                              algo_config=meta.algo_config, _resolve=False)
    critic = custom_instantiate(meta.algo_config.critic, env_config=meta.env_config,
                               algo_config=meta.algo_config, _resolve=False)
    actor.load_state_dict(payload['policy_state_dict'], strict=True)
    critic.load_state_dict(payload['value_state_dict'], strict=True)
    restrict_policy(actor)
    wrapper = PolicyAndValueWrapper(actor, critic).to('cuda').train()
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
    batch = collect(env, wrapper, payload, meta.algo_config, budget, out, critic_buffers_before)
    atomic_json(out / 'capture_budget.json', dict(counters=dict(budget.counts), calls=dict(calls)))
    env.env.close()
    rollout = torch.load(root / 'data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt',
                         map_location='cpu', weights_only=False)
    teacher_obs = cuda_tree(rollout['all_obs_dict'])
    del rollout
    targets = torch.load(root / 'data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt',
                         map_location='cpu', weights_only=False)['initial']['mean'].to('cuda')
    if targets.shape != (16, 24, 29) or batch['mb_actions'].shape != (4, 24, 29):
        raise ValueError('Fixed batch/teacher coverage differs')
    input_before = tree_digest((batch, teacher_obs, targets))
    previous = json.loads((root / 'data/experiments/a3_zhy_adam_direction_20261013_E47/result.json').read_text())
    if actor_before != previous['actor_before_sha256']:
        raise ValueError('E47 and E48 source Actor identities differ')
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
    if not math.isclose(float(teacher.detach()), previous['before_teacher_mse'], abs_tol=1e-7, rel_tol=1e-5):
        raise ValueError('Saved step4 teacher differs from E47 control')
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
    budget.take('mathematical_candidates')
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
    classification = classify(raw['teacher_change_along_descent'], directional, candidate_mse-before_mse)
    if (calls['actor'] != budget.counts['actor_forwards'] or calls['critic'] != budget.counts['critic_forwards'] or
            budget.counts['actor_forwards'] != 34 or budget.counts['critic_forwards'] != 2):
        raise ValueError('Audited forward coverage differs')
    result = dict(status='COMPLETE_FRESH_BATCH_COUNTERFACTUAL', main_answer=classification,
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
        capture=json.loads((out / 'capture.json').read_text()), actual_call_guards=dict(calls), optimizer_step_source=16, mathematical_candidate_step=17,
        full_loss=float(total.detach()), constant_value_actor_control_loss=float(constant_total.detach()),
        critic_buffer_changes=changed, critic_buffers_before_sha256=tree_digest(critic_buffers_before),
        critic_buffers_after_sha256=tree_digest(critic_buffers_after),
        actor_before_sha256=actor_before, actor_after_sha256=actor_after,
        inputs_before_sha256=input_before, inputs_after_sha256=input_after,
        saved_state_before_sha256=saved_before, saved_state_after_sha256=saved_after,
        critic_parameters_before_sha256=critic_parameters_before, critic_parameters_after_sha256=critic_parameters_after,
        optimizer_steps=0, new_rollouts=1, transitions=384, new_deployable_checkpoints=0,
        optimizer_constructed=False, candidate_is_fused_bitwise_replay=False,
        historical_batch_17_19_reconstructed=False, baseline_improvement_proven=False, backup_status='LOCAL_ONLY')
    atomic_json(out / 'result.json', result)
    app.close()
    print(json.dumps({k: result[k] for k in ('status', 'main_answer', 'counters', 'candidate_teacher_rmse', 'candidate_guard_passed')}), flush=True)


def probe(job, root, out, metadata_path):
    budget = Budget()
    try:
        execute(job, root, out, metadata_path, budget)
    finally:
        atomic_json(out / 'progress.json', dict(counters=budget.counts, at=time.time(),
                                               result_exists=(out / 'result.json').exists()))
