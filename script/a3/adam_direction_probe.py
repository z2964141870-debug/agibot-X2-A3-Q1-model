"""One mathematical AdamW candidate from saved moments and a fixed full batch."""

import copy
import json
import math
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from transformers import Trainer
from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper, TRLPPOTrainer
from gear_sonic.trl.utils.common import custom_instantiate

from script.a3.adam_direction_protocol import Budget, candidate, clip_factor, require_group_ids
from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import cached_distribution
from script.a3.cloth_constrained_trainer import restrict_policy
from script.a3.retention_direction_probe import ActorLossHarness, tree_digest, cuda_tree
from script.a3.retention_direction_protocol import direction


class FullLossHarness(ActorLossHarness):
    def __init__(self, args, config, groups):
        super().__init__(copy.deepcopy(args), config)
        # Pure metadata receiver for the original vendor LR rule, not an Optimizer.
        self.optimizer = SimpleNamespace(param_groups=copy.deepcopy(groups))
        self.desired_kl = config.desired_kl
        self.adaptive_lr_min = config.adaptive_lr_min
        self.adaptive_lr_max = config.adaptive_lr_max

    def _adjust_learning_rate_based_on_kl(self, kl, optimizer):
        self.recorded_kl.append(float(kl))
        TRLPPOTrainer._adjust_learning_rate_based_on_kl(self, kl, optimizer)


def norm(values):
    return math.sqrt(sum(float(x.double().square().sum()) for x in values if x is not None))


def dot(left, right):
    return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right)
               if a is not None and b is not None)


def mapped_optimizer(wrapper, payload, job):
    args = payload['args']
    optimizer_class, kwargs = Trainer.get_optimizer_cls_and_kwargs(args, wrapper)
    if optimizer_class is not torch.optim.AdamW or any(k in kwargs for k in ('params', 'model', 'optimizer_dict')):
        raise ValueError('Only original standard AdamW grouping is supported')
    decay = set(Trainer.get_decay_parameter_names(None, wrapper))
    named = [(n, p) for n, p in wrapper.named_parameters() if p.requires_grad]
    grouped = [[(n, p) for n, p in named if n in decay],
               [(n, p) for n, p in named if n not in decay]]
    saved = payload['optimizer_state_dict']
    if len(saved['param_groups']) != 2 or len({id(p) for _, p in named}) != len(named):
        raise ValueError('Unsupported optimizer groups or tied parameters')
    start, mapped, records = 0, {}, []
    for index, (group, members) in enumerate(zip(saved['param_groups'], grouped)):
        require_group_ids(group['params'], len(members), start)
        start += len(members)
        if group.get('param_names') is not None and group['param_names'] != [n for n, _ in members]:
            raise ValueError('Saved optimizer parameter names differ')
        for flag in ('amsgrad', 'maximize', 'capturable', 'differentiable'):
            if group.get(flag, False):
                raise ValueError('Unsupported Adam flag: ' + flag)
        if not group.get('decoupled_weight_decay', True):
            raise ValueError('Requires AdamW decoupled decay')
        if (group['weight_decay'] != (args.weight_decay if index == 0 else 0.) or
                tuple(group['betas']) != (args.adam_beta1, args.adam_beta2) or
                group['eps'] != args.adam_epsilon or group['lr'] != args.learning_rate or
                group.get('fused') != kwargs.get('fused') or group.get('foreach') not in (None, False, True)):
            raise ValueError('Saved optimizer configuration differs from source arguments')
        for identity, (name, parameter) in zip(group['params'], members):
            item = saved['state'][identity]
            if set(item) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Unsupported moment layout')
            if item['step'].numel() != 1 or float(item['step']) != job['saved_optimizer_step']:
                raise ValueError('Saved per-parameter optimizer step differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                value = item[key]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                    raise ValueError('Saved moment shape/dtype/finiteness mismatch: ' + name)
            if torch.any(item['exp_avg_sq'] < 0):
                raise ValueError('Negative saved Adam variance')
            mapped[name] = (item, group)
            records.append(dict(name=name, saved_id=identity, group=index, shape=list(parameter.shape),
                                dtype=str(parameter.dtype), step=int(item['step'])))
    if set(saved['state']) != set(range(start)):
        raise ValueError('Extra or missing saved optimizer state')
    return named, mapped, records, [dict((k, v) for k, v in g.items() if k != 'params')
                                  for g in saved['param_groups']]


def execute(job, root, out, metadata_path, budget):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
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
    captured = torch.load(root / 'logs/a3_zhy_cloth_failure_20261013_E45/training/initial_token_repair.pt',
                          map_location='cpu', weights_only=False)
    batch = cuda_tree(captured['batch'])
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
    previous = json.loads((root / 'data/experiments/a3_zhy_retention_direction_20261013_E46/result.json').read_text())['rows'][1]
    if input_before != previous['inputs_before_sha256'] or actor_before != previous['actor_state_before_sha256']:
        raise ValueError('E46 and E47 fixed Actor/input identities differ')
    audit = [json.loads(line) for line in (root / 'logs/a3_zhy_cloth_failure_20261013_E45/training/actual_optimizer_steps.jsonl').read_text().splitlines()]
    sixteenth = next(row for row in audit if row['attempt_optimizer_step'] == 16)
    if [g['lr'] for g in group_metadata] != sixteenth['applied_group_lrs']:
        raise ValueError('Saved optimizer LR does not match actual step16 receipt')
    atomic_json(out / 'mapping.json', dict(parameters=mapping, groups=group_metadata,
        optimizer=str(payload['args'].optim), saved_source_sha256=saved_before,
        source_actor_sha256=actor_before, fixed_input_sha256=input_before))

    budget.take('actor_forwards')
    actor.forward(batch['mb_obs_dict'], is_training=True, episode_attnmask=batch['episode_attnmask'])
    aux, coefs = actor.aux_losses, actor.aux_loss_coef
    budget.take('actor_forwards')
    policy = cached_distribution(actor, batch['mb_obs_dict'], batch['mb_sampled_tokens'], batch['mb_actions'])
    policy.update(aux_losses=aux, aux_loss_coef=coefs)
    budget.take('critic_forwards')
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
        budget.take('actor_forwards')
        means = actor.forward({n: v[start:start+4] for n, v in teacher_obs.items()}, is_training=False)
        error = means - targets[start:start+4]
        teacher_losses.append(error.square().mean())
        teacher_maxima.append(float(error.detach().abs().max()))
    teacher = torch.stack(teacher_losses).mean()
    total = losses['loss'] + teacher_losses[0]
    constant_total = reference_loss['loss'] + teacher_losses[0]
    if not all(bool(torch.isfinite(t)) for t in (total, teacher, constant_total)):
        raise ValueError('Nonfinite full objective')
    for live, expected in ((float(teacher.detach()), previous['losses']['teacher384']),
                           (float(constant_total.detach()), previous['losses']['total'])):
        if not math.isclose(live, expected, abs_tol=job['scalar_match_atol'], rel_tol=job['scalar_match_rtol']):
            raise ValueError('Fixed E46 objective control mismatch')
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
    if not math.isclose(actor_norm, previous['geometry']['norms']['total'], rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError('E46 Actor gradient norm no longer matches')
    # Match the original float32 L2-of-L2 global clipping expression, without .grad writes.
    full_norm_tensor = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in gradients]), 2)
    factor_tensor = torch.clamp(.1 / (full_norm_tensor + 1e-6), max=1.)
    if not math.isclose(float(factor_tensor), clip_factor(float(full_norm_tensor)), rel_tol=1e-6):
        raise ValueError('Clip formula control failed')
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
            budget.take('actor_forwards')
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
    classification = ('RAW_DESCENT_INWARD_ADAM_OUTWARD' if raw['teacher_change_along_descent'] < 0 and directional > 0 else
        'FIRST_ORDER_INWARD_FINITE_OUTWARD' if directional < 0 and candidate_mse > before_mse else
        'BOTH_ADAM_FIRST_ORDER_AND_FINITE_INWARD' if directional < 0 and candidate_mse < before_mse else
        'OTHER_OR_DEGENERATE_DIRECTION')
    result = dict(status='COMPLETE_SINGLE_ADAM_COUNTERFACTUAL', main_answer=classification,
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
        fixed_old_batch_kl=full.recorded_kl, optimizer_step_source=16, mathematical_candidate_step=17,
        full_loss=float(total.detach()), constant_value_actor_control_loss=float(constant_total.detach()),
        critic_buffer_changes=changed, critic_buffers_before_sha256=tree_digest(critic_buffers_before),
        critic_buffers_after_sha256=tree_digest(critic_buffers_after),
        actor_before_sha256=actor_before, actor_after_sha256=actor_after,
        inputs_before_sha256=input_before, inputs_after_sha256=input_after,
        saved_state_before_sha256=saved_before, saved_state_after_sha256=saved_after,
        critic_parameters_before_sha256=critic_parameters_before, critic_parameters_after_sha256=critic_parameters_after,
        optimizer_steps=0, new_rollouts=0, new_deployable_checkpoints=0,
        optimizer_constructed=False, candidate_is_fused_bitwise_replay=False,
        historical_batch_17_19_reconstructed=False, baseline_improvement_proven=False, backup_status='LOCAL_ONLY')
    atomic_json(out / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'main_answer', 'counters', 'candidate_teacher_rmse', 'candidate_guard_passed')}), flush=True)


def probe(job, root, out, metadata_path):
    budget = Budget()
    try:
        execute(job, root, out, metadata_path, budget)
    finally:
        atomic_json(out / 'progress.json', dict(counters=budget.counts, at=time.time(),
                                               result_exists=(out / 'result.json').exists()))
