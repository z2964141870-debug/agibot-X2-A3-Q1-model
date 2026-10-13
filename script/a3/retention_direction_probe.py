"""E46 fixed-input Actor partial derivatives; no optimizer or environment exists."""

import gc
import hashlib
import json
import math
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from gear_sonic.trl.utils.common import custom_instantiate
from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import cached_distribution
from script.a3.cloth_constrained_trainer import initial_probability, restrict_policy
from script.a3.retention_direction_protocol import Budget, direction


class ActorLossHarness(TRLAuxLossPPOTrainer):
    """Call unchanged vendor loss methods without trainer/env/optimizer setup."""

    def __init__(self, args, config):
        self.args, self.config, self.optimizer = args, config, None
        self.accelerator = SimpleNamespace(device=torch.device('cuda'),
                                          reduce=lambda x, reduction: x)
        self.entropy_coef = config.entropy_coef
        self.multi_critic_advantage_weights = config.get('multi_critic_advantage_weights', None)
        self.use_symmetry = False
        self.compute_imgaug_bc_loss = config.get('compute_imgaug_bc_loss', False)
        self.enable_loss_finite_diagnostics = True
        self.compute_aux_loss = config.get('compute_aux_loss', True)
        self.aux_loss_scale = config.get('aux_loss_scale', 1.)
        self.recorded_kl = []
        if self.compute_imgaug_bc_loss or not self.compute_aux_loss:
            raise ValueError('Unregistered loss configuration')

    def _adjust_learning_rate_based_on_kl(self, kl, optimizer):
        if optimizer is not None:
            raise ValueError('Diagnostic must not own an optimizer')
        self.recorded_kl.append(float(kl))  # Recorded only: no adaptive LR side effect.


def tree_digest(value):
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict')
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        elif item is None or isinstance(item, (bool, int, float, str)):
            result.update(repr((type(item).__name__, item)).encode())
        else:
            raise TypeError('Unsupported state digest type: ' + type(item).__name__)

    visit(value)
    return result.hexdigest()


def cuda_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cuda')
    if isinstance(value, dict):
        return {key: cuda_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cuda_tree(item) for item in value)
    return value


def geometry(gradients):
    norms = {name: math.sqrt(sum(float(g.double().square().sum())
             for g in values if g is not None)) for name, values in gradients.items()}
    rows = {}
    for name, values in gradients.items():
        dot = sum(float((g.double() * t.double()).sum()) for g, t in
                  zip(values, gradients['teacher384']) if g is not None and t is not None)
        rows[name] = direction(dot, norms[name], norms['teacher384'])
    return dict(norms=norms, versus_full_teacher=rows,
                nonzero_tensors={name: sum(bool(torch.count_nonzero(g)) for g in values
                    if g is not None) for name, values in gradients.items()})


def gradient_sum_check(gradients, parameters, job):
    squared_error = squared_total = max_error = 0.
    for index, parameter in enumerate(parameters):
        expected = torch.zeros_like(parameter)
        for name in ('ppo_policy', 'aux', 'teacher96'):
            value = gradients[name][index]
            if value is not None:
                expected = expected + value
        actual = gradients['total'][index]
        if actual is None:
            actual = torch.zeros_like(parameter)
        error = (expected - actual).double()
        squared_error += float(error.square().sum())
        squared_total += float(actual.double().square().sum())
        max_error = max(max_error, float(error.abs().max()))
        if not torch.allclose(expected, actual, atol=job['gradient_sum_atol'],
                              rtol=job['gradient_sum_rtol']):
            raise ValueError('Component gradients do not reconstruct vendor Actor total')
    return dict(passed=True, max_abs=max_error, error_l2=math.sqrt(squared_error),
                total_l2=math.sqrt(squared_total))


def one_state(step, root, meta, algo, batch, teacher_obs, targets, budget, job):
    budget.take('policy_states')
    path = root / ('data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/'
                   f'milestone_{step}/model_step_{step:06d}.pt')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if int(payload['state'].global_step) != step:
        raise ValueError('Saved policy step mismatch')
    optimizer_before = tree_digest(payload['optimizer_state_dict'])
    actor = custom_instantiate(meta.algo_config.actor, env_config=meta.env_config,
                              algo_config=meta.algo_config, _resolve=False)
    actor.load_state_dict(payload['policy_state_dict'], strict=True)
    actor.to('cuda').train()
    scope = restrict_policy(actor)
    if actor.running_mean_std is not None:
        raise ValueError('Actor normalization would mutate state')
    if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p != 0
           for m in actor.modules()):
        raise ValueError('Stochastic dropout invalidates the fixed-input contract')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in actor.parameters()):
        raise ValueError('Requires original FP32 contract')
    actor_before = tree_digest(actor.state_dict())
    inputs_before = tree_digest((batch, teacher_obs, targets))
    named = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]

    # Match training's auxiliary forward, then replace only the distribution with
    # the actual saved sampled tokens. Both calls count against the Actor budget.
    budget.take('actor_forwards')
    actor.forward(batch['mb_obs_dict'], is_training=True,
                  episode_attnmask=batch['episode_attnmask'])
    auxiliary, coefficients = actor.aux_losses, actor.aux_loss_coef
    budget.take('actor_forwards')
    policy = cached_distribution(actor, batch['mb_obs_dict'],
                                 batch['mb_sampled_tokens'], batch['mb_actions'])
    policy.update(aux_losses=auxiliary, aux_loss_coef=coefficients)
    probability = initial_probability(policy, batch) if step == 0 else None
    harness = ActorLossHarness(payload['args'], algo)
    losses = harness._compute_loss(dict(policy_results=policy,
        value_results=batch['mb_values'].detach()), batch)
    ppo, aux = losses['ppo_loss_dict'], losses['aux_loss_dict']
    coefficient = algo.get('ppo_loss_coef', 1.)
    detached_terms = dict(value_constant=ppo['vf_loss'] * harness.args.vf_coef * coefficient,
                         entropy=ppo['entropy_loss'] * harness.entropy_coef * coefficient)
    if any(t.requires_grad for t in detached_terms.values()):
        raise ValueError('Value/entropy unexpectedly connected to Actor parameters')

    teacher_losses, teacher_maxima = [], []
    for start in range(0, 16, 4):
        budget.take('actor_forwards')
        mean = actor.forward({name: value[start:start+4] for name, value in teacher_obs.items()},
                             is_training=False)
        delta = mean - targets[start:start+4]
        teacher_losses.append(delta.square().mean())
        teacher_maxima.append(float(delta.detach().abs().max()))
    full_teacher = torch.stack(teacher_losses).mean()
    if step == 0 and max(teacher_maxima) > 1e-5:
        raise ValueError('Initial teacher target ABI mismatch')
    teacher_rmse = math.sqrt(float(full_teacher.detach()))
    saved_teacher_rmse = None
    if step == 4:
        stopped = json.loads((root / 'data/manifests/a3_zhy_cloth_failure_stop_20261013_E45.json').read_text())
        saved_teacher_rmse = next(r['teacher_action_rmse'] for r in stopped['optimizer_steps']
                                  if r['optimizer_step'] == 16)
        if abs(teacher_rmse - saved_teacher_rmse) > job['saved_teacher_rmse_atol']:
            raise ValueError('Saved step4 teacher metric does not match original step16 receipt')
    terms = dict(ppo_policy=ppo['pg_loss'] * coefficient, aux=aux['total_aux_loss'],
                 teacher96=teacher_losses[0], teacher384=full_teacher,
                 total=losses['loss'] + teacher_losses[0])
    reconstructed = (terms['ppo_policy'] + sum(detached_terms.values()) +
                     terms['aux'] + terms['teacher96'])
    if not torch.allclose(reconstructed, terms['total'], atol=1e-7, rtol=1e-5):
        raise ValueError('Loss decomposition does not reconstruct vendor objective')
    gradients = {}
    for name, term in terms.items():
        if not bool(torch.isfinite(term)):
            raise ValueError('Nonfinite objective term: ' + name)
        if term.requires_grad:
            budget.take('autograd_grad_calls')
            gradients[name] = torch.autograd.grad(term, parameters, retain_graph=True,
                                                  allow_unused=True)
        else:
            gradients[name] = (None,) * len(parameters)
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients[name]):
            raise ValueError('Nonfinite gradient: ' + name)
    sum_check = gradient_sum_check(gradients, parameters, job)
    measured = geometry(gradients)
    torch.cuda.synchronize()
    actor_after = tree_digest(actor.state_dict())
    optimizer_after = tree_digest(payload['optimizer_state_dict'])
    inputs_after = tree_digest((batch, teacher_obs, targets))
    if (actor_before != actor_after or optimizer_before != optimizer_after or
            inputs_before != inputs_after or any(p.grad is not None for p in actor.parameters())):
        raise ValueError('Diagnostic changed model/optimizer/input state or accumulated .grad')
    row = dict(step=step, checkpoint=str(path.relative_to(root)),
        teacher_action_rmse=teacher_rmse, original_saved_teacher_rmse=saved_teacher_rmse,
        teacher_action_max_abs=max(teacher_maxima), teacher_chunk_mse=[float(t.detach()) for t in teacher_losses],
        teacher96_environment_indices=[0, 1, 2, 3], initial_probability=probability,
        losses={name: float(value.detach()) for name, value in {**terms, **detached_terms}.items()},
        loss_sum_abs_error=float((reconstructed-terms['total']).detach().abs()),
        gradient_sum=sum_check, geometry=measured,
        auxiliary={name: dict(raw=float(value.detach()), coefficient=float(coefficients[name]),
            requires_grad=value.requires_grad) for name, value in auxiliary.items()},
        recorded_kl_without_lr_update=harness.recorded_kl,
        actor_state_before_sha256=actor_before, actor_state_after_sha256=actor_after,
        optimizer_before_sha256=optimizer_before, optimizer_after_sha256=optimizer_after,
        inputs_before_sha256=inputs_before, inputs_after_sha256=inputs_after,
        trainable_parameters=sum(p.numel() for p in parameters), trainable_names=scope['trainable'],
        gradients_accumulated=False, optimizer_constructed=False, critic_constructed=False)
    # No checkpoint, gradients, or new rollout are written.
    return row


def probe(job, root, out, metadata_path):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    meta = OmegaConf.create(json.loads(metadata_path.read_text()))
    config = OmegaConf.load(root / 'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/config.yaml')
    algo = OmegaConf.to_container(config.algo.config, resolve=True)
    if algo != OmegaConf.to_container(meta.algo_config, resolve=True):
        raise ValueError('Captured Actor/loss configuration differs from E45')
    algo = OmegaConf.create(algo)
    captured = torch.load(root / 'logs/a3_zhy_cloth_failure_20261013_E45/training/initial_token_repair.pt',
                          map_location='cpu', weights_only=False)
    batch = cuda_tree(captured['batch'])
    del captured
    rollout = torch.load(root / 'data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt',
                         map_location='cpu', weights_only=False)
    teacher_obs = cuda_tree(rollout['all_obs_dict'])
    del rollout
    target_payload = torch.load(root / 'data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt',
                                map_location='cpu', weights_only=False)
    targets = target_payload['initial']['mean'].to('cuda')
    del target_payload
    if targets.shape != (16, 24, 29) or batch['mb_actions'].shape != (4, 24, 29):
        raise ValueError('Fixed teacher/clothing batch ABI changed')
    budget, rows = Budget(), []
    started = time.time()
    try:
        for step in job['policy_steps']:
            row = one_state(step, root, meta, algo, batch, teacher_obs, targets, budget, job)
            rows.append(row)
            atomic_json(out / f'step{step}_direction.json', row)
            gc.collect()
    finally:
        atomic_json(out / 'progress.json', dict(completed_steps=[r['step'] for r in rows],
                                               counters=budget.counts, at=time.time()))
    decision = rows[1]['geometry']['versus_full_teacher']['ppo_policy']['interpretation']
    result = dict(status='COMPLETE_FIXED_INPUT_ACTOR_GRADIENT_DIAGNOSTIC',
        experiment=job['experiment'], started_at=started, ended_at=time.time(),
        budget=budget.LIMITS, counters=budget.counts, rows=rows, main_answer=decision,
        vendor_actor_config_matches_e45=True, optimizer_steps=0, new_rollouts=0,
        value_term=job['value_term'], lr_mutations=0, new_deployable_checkpoints=0,
        fixed_input_counterfactual_not_historical_failure=True,
        actual_adam_or_clipping_step_reconstructed=False, effect_improvement_proven=False,
        backup_status='LOCAL_ONLY')
    atomic_json(out / 'result.json', result)
    print(json.dumps(dict(status=result['status'], counters=budget.counts, main_answer=decision)), flush=True)
