"""One original-vendor rollout without constructing a Trainer or Optimizer."""

import argparse
import copy
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf, open_dict

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
from gear_sonic.utils import average_meters
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.cloth_likelihood_probe import save_capture
from script.a3.fresh_batch_protocol import require_batch_contract
from script.a3.retention_direction_probe import tree_digest
from script.a3.zhy_likelihood_trainer import snapshot


def launch_environment(config, out):
    # Importing the vendor entry registers the same OmegaConf resolvers. Its
    # Hydra main/resume/trainer entry is never invoked.
    from gear_sonic.train_agent_trl import create_manager_env, launch_isaac_app_with_lock
    from gear_sonic.utils.common import seeding
    from gear_sonic.utils.obs_utils import get_group_term_obs_shape
    from isaaclab.app import AppLauncher

    if (config.seed != 0 or config.num_envs != 16 or not config.headless or
            config.multi_gpu or config.use_wandb):
        raise ValueError('Unexpected source simulator configuration')
    config = copy.deepcopy(config)
    with open_dict(config):
        config.experiment_dir = str(out / 'simulator')
        config.manager_env.config.experiment_dir = config.experiment_dir
        config.manager_env.config.save_rendering_dir = str(out / 'simulator/renderings')
    if any(config.manager_env.config.get(k, False) for k in
           ('enable_cameras', 'render_results', 'render_ego', 'overview_camera')):
        raise ValueError('Cameras are outside this diagnostic')
    if config.manager_env.config.get('reinit_dr_freq', 0) or config.get('schedule_dict'):
        raise ValueError('Unimplemented scheduled environment semantics')
    seeding(0)
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.num_envs, args.seed, args.env_spacing = 16, 0, config.manager_env.config.env_spacing
    args.output_dir = str(out / 'simulator/output')
    args.enable_cameras, args.headless = False, True
    args.multi_gpu = args.distributed = False
    args.device = 'cuda:0'
    args.kit_args = '--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error'
    app = launch_isaac_app_with_lock(AppLauncher, args, global_rank=0, local_rank=0,
                                   timeout_seconds=60).app
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    env = create_manager_env(config, 'cuda:0', args)
    if env.action_transform_module is not None or env.motion_command is None:
        raise ValueError('Unexpected additional policy or missing motion command')
    for key, group in (('actor_obs', 'policy'), ('critic_obs', 'critic')):
        dim = env.env.observation_space[group].shape[-1]
        env.config.obs.obs_dims[key] = dim
        env.config.robot.algo_obs_dim_dict[key] = dim
    example = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ('policy', 'critic'):
            dims, names, total = get_group_term_obs_shape(example, key)
            env.config.obs.group_obs_dims[key] = dims
            env.config.obs.group_obs_names[key] = names
            env.config.obs.obs_dims[key] = total
            env.config.robot.algo_obs_dim_dict[key] = total
    env.config.robot.actions_dim = env.env.action_space.shape[-1]
    return app, env


def environment_state(env):
    data = env.env.scene['robot'].data
    state = dict(root_state_w=data.root_state_w, joint_pos=data.joint_pos,
                 joint_vel=data.joint_vel, episode_length_buf=env.env.episode_length_buf,
                 motion_ids=env.motion_command.motion_ids)
    for name in ('time_steps', 'motion_times', 'episode_encoder_index', 'motion_start_times'):
        value = getattr(env.motion_command, name, None)
        if isinstance(value, torch.Tensor):
            state[name] = value
    return snapshot(state)


def install_call_guards(actor, critic, budget):
    counts = dict(actor=0, critic=0)

    def wrap(owner, name, key, budget_key):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            budget.take(budget_key)
            counts[key] += 1
            return original(*args, **kwargs)

        setattr(owner, name, call)

    wrap(actor, 'forward', 'actor', 'actor_forwards')
    wrap(actor.actor_module, 'forward_with_external_tokens', 'actor', 'actor_forwards')
    wrap(critic, 'evaluate', 'critic', 'critic_forwards')
    return counts


class Collector(TRLPPOTrainer):
    """Only initialize attributes used by original storage/rollout/GAE methods."""

    def __init__(self, args, config, env, wrapper, budget):
        self.args, self.config, self.env, self.model = copy.deepcopy(args), config, env, wrapper
        self.policy_model, self.value_model = wrapper.policy, wrapper.value_model
        self.accelerator = SimpleNamespace(device=torch.device('cuda'), num_processes=1,
            process_index=0, unwrap_model=lambda model: model, reduce=lambda x, reduction: x)
        self.state = SimpleNamespace(global_step=4)
        self.episode_env_tensors = average_meters.TensorAverageMeterDict()
        self.learn_normalized_actions = self.policy_model.has_normalized_actions
        self.use_symmetry = False
        self.train_with_evaluating_env = config.get('train_with_evaluating_env', False)
        self.camera_resolution = None
        self.num_critics = env.config.rewards.get('num_critics', 1)
        if (env.config.get('use_symmetry', False) or self.train_with_evaluating_env or
                self.learn_normalized_actions or self.num_critics != 1):
            raise ValueError('Unregistered collector settings')
        self._init_config()
        require_batch_contract(env.num_envs, self.num_steps_per_env,
                               args.per_device_train_batch_size, args.local_batch_size)
        if args.local_mini_batch_size != 4 or args.gradient_accumulation_steps != 1:
            raise ValueError('First vendor minibatch is not four environments')
        self._setup_storage()
        self.budget = budget
        self.tokens = []

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        self.budget.take('environment_steps')
        result = super().policy_step(policy_model, obs_dict, cur_dones)
        token = policy_model.actor_module._last_full_latent_flat
        if token.shape != (16, 1, 64):
            raise ValueError('Actual sampled token ABI differs')
        self.tokens.append(token.detach().clone().reshape(16, 1, 2, 32))
        return result


def collect(env, wrapper, payload, algo_config, budget, out, expected_buffers):
    collector = Collector(payload['args'], algo_config, env, wrapper, budget)
    budget.take('fresh_rollouts')
    obs = {k: v.to('cuda') for k, v in env.reset_all().items()}
    initial_state = environment_state(env)
    save_capture(initial_state, out / 'initial_environment.pt')
    initial_obs = snapshot(obs)
    actor_before = tree_digest(wrapper.policy.state_dict())
    buffers_before = tree_digest(dict(wrapper.value_model.named_buffers()))
    with torch.no_grad():
        final_obs = collector._rollout_step(wrapper, obs)
    if len(collector.tokens) != 24 or collector.storage.step != 24:
        raise ValueError('Incomplete or repeated rollout')
    rollout = collector._get_rollout_data(final_obs.keys())
    tokens = torch.cat(collector.tokens, dim=1)
    collector._train_mode()
    permutation = torch.randperm(payload['args'].local_batch_size, device='cuda')
    batch = collector._get_mb_rollout_data(rollout, permutation[:4])
    batch['mb_sampled_tokens'] = tokens[permutation[:4]]
    final_state = environment_state(env)
    artifact = out / 'fresh_rollout.pt'
    save_capture(dict(initial_obs=initial_obs, final_obs=snapshot(final_obs),
        initial_environment=initial_state, final_environment=final_state,
        rollout=snapshot(rollout), sampled_tokens=snapshot(tokens), batch=snapshot(batch),
        permutation=snapshot(permutation), time_outs=snapshot(collector.storage.time_outs)), artifact)
    buffers_after = tree_digest(dict(wrapper.value_model.named_buffers()))
    if actor_before != tree_digest(wrapper.policy.state_dict()) or buffers_before != buffers_after:
        raise ValueError('Rollout changed protected Actor or eval Critic buffers')
    # An explicit restore makes the comparison independent of collection mode.
    live_buffers = dict(wrapper.value_model.named_buffers())
    with torch.no_grad():
        for name, value in expected_buffers.items():
            live_buffers[name].copy_(value)
    if tree_digest(live_buffers) != tree_digest(expected_buffers):
        raise ValueError('Critic buffers not restored to saved E47 starting state')
    record = dict(fresh_rollouts=1, environment_steps=24, transitions=384,
        environment_indices=permutation[:4].tolist(), permutation=permutation.tolist(),
        initial_environment_sha256=tree_digest(initial_state),
        final_environment_sha256=tree_digest(final_state),
        critic_collection_mode='EVAL_ORIGINAL_VENDOR_TRAIN_ROLLOUT',
        critic_buffers_before_sha256=buffers_before, critic_buffers_after_sha256=buffers_after,
        critic_restored_before_train_forward=True, actor_unchanged=True,
        artifact_sha256=sha256(artifact), artifact_bytes=artifact.stat().st_size,
        historical_rollout_reconstruction=False, optimizer_constructed=False)
    atomic_json(out / 'capture.json', record)
    return batch
