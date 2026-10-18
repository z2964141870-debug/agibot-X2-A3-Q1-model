"""Bounded zzx expert contract, separate from retired unified-policy trials."""

UPDATES = 256
ENVS = 16
STEPS = 24
MINIBATCHES = 4
INTERACTIONS = UPDATES * ENVS * STEPS
OPTIMIZER_STEPS = UPDATES * MINIBATCHES
CHECKPOINTS = (0, UPDATES)
DECODER = 'actor_module.decoders.g1_dyn.'
ROUTE = [1, 0, 0, 1]
# One rollout Actor, one native training Actor and one cached-token decoder.
OUTER_CALLS = UPDATES * (STEPS + 2 * MINIBATCHES)


def ownership(wrapper, optimizer):
    named = dict(wrapper.named_parameters())
    actor = dict(wrapper.policy.named_parameters())
    critic = dict(wrapper.value_model.named_parameters())
    allowed = {n for n in actor if n.startswith(DECODER)}
    if not allowed or {n for n, p in actor.items() if p.requires_grad} != allowed:
        raise ValueError('Only the complete original decoder may be trainable')
    if not critic or any(not p.requires_grad for p in critic.values()):
        raise ValueError('All original Critic parameters required')
    expected = {'policy.' + n for n in allowed} | {'value_model.' + n for n in critic}
    by_id = {id(p): n for n, p in named.items()}
    if len(by_id) != len(named) or set(named) != {'policy.' + n for n in actor} | {
            'value_model.' + n for n in critic}:
        raise ValueError('Unexpected or aliased wrapper parameter')
    groups = [[by_id.get(id(p)) for p in g['params']] for g in optimizer.param_groups]
    flat = [n for g in groups for n in g]
    if len(flat) != len(set(flat)) or set(flat) != expected:
        raise ValueError('Missing, repeated or foreign optimizer parameter')
    return dict(groups=groups, shapes={n: list(named[n].shape) for n in sorted(expected)},
                actor_keys=sorted(allowed), critic_keys=sorted(critic))


def serialized_owners(groups, contract):
    if len(groups) != len(contract['groups']):
        raise ValueError('Optimizer group count changed')
    result = {}
    for group, names in zip(groups, contract['groups']):
        if len(group['params']) != len(names):
            raise ValueError('Optimizer group size changed')
        for identifier, name in zip(group['params'], names):
            if type(identifier) is not int or identifier in result:
                raise ValueError('Invalid or duplicate optimizer ID')
            result[identifier] = name
    return result


def validate_config(config, motionlib):
    algo = config['algo']['config']
    motion = config['manager_env']['commands']['motion']
    if (config['num_envs'] != ENVS or config['seed'] != 0
            or algo['num_learning_iterations'] != UPDATES
            or algo['num_steps_per_env'] != STEPS
            or algo['num_learning_epochs'] != 1
            or algo['num_mini_batches'] != MINIBATCHES):
        raise ValueError('Specialist budget or seed changed')
    if list(motion['motion_lib_cfg']['motion_file']) != [str(motionlib)]:
        raise ValueError('Only the registered complete zzx motion is allowed')
    if motion['motion_lib_cfg']['adaptive_sampling']['enable'] is not True:
        raise ValueError('Existing adaptive sampling must remain enabled')
    if (dict(motion['encoder_sample_probs']) != dict(g1=0., teleop=0., smpl=0., a3_fast=1.)
            or motion['coactivate_g1_when_a3_fast'] is not True
            or list(algo['actor']['backbone']['active_encoders']) != ['g1', 'a3_fast']):
        raise ValueError('Native paired a3_fast route changed')
    if any(config['algo']['trl'].get(k, False) for k in ('bf16', 'fp16', 'tf32')):
        raise ValueError('FP32 required')


def outcome(before, after):
    keys = ('joint_rmse_rad', 'root_pos_error_m_mean', 'wrists_world_mean_m')
    ratios = {key: after['windows']['full_rollout'][key] /
              before['windows']['full_rollout'][key] - 1 for key in keys}
    gates = dict(full_motion_no_fall=not after['fall'],
                 joint_improvement=ratios['joint_rmse_rad'] <= -.01,
                 root_preservation=ratios['root_pos_error_m_mean'] <= .05,
                 wrist_preservation=ratios['wrists_world_mean_m'] <= .05)
    return dict(relative_change=ratios, gates=gates,
                specialist_mujoco_gate_passed=all(gates.values()),
                unified_gate_passed=False, deploy_allowed=False,
                reliable_on_student_visited_states_not_yet_tested=True)


def require_training_receipt(receipt):
    expected = dict(updates=UPDATES, interactions=INTERACTIONS,
                    optimizer_steps=OPTIMIZER_STEPS, rollout_actor_calls=UPDATES * STEPS,
                    native_training_actor_calls=OPTIMIZER_STEPS,
                    cached_decoder_calls=OPTIMIZER_STEPS)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Incomplete specialist budget or forward accounting')
    if receipt.get('teacher_retention_calls') != 0:
        raise ValueError('Expert must not inherit unified teacher restrictions')
    if receipt.get('initial_probability', {}).get('max_abs_ratio_minus_one', 1) > .001:
        raise ValueError('Actual sampled likelihood mismatch')
    if receipt.get('all_actor_buffers_unchanged') is not True:
        raise ValueError('Actor buffer drift')
    return expected
