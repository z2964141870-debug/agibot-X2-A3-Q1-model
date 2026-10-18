"""Exact D03 route/config boundaries without importing tensor libraries."""

import copy

from script.a3.humanplus_finite_verification import recipe as original_recipe

PROBABILITIES = dict(g1=0., teleop=0., smpl=0., a3_fast=1.)
OVERRIDE = 'manager_env.commands.motion.encoder_sample_probs.g1=0.0'


def validate_backbone(backbone):
    if list(backbone['active_encoders']) != ['g1', 'a3_fast']:
        raise ValueError('D03 must retain both original encoder networks')
    probabilities = backbone['encoder_sample_probs']
    if isinstance(probabilities, str):
        if probabilities != '${manager_env.commands.motion.encoder_sample_probs}':
            raise ValueError('Backbone routing interpolation changed')
    elif dict(probabilities) != PROBABILITIES:
        raise ValueError('Backbone sampling differs from command routing')


def validate_config(config):
    motion = config['manager_env']['commands']['motion']
    if dict(motion['encoder_sample_probs']) != PROBABILITIES:
        raise ValueError('D03 requires native a3_fast-only reference sampling')
    if (motion['coactivate_g1_when_a3_fast'] is not True
            or motion.get('optimize_encoders_ratio_for_CHIP', False)):
        raise ValueError('Native G1 pairing must remain enabled')
    validate_backbone(config['algo']['config']['actor']['backbone'])


def recipe(new, old):
    validate_config(new)
    previous = old['manager_env']['commands']['motion']['encoder_sample_probs']
    if dict(previous) != dict(PROBABILITIES, g1=1.):
        raise ValueError('D03 comparison must use the original mixed sampling recipe')
    normalized = copy.deepcopy(new)
    normalized['manager_env']['commands']['motion']['encoder_sample_probs']['g1'] = 1.
    probabilities = normalized['algo']['config']['actor']['backbone']['encoder_sample_probs']
    if not isinstance(probabilities, str):
        probabilities['g1'] = 1.
    original_recipe(normalized, old)


def validate_routes(routes, count):
    if len(routes) != count or not 1 <= count <= 8:
        raise ValueError('Unexpected number of registered rollouts')
    for i, row in enumerate(routes):
        if (row['update'] != i + 1 or row['trainer_global_step'] != i
                or row['transitions'] != 384 or row['routes'] != {'a3_fast': 384}):
            raise ValueError('Rollout is not entirely native a3_fast with original budget')
    return dict(rollouts=count, a3_fast_transitions=384 * count, g1_exclusive_transitions=0,
                native_pair_route=[1, 0, 0, 1], deployment_identity_gating=False)
