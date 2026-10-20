"""E55 training obs hooks (v1.2 native-640): the anchor term stays VENDOR-EXACT
(no geometry change, g1 builds 640); the func only stashes the arm channels on a
module cell for the residual head. 035 untouched - no interleave anywhere."""
import torch
import e54_obs_terms as E54

CHANNEL_CELL = {"vec": None}


def e55_root_channels_a0(env, command_name="motion", non_flatten=True):
    from gear_sonic.envs.manager_env.mdp import observations as vendor_obs
    base = vendor_obs.motion_anchor_ori_b(env, command_name=command_name, non_flatten=non_flatten)
    command = E54._tracking_command(env)
    CHANNEL_CELL["vec"] = E54._raw_channels(command, env)
    env._e55_channels = CHANNEL_CELL["vec"]
    return base


def e55_root_channels_p(env, command_name="motion", non_flatten=True):
    from gear_sonic.envs.manager_env.mdp import observations as vendor_obs
    base = vendor_obs.motion_anchor_ori_b(env, command_name=command_name, non_flatten=non_flatten)
    CHANNEL_CELL["vec"] = E54.e54_root_channels_standardized(env)
    env._e55_channels = CHANNEL_CELL["vec"]
    return base


def e55_root_channels_q(env, command_name="motion", non_flatten=True):
    from gear_sonic.envs.manager_env.mdp import observations as vendor_obs
    base = vendor_obs.motion_anchor_ori_b(env, command_name=command_name, non_flatten=non_flatten)
    CHANNEL_CELL["vec"] = E54.e54_root_channels_errorform(env)
    env._e55_channels = CHANNEL_CELL["vec"]
    return base
