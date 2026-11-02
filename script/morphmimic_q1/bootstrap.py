"""Install explicit adapters before importing the pinned v0.1 Trainer.

This changes Python bindings in this process only. The upstream files and assets
are never edited. The new state269/context62/critic297 contracts are checkpointed
only by the new entry point and are incompatible with published v0.1 policies.
"""

import dataclasses
from functools import cache
import sys

import torch

from assets import presence, q1_spec
from canonical import morphology_with_presence

_INSTALLED = False


def install(robot_name):
    global _INSTALLED
    if _INSTALLED or robot_name not in ("g1", "q1"):
        raise RuntimeError("Install once, selecting g1 or q1 before Trainer imports")
    if "morphmimic.train.trainer" in sys.modules or "morphmimic.model.state" in sys.modules:
        raise RuntimeError("Install the native contracts before importing state or Trainer")
    import morphmimic.constants as constants
    import morphmimic.robots as robots
    old_get = robots.get_robot
    robots.ROBOTS, robots.VARIANTS = ("g1", "q1"), ()

    @cache
    def get_robot(name):
        return q1_spec() if name == "q1" else old_get(name)

    robots.get_robot = get_robot
    constants.STATE_DIM = 269
    constants.CTX_DIM = 62
    import morphmimic.model.state as state_module
    state_module.STATE_DIM = 240
    old_split, old_join = state_module.split_state, state_module.join_state

    def split(state):
        if state.shape[-1] not in (240, 269):
            raise ValueError("Expected physical240 or memory-augmented269 state")
        return old_split(state[..., :240])

    def join(parts):
        physical = old_join(parts)
        return torch.cat((physical, torch.zeros_like(physical[..., :29])), -1)

    state_module.split_state, state_module.join_state = split, join
    spec = get_robot(robot_name)

    import morphmimic.robots.morphology as morphology
    geometry = morphology._geometry
    actuator_terms = morphology._actuator_terms

    @cache
    def morphology_population():
        terms, shapes, masks = [], [], []
        for name in ("g1", "q1"):
            current = get_robot(name)
            if name == "q1":
                from control import parameters
                kp, kd, peak, _, _ = parameters(current.joint_names)
                tr = current.translator
                values = torch.stack((tr.to_canonical_delta(torch.full((22,), .5)),
                    tr.to_canonical_delta(kp), tr.to_canonical_delta(kd),
                    tr.to_canonical_delta(peak)))
            else:
                raw = actuator_terms(current)
                values = torch.stack([torch.as_tensor(raw[k])
                    for k in ("action_scale", "kp", "kd", "effort")])
            terms.append(values.double())
            shapes.append(torch.as_tensor(geometry(current)).double())
            masks.append(presence(current, values))
        vectors, bounds = morphology_with_presence(torch.stack(terms), torch.stack(shapes),
                                                    torch.stack(masks))
        return vectors.float(), bounds, torch.stack(terms), torch.stack(shapes), torch.stack(masks)

    def morphology_vector(name, actor=False, actor_geometry=True):
        index = ("g1", "q1").index(name)
        vec = morphology_population()[0][index]
        return torch.cat((vec[:116], vec[125:])) if actor and not actor_geometry else vec.clone()

    morphology.morphology_vector = morphology_vector
    morphology.MORPH_DIM = 154

    import morphmimic.model.world_model as wm_module
    wm_module._CTX_SCALE = tuple(wm_module._CTX_SCALE) + (50.,) * 29
    from model_adapter import make_world_model
    wm_module.WorldModel = make_world_model(spec)
    import morphmimic.model.policy as policy_module
    policy_module.LOCAL_DIM = 297
    old_decode = policy_module.Decoder.forward

    def decode(self, z, obs_hist_flat, morph=None):
        action = old_decode(self, z, obs_hist_flat, morph)
        morph = self.morph_actor if morph is None else morph
        if morph.shape[-1] not in (145, 154):
            raise ValueError("Native decoder requires morphology and presence mask")
        return torch.where(morph[..., -29:] > .5, action, torch.zeros_like(action))

    policy_module.Decoder.forward = decode

    from native_action import NativeActionCfg
    import morphmimic.env.env_cfg as env_module
    original_cfg = env_module.morphmimic_env_cfg

    def env_cfg(robot, *args, **kwargs):
        delay = kwargs.get("actuator_delay", True)
        max_delay = kwargs.get("actuator_delay_max_s", .02) if delay else 0.
        if max_delay < 0 or max_delay > .02 or not kwargs.get("actuator_delay_fixed", True):
            raise ValueError("Native memory supports episode-fixed delays at most20ms")
        kwargs["actuator_delay"] = False
        cfg = original_cfg(robot, *args, **kwargs)
        dt = .002 if robot == "q1" else .005
        cfg.sim.mujoco.timestep = dt
        cfg.decimation = round(.02 / dt)
        action = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = NativeActionCfg(
            **{field.name: getattr(action, field.name) for field in dataclasses.fields(action)},
            physics_dt=dt, max_delay_s=0. if kwargs.get("eval", False) else max_delay)
        scale = get_robot(robot).height_scale
        for key in ("motion_global_anchor_pos", "motion_relative_body_pos"):
            if key in cfg.rewards:
                cfg.rewards[key].params["std"] *= scale
        for key in ("motion_body_lin_vel",):
            if key in cfg.rewards:
                cfg.rewards[key].params["std"] *= scale
        # Resolve by function names as upstream tracking dictionaries may rename keys.
        for term in cfg.rewards.values():
            name = term.func.__name__
            if name in ("motion_global_anchor_position_error_exp",
                        "motion_relative_body_position_error_exp", "motion_body_linear_velocity_error_exp"):
                term.params["std"] = .3 * scale if "position" in name else 1. * scale
        for term in cfg.terminations.values():
            if term.func.__name__ == "bad_anchor_pos_z_only":
                term.params["threshold"] = .25 * scale
        return cfg

    env_module.morphmimic_env_cfg = env_cfg
    import morphmimic.env.observations as observations
    original_context = observations.sim_context

    class NativeContext(original_context):
        def __call__(self, env, *args, **kwargs):
            base = super().__call__(env, *args, **kwargs)
            return torch.cat((base, env.action_manager.get_term("joint_pos").delay_seconds()), -1)

    observations.sim_context = NativeContext

    import morphmimic.env.morphmimic_env as env_class_module
    original_env = env_class_module.MorphMimicEnv

    class NativeEnv(original_env):
        def project_observations(self, obs):
            term = self.action_manager.get_term("joint_pos")
            for key in ("policy", "policy_clean"):
                x = obs[key]
                obs[key] = torch.cat((term.project(x[..., :29]), term.project(x[..., 29:58]),
                                      x[..., 58:64], term.project(x[..., 64:93])), -1)
            obs["state"] = torch.cat((obs["state"][..., :240], term.executed.clone()), -1)
            return obs

        def step(self, action):
            term = self.action_manager.get_term("joint_pos")
            obs, *rest = super().step(term.project(action))
            return self.project_observations(obs), *rest

        def reset(self, *args, **kwargs):
            obs, extras = super().reset(*args, **kwargs)
            return self.project_observations(obs), extras

    env_class_module.MorphMimicEnv = NativeEnv

    import morphmimic.train.replay as replay_module
    original_add = replay_module.ReplayBuffer.add_step

    def add(self, *args, **kwargs):
        if args:
            raise TypeError("Native replay writes use named fields for masking")
        action = kwargs["action"]
        mask = presence(spec, action)
        kwargs["action"] = torch.where(mask, action, torch.zeros_like(action))
        return original_add(self, **kwargs)

    replay_module.ReplayBuffer.add_step = add
    import morphmimic.train.imagination as imagination
    original_noise = imagination.noise_half_widths

    def noise_widths(noise, device):
        half = original_noise(noise, device)
        mask = presence(spec, half)
        half[:29] = torch.where(mask, half[:29], torch.zeros_like(half[:29]))
        half[29:58] = torch.where(mask, half[29:58], torch.zeros_like(half[29:58]))
        return half

    imagination.noise_half_widths = noise_widths
    # Package task registration imports some aliases before install can replace them.
    rebound = []
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith("morphmimic.") and module is not None:
            for key, previous, replacement in (("get_robot", old_get, get_robot),
                    ("split_state", old_split, split), ("join_state", old_join, join)):
                if getattr(module, key, None) is previous:
                    setattr(module, key, replacement)
                    rebound.append(module_name + "." + key)
    _INSTALLED = True
    return dict(robot=spec, morphology_population=morphology_population, rebound_aliases=rebound,
                contracts=dict(state=269, physical_state=240, local_features=268,
                    critic_features=297, context=62, proprio=93, morphology=154))
