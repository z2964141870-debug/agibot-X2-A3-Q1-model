"""Canonical commands with explicit episode-fixed delay and real-joint masks."""

from dataclasses import dataclass
import re

import torch
from mjlab.envs.mdp.actions import JointPositionAction
from morphmimic.env.actions import CanonicalJointPositionAction, CanonicalJointPositionActionCfg
from morphmimic.robots import get_robot

from assets import presence


class NativeCanonicalAction(CanonicalJointPositionAction):
    def __init__(self, cfg, env):
        JointPositionAction.__init__(self, cfg, env)
        self.spec = get_robot(cfg.robot_name)
        self.tr = self.spec.translator
        if self._target_ids.tolist() != list(range(self.tr.num_native)):
            raise ValueError("Action targets must use complete native entity order")
        self._action_dim = 29
        self._raw_actions = torch.zeros(self.num_envs, 29, device=self.device)
        self.executed = torch.zeros_like(self._raw_actions)
        self.present = presence(self.spec, self.executed)
        self.noise_std = cfg.noise_std
        self.noise_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, self.tr.num_native,
                                      device=self.device, dtype=torch.long)
        self.delay_groups = []
        for actuator in self.spec.entity_cfg_fn().articulation.actuators:
            indices = [i for i, name in enumerate(self.spec.joint_names)
                       if any(re.fullmatch(expr, name) for expr in actuator.target_names_expr)]
            self.delay_groups.append(indices)
        if sorted(i for group in self.delay_groups for i in group) != list(range(self.tr.num_native)):
            raise ValueError("Delay groups must cover every real actuator once")
        self._previous_target = self._offset.clone()
        self._processed_actions = self._offset.clone()
        self._substep = 0

    def project(self, action):
        return torch.where(self.present, action, torch.zeros_like(action))

    def process_actions(self, actions):
        self._previous_target.copy_(self._processed_actions)
        self._raw_actions.copy_(self.project(actions))
        executed = actions
        if self.noise_std:
            executed = actions + self.noise_std * self.noise_scale * torch.randn_like(actions)
        self.executed.copy_(self.project(executed))
        native = self.tr.to_native_delta(self.executed)
        self._processed_actions = self._offset + native * self._scale
        if self.cfg.clip is not None:
            raise ValueError("Delay memory requires unclipped canonical commands")
        self._substep = 0

    def apply_actions(self):
        target = torch.where(self._substep >= self.delay_steps,
                             self._processed_actions, self._previous_target)
        target = target - self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
        self._substep += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(env_ids)
        self.executed[ids] = 0.
        self._processed_actions[ids] = self._offset[ids]
        self._previous_target[ids] = self._offset[ids]
        shape = (self.delay_steps[ids].shape[0], 1)
        max_lag = round(self.cfg.max_delay_s / self.cfg.physics_dt)
        if self.cfg.robot_name == "q1":
            # Primebot draws continuous seconds, then rounds to physics steps.
            lag = (torch.rand(shape, device=self.device) * max_lag).round().long()
            self.delay_steps[ids] = lag
        else:
            lag = torch.zeros_like(self.delay_steps[ids])
            for group in self.delay_groups:
                lag[:, group] = torch.randint(max_lag + 1, shape, device=self.device)
            self.delay_steps[ids] = lag

    def delay_seconds(self):
        native = self.delay_steps * self.cfg.physics_dt
        # Delays do not change sign with the joint coordinate convention.
        pairs = self.tr.perm
        out = native.new_zeros(self.num_envs, 29)
        for slot, index in enumerate(pairs):
            if index is not None:
                out[:, slot] = native[:, index]
        return out


@dataclass(kw_only=True)
class NativeActionCfg(CanonicalJointPositionActionCfg):
    physics_dt: float
    max_delay_s: float

    def build(self, env):
        return NativeCanonicalAction(self, env)
