"""Candidate Q1 input/reward adaptation over the preserved HP MDP.

The runtime retains the upstream Apache-2.0 LICENSE and NOTICE. This module
uses its analytic heads without changing the shared learner or controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .mdp_base import MAX_STEP_REWARD, NUM_J, XI_DIM
from .mdp_base import MDP as BaseMDP
from .mdp_base import RewardCfg as BaseRewardCfg
from .mdp_base import policy_input_spec as base_policy_input_spec
from .motion import CONTROL_DT

_OPTIONS = json.loads(Path(__file__).with_name("candidate_options.json").read_text())
_JOINT_ERROR = _OPTIONS["joint_error_observation"]
_TRACKING_REWARD = _OPTIONS["joint_tracking_reward"]
_ERROR_FIELD = "reference_minus_robot_joint_position"


def policy_input_spec(observation_mode="legacy"):
    spec = base_policy_input_spec(observation_mode)
    if _JOINT_ERROR:
        spec["schema_version"] = 2
        spec["obs_dim"] += NUM_J
        spec["observation_fields"].append(_ERROR_FIELD)
        spec["joint_error_noise"] = "negative_of_joint_position_measurement_noise"
    return spec


@dataclass
class RewardCfg(BaseRewardCfg):
    body_ori_std: float = 0.8
    upper_joint_std: float = 0.8
    lower_joint_std: float = 0.5


class MDP(BaseMDP):
    def __init__(self, kin, motion, default_q, cfg=None, *, observation_mode="legacy"):
        if cfg is None:
            cfg = RewardCfg() if _TRACKING_REWARD else BaseRewardCfg()
        super().__init__(kin, motion, default_q, cfg, observation_mode=observation_mode)
        self.base_obs_dim = self.obs_dim
        self.policy_input_spec = policy_input_spec(observation_mode)
        self.obs_dim = self.policy_input_spec["obs_dim"]
        if _JOINT_ERROR:
            self.obs_noise = torch.cat((self.obs_noise, torch.full_like(default_q, .01)))
            self.obs_scale = torch.cat((self.obs_scale, torch.ones_like(default_q)))

    def obs(self, state, previous_action, context, frames):
        original = super().obs(state, previous_action, context, frames)
        if not _JOINT_ERROR:
            return original
        return torch.cat((original, self.motion.q[frames] - state[4]), -1)

    def add_noise(self, obs):
        if not _JOINT_ERROR:
            return super().add_noise(obs)
        noise = (2*torch.rand_like(obs)-1)*self.obs_noise
        # Both fields refer to the same joint sensor measurement.
        noise[..., self.base_obs_dim:] = -noise[..., :NUM_J]
        return obs + noise

    def reward(self, state, frames, action, previous_action, terms=False):
        if not _TRACKING_REWARD:
            return super().reward(state, frames, action, previous_action, terms=terms)
        value, detail = super().reward(state, frames, action, previous_action, terms=True)
        error = state[4] - self.motion.q[frames]
        upper = torch.exp(-error[..., 14:].square().mean(-1)/self.cfg.upper_joint_std**2)
        lower = torch.exp(-error[..., :14].square().mean(-1)/self.cfg.lower_joint_std**2)
        velocity = (
            torch.exp(-detail["e_body_linvel"]/self.cfg.body_linvel_std**2)
            + torch.exp(-detail["e_body_angvel"]/self.cfg.body_angvel_std**2)
        )
        adjustment = (.5*upper + .5*lower - .5*velocity)*CONTROL_DT
        value = value + adjustment
        if terms:
            return value, {
                **detail,
                "track": detail["track"] + adjustment,
                "joint_upper": .5*upper*CONTROL_DT,
                "joint_lower": .5*lower*CONTROL_DT,
                "velocity_weight_removed": -.5*velocity*CONTROL_DT,
            }
        return value
