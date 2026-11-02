"""Masked physical dynamics with deterministic previous-command state."""

import torch
from torch import nn
from torch.nn import functional as F

import morphmimic.model.world_model as original
from morphmimic.model.nets import run_mlp
from morphmimic.model.rewards_torch import analytic_obs, soft_termination, termination_features
from morphmimic.model.state import canonicalize, decanonicalize, split_state

from assets import presence


def make_world_model(spec):
    class NativeWorldModel(original.WorldModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            first = self.dynamics_mlp[0]
            if not isinstance(first, nn.Linear):
                raise TypeError("Expected upstream dynamics first layer to be Linear")
            self.dynamics_mlp[0] = nn.Linear(first.in_features + 29, first.out_features)
            mask = presence(spec, self.joint_limits)
            self.register_buffer("joint_presence", mask)
            self.register_buffer("feature_presence", torch.cat(
                (mask, mask, torch.ones(210, dtype=torch.bool, device=mask.device))))
            self.height_scale = spec.height_scale
            self.reward_cfg = original.RewardCfg(
                std_pos=.3 * self.height_scale, std_lin_vel=1. * self.height_scale)

        def project(self, action):
            return torch.where(self.joint_presence, action, torch.zeros_like(action))

        def project_physical(self, state):
            return torch.cat((self.project(state[..., :29]),
                              self.project(state[..., 29:58]), state[..., 58:240]), -1)

        def local_features(self, state):
            physical = self.normalizer.normalize(canonicalize(state, self.frame(state)))
            physical = torch.where(self.feature_presence, physical, torch.zeros_like(physical))
            return torch.cat((physical, self.project(state[..., 240:269])), -1)

        def step(self, state, action, ctx):
            if state.shape[-1] != 269 or ctx.shape[-1] != 62:
                raise ValueError("Native dynamics requires state269/context62")
            frame = self.frame(state)
            local = canonicalize(state, frame)
            local = torch.where(self.feature_presence, local, torch.zeros_like(local))
            action = self.project(action)
            parts = [self.normalizer.normalize(local), self.frame_inputs(state, frame),
                     action, self.scale_ctx(ctx), self.project(state[..., 240:269])]
            if self.cfg.morph_dim:
                parts.append(self.morph.expand(*state.shape[:-1], -1))
            delta = run_mlp(self.dynamics_mlp, torch.cat(parts, -1), self.cfg.bf16)
            updated = torch.where(self.feature_presence, local + delta * self.delta_scale(),
                                  torch.zeros_like(local))
            physical = self.project_physical(decanonicalize(updated, frame))
            return torch.cat((physical, action), -1)

        def reward(self, next_state, ref, action, prev_action):
            return super().reward(next_state, ref, self.project(action), self.project(prev_action))

        def obs(self, state, encoder_bias, prev_action):
            obs = analytic_obs(state, self.project(encoder_bias), self.default_joint_pos,
                               self.project(prev_action), self.imu_idx)
            return torch.cat((self.project(obs[..., :29]), self.project(obs[..., 29:58]),
                              obs[..., 58:64], self.project(obs[..., 64:93])), -1)

        def termination_logit(self, next_state, ref):
            e_z, e_tilt = termination_features(next_state, ref)
            if self.cfg.termination == "soft":
                return torch.logit(soft_termination(next_state, ref,
                    z_thresh=.25 * self.height_scale, z_scale=.02 * self.height_scale), eps=1e-6)
            if self.hazard is not None:
                return self.hazard(e_z / self.height_scale, e_tilt)
            # The learned head was built for the original 268 physical features.
            x = torch.cat((self.local_features(next_state)[..., :268],
                           (e_z / self.height_scale).unsqueeze(-1), e_tilt.unsqueeze(-1)), -1)
            return run_mlp(self.term_mlp, x, self.cfg.bf16).squeeze(-1)

        def termination_prob(self, next_state, ref):
            return torch.sigmoid(self.termination_logit(next_state, ref))

        def fit_loss(self, batch, library):
            pred = self.rollout(batch.state[:, 0], batch.action, batch.ctx)
            logged = batch.state[:, 1:]
            frame = self.frame(logged)
            diff = canonicalize(pred, frame) - canonicalize(logged, frame)
            diff = torch.where(self.feature_presence, diff, torch.zeros_like(diff))
            state_loss = (diff / self.delta_scale()).square().sum(-1).mean()
            refs = library.state(batch.frame[:, :-1])
            logits = self.termination_logit(pred, refs)
            term_bce = F.binary_cross_entropy_with_logits(logits, batch.terminated.float())
            loss = state_loss if self.cfg.termination == "soft" else state_loss + term_bce
            with torch.no_grad():
                pp, lp = split_state(pred[:, -1]), split_state(logged[:, -1])
                joint_err = self.project(pp.joint_pos - lp.joint_pos)
                joint_count = self.joint_presence.sum()
                feature_count = self.feature_presence.sum()
                info = dict(state_loss=state_loss.detach(), term_bce=term_bce.detach(),
                    pred_err_8step=((diff[:, -1] / self.normalizer.std).square().sum(-1)
                                   / feature_count).mean().sqrt(),
                    err8_pelvis_m=(pp.body_pos[:, 0] - lp.body_pos[:, 0]).square().sum(-1).mean().sqrt(),
                    err8_joint_rad=(joint_err.square().sum(-1) / joint_count).mean().sqrt(),
                    term_acc=((logits > 0) == batch.terminated).float().mean())
            return loss, info

    return NativeWorldModel
