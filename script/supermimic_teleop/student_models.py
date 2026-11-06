"""Trainable encoders sharing a frozen, copied native decoder.

No checkpoint loader, optimizer loop or server launch is implicit. The admission
controller must load the audited 32k teacher and fit f0 statistics on train only.
"""

import copy

import torch
from torch import nn

from student_inputs import FRAME_DIM, GEOMETRY_DIM, HISTORY


def masked_preview_goal(features, heading, horizons):
    if (features.ndim != 2 or features.shape[1] != 2160
            or heading.shape != (len(features), 60)
            or horizons.shape != (len(features),) or horizons.dtype != torch.long
            or bool(((horizons < 1) | (horizons > 10)).any())):
        raise ValueError("Require normalized native features, heading and explicit H1..H10")
    frames = torch.arange(10, device=features.device)[None] < horizons[:, None]
    points = torch.ones((len(features), 24), dtype=features.dtype, device=features.device)
    visible = torch.where(frames[:, :, None], features.reshape(-1, 10, 216), 0.)
    relative = torch.where(frames[:, :, None], heading.reshape(-1, 10, 6), 0.)
    return torch.cat((visible.flatten(1), points, frames.to(features.dtype), relative.flatten(1)), -1)


def native_preview_input(native, windows, robot_quat):
    from morphmimic.model.human import (fixed_human_masks, heading_from_goal,
        human_features_from_goal, pack_human_goal)
    from preview_policy import validate_window

    counts = [validate_window(window) for window in windows]
    if not counts or robot_quat.shape != (len(counts), 4):
        raise ValueError("Require one current robot torso for each preview")
    device, dtype = robot_quat.device, robot_quat.dtype
    if any(norm.training for norm in native.norms.values()):
        raise ValueError("Teacher normalizers must remain frozen in evaluation mode")
    positions = torch.tensor([[pose.positions for pose in window.poses] for window in windows],
                             dtype=dtype, device=device)
    rotations = torch.tensor([[pose.orientations for pose in window.poses] for window in windows],
                             dtype=dtype, device=device)
    with torch.no_grad():
        raw = pack_human_goal(positions, rotations, robot_quat)
        points, _, _ = fixed_human_masks(len(counts), "full", 10, device, heading=True)
        features = native.norms["human_norm"](human_features_from_goal(raw, points))
        heading = native.norms["heading_norm"](heading_from_goal(raw, points))
        return masked_preview_goal(features, heading, torch.tensor(counts, device=device))


class CurrentHistoryEncoder(nn.Module):
    def __init__(self, latent_dim, statistics):
        super().__init__()
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("Read the positive latent width from the real teacher")
        if (statistics.get("fitted_split") != "train" or statistics.get("count", 0) <= 0
                or statistics.get("schema") != "q1_current_geometry_stats_v1"
                or statistics.get("geometry_dim") != GEOMETRY_DIM
                or statistics.get("clipping") != 10.):
            raise ValueError("Require frozen train-only current geometry statistics")
        mean = torch.tensor(statistics["mean"], dtype=torch.float32)
        scale = torch.tensor(statistics["scale"], dtype=torch.float32)
        if (mean.shape != (GEOMETRY_DIM,) or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or bool((scale < .01).any())):
            raise ValueError("Invalid current geometry normalization")
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.net = nn.Sequential(nn.Linear(HISTORY*FRAME_DIM, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, latent_dim), nn.Tanh())

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (HISTORY, FRAME_DIM):
            raise ValueError("Require oldest-first f0/history input, not native future slots")
        valid = values[..., GEOMETRY_DIM:GEOMETRY_DIM+1] == 1.
        geometry = ((values[..., :GEOMETRY_DIM]-self.mean)/self.scale).clamp(-10., 10.)
        geometry = torch.where(valid, geometry, 0.)
        flags_age = torch.where(valid, values[..., GEOMETRY_DIM:], 0.)
        return self.net(torch.cat((geometry, flags_age), -1).flatten(1))


class FrozenDecoderStudent(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = copy.deepcopy(decoder).eval().requires_grad_(False)
        reference = next(self.decoder.parameters())
        self.encoder.to(device=reference.device, dtype=reference.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(self, normalized_history, goal):
        return self.decoder(self.encoder(goal), normalized_history)

    def trainable_parameters(self):
        return self.encoder.parameters()


def short_preview_student(native):
    if native.policy.human_encoder is None:
        raise ValueError("The audited teacher has no human encoder")
    return FrozenDecoderStudent(native.policy.human_encoder, native.policy.decoder)


def current_student(native, statistics):
    return FrozenDecoderStudent(CurrentHistoryEncoder(native.policy.cfg.latent, statistics),
                                native.policy.decoder)


def action_distillation_loss(student_action, teacher_action, present):
    if (student_action.ndim != 2 or student_action.shape != teacher_action.shape
            or student_action.shape[-1] != 29 or present.shape != (29,)
            or present.dtype != torch.bool or not bool(present.any())):
        raise ValueError("Require matched canonical actions and a real morphology presence mask")
    if not bool(torch.isfinite(student_action).all() & torch.isfinite(teacher_action).all()):
        raise ValueError("Nonfinite distillation action")
    return ((student_action[:, present]-teacher_action.detach()[:, present])**2).mean()
