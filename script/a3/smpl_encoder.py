"""Human SMPL student and frozen A3 bridge; training needs qualified pairs."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from torch import nn

FRAMES = 10
FRAME_DIM = 90
LATENT_DIM = 64
PARENTS = np.asarray([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                      9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
SMPL_TO_ROBOT = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def rotations(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError('Expected finite rotation matrices')
    error = np.linalg.norm(value.swapaxes(-1, -2) @ value - np.eye(3), axis=(-2, -1))
    determinant = np.linalg.det(value)
    if np.any(error > .01) or np.any(determinant <= 0) or np.any(np.abs(determinant - 1) > .01):
        raise ValueError('Rotation exceeds SO(3) repair budget')
    return Rotation.from_matrix(value.reshape(-1, 3, 3)).as_matrix().reshape(value.shape)


def feature_frames(joints, local_pose, anchor_rotation, root_alignment=None):
    """Root-local metres, robot-relative root, and two parent-local wrist poses."""
    joints = np.asarray(joints, dtype=np.float64)
    local_pose = rotations(local_pose)
    if (joints.ndim != 3 or joints.shape[1:] != (24, 3)
            or local_pose.shape != (len(joints), 24, 3, 3) or not np.isfinite(joints).all()):
        raise ValueError('Expected aligned 24-joint SMPL positions and local poses')
    anchor = rotations(anchor_rotation)
    if anchor.shape != (3, 3):
        raise ValueError('One current robot anchor is required for each window')
    alignment = rotations(np.eye(3) if root_alignment is None else root_alignment)
    if alignment.shape != (3, 3):
        raise ValueError('Expected one fixed session root alignment')
    root = local_pose[:, 0]
    local = np.einsum('tbj,tjk->tbk', joints - joints[:, :1], root) @ SMPL_TO_ROBOT.T
    robot_root = alignment @ SMPL_TO_ROBOT @ root @ SMPL_TO_ROBOT.T
    relative = anchor.T @ robot_root
    wrists = SMPL_TO_ROBOT @ local_pose[:, [20, 21]] @ SMPL_TO_ROBOT.T
    # Keep the pinned A3 row-major first-two-columns convention for 6D rotations.
    result = np.concatenate((local.reshape(-1, 72), relative[..., :2].reshape(-1, 6),
                             wrists[..., :2].reshape(-1, 12)), axis=-1)
    return result.astype(np.float32)


def smpl_window(times, joints, local_pose, anchor_rotation, target_time,
                mode='offline_future', arrival_time=None, root_alignment=None):
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid source times')
    if not np.isfinite(target_time) or mode not in ('offline_future', 'causal_history'):
        raise ValueError('Invalid target time or input mode')
    offsets = np.arange(FRAMES) * .02
    if mode == 'causal_history':
        offsets -= .18
        if arrival_time is None or not np.isfinite(arrival_time):
            raise ValueError('Causal history requires an arrival time')
    query = target_time + offsets
    if query[0] < times[0] - 1e-10 or query[-1] > times[-1] + 1e-10:
        raise ValueError('Window would extrapolate outside its session')
    query = np.clip(query, times[0], times[-1])
    upper = np.searchsorted(times, query - 1e-10)
    upper = np.minimum(upper, len(times) - 1)
    if mode == 'causal_history' and times[upper].max() > arrival_time + 1e-10:
        raise ValueError('Interpolation would read a sample not yet arrived')
    joints = np.asarray(joints, dtype=np.float64)
    local_pose = np.asarray(local_pose, dtype=np.float64)
    if joints.shape != (len(times), 24, 3) or local_pose.shape != (len(times), 24, 3, 3):
        raise ValueError('Source times and full SMPL arrays differ')
    # Validate and interpolate only the bracketing samples available to this window.
    lower = max(0, np.searchsorted(times, query[0], side='right') - 1)
    selected = slice(lower, int(upper[-1]) + 1)
    times, joints, local_pose = times[selected], joints[selected], local_pose[selected]
    if len(times) < 2 or np.diff(times).max() > .2:
        raise ValueError('Source gap exceeds 200ms budget')
    local_pose = rotations(local_pose)
    if not np.isfinite(joints).all():
        raise ValueError('Nonfinite SMPL joints')
    sampled_joints = np.column_stack([np.interp(query, times, column)
                                     for column in joints.reshape(len(times), -1).T]).reshape(FRAMES, 24, 3)
    sampled_pose = np.stack([Slerp(times, Rotation.from_matrix(local_pose[:, j]))(query).as_matrix()
                             for j in range(24)], axis=1)
    return feature_frames(sampled_joints, sampled_pose, anchor_rotation, root_alignment), query


def pack_teacher(joint_pos, joint_vel, orientation6d):
    q, v, ori = [np.asarray(value, dtype=np.float32) for value in (joint_pos, joint_vel, orientation6d)]
    if q.shape != (FRAMES, 29) or v.shape != q.shape or ori.shape != (FRAMES, 6):
        raise ValueError('Teacher requires 10x29 position/velocity and 10x6 orientation')
    if any(not np.isfinite(value).all() for value in (q, v, ori)):
        raise ValueError('Nonfinite teacher input')
    # The public ABI groups all q, then all v, before reshaping into ten rows.
    command = np.concatenate((q.reshape(-1), v.reshape(-1))).reshape(FRAMES, 58)
    return np.concatenate((command, ori), axis=-1).reshape(640)


def require_training_pairs(metadata):
    if metadata.get('usage') != 'TRAINING_PAIRED_SMPL_A3' or metadata.get('training_allowed') is not True:
        raise ValueError('Pairs are not authorized for training; TEST_ONLY remains test-only')
    for key in ('quality_verified', 'source_use_verified', 'pair_alignment_verified'):
        if metadata.get(key) is not True:
            raise ValueError('Missing pair requirement: ' + key)
    if metadata.get('input_mode') not in ('offline_future', 'causal_history'):
        raise ValueError('Input timing mode must be explicit')
    partitions = [set(metadata.get(key, [])) for key in ('train_sessions', 'validation_sessions', 'test_sessions')]
    if not all(partitions) or any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError('Need separate, nonempty train/validation/test sessions')


class SMPLStudent(nn.Module):
    """A separate human input branch; not the unused vendor SMPL definition."""

    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        dimensions = (FRAMES * FRAME_DIM, *hidden, LATENT_DIM)
        layers = []
        for i, (before, after) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(before, after))
            if i < len(dimensions) - 2:
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)
        self.register_buffer('input_mean', torch.zeros(FRAMES * FRAME_DIM))
        self.register_buffer('input_std', torch.ones(FRAMES * FRAME_DIM))

    @torch.no_grad()
    def fit_normalization(self, training_inputs, metadata):
        require_training_pairs(metadata)
        flat = self._flatten(training_inputs)
        if len(flat) < 2:
            raise ValueError('Need at least two qualified training windows')
        mean, std = flat.mean(0), flat.std(0, unbiased=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError('Nonfinite training normalization')
        self.input_mean.copy_(mean)
        self.input_std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))

    def _flatten(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (FRAMES, FRAME_DIM) or not torch.isfinite(inputs).all():
            raise ValueError('Expected finite batch x 10 x 90 human features')
        return inputs.flatten(1)

    def forward(self, inputs):
        flat = self._flatten(inputs)
        return self.module((flat - self.input_mean) / self.input_std)


class FrozenA3Bridge(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy.requires_grad_(False).eval()
        if policy.actor_obs_dim != 930 or policy.encoder_input_dim != 640:
            raise ValueError('Requires the checked 035 A3-fast interface')

    @torch.no_grad()
    def teacher_latent(self, encoder_input):
        if encoder_input.ndim != 2 or encoder_input.shape[-1] != 640 or not torch.isfinite(encoder_input).all():
            raise ValueError('Expected finite batch x 640 teacher input')
        return self.policy.encoder(encoder_input)

    @torch.no_grad()
    def decode_latent(self, latent, proprioception):
        if (latent.ndim != 2 or latent.shape[-1] != LATENT_DIM
                or proprioception.shape != (len(latent), 930)
                or not torch.isfinite(latent).all() or not torch.isfinite(proprioception).all()):
            raise ValueError('Invalid latent or robot proprioception')
        tokens = self.policy.fsq(latent.reshape(-1, 2, 32)).flatten(1)
        actions = self.policy.decoder(torch.cat((tokens, proprioception), dim=-1))
        if not torch.isfinite(actions).all():
            raise ValueError('Nonfinite decoder actions')
        return tokens, actions


def distillation_step(student, optimizer, human_inputs, teacher_input, bridge, metadata):
    require_training_pairs(metadata)
    eligible = {id(parameter) for parameter in student.parameters()}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    if actual != eligible:
        raise ValueError('Optimizer must contain exactly the new student parameters')
    if any(parameter.requires_grad for parameter in bridge.parameters()):
        raise ValueError('Official bridge must remain frozen')
    student.train()
    optimizer.zero_grad(set_to_none=True)
    target = bridge.teacher_latent(teacher_input)
    output = student(human_inputs)
    if output.shape != target.shape:
        raise ValueError('Paired human/teacher batch sizes differ')
    loss = (output - target).square().mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite distillation objective')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in student.parameters()):
        raise ValueError('Nonfinite student parameters')
    return dict(loss=float(loss.detach()), gradient_norm=float(norm))
