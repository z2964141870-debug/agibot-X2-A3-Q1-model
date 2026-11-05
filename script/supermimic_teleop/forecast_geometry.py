"""History-only human forecast coordinates and time-consistent pose decoding.

The local frame removes a common world yaw and translation, not body posture.
This is an independent prediction representation, not a policy goal schema.
Quaternion math reuses the existing causal-tail Hamilton product convention.
No bone-length, contact, collision or force constraints are imposed here.
"""

from dataclasses import dataclass
import math

import numpy as np

from causal_tail import multiply
from forecast_data import DT_NS


FEATURE_DIM = 216
TARGET_DIM = 144


def immutable(values):
    values = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite forecast values")
    values.setflags(write=False)
    return values


def checked_poses(positions, quaternions):
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if (positions.ndim != 3 or not len(positions) or positions.shape[1:] != (24, 3)
            or quaternions.shape != (len(positions), 24, 4)
            or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()):
        raise ValueError("Require finite [time,24,3] positions and [time,24,4] wxyz")
    length = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.max(np.abs(length-1)) > 1e-4:
        raise ValueError("Require unit wxyz quaternions")
    return positions, quaternions / length


def product(a, b):
    return np.stack(multiply(np.moveaxis(a, -1, 0), np.moveaxis(b, -1, 0)), axis=-1)


def inverse(quaternions):
    return quaternions * np.array([1., -1., -1., -1.])


def log_rotation(quaternions):
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    # q and -q are the same rotation; take its shortest principal rotation vector.
    quaternions = np.where(quaternions[..., :1] < 0., -quaternions, quaternions)
    length = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    angle = 2*np.arctan2(length, quaternions[..., :1])
    factor = np.divide(angle, length, out=np.full_like(length, 2.), where=length > 1e-10)
    return quaternions[..., 1:]*factor


def exp_rotation(vectors):
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    factor = np.divide(np.sin(angle/2), angle, out=np.full_like(angle, .5), where=angle > 1e-10)
    quaternions = np.concatenate((np.cos(angle/2), vectors*factor), axis=-1)
    return quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)


def rotation_columns(quaternions):
    """First and second rotation-matrix columns, concatenated as two 3-vectors."""
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                     2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), axis=-1)


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))


@dataclass(frozen=True)
class ForecastAnchor:
    root_position: np.ndarray
    world_from_heading: np.ndarray
    last_relative_body: np.ndarray
    last_quaternions: np.ndarray
    last_positions: np.ndarray


@dataclass(frozen=True)
class EncodedHistory:
    features: np.ndarray
    anchor: ForecastAnchor


@dataclass(frozen=True)
class DecodedForecast:
    positions: np.ndarray
    quaternions: np.ndarray
    linear_velocity_w: np.ndarray
    angular_velocity_w: np.ndarray
    target_offsets_ns: tuple
    bone_contact_constraints_applied: bool = False


def encode_history(positions, quaternions):
    """Accept only the visible history: no target, full-clip or future heading argument."""
    positions, quaternions = checked_poses(positions, quaternions)
    if len(positions) < 2:
        raise ValueError("Require at least two visible history frames")
    w, x, y, z = quaternions[-1, 0]
    forward_x, forward_y = 1-2*(y*y+z*z), 2*(w*z+x*y)
    if math.hypot(forward_x, forward_y) < 1e-6:
        raise ValueError("Pelvis heading is undefined when its forward axis is vertical")
    yaw = math.atan2(forward_y, forward_x)
    rotation = yaw_matrix(yaw)
    root = positions[-1, 0]
    root_history = (positions[:, 0]-root) @ rotation
    relative = (positions[:, 1:]-positions[:, :1]) @ rotation
    undo_yaw = np.array([math.cos(yaw/2), 0., 0., -math.sin(yaw/2)])
    heading_quaternions = product(undo_yaw, quaternions)
    features = np.concatenate((root_history, relative.reshape(len(positions), -1),
                              rotation_columns(heading_quaternions).reshape(len(positions), -1)), axis=-1)
    if features.shape != (len(positions), FEATURE_DIM):
        raise RuntimeError("History feature schema changed")
    anchor = ForecastAnchor(immutable(root), immutable(rotation), immutable(relative[-1]),
        immutable(quaternions[-1]), immutable(positions[-1]))
    return EncodedHistory(immutable(features), anchor)


def encode_targets(positions, quaternions, anchor):
    """Future labels only; the anchor must already have been made from history."""
    positions, quaternions = checked_poses(positions, quaternions)
    if not isinstance(anchor, ForecastAnchor) or len(positions) > 10:
        raise ValueError("Require a history anchor and one to ten future targets")
    root = (positions[:, 0]-anchor.root_position) @ anchor.world_from_heading
    relative = (positions[:, 1:]-positions[:, :1]) @ anchor.world_from_heading
    body_delta = relative-anchor.last_relative_body
    rotation = log_rotation(product(inverse(anchor.last_quaternions), quaternions))
    return immutable(np.concatenate((root, body_delta.reshape(len(root), -1),
                                     rotation.reshape(len(root), -1)), axis=-1))


def decode_targets(targets, anchor):
    targets = np.asarray(targets, dtype=np.float64)
    if (not isinstance(anchor, ForecastAnchor) or targets.ndim != 2 or not 1 <= len(targets) <= 10
            or targets.shape[1] != TARGET_DIM or not np.isfinite(targets).all()):
        raise ValueError("Require finite [one-to-ten,144] forecasts and a history anchor")
    count = len(targets)
    root = anchor.root_position + targets[:, :3] @ anchor.world_from_heading.T
    relative = anchor.last_relative_body + targets[:, 3:72].reshape(count, 23, 3)
    positions = np.concatenate((root[:, None],
        root[:, None] + relative @ anchor.world_from_heading.T), axis=1)
    quaternions = product(anchor.last_quaternions,
                         exp_rotation(targets[:, 72:].reshape(count, 24, 3)))
    dt = DT_NS/1e9
    previous_positions = np.concatenate((anchor.last_positions[None], positions[:-1]))
    previous_quaternions = np.concatenate((anchor.last_quaternions[None], quaternions[:-1]))
    # Left relative rotations give angular increments in the world frame.
    angular = log_rotation(product(quaternions, inverse(previous_quaternions))) / dt
    linear = (positions-previous_positions) / dt
    return DecodedForecast(*(immutable(v) for v in (positions, quaternions, linear, angular)),
                           tuple((i+1)*DT_NS for i in range(count)))


def error_by_horizon(prediction, positions, quaternions):
    positions, quaternions = checked_poses(positions, quaternions)
    if positions.shape != prediction.positions.shape:
        raise ValueError("Prediction and labels must use the same unshifted target slots")
    delta = prediction.positions-positions
    angle = np.linalg.norm(log_rotation(product(inverse(quaternions), prediction.quaternions)), axis=-1)
    return dict(position_rmse_m=np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=-1)).tolist(),
        root_position_error_m=np.linalg.norm(delta[:, 0], axis=-1).tolist(),
        orientation_geodesic_rmse_rad=np.sqrt(np.mean(angle**2, axis=-1)).tolist(),
        target_offsets_ns=list(prediction.target_offsets_ns), temporal_realignment_applied=False,
        bone_contact_constraints_applied=False, measured_robot_latency_ns=None)
