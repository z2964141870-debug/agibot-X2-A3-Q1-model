"""Analytical planar contracts, not SMPL FK, SLAM, or a robot controller.

Pose AB maps B into A. z/roll/pitch and biomechanical effects are intentionally
outside this small oracle; passing it does not validate a real 3-D pipeline.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.x, self.y, self.yaw)):
            raise ValueError('non-finite pose')

    def compose(self, other):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose(self.x + c * other.x - s * other.y,
                    self.y + s * other.x + c * other.y,
                    self.yaw + other.yaw)

    def inverse(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose(-c * self.x - s * self.y, s * self.x - c * self.y, -self.yaw)


def pelvis_from_camera(world_camera, pelvis_head, head_camera):
    return world_camera.compose(head_camera.inverse()).compose(pelvis_head.inverse())


def camera_in_continuous_odom(map_camera, map_odom):
    """Requires a valid map-to-continuous-odom transform; not a jump detector."""
    return map_odom.inverse().compose(map_camera)


def velocity(previous, current, dt, previous_epoch, current_epoch, max_gap_s=.2):
    if previous_epoch != current_epoch: raise ValueError('coordinate/clock epoch changed')
    if not math.isfinite(dt) or not 0 < dt <= max_gap_s:
        raise ValueError('missing, backward, duplicate or excessive time interval')
    return ((current.x - previous.x) / dt, (current.y - previous.y) / dt)


@dataclass(frozen=True)
class ClockAlignment:
    """Explicit calibrated mapping of unwrapped device ticks to host seconds."""
    epoch: str
    seconds_per_tick: float
    offset_s: float
    calibration_source: str

    def host_time(self, ticks, epoch):
        if self.calibration_source != 'independent_sync_events':
            raise ValueError('arrival times alone do not establish exposure alignment')
        if self.epoch != epoch: raise ValueError('device reset requires new calibration epoch')
        if not all(math.isfinite(x) for x in (ticks, self.seconds_per_tick, self.offset_s)):
            raise ValueError('invalid clock mapping')
        if ticks < 0 or self.seconds_per_tick <= 0: raise ValueError('invalid clock scale')
        return ticks * self.seconds_per_tick + self.offset_s


def observation_age(capture_s, arrival_s, use_s):
    if not all(math.isfinite(x) for x in (capture_s, arrival_s, use_s)):
        raise ValueError('unknown observation time')
    if not capture_s <= arrival_s <= use_s: raise ValueError('inconsistent clock order')
    return {'capture_to_use_s': use_s - capture_s, 'host_buffer_to_use_s': use_s - arrival_s}
