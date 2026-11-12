"""Explicit human timing/coordinates and X2 sensor adaptation. No device I/O."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
DT_NS = 20_000_000
K = 10


def array(x, shape, name):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"{name}: expected finite {shape}, got {x.shape}")
    return x


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norm < 1e-8).any():
        raise ValueError("invalid quaternion")
    return x / norm


def mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def inv(q):
    return unit(q) * np.array([1, -1, -1, -1])


def rotate(q, v):
    q = unit(q)
    t = 2*np.cross(q[..., 1:], v)
    return v + q[..., :1]*t + np.cross(q[..., 1:], t)


def axis_angle(v):
    v = np.asarray(v, np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.concatenate((np.cos(theta/2), v * (.5*np.sinc(theta/(2*np.pi)))), -1)


def matrix_to_quat(rot_mats, name="rot_mats", tol=1e-3):
    """Convert proper SO(3) rotation matrices [..., 3, 3] to unit quaternions [..., 4] (wxyz)."""
    m = np.asarray(rot_mats, dtype=np.float64)
    if m.ndim < 2 or m.shape[-2:] != (3, 3) or not np.isfinite(m).all():
        raise ValueError(f"{name}: expected finite [..., 3, 3], got {m.shape}")
    flat = m.reshape(-1, 3, 3)
    eye = np.eye(3, dtype=np.float64)
    ortho_err = np.max(np.abs(np.matmul(np.swapaxes(flat, -1, -2), flat) - eye))
    det_err = np.max(np.abs(np.linalg.det(flat) - 1.0))
    if ortho_err > tol or det_err > tol:
        raise ValueError(f"{name}: non-SO(3) matrix (ortho_err={ortho_err:.3e}, det_err={det_err:.3e})")
    out = np.empty((len(flat), 4), dtype=np.float64)
    for idx, r in enumerate(flat):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            out[idx] = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            out[idx] = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            out[idx] = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            out[idx] = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    out = unit(out)
    out = out * np.where(out[:, :1] < 0, -1.0, 1.0)
    return out.reshape(m.shape[:-2] + (4,))


def interpolate(a, b, t):
    a, b = unit(a), unit(b)
    dot = np.sum(a*b, -1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), -1, 1))
    den = np.sin(angle)
    safe = np.where(den < 1e-8, 1, den)
    wa = np.where(den < 1e-8, 1-t, np.sin((1-t)*angle)/safe)
    wb = np.where(den < 1e-8, t, np.sin(t*angle)/safe)
    return unit(wa*a+wb*b)


def fk(root, world_quat, bones):
    pos = np.empty((24, 3))
    pos[0] = root
    for i in range(1, 24):
        pos[i] = pos[PARENTS[i]] + rotate(world_quat[PARENTS[i]], bones[i])
    return pos


@dataclass(frozen=True)
class Calibration:
    """SMPL y-up rest axes -> training z-up. Not a PICO/garment guessed transform.

    Translation is metric pelvis position. Scale affects translation, NOT fixed bones.
    Device adapters must explicitly choose this calibration after axis/rest checking.
    """
    id: str
    world_from_smpl_wxyz: tuple = (0.7071067811865476, 0.7071067811865476, 0., 0.)
    translation_scale: float = 1.0
    world_offset_m: tuple = (0., 0., 0.)

    def __post_init__(self):
        if not self.id or self.id.startswith("REPLACE") or not np.isfinite(self.translation_scale) or self.translation_scale <= 0:
            raise ValueError("explicit calibration id and positive scale required")
        unit(array(self.world_from_smpl_wxyz, (4,), "world_from_smpl"))
        array(self.world_offset_m, (3,), "offset")


class SmplAdapter:
    def __init__(self, skeleton: Path, calibration: Calibration):
        self.skeleton_id = hashlib.sha256(skeleton.read_bytes()).hexdigest()
        self.doc = json.loads(skeleton.read_text())
        self.bones = array(self.doc["bones"], (24, 3), "bones")
        self.calibration = calibration

    def _fk_from_local_quats(self, local_quat_24x4, root_translation_m):
        world = np.empty((24, 4))
        c = unit(np.asarray(self.calibration.world_from_smpl_wxyz))
        world[0] = mul(c, local_quat_24x4[0])
        for i in range(1, 24):
            world[i] = mul(world[PARENTS[i]], local_quat_24x4[i])
        root = rotate(c, array(root_translation_m, (3,), "root_translation_m"))
        root = root*self.calibration.translation_scale + self.calibration.world_offset_m
        return fk(root, world, self.bones), unit(world)

    def convert(self, local_axis_angle, root_translation_m):
        """SMPL24 local rotations, not SMPL-X55 or already-world garment quaternions."""
        q = axis_angle(array(local_axis_angle, (24, 3), "SMPL24 local axis angle"))
        return self._fk_from_local_quats(q, root_translation_m)

    def convert_matrices(self, local_rot_mats, root_translation_m):
        """SMPL24 local rotation matrices [24, 3, 3] (Latest FGP / CLOTHO output)."""
        m = array(local_rot_mats, (24, 3, 3), "SMPL24 local rotation matrices")
        q = matrix_to_quat(m, name="SMPL24 local rotation matrices")
        return self._fk_from_local_quats(q, root_translation_m)

    def packet(self, local_axis_angle, root_translation_m, *, sequence, capture_ns,
               capture_clock, arrival_ns, arrival_clock):
        pos, q = self.convert(local_axis_angle, root_translation_m)
        return dict(kind="morph_human_frame", version=1, sequence=sequence,
                    capture_ns=capture_ns, capture_clock=capture_clock,
                    arrival_ns=arrival_ns, arrival_clock=arrival_clock,
                    calibration_id=self.calibration.id, skeleton_id=self.skeleton_id,
                    convention="world_zup_m_smpl_rest_wxyz", valid=[True]*24,
                    positions=pos.tolist(), quaternions=q.tolist())

    def packet_from_matrices(self, local_rot_mats, root_translation_m, *, sequence, capture_ns,
                             capture_clock, arrival_ns, arrival_clock):
        pos, q = self.convert_matrices(local_rot_mats, root_translation_m)
        return dict(kind="morph_human_frame", version=1, sequence=sequence,
                    capture_ns=capture_ns, capture_clock=capture_clock,
                    arrival_ns=arrival_ns, arrival_clock=arrival_clock,
                    calibration_id=self.calibration.id, skeleton_id=self.skeleton_id,
                    convention="world_zup_m_smpl_rest_wxyz", valid=[True]*24,
                    positions=pos.tolist(), quaternions=q.tolist())


class NotReady(ValueError):
    pass


class HumanBuffer:
    """Full real preview only. Raw support/arrival retained; no extrapolation.

    capture clocks need not equal the receiver clock. No subtraction across clocks.
    3s receiver staleness matches the pinned upstream teleop timeout, not a new
    hardware abort. NotReady produces NO target; the robot owner decides its mode.
    """
    def __init__(self, skeleton_id, max_age_ns=3_000_000_000, max_interp_gap_ns=2*DT_NS):
        if type(max_interp_gap_ns) is not int or max_interp_gap_ns < DT_NS:
            raise ValueError("max_interp_gap_ns must be an integer >= DT_NS")
        self.skeleton_id = skeleton_id
        self.max_age_ns = max_age_ns
        self.max_interp_gap_ns = max_interp_gap_ns
        self.frames = deque(maxlen=512)
        self.epoch = None
        self.origin_capture_ns = None

    def reset(self):
        self.frames.clear()
        self.epoch = None
        self.origin_capture_ns = None

    def push(self, p):
        if p.get("kind") != "morph_human_frame" or p.get("version") != 1:
            raise ValueError("expected morph_human_frame v1; legacy positions-only input refused")
        if p.get("skeleton_id") != self.skeleton_id or p.get("convention") != "world_zup_m_smpl_rest_wxyz":
            raise ValueError("skeleton/convention mismatch")
        if np.asarray(p.get("valid")).shape != (24,) or not all(v is True for v in p["valid"]):
            raise ValueError("all 24 joints must be valid in full mode")
        for key in ("sequence", "capture_ns", "arrival_ns"):
            if type(p.get(key)) is not int or not 0 <= p[key] < 2**63:
                raise ValueError(f"invalid integer {key}")
        epoch = tuple(p.get(k) for k in ("capture_clock", "arrival_clock", "calibration_id"))
        if not all(isinstance(x, str) and x for x in epoch):
            raise ValueError("clock and calibration identifiers required")
        if self.epoch is not None and self.epoch != epoch:
            raise ValueError("clock/calibration changed: explicit reset required")
        pos = array(p["positions"], (24, 3), "positions").copy()
        quat = array(p["quaternions"], (24, 4), "quaternions")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-3):
            raise ValueError("non-unit quaternion")
        quat = unit(quat).copy()
        if self.frames:
            old = self.frames[-1]
            if p["sequence"] <= old[0] or p["capture_ns"] <= old[1] or p["arrival_ns"] < old[2]:
                raise ValueError("out-of-order/duplicate frame")
        self.epoch = epoch
        if self.origin_capture_ns is None:
            self.origin_capture_ns = p["capture_ns"]
        self.frames.append((p["sequence"], p["capture_ns"], p["arrival_ns"], pos, quat))

    def window(self, now_ns, receiver_clock):
        if self.epoch is None or receiver_clock != self.epoch[1]:
            raise NotReady("no frames or receiver clock mismatch")
        available = [f for f in self.frames if f[2] <= now_ns]
        if not available or now_ns-available[-1][2] > self.max_age_ns:
            raise NotReady("human input unavailable/stale")
        times = np.array([f[1] for f in available], dtype=np.int64)
        origin = self.origin_capture_ns
        last = origin + ((int(times[-1])-origin)//DT_NS)*DT_NS
        target = last - DT_NS*np.arange(K-1, -1, -1, dtype=np.int64)
        if target[0] < times[0]:
            raise NotReady("need 10 real 50Hz slots (180ms span)")
        pos, quat, support, gaps_ns = [], [], [], []
        for t in target:
            right = int(np.searchsorted(times, t))
            if right >= len(times):
                raise NotReady("missing future interpolation support")
            if times[right] == t:
                f = available[right]
                pos.append(f[3]); quat.append(f[4]); support.append(int(times[right]))
                gaps_ns.append(0)
            else:
                left = right-1
                gap = int(times[right]) - int(times[left]) if left >= 0 else self.max_interp_gap_ns + 1
                if left < 0 or gap > self.max_interp_gap_ns:
                    raise NotReady("input gap exceeds configured source interpolation limit")
                a = (int(t)-int(times[left]))/gap
                pos.append((1-a)*available[left][3]+a*available[right][3])
                quat.append(interpolate(available[left][4], available[right][4], a))
                support.append(int(times[right]))
                gaps_ns.append(gap)
        frames = np.concatenate((np.asarray(pos), np.asarray(quat)), -1).reshape(K, 168)
        return frames, dict(slot_capture_ns=target.tolist(), raw_support_end_ns=support,
                            capture_clock=self.epoch[0], arrival_clock=self.epoch[1],
                            calibration_id=self.epoch[2], measured_preview_span_ms=180.,
                            newest_arrival_age_ms=(now_ns-available[-1][2])/1e6,
                            max_source_gap_ms=max(gaps_ns)/1e6,
                            max_interp_gap_ms=self.max_interp_gap_ns/1e6,
                            hardware_end_to_end_latency_ms=None)


def pelvis_from_sensor(robot, q, dq, sensor_quat, sensor_gyro, body_name, body_from_sensor):
    """Transform calibrated sensor -> FK body -> pelvis, including joint-rate gyro.

    body_from_sensor rotates sensor vectors into body axes (q_BS); q_WS*q_SB=q_WB.
    Relative angular velocity is accumulated analytically in root axes. Translational
    acceleration is not reconstructed/used by this policy.
    """
    qb = unit(np.asarray(body_from_sensor))
    qw_body = mul(unit(sensor_quat), inv(qb))
    gyro_body = rotate(qb, sensor_gyro)
    doc_names = getattr(robot, "tree_names", None)
    if doc_names is None or body_name not in doc_names:
        raise ValueError("sensor body not in exported FK tree")
    index = doc_names.index(body_name)
    rot, omega = [], []
    for b, parent in enumerate(robot.fk_parent):
        r = np.array([1., 0, 0, 0]) if parent < 0 else mul(rot[parent], robot.fk_quat[b])
        w = np.zeros(3) if parent < 0 else omega[parent].copy()
        for j, axis, ref in robot.fk_joints[b]:
            w += rotate(r, axis)*dq[j]
            r = mul(r, axis_angle(axis*(q[j]-ref)))
        rot.append(r); omega.append(w)
    root = mul(qw_body, inv(rot[index]))
    root_omega = rotate(rot[index], gyro_body)-omega[index]
    imu = robot.fk_imu
    return unit(mul(root, rot[imu])), rotate(inv(rot[imu]), root_omega+omega[imu])
