"""Pure NumPy + ONNXRuntime Q1 V2 autonomous fall-recovery ownership supervisor (no PyTorch/MuJoCo)."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from q1_autonomous_recovery import Q1AutonomousRecovery
from recovery import Gate

DEFAULT_BLEND_TICKS = 16
DEFAULT_LEG_SPEED_CAP = 8.5
DEFAULT_LEG_BRAKE_GAIN = 0.5
GATE_CONFIG = {
    "h_min": 0.355,
    "up_z_min": 0.965,
    "rms_leg_max": 0.35,
    "w_norm_max": 0.70,
    "qd_rms_max": 0.95,
    "calm_ticks": 4,
}


class CalmUprightGate(Gate):
    """Switch from recovery expert to private V2b stand only once upright and calm (or standard stand dwell)."""

    def reset(self):
        super().reset()
        self.calm_ticks = 0

    def advance(self, x, width):
        if self.mode == "recover":
            qd = x[22:44] * 10.0
            w = x[44:47] * 4.0
            upright_calm = (
                bool(x[-1] > 0.5)
                and float(x[-2]) > GATE_CONFIG["h_min"]
                and -float(x[49]) > GATE_CONFIG["up_z_min"]
                and float(np.sqrt(np.mean(x[:12] ** 2))) < GATE_CONFIG["rms_leg_max"]
                and float(np.linalg.norm(w)) < GATE_CONFIG["w_norm_max"]
                and float(np.sqrt(np.mean(qd ** 2))) < GATE_CONFIG["qd_rms_max"]
            )
            self.calm_ticks = self.calm_ticks + 1 if upright_calm else 0
            if self.calm_ticks >= GATE_CONFIG["calm_ticks"]:
                self.mode = "track"
                self.tilt_ticks = self.stand_ticks = self.calm_ticks = self.blend_ticks = 0
                return 0.0, True
        else:
            self.calm_ticks = 0
        return super().advance(x, width)


class Q1SoftRecoveryV2Owner(Q1AutonomousRecovery):
    """Exclusive V2 recovery supervisor with protective fall unload, calm handoff, and leg speed braking."""

    def __init__(
        self,
        tracker,
        expert_onnx: str | Path,
        q1_xml: str | Path,
        *,
        blend_ticks: int = DEFAULT_BLEND_TICKS,
        leg_speed_cap: float = DEFAULT_LEG_SPEED_CAP,
        leg_brake_gain: float = DEFAULT_LEG_BRAKE_GAIN,
    ):
        self.blend_ticks_cfg = int(blend_ticks)
        self.leg_speed_cap = float(leg_speed_cap)
        self.leg_brake_gain = float(leg_brake_gain)
        super().__init__(tracker, expert_onnx, q1_xml)
        self.rise.expert.gate = CalmUprightGate()
        self.rise.expert.gate.reset()

    def reset(self):
        super().reset()
        self.origin_target = None
        self.blend_index = self.blend_ticks_cfg
        self.protecting = False
        self.imminent_ticks = 0
        self.protect_pose = None

    def step(self, R, w, q, qd, previous, base, anchor, *, sample_time_s, teleop_stamp_s):
        c = self.tracker
        if (
            not math.isfinite(sample_time_s)
            or (self.last is not None and sample_time_s <= self.last)
            or not all(np.isfinite(x).all() for x in (R, w, q, qd, previous))
        ):
            self.since = None
            self.standing_confirmed = False
            self.fall_count = 0
            self.imminent_ticks = 0
            raise ValueError("expected finite proprioception and increasing sample time")

        fresh = (
            teleop_stamp_s is not None
            and math.isfinite(teleop_stamp_s)
            and 0.0 <= sample_time_s - teleop_stamp_s <= 0.1 + 1e-9
        )
        if self.owner == "teleop" and fresh and not self.protecting:
            up = R[2]
            vertical_rate = float(np.cross(up, w)[2])
            imminent = (
                up[2] < 0.88 and vertical_rate < -2.0 and up[2] + 0.18 * vertical_rate < 0.50
            ) or (up[2] < 0.71 and vertical_rate < -0.6)
            severe = (up[2] < 0.82 and vertical_rate < -2.1) or (up[2] < 0.70)
            self.imminent_ticks = self.imminent_ticks + 1 if imminent else 0
            if self.imminent_ticks >= 2 or severe:
                self.owner = "recovery"
                self.phase = "protect"
                self.entered = sample_time_s
                self.standing_confirmed = False
                self.since = self.hold_R = None
                self.fall_count = 0
                self.protecting = True
                self.imminent_ticks = 0
                self.rise.reset()
                self.rise.expert.gate.mode = "recover"
                self.protect_pose = None
        elif not fresh or self.owner != "teleop":
            self.imminent_ticks = 0

        if self.protecting:
            age = sample_time_s - self.entered
            exit_protect = (
                age >= 0.20
                and abs(R[2, 2]) < 0.42
                and float(np.linalg.norm(w)) < 1.3
                and float(np.max(np.abs(qd))) < 4.5
            ) or age >= 0.36
            if not exit_protect:
                self.last = sample_time_s
                self.rise.expert.state.fill(0.0)
                target = q.copy()
                target[15] = q[15] + 0.35 * (max(q[15], 0.65) - q[15])
                target[19] = q[19] + 0.35 * (min(q[19], -0.65) - q[19])
                target[17] = q[17] + 0.25 * (min(q[17], 0.30) - q[17])
                target[21] = q[21] + 0.25 * (min(q[21], 0.30) - q[21])
                target = np.clip(target, c.default_q - 2.0, c.default_q + 2.0)
                tau = np.clip(c.kp * (target - q) - c.kd * qd, -c.eff, c.eff)
                return target, tau, 1.0
            self.protecting = False
            self.phase = "rise"

        before = self.rise.expert.gate.mode
        target, tau, owned = super().step(
            R,
            w,
            q,
            qd,
            previous,
            base,
            anchor,
            sample_time_s=sample_time_s,
            teleop_stamp_s=teleop_stamp_s,
        )
        after = self.rise.expert.gate.mode
        if owned == 0.0:
            self.protecting = False
            self.imminent_ticks = 0
            self.protect_pose = None
            self.origin_target = None
            self.blend_index = self.blend_ticks_cfg
            return target, tau, owned
        if before == "recover" and after in ("track", "handoff") and owned:
            self.origin_target = c.default_q + c.action_scale * previous
            self.blend_index = 0
        if after == "recover":
            self.blend_index = self.blend_ticks_cfg
            return target, tau, owned

        out = target.copy()
        if self.blend_index < self.blend_ticks_cfg and self.origin_target is not None:
            self.blend_index += 1
            weight = 0.5 - 0.5 * math.cos(math.pi * self.blend_index / self.blend_ticks_cfg)
            out = self.origin_target + weight * (out - self.origin_target)

        tau_cmd = np.clip(c.kp * (out - q) - c.kd * qd, -c.eff, c.eff)
        leg_idx = slice(0, 12)
        excess = np.maximum(np.abs(qd[leg_idx]) - self.leg_speed_cap, 0.0)
        if excess.any():
            tau_cmd[leg_idx] = np.clip(
                tau_cmd[leg_idx] - np.sign(qd[leg_idx]) * excess * (self.leg_brake_gain * c.kd[leg_idx]),
                -c.eff[leg_idx],
                c.eff[leg_idx],
            )
        guarded_target = np.clip(q + (tau_cmd + c.kd * qd) / c.kp, c.default_q - 2.0, c.default_q + 2.0)
        tau_cmd = np.clip(c.kp * (guarded_target - q) - c.kd * qd, -c.eff, c.eff)
        return guarded_target, tau_cmd, owned


def check_v2_ownership(
    tracker,
    expert_onnx: str | Path,
    q1_xml: str | Path,
    blend_ticks: int = DEFAULT_BLEND_TICKS,
) -> dict:
    """Deterministic ownership, finite/monotonic timestamp guard, and stale-input regression check."""
    branch = Q1SoftRecoveryV2Owner(tracker, expert_onnx, q1_xml, blend_ticks=blend_ticks)
    q = tracker.default_q.copy()
    zero = np.zeros_like(q)
    w = np.zeros(3, dtype=np.float32)
    upright = np.eye(3, dtype=np.float32)
    down = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    calls = []

    def live():
        calls.append("teleop")
        return q.copy(), zero.copy(), {}

    def anchor():
        calls.append("anchor")

    def tick(t, stamp, R=upright, rate=w, joints=q):
        return branch.step(
            R, rate, joints, zero, zero, live, anchor, sample_time_s=t, teleop_stamp_s=stamp
        )[2]

    assert tick(0.0, 0.0) == 0.0 and branch.owner == "teleop"
    assert calls == ["teleop"]
    calls.clear()

    assert tick(0.02, 0.02, down) == 1.0
    assert branch.owner == "recovery" and branch.phase == "protect" and branch.protecting
    assert not calls

    try:
        tick(0.02, 0.02, down)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate timestamp accepted during protect")
    nan_q = q.copy()
    nan_q[0] = float("nan")
    try:
        tick(0.03, 0.03, upright, w, nan_q)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite proprioception accepted during protect")

    for k in range(2, 19):
        assert tick(k * 0.02, k * 0.02, down) == 1.0
    assert not calls
    original_step = branch.rise.step
    branch.rise.step = lambda *args: (q.copy(), zero.copy(), 0.0)

    for k in range(19, 60):
        assert tick(k * 0.02, None) == 1.0
    assert branch.standing_confirmed and not calls and branch.owner == "recovery"

    assert tick(1.22, 0.92) == 1.0
    assert tick(1.24, 1.26) == 1.0
    assert tick(1.26, 0.01) == 1.0
    assert not calls
    assert tick(1.28, 1.28, rate=np.ones(3, np.float32)) == 1.0
    assert not branch.standing_confirmed and not calls

    for k in range(65, 81):
        assert tick(k * 0.02, k * 0.02) == 1.0
    assert tick(2.30, 2.30) == 1.0 and not calls
    for k in range(116, 140):
        assert tick(k * 0.02, k * 0.02) == 1.0
    assert not calls
    assert tick(2.80, 2.80) == 0.0
    assert calls == ["anchor", "teleop"] and branch.owner == "teleop"

    calls.clear()
    assert tick(2.82, 2.82, down) == 1.0 and branch.owner == "recovery" and not calls

    branch.reset()
    assert branch.owner == "teleop" and not branch.standing_confirmed
    assert tick(0.0, None, down) == 1.0
    branch.rise.step = original_step
    for k in range(1, 1501):
        assert tick(k * 0.02, k * 0.02, down) == 1.0
    assert not calls and branch.owner == "recovery"

    return {
        "automatic_takeover": True,
        "protect_duplicate_timestamp_rejected": True,
        "protect_nonfinite_rejected": True,
        "offline_hold": True,
        "stale_future_pre_takeover_rejected": True,
        "instability_restarts_dwell": True,
        "sample_gap_restarts_dwell": True,
        "anchor_before_teleop": True,
        "second_fall": True,
        "timestamp_reset": True,
        "persistent_down_no_timeout_s": 30,
    }
