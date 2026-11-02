"""Q1 software PD law; to be wired into mjlab's per-physics-step actuator.

Constants mirror PrimebotTech/SuperMimic 118338b. Numerical source parity is
checked by verify_contract.py; this module alone is not a simulator adapter.
"""

import math

import torch


PD = {
    "hip_pitch_joint": (40., 1.), "hip_roll_joint": (40., 1.),
    "hip_yaw_joint": (40., 1.), "knee_joint": (40., 1.),
    "ankle_pitch_joint": (20., 1.), "ankle_roll_joint": (10., 1.),
    "waist_yaw_joint": (80., 2.), "waist_roll_joint": (80., 2.),
    "shoulder": (80., 2.), "elbow": (80., 2.),
}


def parameters(joint_names, *, device="cpu", dtype=torch.float32):
    rows = []
    for name in joint_names:
        hits = [value for key, value in PD.items() if key in name]
        big = any(part in name for part in ("hip_", "knee_", "waist_"))
        small = any(part in name for part in ("ankle_", "shoulder_", "elbow_"))
        if len(hits) != 1 or big == small:
            raise ValueError(f"Unknown or ambiguous Q1 actuator {name!r}")
        rows.append((*hits[0], *((34., 7.5, 17.) if big else (21., 8., 16.))))
    if not rows:
        raise ValueError("Expected nonempty Q1 joint names")
    return torch.tensor(rows, device=device, dtype=dtype).unbind(-1)


def torque_bounds(velocity, peak, knee, cutoff):
    slope = peak / (cutoff - knee)
    upper = (peak - slope * (velocity - knee)).clamp(min=0).minimum(peak)
    lower = (-peak - slope * (velocity + knee)).clamp(max=0).maximum(-peak)
    return lower, upper


def delayed_target(current, previous, delay_s, substep, physics_dt=0.002):
    if (not math.isfinite(physics_dt) or physics_dt <= 0 or
            not math.isclose(round(0.02 / physics_dt) * physics_dt, 0.02)):
        raise ValueError("Physics dt must divide the 20ms control period")
    if type(substep) is not int or not 0 <= substep < round(0.02 / physics_dt):
        raise ValueError("Substep outside the 20ms control period")
    # The source rounds delays to physics steps before comparing the step index.
    delay = (delay_s / physics_dt).round().long()
    return torch.where(substep >= delay, current, previous)


def motor_control(target, position, velocity, kp, kd, peak, knee, cutoff,
                  gear, ctrl_limits):
    torque = kp * (target - position) - kd * velocity
    lower, upper = torque_bounds(velocity, peak, knee, cutoff)
    torque = torque.maximum(lower).minimum(upper)
    return (torque / gear).maximum(ctrl_limits[..., 0]).minimum(ctrl_limits[..., 1])
