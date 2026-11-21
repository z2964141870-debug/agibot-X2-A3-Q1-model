"""Historical replay with 5 ms physics. Command ramp mode groups ten target changes
into four physics steps. Use the exclusive-owner replay for 2 ms command integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("Q1_RECOVERY_RUN_ROOT", HERE)).resolve()
PROJECT_ROOT = Path(
    os.environ.get("SUPERMIMIC_ROOT", RUN.parent.parent if RUN.parent.name == "runs" else RUN.parent)
).resolve()
BUNDLE = Path(
    os.environ.get(
        "Q1_V2B_BUNDLE",
        RUN / "formal_03" / "pinned" / "q1_v2b_gantry_deploy_bundle",
    )
).resolve()
EXPERT = Path(
    os.environ.get("Q1_RECOVERY_EXPERT", RUN / "formal_03" / "q1_recovery_expert.onnx")
).resolve()
H3 = Path(os.environ.get("H003_BWM_ROOT", PROJECT_ROOT / "runs" / "h003_causal_bwm")).resolve()
SOURCE_ROOT = Path(os.environ.get("Q1_SOURCE_ROOT", PROJECT_ROOT / "src")).resolve()
sys.path[:0] = [str(RUN), str(BUNDLE / "code"), str(H3), str(SOURCE_ROOT)]

from bench import (  # noqa: E402
    BUNDLE as BENCH_BUNDLE,
    Q1V2bOnnxController,
    ContactGatedFootAnchorNP,
    LegHeight,
    build_obs_80_np,
    build_robot_pose_590_np,
    scene,
)
from q1_kinematics_np import (  # noqa: E402
    Q1KinematicsNP,
    mat_to_quat_wxyz,
    quat_wxyz_to_mat,
    quat_xyzw_to_mat,
    rotz_np,
    yaw_mat_np,
    yaw_angle_from_mat,
)
from q1_recovery_runtime import Q1RecoveryBranch  # noqa: E402
from recovery import Gate as BaseGate, Recovery as BaseRecovery  # noqa: E402
from q1_v2b_runtime import apply_torque_safe_pd_np  # noqa: E402
from reference import HandoffReference  # noqa: E402


DT_CONTROL = 0.02
STATE_DELAY_S = 0.011
MAX_TARGET_STEP = 0.2
COMMAND_TICKS_PER_INFERENCE = 10
COMMAND_TICKS_PER_PHYSICS = (2, 2, 3, 3)


class TunedGate(BaseGate):
    def __init__(self, tilt_ticks_required: int):
        self.tilt_ticks_required = int(tilt_ticks_required)
        self.reset()

    def advance(self, x, width):
        cosine = -float(x[49])
        self.tilt_ticks = self.tilt_ticks + 1 if cosine < .65 or (x[-1] and x[-2] < .19) else 0
        reset_anchor = False
        if self.mode in ("track", "handoff") and self.tilt_ticks >= self.tilt_ticks_required:
            self.mode = "recover"
            self.stand_ticks = self.blend_ticks = 0
        if self.mode == "recover":
            stable = cosine > .92 and np.sqrt(np.mean(x[:22] ** 2)) < .32 and x[-1] and x[-2] > .35
            self.stand_ticks = self.stand_ticks + 1 if stable else 0
            dwell = 40 if width > .35 else 25
            if self.stand_ticks >= dwell:
                self.mode = "handoff"
                reset_anchor = True
        if self.mode == "handoff":
            self.blend_ticks += 1
            if self.blend_ticks >= 30:
                self.mode = "track"
                return 0., reset_anchor
            return .5 * (1 + np.cos(np.pi * self.blend_ticks / 30)), reset_anchor
        return float(self.mode == "recover"), reset_anchor


class TunedRecovery(BaseRecovery):
    def __init__(self, ctrl, path, tilt_ticks_required: int):
        super().__init__(ctrl, path)
        self.gate = TunedGate(tilt_ticks_required)


class TunedRecoveryBranch(Q1RecoveryBranch):
    def __init__(self, tracker, expert_onnx, q1_xml, tilt_ticks_required: int):
        self.tracker = tracker
        self.legs = LegHeight(Path(q1_xml))
        self.expert = TunedRecovery(tracker, expert_onnx, tilt_ticks_required)


class CommandRamp:
    def __init__(self, initial: np.ndarray, max_command_step: float):
        self.current = np.asarray(initial, dtype=np.float32).copy()
        self.desired = self.current.copy()
        self.max_command_step = np.broadcast_to(
            np.asarray(max_command_step, dtype=np.float32), self.current.shape
        ).copy()

    def run_period(self, desired: np.ndarray):
        self.desired = np.asarray(desired, dtype=np.float32).copy()
        outputs = []
        max_delta = 0.0
        for command_count in COMMAND_TICKS_PER_PHYSICS:
            for _ in range(command_count):
                delta = np.clip(
                    self.desired - self.current,
                    -self.max_command_step,
                    self.max_command_step,
                )
                self.current = self.current + delta
                max_delta = max(max_delta, float(np.abs(delta).max()))
            outputs.append(self.current.copy())
        return outputs, max_delta


def blend_rotation(previous: np.ndarray, current: np.ndarray, delay_ratio: float) -> np.ndarray:
    mixed = (1.0 - delay_ratio) * current + delay_ratio * previous
    u, _, vh = np.linalg.svd(mixed)
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result.astype(np.float32)


class DelayedState:
    def __init__(self, delay_s: float):
        self.ratio = float(np.clip(delay_s / DT_CONTROL, 0.0, 1.0))
        self.previous = None

    def reset(self, state):
        self.previous = tuple(np.array(x, copy=True) for x in state)

    def read(self, state):
        if self.previous is None:
            self.reset(state)
        previous = self.previous
        current = state
        delayed = (
            blend_rotation(previous[0], current[0], self.ratio),
            (1.0 - self.ratio) * current[1] + self.ratio * previous[1],
            (1.0 - self.ratio) * current[2] + self.ratio * previous[2],
            (1.0 - self.ratio) * current[3] + self.ratio * previous[3],
        )
        self.previous = tuple(np.array(x, copy=True) for x in current)
        return delayed


def state_from_data(data):
    return (
        quat_wxyz_to_mat(data.qpos[3:7]).astype(np.float32),
        data.qvel[3:6].astype(np.float32).copy(),
        data.qpos[7:].astype(np.float32).copy(),
        data.qvel[6:].astype(np.float32).copy(),
    )


def clamp_target(target, previous, lo, hi, max_target_step):
    target = np.asarray(target, dtype=np.float32)
    if np.isfinite(max_target_step):
        clipped = np.clip(target, previous - max_target_step, previous + max_target_step)
    else:
        clipped = target.copy()
    clipped = np.clip(clipped, lo, hi)
    changed = np.abs(clipped - target) > 1e-7
    return clipped.astype(np.float32), changed


def trace_row(data, tau, previous_action, action, gate, target_changed, previous_qd,
              target, previous_target, target_step_override=None):
    rotation = quat_wxyz_to_mat(data.qpos[3:7])
    qd = data.qvel[6:].astype(np.float32)
    qdd = (qd - previous_qd) / DT_CONTROL
    depth = max(0.0, -min((contact.dist for contact in data.contact), default=0.0))
    return [
        float(data.qpos[2]),
        float(rotation[2, 2]),
        float(np.abs(tau).max()),
        float(np.count_nonzero(np.abs(tau) > 0.98 * 36.0)),
        float(np.abs(action - previous_action).mean()),
        float(gate),
        float(depth),
        float(np.abs(qd).max()),
        float(np.abs(qdd).max()),
        float(np.abs(target - previous_target).max() if target_step_override is None else target_step_override),
        float(target_changed.mean()),
    ]


def score_e1(trace):
    upright = (trace[:, 0] >= 0.39) & (trace[:, 1] >= 0.85)
    count = 0
    first = None
    for step, ok in enumerate(upright, 1):
        count = count + 1 if ok else 0
        if count == 50 and first is None:
            first = step * DT_CONTROL
    return {
        "success": bool(count >= 50),
        "first_stable_s": first,
        "end_z": float(trace[-1, 0]),
        "end_cos": float(trace[-1, 1]),
        "terminal_stable_ticks": count,
        "torque_near_limit_ticks": int(np.count_nonzero(trace[:, 3])),
        "target_clip_ticks": int(np.count_nonzero(trace[:, 10] > 0.0)),
        "target_clip_fraction": float(trace[:, 10].mean()),
        "max_penetration_mm": float(trace[:, 6].max() * 1000.0),
        "max_joint_speed_est_rad_s": float(trace[:, 7].max()),
        "p99_joint_speed_est_rad_s": float(np.percentile(trace[:, 7], 99)),
        "max_joint_accel_est_rad_s2": float(trace[:, 8].max()),
        "max_target_step_rad": float(trace[:, 9].max()),
    }


def evaluate_e1(output: Path, seed: int, state_delay_s: float, max_target_step: float,
                expert_path: Path, e1_steps: int, tilt_ticks_required: int,
                command_ramp: bool, command_frame_step: float,
                upper_body_command_step: float | None):
    output.mkdir(parents=True, exist_ok=False)
    model = scene(output / "scene")
    ctrl = Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    q1_xml = BUNDLE / "robot_model" / "supermimic.xml"
    kin = ContactGatedFootAnchorNP(Q1KinematicsNP(q1_xml))
    legs = LegHeight(q1_xml)
    results = []
    rng = np.random.default_rng(seed)
    angles = [(0, np.deg2rad(88 + rng.uniform(-5, 5))) for _ in range(8)]
    angles += [(0, np.deg2rad(-88 + rng.uniform(-5, 5))) for _ in range(8)]
    angles += [(np.deg2rad((1 if i % 2 == 0 else -1) * (88 + rng.uniform(-5, 5))), 0) for i in range(8)]

    for i, (roll, pitch) in enumerate(angles):
        yaw = rng.uniform(-np.pi, np.pi)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        quat = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
        q = np.clip(ctrl.default_q + rng.uniform(-0.15, 0.15, 22), ctrl.q_lo, ctrl.q_hi)
        qpos = np.r_[0, 0, 0.16, quat, q]
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.ctrl[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data, nstep=20)
        ctrl.reset(data.qpos[7:].astype(np.float32))
        expert = TunedRecoveryBranch(ctrl, expert_path, BUNDLE / "robot_model/supermimic.xml", tilt_ticks_required)
        expert.reset()
        delayed = DelayedState(state_delay_s)
        ref_R = np.eye(3, dtype=np.float32)
        modal = build_robot_pose_590_np(
            np.zeros((1, 3), np.float32), ref_R[None], ctrl.default_q[None], 0
        )
        previous_action = np.zeros(22, dtype=np.float32)
        previous_target = data.qpos[7:].astype(np.float32).copy()
        command_step = np.full(22, command_frame_step, dtype=np.float32)
        if upper_body_command_step is not None:
            command_step[13:] = upper_body_command_step
        ramp = CommandRamp(previous_target, command_step) if command_ramp else None
        previous_qd = data.qvel[6:].astype(np.float32).copy()
        first_state = state_from_data(data)
        delayed.reset(first_state)
        trace = []
        states = []
        for _ in range(e1_steps):
            delayed_state = delayed.read(state_from_data(data))
            R, omega, q, qd = delayed_state
            if len(trace) == 0:
                kin.reset(R, omega, q, qd, np.zeros(2, dtype=np.float32), R_ref=ref_R)

            def observation(exy):
                return build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R,
                                       w_body=omega, a_prev=previous_action, R_ref=ref_R,
                                       cgfa_exy=exy)

            def base():
                exy, _ = kin.step(R, omega, q, qd, np.zeros(2, dtype=np.float32), R_ref=ref_R)
                return ctrl.step(observation(exy), modal, q, qd)

            def reanchor():
                ref_R[:] = yaw_mat_np(R)
                kin.reset(R, omega, q, qd, np.zeros(2, dtype=np.float32), R_ref=ref_R)
                ctrl.reset(q)

            desired_target, tau, gate = expert.step(R, omega, q, qd, previous_action, base, reanchor)
            if command_ramp:
                desired_target = np.clip(desired_target, ctrl.q_lo, ctrl.q_hi)
                command_targets, command_max_step = ramp.run_period(desired_target)
                for command_target in command_targets:
                    data.ctrl[:] = command_target
                    mujoco.mj_step(model, data, nstep=1)
                target = command_targets[-1]
                changed = np.abs(target - desired_target) > 1e-7
            else:
                target, changed = clamp_target(desired_target, previous_target, ctrl.q_lo, ctrl.q_hi, max_target_step)
                command_max_step = None
                data.ctrl[:] = target
                mujoco.mj_step(model, data, nstep=4)
            action = (target - ctrl.default_q) / ctrl.action_scale
            trace.append(trace_row(data, tau, previous_action, action, gate, changed, previous_qd,
                                   target, previous_target, command_max_step))
            states.append(data.qpos.copy())
            previous_action = action
            previous_target = target
            previous_qd = data.qvel[6:].astype(np.float32).copy()
        trace = np.asarray(trace, dtype=np.float64)
        row = {"trial": i, "pose": ("prone", "supine", "side")[i // 8], **score_e1(trace)}
        results.append(row)
        np.savez_compressed(output / f"trial_{i:02}.npz", trace=trace, qpos=np.asarray(states), initial=qpos)
        (output / "trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def evaluate_relay(output: Path, tracks_dir: Path, state_delay_s: float, max_target_step: float,
                   expert_path: Path, tilt_ticks_required: int, command_ramp: bool,
                   command_frame_step: float, upper_body_command_step: float | None):
    output.mkdir(parents=True, exist_ok=False)
    ctrl = Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    report = []
    for i, (track, kick) in enumerate([
        ("straight_walk_50hz.csv", (0, 2.7)),
        ("straight_walk_50hz.csv", (0, -2.7)),
        ("loop_figure8_walk.csv", (-2.7, 0)),
        ("loop_figure8_walk.csv", (2.7, 0)),
        ("loop_figure8_walk.csv", (1.91, 1.91)),
    ]):
        clip = np.loadtxt(tracks_dir / track, delimiter=",").astype(np.float32)
        pos = clip[:, :3]
        rot = quat_xyzw_to_mat(clip[:, 3:7])
        joints = clip[:, 7:]
        model_dir = output / f"model_{i:02}"
        model_dir.mkdir()
        model = scene(model_dir)
        data = mujoco.MjData(model)
        data.qpos[:] = np.r_[pos[0], mat_to_quat_wxyz(rot[0]), joints[0]]
        data.ctrl[:] = joints[0]
        mujoco.mj_forward(model, data)
        expert = TunedRecoveryBranch(ctrl, expert_path, BUNDLE / "robot_model/supermimic.xml", tilt_ticks_required)
        expert.reset()
        kin = ContactGatedFootAnchorNP(Q1KinematicsNP(BUNDLE / "robot_model" / "supermimic.xml"))
        delayed = DelayedState(state_delay_s)
        delayed.reset(state_from_data(data))
        previous_action = np.zeros(22, dtype=np.float32)
        ctrl.reset(joints[0])
        previous_target = joints[0].copy()
        command_step = np.full(22, command_frame_step, dtype=np.float32)
        if upper_body_command_step is not None:
            command_step[13:] = upper_body_command_step
        ramp = CommandRamp(previous_target, command_step) if command_ramp else None
        previous_qd = data.qvel[6:].astype(np.float32).copy()
        align = np.eye(3, dtype=np.float32)
        offset = np.zeros(3, dtype=np.float32)
        display_offset = np.zeros(3, dtype=np.float32)
        ref_window = None
        idx = 0
        fallen = False
        handed = False
        walked = 0
        released = False
        refall = False
        states, ghosts, trace, widths = [], [], [], []
        contact_after = False

        for tick in range(850):
            if tick == 100:
                data.qvel[:2] += kick
            current = state_from_data(data)
            R_now, _, _, _ = current
            delayed_state = delayed.read(current)
            R, omega, q, qd = delayed_state
            _, _, _, width_now = kin.compute_sole_fk(*current)
            widths.append(width_now)

            goal_p, goal_R, goal_q = align @ pos[idx] + offset, align @ rot[idx], joints[idx]

            def observation(exy):
                return build_obs_80_np(q=q, qd=qd, default_q=ctrl.default_q, R_imu=R,
                                       w_body=omega, a_prev=previous_action, R_ref=goal_R,
                                       cgfa_exy=exy)

            def base():
                nonlocal goal_p, goal_R, goal_q
                if ref_window is None:
                    modal = build_robot_pose_590_np(pos, rot, joints, idx)
                else:
                    goal_p, goal_R, goal_q, modal = ref_window.step(align @ pos[idx] + offset, align @ rot[idx], joints[idx])
                exy, _ = kin.step(R, omega, q, qd, goal_p[:2], R_ref=goal_R)
                return ctrl.step(observation(exy), modal, q, qd)

            def reanchor():
                nonlocal align, offset, ref_window
                align = rotz_np(yaw_angle_from_mat(R) - yaw_angle_from_mat(rot[idx]))
                offset[:2] = -(align @ pos[idx])[:2]
                display_offset[:2] = data.qpos[:2]
                ref_window = HandoffReference(align @ pos[idx] + offset, R, q)
                kin.reset(R, omega, q, qd, ref_window.origin[:2], R_ref=ref_window.rotation)

            desired_target, tau, gate = expert.step(R, omega, q, qd, previous_action, base, reanchor)
            if command_ramp:
                desired_target = np.clip(desired_target, ctrl.q_lo, ctrl.q_hi)
                command_targets, command_max_step = ramp.run_period(desired_target)
                target = command_targets[-1]
                changed = np.abs(target - desired_target) > 1e-7
            else:
                target, changed = clamp_target(desired_target, previous_target, ctrl.q_lo, ctrl.q_hi, max_target_step)
                command_max_step = None

            if tick > 100:
                for contact in data.contact:
                    names = [model.geom(int(j)).name or "" for j in (contact.geom1, contact.geom2)]
                    if "floor" in names and any(x in n for x in ("pelvis", "elbow", "wrist", "torso", "shoulder") for n in names):
                        contact_after = True
                fallen |= bool(contact_after and (R_now[2, 2] < 0.65 or data.qpos[2] < 0.22))
                if fallen and gate > 0:
                    handed = True
                if released and gate == 1:
                    refall = True
                if handed and gate == 0:
                    released = True
                if handed and gate == 0 and data.qpos[2] >= 0.35 and R_now[2, 2] > 0.85:
                    walked += 1
                else:
                    walked = 0

            action = (target - ctrl.default_q) / ctrl.action_scale
            trace.append(trace_row(data, tau, previous_action, action, gate, changed, previous_qd,
                                   target, previous_target, command_max_step))
            ghosts.append(np.r_[goal_p + display_offset, mat_to_quat_wxyz(goal_R), goal_q])
            states.append(data.qpos.copy())
            previous_action = action
            previous_target = target
            previous_qd = data.qvel[6:].astype(np.float32).copy()
            if gate < 1.0 and (ref_window is None or ref_window.tick >= 30):
                idx = min(idx + 1, len(pos) - 1)
            if not command_ramp:
                data.ctrl[:] = target
                mujoco.mj_step(model, data, nstep=4)
            else:
                for command_target in command_targets:
                    data.ctrl[:] = command_target
                    mujoco.mj_step(model, data, nstep=1)
            if walked == 150 or (tick >= 700 and not released):
                break

        trace = np.asarray(trace, dtype=np.float64)
        width = float(np.mean(widths[-150:])) if walked == 150 else None
        displacement = float(np.linalg.norm(states[-1][:2] - states[-150][:2])) if walked == 150 else 0.0
        row = {
            "trial": i,
            "track": track,
            "kick": kick,
            "fallen": fallen,
            "took_control": handed,
            "walking_ticks": walked,
            "foot_width_cm": width,
            "walk_disp_m": displacement,
            "target_clip_ticks": int(np.count_nonzero(trace[:, 10] > 0.0)),
            "target_clip_fraction": float(trace[:, 10].mean()),
            "max_penetration_mm": float(trace[:, 6].max() * 1000.0),
            "max_joint_speed_est_rad_s": float(trace[:, 7].max()),
            "p99_joint_speed_est_rad_s": float(np.percentile(trace[:, 7], 99)),
            "max_joint_accel_est_rad_s2": float(trace[:, 8].max()),
            "max_target_step_rad": float(trace[:, 9].max()),
            "refall": refall,
            "success": bool(fallen and handed and not refall and walked == 150 and width <= 22 and displacement > 0.25),
        }
        report.append(row)
        np.savez_compressed(output / f"relay_{i}.npz", qpos=np.asarray(states), reference=np.asarray(ghosts), trace=trace, foot_width_cm=np.asarray(widths))
        (output / "relay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--tracks", type=Path, default=Path("/tmp/q1_v2b_narrow_tracks"))
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--state-delay-ms", type=float, default=11.0)
    parser.add_argument("--max-target-step-rad", type=float, default=0.2)
    parser.add_argument("--expert", type=Path, default=EXPERT)
    parser.add_argument("--e1-steps", type=int, default=300)
    parser.add_argument("--gate-tilt-ticks", type=int, default=3)
    parser.add_argument("--command-ramp", action="store_true")
    parser.add_argument("--command-frame-step-rad", type=float, default=0.2)
    parser.add_argument("--upper-body-command-frame-step-rad", type=float, default=None)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.tracks = args.tracks.resolve()
    args.expert = args.expert.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "expert": str(args.expert),
        "expert_sha256": hashlib.sha256(args.expert.read_bytes()).hexdigest(),
        "state_delay_s": args.state_delay_ms / 1000.0,
        "control_hz": 50.0,
        "command_repeat_hz": 500.0,
        "max_target_step_rad": args.max_target_step_rad,
        "gate_tilt_ticks": args.gate_tilt_ticks,
        "command_ramp": args.command_ramp,
        "command_frame_step_rad": args.command_frame_step_rad,
        "upper_body_command_frame_step_rad": args.upper_body_command_frame_step_rad,
        "physics_timestep_s": 0.005,
        "physics_steps_per_control": 4,
        "trace_columns": ["pelvis_z", "upright_cos", "max_abs_tau", "near_tau_limit_count",
                           "action_change_mean", "gate", "penetration_m", "max_abs_joint_speed",
                           "max_abs_joint_accel", "max_target_step", "target_clip_fraction"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    e1_dir = args.output / "e1"
    relay_dir = args.output / "relay"
    e1 = evaluate_e1(e1_dir, args.seed, args.state_delay_ms / 1000.0, args.max_target_step_rad,
                     args.expert, args.e1_steps, args.gate_tilt_ticks,
                     args.command_ramp, args.command_frame_step_rad,
                     args.upper_body_command_frame_step_rad)
    relay = evaluate_relay(relay_dir, args.tracks, args.state_delay_ms / 1000.0,
                           args.max_target_step_rad, args.expert, args.gate_tilt_ticks,
                           args.command_ramp, args.command_frame_step_rad,
                           args.upper_body_command_frame_step_rad)
    summary = {
        "e1": {
            "successes": sum(r["success"] for r in e1),
            "total": len(e1),
            "by_pose": {pose: sum(r["success"] for r in e1 if r["pose"] == pose) for pose in ("prone", "supine", "side")},
            "max_penetration_mm": max(r["max_penetration_mm"] for r in e1),
            "max_speed_est_rad_s": max(r["max_joint_speed_est_rad_s"] for r in e1),
            "max_accel_est_rad_s2": max(r["max_joint_accel_est_rad_s2"] for r in e1),
        },
        "relay": {
            "successes": sum(r["success"] for r in relay),
            "total": len(relay),
            "max_penetration_mm": max(r["max_penetration_mm"] for r in relay),
            "max_speed_est_rad_s": max(r["max_joint_speed_est_rad_s"] for r in relay),
            "max_accel_est_rad_s2": max(r["max_joint_accel_est_rad_s2"] for r in relay),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
