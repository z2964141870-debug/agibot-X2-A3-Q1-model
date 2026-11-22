"""Run Q1 MuJoCo with recorded or live FGP input."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from contextlib import contextmanager, nullcontext

import mujoco
import numpy as np

from q1_cgfa_obs_np import ContactGatedFootAnchorNP, PrebakedClipLibraryNP, build_obs_80_np
from q1_fgp_client import FGPHttpSource
from q1_fgp_smpl import FGPSMPLStream, SMPLTick
from q1_kinematics_np import Q1KinematicsNP, yaw_mat_np
from q1_v2b_runtime import Q1V2bOnnxController
from replay_q1_walk_start import load_standing_feedback
from sim_q1_stand_start import load_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/home/yu/projects/SuperMimicHP/src/unisupermimic/assets/robots/q1/assets")


def iter_fgp_ticks(
    path: Path, dt: float, duration_s: float, start_s: float = 0.0,
    *, standing_height_m: float | None = None,
):
    stream = FGPSMPLStream(dt, standing_height_m=standing_height_m)
    first_stamp = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            stamp = int(sample["source_ns"])
            if first_stamp is None:
                first_stamp = stamp
            elapsed = (stamp - first_stamp) * 1e-9
            if elapsed < start_s:
                continue
            if elapsed >= start_s + duration_s:
                break
            yield from stream.push_sample(sample)


def iter_live_ticks(source: FGPHttpSource, dt: float, duration_s: float):
    deadline = next_tick = time.monotonic()
    deadline += duration_s
    source.start()
    try:
        while max(next_tick, time.monotonic()) < deadline:
            time.sleep(max(0.0, next_tick - time.monotonic()))
            queued, _ = source.get_latest()
            age = None if queued is None else time.monotonic_ns() - queued.arrival_ns
            yield queued.tick if age is not None and 0 <= age <= source.fresh_ns else None
            next_tick = max(next_tick + dt, time.monotonic())
    finally:
        source.stop()


@contextmanager
def passive_viewer(model, data):
    from mujoco.viewer import launch_passive

    display = launch_passive(model, data)
    try:
        yield display
    finally:
        display.close()
        # Close requests an exit. Wait for teardown before Python finalizes GLFW.
        deadline = time.monotonic() + 2.0
        try:
            while time.monotonic() < deadline and display.m is not None:
                time.sleep(0.01)
        except mujoco.UnexpectedError:
            pass


def rollout(
    model,
    controller,
    ticks,
    kin,
    prestand_s: float,
    initial: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    video_path: Path | None = None,
    *,
    modality: str = "human_smpl",
    viewer: bool = False,
    xy_feedback: str = "cgfa",
) -> dict:
    names = controller.cfg["joint_names"]
    pose_ids = [model.joint(name).qposadr[0] for name in names]
    velocity_ids = [model.joint(name).dofadr[0] for name in names]
    actuator_ids = [model.actuator(name).id for name in names]
    pelvis = model.body("pelvis").id
    root_angular_start = int(model.jnt_dofadr[model.body_jntadr[pelvis]]) + 3
    left_foot = model.geom("left_ankle_roll_link_collision_1").id
    right_foot = model.geom("right_ankle_roll_link_collision_1").id
    floor = model.geom("floor").id
    data = mujoco.MjData(model)
    initial_q = controller.default_q if initial is None else initial[0]
    initial_qd = np.zeros(22, dtype=np.float32) if initial is None else initial[1]
    initial_quat_xyzw = np.zeros(4, dtype=np.float32) if initial is None else initial[2]
    initial_gyro = np.zeros(3, dtype=np.float32) if initial is None else initial[3]
    if initial is None:
        initial_quat_xyzw[:] = [0.0, 0.0, 0.0, 1.0]
    data.qpos[0:3] = [0.0, 0.0, 0.42132428111574716]
    data.qpos[3:7] = [initial_quat_xyzw[3], initial_quat_xyzw[0], initial_quat_xyzw[1], initial_quat_xyzw[2]]
    data.qvel[root_angular_start:root_angular_start + 3] = initial_gyro
    data.qpos[pose_ids] = initial_q
    data.qvel[velocity_ids] = initial_qd
    data.ctrl[actuator_ids] = initial_q
    mujoco.mj_forward(model, data)

    dt = float(controller.cfg["dt_s"])
    substeps = round(dt / float(model.opt.timestep))
    if not np.isclose(substeps * model.opt.timestep, dt):
        raise ValueError("MuJoCo timestep must divide the 50 Hz policy timestep")
    cgfa = ContactGatedFootAnchorNP(kin, dt=dt)
    previous_action = np.zeros(22, dtype=np.float32)
    previous_target = controller.default_q.copy()
    segment = None
    heading = None
    origin = None
    reference = None
    rows = []
    video = None
    renderer = None
    if video_path is not None:
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(model, height=480, width=640)
        video = imageio.get_writer(video_path, fps=round(1 / dt), codec="libx264", quality=7)

    stop_reason = "completed"
    source_ticks = 0
    policy_steps = 0
    target_errors = []
    target_steps = []
    max_tilt = 0.0
    max_qd = 0.0
    stamp_ns = []
    reference_segments = []
    idle_ticks = 0
    window = passive_viewer(model, data) if viewer else nullcontext()
    try:
        for _ in range(round(prestand_s / dt)):
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if video is not None:
                renderer.update_scene(data)
                video.append_data(renderer.render())
        with window as display:
            for tick in ticks:
                if display is not None:
                    if not display.is_running():
                        stop_reason = "viewer_closed"
                        break
                    display.sync()
                if tick is None:
                    segment = None
                    idle_ticks += 1
                    continue
                rotation = data.xmat[pelvis].reshape(3, 3).astype(np.float32)
                # The free joint stores angular velocity in the pelvis frame used by obs_80.
                omega = data.qvel[root_angular_start:root_angular_start + 3].astype(np.float32)
                q = data.qpos[pose_ids].astype(np.float32)
                qd = data.qvel[velocity_ids].astype(np.float32)
                source_ticks += 1

                if tick.segment != segment:
                    segment = tick.segment
                    heading = yaw_mat_np(rotation) @ yaw_mat_np(tick.root_rotation).T
                    origin = tick.root_xy.astype(np.float32).copy()
                    world_origin = data.qpos[:2].copy()
                    reference = heading @ tick.root_rotation
                    cgfa.reset(rotation, omega, q, qd, np.zeros(2, dtype=np.float32), R_ref=reference)
                    controller.reset(q)
                    previous_action.fill(0.0)
                    previous_target = q.copy()

                p_ref_xy = heading[:2, :2] @ (tick.root_xy - origin)
                reference = heading @ tick.root_rotation
                if xy_feedback == "sim-truth":
                    delta = world_origin + p_ref_xy - data.qpos[:2]
                    foot_error = yaw_mat_np(rotation)[:2, :2].T @ delta
                    foot_error *= min(1.0, cgfa.bound_m / max(np.linalg.norm(foot_error), 1e-8))
                    foot_error = foot_error.astype(np.float32)
                else:
                    mode = "zeroed" if xy_feedback == "zeroed" else "cgfa_clipped_norm"
                    foot_error, _ = cgfa.step(rotation, omega, q, qd, p_ref_xy, R_ref=reference, mode=mode)
                obs = build_obs_80_np(
                    q=q,
                    qd=qd,
                    default_q=controller.default_q,
                    R_imu=rotation,
                    w_body=omega,
                    a_prev=previous_action,
                    R_ref=reference,
                    cgfa_exy=foot_error,
                )
                target, _, _ = controller.step(obs, tick.feature, q, qd, modality=modality)
                target = np.clip(target, previous_target - 0.2, previous_target + 0.2)
                target = np.clip(target, controller.default_q - 2.0, controller.default_q + 2.0)
                target_errors.append(float(np.max(np.abs(target - q))))
                target_steps.append(float(np.max(np.abs(target - previous_target))))
                previous_action = (target - controller.default_q) / controller.action_scale
                previous_target = target.copy()
                data.ctrl[actuator_ids] = target
                policy_steps += 1

                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)

                post_rotation = data.xmat[pelvis].reshape(3, 3)
                tilt = math.degrees(math.acos(float(np.clip(post_rotation[2, 2], -1.0, 1.0))))
                max_tilt = max(max_tilt, tilt)
                max_qd = max(max_qd, float(np.max(np.abs(data.qvel[velocity_ids]))))
                contacts = {"left": 0.0, "right": 0.0}
                for index in range(data.ncon):
                    contact = data.contact[index]
                    if floor not in (contact.geom1, contact.geom2):
                        continue
                    other = contact.geom2 if contact.geom1 == floor else contact.geom1
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    if other == left_foot:
                        contacts["left"] += float(force[0])
                    elif other == right_foot:
                        contacts["right"] += float(force[0])
                rows.append(
                    (
                        float(data.time),
                        float(data.qpos[0]),
                        float(data.qpos[1]),
                        float(data.qpos[2]),
                        tilt,
                        contacts["left"],
                        contacts["right"],
                        target_errors[-1],
                        target_steps[-1],
                        *p_ref_xy,
                        *foot_error,
                        math.degrees(math.acos(float(np.clip(reference[2, 2], -1.0, 1.0)))),
                    )
                )
                stamp_ns.append(tick.stamp_ns)
                reference_segments.append(tick.segment)
                if video is not None:
                    renderer.update_scene(data)
                    video.append_data(renderer.render())
                if data.qpos[2] < 0.15 or tilt >= 65.0:
                    stop_reason = "tilt_or_height_limit"
                    break
    finally:
        if renderer is not None:
            renderer.close()
        if video is not None:
            video.close()

    trace = np.asarray(rows, dtype=np.float64).reshape(-1, 14)
    if len(trace):
        tail = trace[-min(len(trace), 50):]
        bilateral = (tail[:, 5] > 5.0) & (tail[:, 6] > 5.0)
        final = trace[-1]
    else:
        tail = np.empty((0, 14))
        bilateral = np.empty(0, dtype=bool)
        final = np.full(14, np.nan)
    return {
        "input_ticks": source_ticks,
        "policy_steps": policy_steps,
        "simulated_s": round(float(data.time), 3),
        "stop_reason": stop_reason if policy_steps else "no_reference_ticks",
        "final_root_xyz_m": [round(float(x), 5) for x in final[1:4]] if len(trace) else None,
        "final_tilt_deg": round(float(final[4]), 3) if len(trace) else None,
        "max_tilt_deg": round(max_tilt, 3),
        "max_joint_velocity_rad_s": round(max_qd, 4),
        "target_error_p95_rad": round(float(np.percentile(target_errors, 95)), 5) if target_errors else None,
        "target_step_max_rad": round(max(target_steps, default=0.0), 5),
        "tail_bilateral_contact_fraction": round(float(np.mean(bilateral)), 3) if len(bilateral) else None,
        "trace": trace,
        "stamp_ns": np.asarray(stamp_ns, dtype=np.int64),
        "reference_segments": np.asarray(reference_segments, dtype=np.int64),
        "idle_ticks": idle_ticks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smpl-jsonl", type=Path)
    source.add_argument("--fgp-url")
    source.add_argument("--reference-clip", type=Path)
    parser.add_argument("--modality", choices=("human_smpl", "robot_pose", "planner_cmd"), default="human_smpl")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--fgp-standing-height", type=float, help="Measured standing pelvis height in meters")
    parser.add_argument("--prestand-s", type=float, default=0.2)
    parser.add_argument("--physics-ms", type=int, choices=(2, 5), default=2)
    parser.add_argument("--xy-feedback", choices=("cgfa", "zeroed", "sim-truth"), default="cgfa",
                        help="Root XY observation for MuJoCo comparisons")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--initial-standing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not args.assets.is_dir():
        parser.error(f"Q1 mesh assets not found: {args.assets}")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    if not np.isfinite(args.prestand_s) or args.prestand_s < 0:
        parser.error("--prestand-s must be nonnegative and finite")
    if not np.isfinite(args.start_s) or args.start_s < 0 or (args.start_s and not args.smpl_jsonl):
        parser.error("--start-s must be nonnegative and is only used with --smpl-jsonl")
    if not args.reference_clip and args.modality != "human_smpl":
        parser.error("FGP input requires human_smpl")
    if args.fgp_standing_height is not None and (
        args.reference_clip or not np.isfinite(args.fgp_standing_height) or args.fgp_standing_height <= 0
    ):
        parser.error("--fgp-standing-height requires FGP input and a positive finite height")
    if args.viewer and not args.fgp_url:
        parser.error("--viewer requires a live FGP URL")

    args.output.mkdir(parents=True)
    controller = Q1V2bOnnxController(ROOT, modalities=(args.modality,))
    dt = controller.cfg["dt_s"]
    kin = Q1KinematicsNP(ROOT / "robot_model/supermimic.xml")
    initial = None
    if args.initial_standing:
        feedback = load_standing_feedback(args.initial_standing, controller)
        initial = tuple(values[-1] for values in feedback[:4])
    http = None
    if args.fgp_url:
        http = FGPHttpSource(args.fgp_url, dt, standing_height_m=args.fgp_standing_height)
        ticks = iter_live_ticks(http, dt, args.duration)
    elif args.reference_clip:
        clip = PrebakedClipLibraryNP(args.reference_clip)
        ticks = (SMPLTick(1, round(i * dt * 1e9), *clip.get_step_inputs(i, args.modality))
                 for i in range(round(args.duration / dt)))
    else:
        ticks = iter_fgp_ticks(
            args.smpl_jsonl, dt, args.duration, args.start_s, standing_height_m=args.fgp_standing_height,
        )
    video_path = args.output / "q1_fgp_human_smpl_sim.mp4" if args.video else None
    model = load_model(args.assets)
    model.opt.timestep = args.physics_ms / 1000
    try:
        result = rollout(model, controller, ticks, kin, args.prestand_s, initial, video_path,
                         modality=args.modality, viewer=args.viewer, xy_feedback=args.xy_feedback)
    finally:
        ticks.close()
    trace = result.pop("trace")
    stamp_ns = result.pop("stamp_ns")
    reference_segments = result.pop("reference_segments")
    np.savez_compressed(
        args.output / "trace.npz",
        time_s=trace[:, 0],
        root_xyz_m=trace[:, 1:4],
        tilt_deg=trace[:, 4],
        left_contact=trace[:, 5],
        right_contact=trace[:, 6],
        target_error_rad=trace[:, 7],
        target_step_rad=trace[:, 8],
        reference_tick_stamp_ns=stamp_ns,
        reference_segment=reference_segments,
        reference_xy_m=trace[:, 9:11],
        cgfa_error_xy_m=trace[:, 11:13],
        reference_tilt_deg=trace[:, 13],
    )
    report = {
        "input": str(args.smpl_jsonl or args.reference_clip or args.fgp_url),
        "input_kind": "http" if http else "reference_clip" if args.reference_clip else "recorded_smpl",
        "policy_modality": args.modality,
        "xy_feedback": args.xy_feedback,
        "physics_ms": args.physics_ms,
        "prestand_s": args.prestand_s,
        "requested_duration_s": args.duration,
        "recording_start_s": args.start_s,
        "fgp_standing_height_m": args.fgp_standing_height,
        "http_status": http.status() if http else None,
        "reference_segments": np.unique(reference_segments).tolist(),
        "scene": str(ROOT / "robot_model/scene_supermimic_hard_floor.xml"),
        **result,
        "trace_path": str(args.output / "trace.npz"),
        "video_path": str(video_path) if video_path else None,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if result["stop_reason"] in ("completed", "viewer_closed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
