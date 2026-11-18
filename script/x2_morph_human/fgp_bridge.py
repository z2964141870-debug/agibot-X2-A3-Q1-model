"""Latest-FGP (CLOTHO [24,3,3]) -> morph_human_frame bridge and deterministic SMPL24 presets."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from . import artifacts
from .core import Calibration, DT_NS, PARENTS, SmplAdapter, axis_angle, inv, mul, rotate


def quat_to_matrix(q):
    """Unit quaternion [..., 4] (wxyz) -> rotation matrix [..., 3, 3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], axis=-1),
            np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], axis=-1),
            np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], axis=-1),
        ],
        axis=-2,
    )


def stand_local_quats_and_root(skel_doc: dict, world_from_smpl_wxyz):
    stand = skel_doc["stand"]
    world = np.asarray(stand["quat"], dtype=np.float64)
    c = np.asarray(world_from_smpl_wxyz, dtype=np.float64)
    local_q = np.empty((24, 4), dtype=np.float64)
    local_q[0] = mul(inv(c), world[0])
    for i in range(1, 24):
        local_q[i] = mul(inv(world[PARENTS[i]]), world[i])
    local_q = local_q * np.where(local_q[:, :1] < 0, -1.0, 1.0)
    root_smpl = rotate(inv(c), np.asarray(stand["pos"], dtype=np.float64)[0])
    return local_q, root_smpl


def convert_fgp_npz(
    input_npz: Path,
    calibration_path: Path,
    output_jsonl: Path,
    *,
    align_initial_ground: bool = False,
    root: Path | None = None,
) -> dict:
    root = artifacts.activate(root)
    cal_raw = json.loads(calibration_path.read_text())
    cal = Calibration(**cal_raw)
    skel_path = root / "upstream/deploy/vr_teleop/human_skeleton.json"
    smpl = SmplAdapter(skel_path, cal)

    with np.load(input_npz, allow_pickle=False) as d:
        poses = np.asarray(d["pose_mat"], dtype=np.float64)
        roots = np.asarray(d["root_translation_m"], dtype=np.float64)
        times = np.asarray(d["capture_ns"], dtype=np.int64)
        clock = str(d["capture_clock"].item())

    if poses.ndim != 4 or poses.shape[1:] != (24, 3, 3) or roots.shape != (len(poses), 3):
        raise ValueError(f"Expected pose_mat[T,24,3,3] and root_translation_m[T,3], got {poses.shape}, {roots.shape}")
    if times.shape != (len(poses),) or len(times) < 10 or (np.diff(times) <= 0).any():
        raise ValueError("capture_ns must strictly increase with at least 10 frames")

    ground_z_shift_m = 0.0
    if align_initial_ground:
        pos0, _ = smpl.convert_matrices(poses[0], roots[0])
        stand_pos = np.asarray(smpl.doc["stand"]["pos"], dtype=np.float64)
        target_ankle_z = float(np.min(stand_pos[[7, 8], 2]))
        current_ankle_z = float(np.min(pos0[[7, 8], 2]))
        ground_z_shift_m = target_ankle_z - current_ankle_z
        new_offset = (
            float(cal.world_offset_m[0]),
            float(cal.world_offset_m[1]),
            float(cal.world_offset_m[2] + ground_z_shift_m),
        )
        cal = Calibration(
            id=f"{cal.id}_initground",
            world_from_smpl_wxyz=cal.world_from_smpl_wxyz,
            translation_scale=cal.translation_scale,
            world_offset_m=new_offset,
        )
        smpl = SmplAdapter(skel_path, cal)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    t0 = int(times[0])
    with open(output_jsonl, "x", encoding="utf-8") as f:
        for i, (pose_m, root_m, t_ns) in enumerate(zip(poses, roots, times)):
            pkt = smpl.packet_from_matrices(
                pose_m,
                root_m,
                sequence=i,
                capture_ns=int(t_ns),
                capture_clock=clock,
                arrival_ns=int(t_ns - t0),
                arrival_clock="replay_monotonic",
            )
            f.write(json.dumps(pkt, allow_nan=False) + "\n")

    dt_ms = np.diff(times) / 1e6
    return {
        "status": "FGP_NPZ_CONVERTED",
        "frames": int(len(times)),
        "duration_s": float((times[-1] - times[0]) / 1e9),
        "source_dt_ms": {
            "median": float(np.median(dt_ms)),
            "p95": float(np.percentile(dt_ms, 95)),
            "max": float(np.max(dt_ms)),
        },
        "calibration_id": smpl.calibration.id,
        "world_offset_m": list(smpl.calibration.world_offset_m),
        "ground_z_shift_m": float(ground_z_shift_m),
        "input_sha256": artifacts.sha(input_npz),
        "output": str(output_jsonl),
        "output_sha256": artifacts.sha(output_jsonl),
    }


def generate_synth_smpl_npz(preset: str, frames: int, output_npz: Path, *, root: Path | None = None) -> dict:
    """Generate deterministic SMPL24 [T,24,3,3] Y-up motions for controlled locomotion & arm tests."""
    if frames < 60 or frames > 2000:
        raise ValueError("frames must be in [60, 2000]")
    root = artifacts.activate(root)
    skel_doc = json.loads((root / "upstream/deploy/vr_teleop/human_skeleton.json").read_text())
    cal = Calibration("synth_smpl_yup_v1")
    base_q, base_root_yup = stand_local_quats_and_root(skel_doc, cal.world_from_smpl_wxyz)
    # Shift base_root_yup.y by -0.97 so that clotho_fgp_yup.json (+0.97 m in world Z) lands exact stand height!
    base_root_clotho_yup = base_root_yup.copy()
    base_root_clotho_yup[1] -= 0.97

    dt = 0.02
    poses = np.empty((frames, 24, 3, 3), dtype=np.float64)
    roots = np.empty((frames, 3), dtype=np.float64)
    capture_ns = np.arange(frames, dtype=np.int64) * DT_NS

    # Note: In SMPL Y-up rest coordinates:
    # +X is left, +Y is up, +Z is forward!
    # Pitching leg forward (flexion) rotates around -X; knee flexion rotates around +X; ankle dorsiflexion around -X.
    if preset == "walk_start_stop":
        x_fwd = 0.0
        for i in range(frames):
            t = i * dt
            total_t = (frames - 1) * dt
            # Smooth envelope: 0..1.2s stand, 1.2..2.4s ramp up, cruise, ramp down last 2.0s to stand
            if t < 1.0:
                env = 0.0
            elif t < 2.2:
                u = (t - 1.0) / 1.2
                env = u * u * (3 - 2 * u)
            elif t < total_t - 2.2:
                env = 1.0
            elif t < total_t - 1.0:
                u = (total_t - 1.0 - t) / 1.2
                env = u * u * (3 - 2 * u)
            else:
                env = 0.0

            freq = 0.95  # Hz gait cycle (~1.9 steps/s)
            phase = 2.0 * np.pi * freq * max(0.0, t - 1.0)
            v_fwd = 0.30 * env  # 0.30 m/s forward in SMPL +Z
            x_fwd += v_fwd * dt

            q_frame = base_q.copy()
            s_l = np.sin(phase)
            s_r = np.sin(phase + np.pi)
            swing_l = max(0.0, s_l)
            swing_r = max(0.0, s_r)

            # Left/Right hip flexion (joints 1, 2), knee flexion (4, 5), ankle pitch (7, 8)
            hip_amp = 0.28 * env
            knee_amp = 0.52 * env
            q_frame[1] = mul(q_frame[1], axis_angle(np.array([-hip_amp * s_l, 0.0, 0.02 * env * s_l])))
            q_frame[2] = mul(q_frame[2], axis_angle(np.array([-hip_amp * s_r, 0.0, -0.02 * env * s_r])))
            q_frame[4] = mul(q_frame[4], axis_angle(np.array([knee_amp * swing_l, 0.0, 0.0])))
            q_frame[5] = mul(q_frame[5], axis_angle(np.array([knee_amp * swing_r, 0.0, 0.0])))
            q_frame[7] = mul(q_frame[7], axis_angle(np.array([-0.22 * env * swing_l + 0.08 * env * s_l, 0.0, 0.0])))
            q_frame[8] = mul(q_frame[8], axis_angle(np.array([-0.22 * env * swing_r + 0.08 * env * s_r, 0.0, 0.0])))

            # Counter-phase arm swing (shoulders 16, 17; elbows 18, 19) + slight pelvis sway
            q_frame[0] = mul(q_frame[0], axis_angle(np.array([0.0, 0.03 * env * s_l, 0.025 * env * np.cos(phase)])))
            q_frame[16] = mul(q_frame[16], axis_angle(np.array([0.22 * env * s_l, 0.0, 0.0])))
            q_frame[17] = mul(q_frame[17], axis_angle(np.array([0.22 * env * s_r, 0.0, 0.0])))
            q_frame[18] = mul(q_frame[18], axis_angle(np.array([-0.18 * env * max(0.0, -s_l), 0.0, 0.0])))
            q_frame[19] = mul(q_frame[19], axis_angle(np.array([-0.18 * env * max(0.0, -s_r), 0.0, 0.0])))

            poses[i] = quat_to_matrix(q_frame)
            root_cur = base_root_clotho_yup.copy()
            root_cur[0] += 0.018 * env * np.cos(phase)  # slight lateral CoM shift toward stance foot
            root_cur[1] += -0.012 * env * (1.0 - np.abs(np.cos(phase)))
            root_cur[2] += x_fwd
            roots[i] = root_cur

    elif preset == "turn_and_wave":
        for i in range(frames):
            t = i * dt
            total_t = (frames - 1) * dt
            if t < 0.8:
                env = 0.0
            elif t < 1.8:
                u = (t - 0.8) / 1.0
                env = u * u * (3 - 2 * u)
            elif t < total_t - 1.8:
                env = 1.0
            elif t < total_t - 0.8:
                u = (total_t - 0.8 - t) / 1.0
                env = u * u * (3 - 2 * u)
            else:
                env = 0.0

            q_frame = base_q.copy()
            # Pelvis + spine gentle yaw turn around SMPL +Y
            yaw = 0.35 * env * np.sin(2.0 * np.pi * 0.22 * max(0.0, t - 0.8))
            q_frame[0] = mul(q_frame[0], axis_angle(np.array([0.0, yaw * 0.65, 0.0])))
            q_frame[3] = mul(q_frame[3], axis_angle(np.array([0.0, yaw * 0.35, 0.0])))
            # Distinct elbow/shoulder articulation so full (24-joint) vs five (5-point) differs measurably
            wave = np.sin(2.0 * np.pi * 0.85 * max(0.0, t - 0.8))
            q_frame[16] = mul(q_frame[16], axis_angle(np.array([-0.45 * env, 0.25 * env * wave, 0.35 * env])))
            q_frame[18] = mul(q_frame[18], axis_angle(np.array([-0.75 * env * (0.5 + 0.5 * wave), -0.45 * env, 0.0])))
            q_frame[17] = mul(q_frame[17], axis_angle(np.array([-0.35 * env * (0.5 - 0.5 * wave), -0.20 * env * wave, -0.30 * env])))
            q_frame[19] = mul(q_frame[19], axis_angle(np.array([-0.65 * env * (0.5 - 0.5 * wave), 0.40 * env, 0.0])))

            poses[i] = quat_to_matrix(q_frame)
            root_cur = base_root_clotho_yup.copy()
            root_cur[0] += 0.02 * env * np.sin(2.0 * np.pi * 0.22 * max(0.0, t - 0.8))
            roots[i] = root_cur
    else:
        raise ValueError(f"Unknown preset: {preset}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        pose_mat=poses,
        root_translation_m=roots,
        capture_ns=capture_ns,
        capture_clock=np.array(f"synthetic_{preset}_50hz"),
    )
    return {
        "status": "SYNTH_SMPL_NPZ_CREATED",
        "preset": preset,
        "frames": int(frames),
        "duration_s": float((capture_ns[-1] - capture_ns[0]) / 1e9),
        "output_npz": str(output_npz),
        "output_sha256": artifacts.sha(output_npz),
    }
