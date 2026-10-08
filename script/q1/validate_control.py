#!/usr/bin/env python3
"""Diagnose named-joint control with artificial support and a free-base contrast."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from q1_sim import JointReference, Q1Sim


def synthetic_reference(sim):
    times = np.arange(0, 24 + sim.reference_dt / 2, sim.reference_dt)
    positions = np.tile(sim.neutral, (len(times), 1))
    amplitudes = np.minimum(0.06, sim.upper - sim.neutral)
    for joint in range(22):
        phase = times - (joint + 1)
        active = (phase >= 0) & (phase <= 1)
        positions[active, joint] += amplitudes[joint] * (1 - np.cos(2 * np.pi * phase[active])) / 2
    reference = JointReference(times, positions, sim.names, sim.names)
    reference.validate_limits(sim)
    return reference, amplitudes


def render(sim):
    with mujoco.Renderer(sim.model, height=600, width=480) as renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, sim.initial_height + 0.08]
        camera.distance = 1.7
        camera.azimuth = 130
        camera.elevation = -12
        renderer.update_scene(sim.data, camera=camera)
        return renderer.render().copy()


def run_supported(config, directory, images):
    sim = Q1Sim(config, supported=True)
    reference, amplitudes = synthetic_reference(sim)
    reference.save(directory / "synthetic_joint_reference.npz")
    reference = JointReference.load(directory / "synthetic_joint_reference.npz", sim.names)
    reference.validate_limits(sim)
    if images is not None:
        images.append(("Artificial base support: initial", render(sim)))
    times, targets, states, velocities, torques = [], [], [], [], []
    started = time.perf_counter()
    while sim.data.time < reference.times[-1] - sim.dt / 2:
        target, velocity = reference.sample(sim.data.time)
        obs = sim.advance(target, velocity)
        times.append(obs["time_s"])
        targets.append(target)
        states.append(obs["joint_position_rad"])
        velocities.append(obs["joint_velocity_rad_s"])
        torques.append(sim.last_torque.copy())
        if images is not None and len(times) == 825:
            images.append(("Artificial support: arm pulse", render(sim)))
    elapsed = time.perf_counter() - started
    times, targets, states = map(np.asarray, (times, targets, states))
    errors = states - targets
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    excursions = []
    for joint in range(22):
        active = (times > joint + 1) & (times < joint + 2)
        before = np.searchsorted(times, joint + 1)
        excursions.append(float(np.max(states[active, joint]) - states[before, joint]))
    np.savez_compressed(directory / "supported_trace.npz", time_s=times,
                        target_rad=targets, position_rad=states,
                        velocity_rad_s=np.asarray(velocities), torque_nm=np.asarray(torques))
    result = {
        "scope": "SUPPORTED_SYNTHETIC_JOINT_DIAGNOSTIC_NOT_BALANCE",
        "duration_sim_s": float(sim.data.time),
        "wall_s": elapsed,
        "reference_frames": len(times),
        "max_joint_rmse_rad": float(rmse.max()),
        "max_abs_tracking_error_rad": float(np.abs(errors).max()),
        "torque_saturation_fraction": sim.torque_saturated / sim.torque_samples,
        "finite": bool(np.isfinite(states).all()),
        "all_joint_pulses_detected": all(e > 0.015 for e in excursions),
        "pipeline_passed": bool(rmse.max() < 0.08 and all(e > 0.015 for e in excursions)),
        "joints": [{"name": n, "target_amplitude_rad": float(amplitudes[i]),
                    "rmse_rad": float(rmse[i]), "measured_excursion_rad": excursions[i]}
                   for i, n in enumerate(sim.names)],
    }
    return result, sim.contract()


def run_free(config, directory, images):
    sim = Q1Sim(config, supported=False)
    if images is not None:
        images.append(("Free base: initial", render(sim)))
    rows = []
    while sim.data.time < 5 - sim.dt / 2:
        obs = sim.advance(sim.neutral)
        rows.append([obs["time_s"], *obs["root_position_m_world_privileged"],
                     *obs["projected_gravity_body"]])
        if sim.fallen():
            break
    if images is not None:
        images.append(("Free base: final (PD only)", render(sim)))
    np.savez_compressed(directory / "free_base_trace.npz", state=np.asarray(rows),
                        columns=np.asarray(["time_s", "x_m", "y_m", "z_m",
                                            "gravity_x", "gravity_y", "gravity_z"]))
    obs = sim.observe()
    return {
        "scope": "FREE_BASE_PD_ONLY_STATIC_BASELINE_NO_BALANCE_POLICY",
        "fell": bool(sim.fallen()),
        "duration_sim_s": float(sim.data.time),
        "initial_height_m": sim.initial_height,
        "final_height_m": float(obs["root_position_m_world_privileged"][2]),
        "final_tilt_deg": float(np.rad2deg(np.arccos(np.clip(-obs["projected_gravity_body"][2], -1, 1)))),
        "torque_saturation_fraction": sim.torque_saturated / sim.torque_samples,
        "balance_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sim_config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new experiment output directory; existing results are preserved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [] if args.render else None
    supported, contract = run_supported(args.config, args.output_dir, images)
    free = run_free(args.config, args.output_dir, images)
    if images is not None:
        from PIL import Image, ImageDraw

        montage = Image.new("RGB", (960, 1280), "white")
        for index, (label, pixels) in enumerate(images):
            x, y = (index % 2) * 480, (index // 2) * 640
            montage.paste(Image.fromarray(pixels), (x, y + 40))
            ImageDraw.Draw(montage).text((x + 10, y + 10), label, fill="black")
        montage.save(args.output_dir / "control_overview.png")
    repo = args.config.resolve().parents[2]
    report = {
        "stage": args.output_dir.name,
        "code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_files": [{"path": str(path.relative_to(repo)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in sorted(Path(__file__).resolve().parent.glob("*.py"))],
        "mujoco_version": mujoco.__version__,
        "source_is_smpl": False,
        "policy_trained": False,
        "hardware_control": False,
        "contract": contract,
        "supported": supported,
        "free_base": free,
        "artifacts": [{"path": str(path.resolve().relative_to(repo)), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(args.output_dir.iterdir()) if path.is_file()],
        "baidu_backup_status": "LOCAL_ONLY",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"supported": supported, "free_base": free}, indent=2))
    return 0 if supported["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
