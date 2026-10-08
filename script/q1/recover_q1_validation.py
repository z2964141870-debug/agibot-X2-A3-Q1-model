#!/usr/bin/env python3
"""Recover numerical evidence from saved validation traces without replay/render."""

import argparse
import json
from pathlib import Path

import numpy as np

from egolocate_input import convert_bundle
from infer_clotho_fgp import identity
from q1_retarget import BodyReference, Q1Retargeter
from q1_sim import JointReference


def load_trace(path):
    with np.load(path, allow_pickle=False) as source:
        if source["usage"].item() != "TEST_ONLY" or source["training_allowed"].item() is not False:
            raise ValueError("Expected TEST_ONLY saved output")
        result = {name: source[name].copy() for name in source.files}
    for value in result.values():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite persisted output: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-validator-sha256", required=True)
    args = parser.parse_args()
    if (len(args.original_validator_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.original_validator_sha256)):
        parser.error("Record the original validator hash from the execution checkpoint")
    destination = args.output_dir / "recovered_summary.json"
    if destination.exists():
        parser.error("Preserve the existing recovery summary")
    retargeter = Q1Retargeter()
    sim = retargeter.sim
    source, conversion = convert_bundle(args.input, retargeter.names, sim.config["reference_hz"])
    body = BodyReference.load(args.output_dir / "body_reference.npz", retargeter.names)
    for name in ("times", "positions", "orientations"):
        if not np.allclose(getattr(source, name), getattr(body, name), atol=1e-9, rtol=0):
            raise ValueError("Persisted body reference differs from the source bundle")
    load_trace(args.output_dir / "body_reference.npz")
    load_trace(args.output_dir / "joint_reference.npz")
    reference = JointReference.load(args.output_dir / "joint_reference.npz", sim.names)
    reference.validate_limits(sim)
    trace = load_trace(args.output_dir / "kinematic_trace.npz")
    replay = load_trace(args.output_dir / "supported_replay.npz")
    frames = len(source.times)
    if (trace["qpos"].shape != (frames, sim.model.nq)
            or trace["body_position_error_m"].shape != (frames, len(source.names), 3)
            or trace["body_orientation_error_rad"].shape != (frames, len(source.names))
            or not np.array_equal(trace["time_s"], source.times)
            or not np.array_equal(reference.times, source.times)
            or not np.array_equal(trace["qpos"][:, sim.q_indices], reference.positions)):
        raise ValueError("Kinematic trace and joint/body references disagree")
    times = replay["time_s"]
    expected_steps = round(reference.times[-1] / sim.reference_dt)
    if (times.shape != (expected_steps,)
            or not np.allclose(times, np.arange(1, expected_steps + 1) * sim.reference_dt, atol=1e-8, rtol=0)
            or any(replay[name].shape != (expected_steps, 22) for name in ("position_rad", "target_rad", "torque_nm"))):
        raise ValueError("Persisted replay does not cover the full reference timeline")
    expected_targets = np.asarray([reference.sample(t - sim.reference_dt)[0] for t in times])
    if not np.allclose(replay["target_rad"], expected_targets, atol=1e-7, rtol=0):
        raise ValueError("Persisted replay targets differ from the joint reference")
    torque = replay["torque_nm"]
    if np.any(torque < sim.torque_bounds[:, 0]) or np.any(torque > sim.torque_bounds[:, 1]):
        raise ValueError("Persisted torques exceed configured simulation limits")
    errors = replay["position_rad"] - replay["target_rad"]
    position_norms = np.linalg.norm(trace["body_position_error_m"], axis=-1)
    margin = np.minimum(reference.positions - sim.lower, sim.upper - reference.positions)
    report = {
        "stage": args.output_dir.name, "usage": "TEST_ONLY", "training_allowed": False,
        "scope": "PERSISTED_KINEMATIC_AND_SUPPORTED_PD_TRACES_NUMERICAL_RECOVERY",
        "original_validator_sha256_from_checkpoint": args.original_validator_sha256,
        "recovery_script": identity(Path(__file__).resolve()), "conversion": conversion,
        "input_decoded": True, "finite_kinematic_output": True, "supported_replay_finite": True,
        "persisted_body_reference_matches_input": True, "full_replay_timeline_verified": True,
        "persisted_joint_reference_matches_qpos": True, "persisted_replay_targets_verified": True,
        "kinematic_body_position_error_p95_m": float(np.quantile(position_norms, .95)),
        "kinematic_max_body_position_error_m": float(position_norms.max()),
        "kinematic_body_orientation_error_p95_rad": float(np.quantile(trace["body_orientation_error_rad"], .95)),
        "reference_joint_limit_violations": int(np.count_nonzero(margin < -1e-8)),
        "reference_max_joint_speed_rad_s": float(np.abs(reference.slopes).max()),
        "reference_fraction_joint_samples_within_0_001_rad_of_limit": float(np.mean(margin < .001)),
        "supported_replay_duration_s": float(times[-1]),
        "supported_replay_max_joint_rmse_rad": float(np.sqrt(np.mean(errors ** 2, axis=0)).max()),
        "supported_replay_max_abs_joint_error_rad": float(np.abs(errors).max()),
        "supported_replay_torque_saturation_fraction": None,
        "torque_saturation_limitation": "FULL_PHYSICS_SUBSTEP_COUNTER_NOT_PERSISTED; ONLY_LAST_SUBSTEP_TORQUE_SAMPLES_SAVED",
        "saved_torque_bounds_verified": True,
        "render_completed": (args.output_dir / "body_q1_overview.png").is_file(),
        "pose_fidelity_validated": False, "root_translation_tracking_tested": False,
        "balance_verified": False, "policy_trained": False, "hardware_control": False,
        "baidu_backup_status": "LOCAL_ONLY",
        "artifacts": [identity(args.output_dir / name) for name in
                      ("body_reference.npz", "joint_reference.npz", "kinematic_trace.npz", "supported_replay.npz")],
    }
    with destination.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
