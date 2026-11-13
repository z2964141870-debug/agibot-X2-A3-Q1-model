"""Check the built native dynamics on the author's audited SMPL cache."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .assets import file_digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing check output")
    spec = importlib.util.spec_from_file_location("carticulate", args.module)
    cart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cart)
    with np.load(args.cache, allow_pickle=False) as cache:
        armature = cart.DynamicArmature()
        armature.n_joints = 24
        armature.name = "fgp_native_check"
        armature.parent = cache["parent"].tolist()
        armature.gravity = np.array([0., -9.8, 0.])
        for key in ("mass", "bone", "com", "inertia"):
            setattr(armature, key, cache[key].copy())
    model = cart.DynamicModel(armature)
    rng = np.random.default_rng(20261113)
    pose = Rotation.from_rotvec(rng.normal(0, .08, (24, 3))).as_matrix().astype(np.float32)
    root = np.array([.1, .03, -.2], dtype=np.float32)
    vel = rng.normal(0, .03, 75).astype(np.float32)
    model.set_state_R(pose, root, vel)
    restored_pose, restored_root, restored_vel = model.get_state_R()
    np.testing.assert_allclose(restored_pose, pose, atol=2e-6, rtol=0)
    np.testing.assert_allclose(restored_root, root, atol=2e-6, rtol=0)
    np.testing.assert_allclose(restored_vel, vel, atol=2e-6, rtol=0)
    mass = model.mass_matrix().astype(np.float64)
    symmetry = float(np.max(np.abs(mass - mass.T)))
    np.testing.assert_allclose(mass, mass.T, atol=1e-4, rtol=1e-5)
    minimum_eigenvalue = float(np.linalg.eigvalsh((mass + mass.T) / 2).min())
    if minimum_eigenvalue <= 0:
        raise ValueError("Native mass matrix must be positive definite")
    acc = rng.normal(0, .1, 75).astype(np.float32)
    bias = model.inverse_dynamics(np.zeros(75))
    force = model.inverse_dynamics(acc)
    inverse_residual = float(np.max(np.abs(force - bias - mass @ acc)))
    np.testing.assert_allclose(force-bias, mass @ acc, atol=1e-3, rtol=1e-4)
    recovered_acc = model.forward_dynamics(force)
    np.testing.assert_allclose(recovered_acc, acc, atol=2e-3, rtol=2e-3)
    jacobian_errors, finite_difference_errors = [], []
    for joint in (0, 10, 11, 22, 23):
        model.set_state_R(pose, root, vel)
        position = model.get_position(joint)
        jacobian = model.get_linear_Jacobian(joint)
        expected_velocity = jacobian @ vel
        actual_velocity = model.get_linear_velocity(joint)
        np.testing.assert_allclose(actual_velocity, expected_velocity, atol=2e-6, rtol=1e-5)
        jacobian_errors.append(float(np.max(np.abs(actual_velocity-expected_velocity))))
        model.update_state(np.zeros(75), .001)
        finite_difference = (model.get_position(joint)-position)/.001
        np.testing.assert_allclose(finite_difference, expected_velocity, atol=1e-3, rtol=1e-2)
        finite_difference_errors.append(float(np.max(np.abs(finite_difference-expected_velocity))))
    report = dict(status="NATIVE_DYNAMICS_INVARIANTS_PASSED", module_sha256=file_digest(args.module),
        cache_sha256=file_digest(args.cache), roundtrip_state=True, mass_symmetry_max_abs=symmetry,
        mass_min_eigenvalue=minimum_eigenvalue, inverse_dynamics_residual_max_abs=inverse_residual,
        forward_inverse_acceleration_max_abs=float(np.max(np.abs(recovered_acc-acc))),
        jacobian_velocity_max_abs=max(jacobian_errors),
        integration_finite_difference_max_abs=max(finite_difference_errors), contact_joints_checked=5,
        windows_binary_equivalence="NOT_ESTABLISHED", hardware_commands=0,
        limitation="Invariant checks on one deterministic SMPL state; not Windows-reference parity or accuracy validation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
