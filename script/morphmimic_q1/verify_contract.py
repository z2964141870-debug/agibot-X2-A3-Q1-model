"""CPU-only Q1 model and pinned Primebot equation audit, without simulation steps."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from canonical import SparseJointMap, morphology_with_presence
from control import delayed_target, motor_control, parameters, torque_bounds


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def source_equations(path):
    # The review snapshot is intentionally partial. Execute only the three named
    # tensor equations and two literal tables, without importing its simulator.
    tree = ast.parse(path.read_text())
    tables = {"STIFFNESS", "DAMPING"}
    functions = {"pd_gains", "torque_speed_parameters", "torque_speed_limits"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in tables for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
    if len(body) != 5:
        raise ValueError("Pinned Primebot source no longer has the expected equation API")
    namespace = {"torch": torch, "Tensor": torch.Tensor}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def run(args):
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit must not run in an initialized CUDA process")
    sys.path.insert(0, str(args.morphmimic_source / "src"))
    from morphmimic.constants import JOINT_NAMES, TRACKED_BODY_NAMES
    from morphmimic.robots import get_robot

    inputs = [args.q1_xml, args.primebot_source / "src/supermimic/simulator.py",
              args.primebot_source / "configs/q1-fresh-single.example.toml",
              args.morphmimic_source / "src/morphmimic/robots/spec.py",
              args.morphmimic_source / "src/morphmimic/robots/morphology.py"]
    before = [identity(p) for p in inputs]
    model = mujoco.MjModel.from_xml_path(str(args.q1_xml))
    if (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError("Expected a free 22-hinge Q1 model")
    native = tuple(model.joint(i).name for i in range(1, model.njnt))
    mapping = SparseJointMap.q1(JOINT_NAMES, native)
    if any(model.body(b).id < 1 for b in TRACKED_BODY_NAMES):
        raise ValueError("Q1 is missing one of the 14 tracked bodies")
    if not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE):
        raise ValueError("Q1 must use torque motors, not implicit position servos")
    actuator_joints = model.actuator_trnid[:, 0]
    if sorted(actuator_joints.tolist()) != list(range(1, 23)):
        raise ValueError("Each real Q1 hinge needs exactly one actuator")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g1 = get_robot("g1").entity_cfg_fn().spec_fn().compile()
    g1_data = mujoco.MjData(g1)
    mujoco.mj_forward(g1, g1_data)
    axes = []
    for name in native:
        a, b = data.xaxis[model.joint(name).id], g1_data.xaxis[g1.joint(name).id]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if cosine < .8:
            raise ValueError(f"Q1/G1 positive joint-axis agreement requires review: {name}")
        axes.append({"joint": name, "positive_axis_cosine_at_qpos0": cosine})
    # Zero-pose forearm directions substantiate a zero elbow offset independently
    # of X2's different -pi/2 convention. This does not equate whole-body FK.
    forearms = []
    for side in ("left", "right"):
        vectors = []
        for m, d in ((model, data), (g1, g1_data)):
            delta = (d.body(f"{side}_wrist_yaw_link").xpos -
                     d.body(f"{side}_elbow_link").xpos)
            vectors.append(delta / np.linalg.norm(delta))
        cosine = float(np.dot(*vectors))
        if cosine < .8:
            raise ValueError("Zero elbow offset requires a new frame mapping")
        forearms.append({"side": side, "zero_pose_forearm_cosine": cosine})

    rng = np.random.default_rng(0)
    max_roundtrip, max_fk = 0., 0.
    for _ in range(64):
        q = torch.from_numpy(rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1]))
        canonical = mapping.canonical(q)
        canonical[~mapping.mask(canonical)] = float("nan")
        restored = mapping.native(canonical).numpy()
        max_roundtrip = max(max_roundtrip, float(np.max(np.abs(restored - q.numpy()))))
        data.qpos[7:] = q.numpy()
        mujoco.mj_forward(model, data)
        expected = data.xpos.copy()
        data.qpos[7:] = restored
        mujoco.mj_forward(model, data)
        max_fk = max(max_fk, float(np.max(np.abs(data.xpos - expected))))
    if max_roundtrip > 1e-12 or max_fk > 1e-12:
        raise ValueError("Sparse translation changed physical joint positions or body FK")

    equations = source_equations(inputs[1])
    kp, kd, peaks, knees, cutoffs = parameters(native)
    ref_kp, ref_kd = equations["pd_gains"](native)
    torch.testing.assert_close(kp, torch.tensor(ref_kp), rtol=0, atol=0)
    torch.testing.assert_close(kd, torch.tensor(ref_kd), rtol=0, atol=0)
    for actual, reference in zip((peaks, knees, cutoffs),
                                 equations["torque_speed_parameters"](native, "cpu")):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    velocity = torch.from_numpy(rng.uniform(-25, 25, size=(128, 22))).float()
    low, high = torque_bounds(velocity, peaks, knees, cutoffs)
    ref_low, ref_high = equations["torque_speed_limits"](velocity, peaks, knees, cutoffs)
    torch.testing.assert_close(low, ref_low, rtol=0, atol=0)
    torch.testing.assert_close(high, ref_high, rtol=0, atol=0)
    current = torch.from_numpy(rng.normal(size=(128, 22))).float() * .5
    previous = torch.from_numpy(rng.normal(size=(128, 22))).float() * .5
    position = torch.from_numpy(rng.normal(size=(128, 22))).float()
    gear = torch.from_numpy(model.actuator_gear[:, 0]).float()
    index = torch.from_numpy(model.jnt_qposadr[actuator_joints] - 7)
    limits = torch.from_numpy(model.actuator_ctrlrange).float()
    delay = torch.from_numpy(rng.uniform(0, .02, size=(128, 1))).float()
    max_torque_error = 0.
    for substep in range(10):
        target = delayed_target(current, previous, delay, substep)
        expected_target = torch.where(substep >= (delay / .002).round().long(),
                                      current, previous)
        torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
        # Evaluate motor controls in native hinge order, then the model's actual
        # actuator order; gear and ctrl bounds belong to actuators.
        torque = kp * (target - position) - kd * velocity
        actual = motor_control(target, position, velocity, kp, kd, peaks, knees, cutoffs,
                               torch.ones(22), torch.stack((-peaks, peaks), -1))
        actual = (actual[:, index] / gear).maximum(limits[:, 0]).minimum(limits[:, 1])
        expected = (torque.maximum(ref_low).minimum(ref_high)[:, index] / gear)
        expected = expected.maximum(limits[:, 0]).minimum(limits[:, 1])
        max_torque_error = max(max_torque_error, (actual - expected).abs().max().item())
    if max_torque_error != 0:
        raise ValueError("Q1 control law differs from pinned source equations")
    q1_terms = torch.stack((torch.full_like(kp, .5), kp, kd, peaks))
    canon_terms = torch.stack([mapping.canonical(row, delta=True) for row in q1_terms])
    descriptor, bounds = morphology_with_presence(canon_terms[None], torch.ones(1, 9),
                                                 mapping.mask(kp)[None])
    after = [identity(p) for p in inputs]
    if before != after or torch.cuda.is_initialized():
        raise RuntimeError("Source identity changed or a CUDA context was created")
    result = {
        "status": "CPU_CONTRACT_PASSED", "utc": datetime.now(timezone.utc).isoformat(),
        "inputs": before, "inputs_unchanged": True,
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                  "nbody": model.nbody, "ngeom": model.ngeom,
                  "native_joint_names": native, "tracked_bodies": TRACKED_BODY_NAMES,
                  "nominal_pelvis_height_m": float(model.qpos0[2]),
                  "original_xml_physics_dt_s": float(model.opt.timestep)},
        "missing": sorted(set(JOINT_NAMES) - set(native)), "axes": axes,
        "zero_pose_forearms": forearms,
        "static_fk": {"poses": 64, "max_roundtrip_rad": max_roundtrip,
                      "max_body_position_change_m": max_fk},
        "control_equations": {"sample_rows": 128, "substeps": 10,
                              "motor_elements": 128 * 10 * 22,
                              "max_motor_control_error": max_torque_error,
                              "equation_loading": "AST_subset_not_full_simulator",
                              "kp": kp.tolist(), "kd": kd.tolist()},
        "morphology_test": {"dim": descriptor.shape[-1],
                            "geometry": "synthetic_ones_not_actual_descriptor"},
        "integration_status": "NOT_WIRED_INTO_NATIVE_TRAINER",
        "physics_integration_steps": 0, "policy_inferences": 0,
        "optimizer_updates": 0, "cuda_context_created": False,
        "user_accepted_checkpoint": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("morphmimic-source", "primebot-source", "q1-xml", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
