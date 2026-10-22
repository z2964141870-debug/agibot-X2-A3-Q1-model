#!/usr/bin/env python3
"""Independent T13 shoulder-landmark correction; CPU kinematic diagnosis only."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from a3_landmark_geometry_audit import BASIS, IDS, PARENTS, identity, source_from_paired


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def current_source(artifact, data_zip, sequence, names):
    from q1_retarget import BodyReference

    with np.load(artifact, allow_pickle=False) as saved:
        frames = saved["frame_index"]
        selected = frames >= 59
        frames = frames[selected]
        label_pose = saved["label_pose"][selected].copy()
        prediction = saved["tanh_pose"][selected].copy()
        bodies = {"label": saved["label_rootrelative_joints"][selected].copy(),
                  "tanh": saved["tanh_rootrelative_joints"][selected].copy()}
        rest = saved["rest_bone_vectors"].copy()
    with zipfile.ZipFile(data_zip) as package:
        payload = package.read(f"train_released/{sequence}/joint.pt")
        label_joints = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True).numpy()[frames]
    root = label_joints[:, :1]
    neutral = np.zeros((24, 3))
    for joint in range(1, 24):
        neutral[joint] = neutral[PARENTS[joint]] + rest[joint - 1]
    calibration = (neutral[IDS] + root[0, 0]) @ BASIS.T
    sources = {}
    for arm, pose in (("label", label_pose), ("tanh", prediction)):
        local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        world = np.empty_like(local)
        for joint, parent in enumerate(PARENTS):
            world[:, joint] = local[:, joint] if parent < 0 else world[:, parent] @ local[:, joint]
        rotations = BASIS @ world[:, IDS] @ BASIS.T
        orientations = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 14, 4)
        sources[arm] = BodyReference(np.arange(len(frames)) / 30, list(names),
            (bodies[arm] + root)[:, IDS] @ BASIS.T, orientations, calibration,
            np.tile([1., 0, 0, 0], (14, 1)), list(names), "OFFLINE_LABEL_ASSISTED_ROOT_CONTACT")
    return sources, bodies["label"] + root, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "t10", "t11", "t12", "data", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(args.root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "data/references/a3_smpl_q1_helpers_20261012"))
    import mink
    import mujoco
    from script.a3 import mocap_adapter
    from script.a3.smpl_base_retarget import base_pair_quality
    from script.a3.smpl_hull_floor import HullA3Retargeter

    t10 = json.loads((Path(args.t10) / "receipt.json").read_text())
    t11 = json.loads(Path(args.t11).read_text())
    t12 = json.loads(Path(args.t12).read_text())
    for record in t12["source_files"] + [t12["mjcf"], t10["paired_artifact"], t11["inputs"]["data"]]:
        if identity(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("Registered upstream source or input changed")
    if identity(args.data)["sha256"] != t11["inputs"]["data"]["sha256"]:
        raise ValueError("Data source differs from T11")
    for record in t11["sequences"]:
        if identity(record["artifact"]["path"])["sha256"] != record["artifact"]["sha256"]:
            raise ValueError("Saved T11 source changed")

    class PositionOnlyTask(mink.FrameTask):
        def set_orientation_cost(self, unused_cost):
            super().set_orientation_cost(0.)

    class OrientationOnlyTask(mink.FrameTask):
        def set_position_cost(self, unused_cost):
            super().set_position_cost(0.)

    original_map = mocap_adapter.BODY_MAP
    corrected_map = dict(original_map)
    for side in ("left", "right"):
        corrected_map[side + "_shoulder"] = side + "_shoulder_roll_Link"

    def create():
        # The historical module is untouched; the override exists only in this constructor.
        mocap_adapter.BODY_MAP = corrected_map
        try:
            retargeter = HullA3Retargeter()
        finally:
            mocap_adapter.BODY_MAP = original_map
        tasks = []
        for index, task in retargeter.tasks:
            name = retargeter.names[index]
            if name in ("left_shoulder", "right_shoulder"):
                tasks.append((index, PositionOnlyTask(corrected_map[name], "body", 20., 0., lm_damping=1.)))
                tasks.append((index, OrientationOnlyTask(original_map[name], "body", 0., 1., lm_damping=1.)))
                yaw_id = mujoco.mj_name2id(retargeter.model, mujoco.mjtObj.mjOBJ_BODY, original_map[name])
                yaw_rest = retargeter.configuration.data.xmat[yaw_id].reshape(3, 3)
                if not np.allclose(yaw_rest, retargeter.rest_rotation[index], atol=1e-8, rtol=0):
                    raise ValueError("Requires coincident rest roll/yaw orientation for this correction")
            else:
                tasks.append((index, task))
        retargeter.tasks = tasks
        return retargeter

    admission = dict(experiment="T13_ANATOMICAL_SHOULDER", status="ADMITTED_NOT_COMPLETED",
        driver=identity(__file__), input_helper=identity(Path(__file__).with_name("a3_landmark_geometry_audit.py")),
        t10=identity(Path(args.t10) / "receipt.json"), t11=identity(args.t11), t12=identity(args.t12),
        original_map=original_map, corrected_position_map=corrected_map,
        shoulder_orientation_map={name: original_map[name] for name in ("left_shoulder", "right_shoulder")},
        unchanged="leg mapping, limits, floor, solver budget, contact, 5cm gate",
        cases="T10 two original 30-frame arms plus three T11 label/tanh pairs on source frames 59..299",
        policy_loaded=False, dynamics_step_executed=False, training_started=False)
    write_json(out / "admission.json", admission)
    start = time.perf_counter()
    cases = []
    for arm, key in (("label_pose", "label_pose"), ("onnx_pose", "pred_pose")):
        source, contacts = source_from_paired(t10["paired_artifact"]["path"], args.data, list(original_map), key)
        cases.append(("T10_walk_" + arm, source, contacts, t10["scale"], Path(args.t10) / arm / "kinematic_trace.npz"))
    for record in t11["sequences"]:
        sources, contacts, frames = current_source(record["artifact"]["path"], args.data, record["sequence"], list(original_map))
        probe = create()
        ratios = []
        for side in ("left", "right"):
            chain = [probe.names.index(name) for name in ("pelvis", side + "_hip", side + "_knee", side + "_ankle")]
            ratios.append(np.linalg.norm(np.diff(probe.rest_position[chain], axis=0), axis=-1).sum() /
                          np.linalg.norm(np.diff(sources["label"].calibration_positions[chain], axis=0), axis=-1).sum())
        scale = float(np.mean(ratios))
        for arm, source in sources.items():
            cases.append((record["sequence"].split("/")[-1] + "_" + arm, source, contacts, scale, None))
    results = []
    for name, source, contacts, scale, prior_trace in cases:
        directory = out / name
        directory.mkdir()
        retargeter = create()
        retargeter.set_human_source(contacts)
        summary = retargeter.solve(source, scale, directory)
        with np.load(directory / "kinematic_trace.npz", allow_pickle=False) as saved:
            trace = {key: saved[key].copy() for key in saved.files}
        quality = base_pair_quality(summary, trace, contacts, retargeter)
        data = mujoco.MjData(retargeter.model)
        actual = []
        old_actual = []
        old_ids = [mujoco.mj_name2id(retargeter.model, mujoco.mjtObj.mjOBJ_BODY, body) for body in original_map.values()]
        for qpos in trace["qpos"]:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(retargeter.model, data)
            actual.append(data.xpos[retargeter.bodies].copy())
            old_actual.append(data.xpos[old_ids].copy())
        actual = np.asarray(actual)
        if not np.allclose(actual, trace["actual_position_m"], atol=1e-9, rtol=0):
            raise ValueError("Saved corrected landmark position disagrees with independent FK")
        error = np.linalg.norm(actual - trace["target_position_m"], axis=-1)
        if abs(float(np.quantile(error, .95)) - summary["ik_position_p95_m"]) > 1e-10:
            raise ValueError("Summary position error is not reproducible")
        old_metric = None
        if prior_trace is not None:
            with np.load(prior_trace, allow_pickle=False) as saved:
                old_error = np.linalg.norm(np.asarray(old_actual) - saved["target_position_m"], axis=-1)
            old_metric = dict(p95_m=float(np.quantile(old_error, .95)), mean_m=float(old_error.mean()),
                interpretation="Old yaw-origin target contract retained for transparency; contracts are different")
        result = dict(name=name, frames=len(source.times), scale=scale, summary=summary, quality=quality,
            independent_fk_verified=True, per_body_p95_m=dict(zip(retargeter.names, np.quantile(error, .95, axis=0).tolist())),
            legacy_target_error=old_metric, trace=identity(directory / "kinematic_trace.npz"),
            reference=identity(directory / "reference.csv"))
        results.append(result)
        write_json(directory / "receipt.json", result)
        print("CASE", name, json.dumps({"p95_m": summary["ik_position_p95_m"], "quality": quality}), flush=True)
    report = dict(**{k: v for k, v in admission.items() if k != "status"}, status="COMPLETED_KINEMATIC_DIAGNOSIS",
        rows=results, qualified=sum(item["quality"]["passed"] for item in results), wall_s=time.perf_counter() - start,
        label_root_and_contact_assisted=True, independent_test=False, physical_balance_verified=False,
        warning="Corrected task landmarks change target semantics. No claim of policy or world-root improvement.")
    write_json(out / "receipt.json", report)
    print("COMPLETED", report["qualified"], "/", len(results), "wall_s", report["wall_s"], flush=True)


if __name__ == "__main__":
    main()
