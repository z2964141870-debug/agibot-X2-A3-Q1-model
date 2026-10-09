"""TEST_ONLY SMPL normalization and A3 Mink IK using pinned project assets."""

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.fullchain_support import DATA, ROOT, VENDOR, digest, mark, write_json

Q1 = Path("/media/yu/FAFF-E977/YuanQi_Q1")
sys.path.insert(0, str(Q1 / "script/q1"))
from egolocate_input import convert_bundle, save_test_only
from q1_retarget import BodyReference, heading_rotation

BODY_MAP = {
    "pelvis": "pelvis_link", "spine3": "torso_Link",
    "left_hip": "left_hip_yaw_Link", "right_hip": "right_hip_yaw_Link",
    "left_knee": "left_knee_Link", "right_knee": "right_knee_Link",
    "left_ankle": "left_ankle_roll_Link", "right_ankle": "right_ankle_roll_Link",
    "left_shoulder": "left_shoulder_yaw_Link", "right_shoulder": "right_shoulder_yaw_Link",
    "left_elbow": "left_elbow_Link", "right_elbow": "right_elbow_Link",
    "left_wrist": "left_wrist_yaw_Link", "right_wrist": "right_wrist_yaw_Link",
}
MJCF = VENDOR / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml"
URDF = VENDOR / "gear_sonic/data/assets/robot_description/urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf"
SOURCES = {
    "cloth_walk": Q1 / "data/experiments/q1_cloth_fgp_20261008_E003/body_only.npz",
    "cloth_oneleg": Q1 / "data/experiments/q1_cloth_fgp_20261008_E004/body_only.npz",
    "recording_0923": Q1 / "data/test_inputs/q1_mocap_20261008/hand_follow_20260923_191240/body_only.npz",
    "recording_0924": Q1 / "data/test_inputs/q1_mocap_20261008/hand_follow_20260924_150132/body_only.npz",
}


def audit():
    report = {"usage": "TEST_ONLY", "training_allowed": False, "cases": [],
              "q1_normalizer_sha256": digest(Q1 / "script/q1/egolocate_input.py")}
    mark("input_audit", "running")
    for name, path in SOURCES.items():
        row = {"name": name, "source": str(path), "status": "running"}
        report["cases"].append(row)
        try:
            source, details = convert_bundle(path, list(BODY_MAP), hz=30)
            out = DATA / "mocap" / name
            out.mkdir(parents=True, exist_ok=True)
            save_test_only(source, out / "body_reference.npz")
            with np.load(path, allow_pickle=False) as arrays:
                raw_frames = arrays["frame_index"].copy()
                row["finite_numeric_arrays"] = all(
                    np.isfinite(arrays[key]).all() for key in arrays.files
                    if np.issubdtype(arrays[key].dtype, np.number))
            if not row["finite_numeric_arrays"]:
                raise ValueError("Nonfinite numeric source array")
            trace = path.with_name("fgp_trace.npz")
            if trace.exists():
                with np.load(trace, allow_pickle=False) as arrays:
                    ids = arrays["frame_index"]
                    index = {int(value): i for i, value in enumerate(ids)}
                    aligned = np.asarray([index[int(value)] for value in raw_frames])
                    row["trace_alignment"] = {"method": "frame_index_not_array_offset",
                        "matched_frames": len(aligned), "first_trace_index": int(aligned[0]),
                        "trace_sha256": digest(trace)}
                    if "foot_contact_probability" in arrays:
                        contact = arrays["foot_contact_probability"][aligned]
                        row["contact_probability_range"] = [float(contact.min()), float(contact.max())]
            row.update(status="passed_structure_only", sha256=digest(path), bytes=path.stat().st_size,
                       normalized_path=str(out / "body_reference.npz"), details=details,
                       semantic_quality="UNVALIDATED_ROOT_CONTACT_POSE_ACCURACY",
                       high_fidelity_verified=False)
            write_json(out / "input_audit.json", row)
        except Exception as error:
            row.update(status="deferred", error=f"{type(error).__name__}: {error}")
        write_json(DATA / "input_audit.json", report)
    valid = sum(row["status"] == "passed_structure_only" for row in report["cases"])
    mark("input_audit", "passed" if valid == 4 else "deferred", valid_cases=valid,
         evidence=str(DATA / "input_audit.json"), scope="structure_only_not_pose_fidelity")
    return report


class A3Retargeter:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.key_qpos[0] if self.model.nkey else self.model.qpos0
        mujoco.mj_forward(self.model, self.data)
        self.names = list(BODY_MAP)
        self.bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                       for name in BODY_MAP.values()]
        if min(self.bodies) < 0:
            raise ValueError("A3 body mapping references a missing body")
        self.rest_position = self.data.xpos[self.bodies].copy()
        self.rest_rotation = self.data.xmat[self.bodies].reshape(-1, 3, 3).copy()
        self.joint_names, velocities = [], {}
        for joint in ET.parse(URDF).getroot().findall("joint"):
            name = joint.attrib["name"]
            if joint.attrib["type"] == "fixed" or name.startswith("head_") or "foot_" in name:
                continue
            self.joint_names.append(name)
            velocities[name] = float(joint.find("limit").attrib["velocity"])
        if len(self.joint_names) != 29:
            raise ValueError(f"Expected 29 active joints, got {len(self.joint_names)}")
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                          for name in self.joint_names]
        if min(self.joint_ids) < 0:
            raise ValueError("URDF policy joint missing from kinematic MJCF")
        self.q_indices = self.model.jnt_qposadr[self.joint_ids]
        self.configuration = mink.Configuration(self.model, q=self.data.qpos.copy())
        self.tasks = [(i, mink.FrameTask(name, "body", 20.0, 1.0, lm_damping=1.0))
                      for i, name in enumerate(BODY_MAP.values()) if i != 0]
        self.root_task = mink.FrameTask("pelvis_link", "body", 1.0, 1.0)
        costs = np.full(self.model.nv, .001)
        costs[:6] = 0
        self.posture = mink.PostureTask(self.model, costs)
        self.posture.set_target(self.data.qpos)
        for jid in range(1, self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            velocities.setdefault(name, 1e-8)
        self.limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=.001),
                       mink.VelocityLimit(self.model, velocities)]
        self.foot_geometries = []
        for side in ("left", "right"):
            selected = []
            for geom in range(self.model.ngeom):
                body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[geom]))
                if body.startswith(side + "_") and ("ankle_roll" in body or "foot_" in body):
                    kind = self.model.geom_type[geom]
                    if kind == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh = self.model.geom_dataid[geom]
                        start = self.model.mesh_vertadr[mesh]
                        local = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]].copy()
                    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                        from itertools import product
                        local = np.asarray(list(product((-1, 1), repeat=3))) * self.model.geom_size[geom]
                    else:
                        raise ValueError("Unsupported foot geometry in penetration check")
                    selected.append((geom, local))
            if not selected:
                raise ValueError("No actual foot geometry available")
            self.foot_geometries.append(selected)

    def foot_min_z(self):
        return [min(float((local @ self.configuration.data.geom_xmat[geom].reshape(3, 3).T +
                           self.configuration.data.geom_xpos[geom])[:, 2].min())
                    for geom, local in selected) for selected in self.foot_geometries]

    def targets(self, source, scale):
        cal_rotation = Rotation.from_quat(source.calibration_orientations, scalar_first=True).as_matrix()
        alignment = self.rest_rotation[0] @ cal_rotation[0].T
        yaw = heading_rotation(self.rest_rotation[0]) @ heading_rotation(cal_rotation[0]).T
        human_rest = (source.calibration_positions - source.calibration_positions[0]) @ cal_rotation[0]
        robot_rest = (self.rest_position - self.rest_position[0]) @ self.rest_rotation[0]
        human_rotation = Rotation.from_quat(source.orientations.reshape(-1, 4), scalar_first=True).as_matrix()
        human_rotation = human_rotation.reshape(len(source.times), len(self.names), 3, 3)
        root_rotation = alignment @ human_rotation[:, 0]
        root_position = self.rest_position[0] + scale * (source.positions[:, 0] - source.calibration_positions[0]) @ yaw.T
        local = np.einsum("tbi,tij->tbj", source.positions - source.positions[:, :1], human_rotation[:, 0])
        positions = root_position[:, None] + np.einsum("tbi,tji->tbj", robot_rest + scale * (local - human_rest), root_rotation)
        rotations = alignment @ human_rotation @ cal_rotation.transpose(0, 2, 1) @ alignment.T @ self.rest_rotation
        return positions, rotations

    def solve(self, source, scale, directory):
        positions, rotations = self.targets(source, scale)
        self.configuration.update(self.data.qpos.copy())
        qposes, actual, actual_rotations, foot_minima = [], [], [], []
        for frame in range(len(source.times)):
            q = self.configuration.data.qpos.copy()
            q[:3] = positions[frame, 0]
            q[3:7] = Rotation.from_matrix(rotations[frame, 0]).as_quat(scalar_first=True)
            self.configuration.update(q)
            self.root_task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotations[frame, 0]), positions[frame, 0]))
            for index, task in self.tasks:
                task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(rotations[frame, index]), positions[frame, index]))
            for _ in range(4):
                velocity = mink.solve_ik(self.configuration, [task for _, task in self.tasks] + [self.posture],
                                         1 / 120, "quadprog", damping=.001, safety_break=True,
                                         limits=self.limits, constraints=[self.root_task])
                self.configuration.integrate_inplace(velocity, 1 / 120)
            q = self.configuration.data.qpos.copy()
            if not np.isfinite(q).all():
                raise ValueError("Nonfinite IK output")
            qposes.append(q)
            actual.append(self.configuration.data.xpos[self.bodies].copy())
            actual_rotations.append(self.configuration.data.xmat[self.bodies].reshape(-1, 3, 3).copy())
            foot_minima.append(self.foot_min_z())
            if frame % 300 == 0:
                print(json.dumps({"frame": frame, "total": len(source.times)}), flush=True)
        qposes, actual = np.asarray(qposes), np.asarray(actual)
        joints = qposes[:, self.q_indices]
        limits = self.model.jnt_range[self.joint_ids]
        violations = int(np.count_nonzero((joints < limits[:, 0] - 1e-6) | (joints > limits[:, 1] + 1e-6)))
        error = np.linalg.norm(actual - positions, axis=-1)
        orientation_error = (Rotation.from_matrix(np.asarray(actual_rotations).reshape(-1, 3, 3)) *
                             Rotation.from_matrix(rotations.reshape(-1, 3, 3)).inv()).magnitude()
        velocities = np.diff(joints, axis=0) / np.diff(source.times)[:, None]
        feet = [self.names.index(f"{side}_ankle") for side in ("left", "right")]
        foot_position = actual[:, feet]
        foot_speed = np.linalg.norm(np.diff(foot_position, axis=0) / np.diff(source.times)[:, None, None], axis=-1)
        np.savez_compressed(directory / "kinematic_trace.npz", time_s=source.times, qpos=qposes,
                            joint_names=np.asarray(self.joint_names), target_position_m=positions,
                            actual_position_m=actual, position_error_m=error,
                            foot_geometry_min_z_m=np.asarray(foot_minima),
                            usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
        fields = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
                  "root_rotateX", "root_rotateY", "root_rotateZ", *self.joint_names,
                  "head_yaw_joint", "head_pitch_joint"]
        with (directory / "reference.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            euler = Rotation.from_quat(qposes[:, 3:7], scalar_first=True).as_euler("xyz", degrees=True)
            for i in range(len(qposes)):
                writer.writerow([i, *(qposes[i, :3] * 100), *euler[i], *np.rad2deg(joints[i]), 0, 0])
        report = {"finite": True, "frames": len(qposes), "fps": 30, "duration_s": float(source.times[-1]),
                  "joint_limit_violations": violations, "ik_position_p95_m": float(np.percentile(error, 95)),
                  "ik_position_max_m": float(error.max()), "ik_orientation_p95_rad": float(np.percentile(orientation_error, 95)),
                  "joint_range_rad": np.ptp(joints, axis=0).tolist(), "joint_names": self.joint_names,
                  "joint_velocity_max_rad_s": float(np.abs(velocities).max()),
                  "foot_ankle_min_z_m": float(foot_position[:, :, 2].min()),
                  "foot_speed_p95_m_s": float(np.percentile(foot_speed, 95)),
                  "foot_geometry_min_z_m": float(np.min(foot_minima)),
                  "foot_geometry_penetration_frames": int(np.count_nonzero(np.min(foot_minima, axis=1) < -.001)),
                  "foot_metrics_scope": "ANKLE_PROXY_NOT_SOLE_CONTACT_OR_SLIP_VALIDATION",
                  "scale": scale, "calibration": "FIRST_FRAME_RELATIVE_DIAGNOSTIC",
                  "physical_balance_verified": False, "high_fidelity_verified": False,
                  "usage": "TEST_ONLY", "training_allowed": False}
        write_json(directory / "retarget_summary.json", report)
        if violations:
            raise ValueError("A3 reference has joint-limit violations")
        return report


def retarget():
    mark("retarget", "running")
    prototype = A3Retargeter()
    write_json(DATA / "a3_retarget_config.json", {"body_map": BODY_MAP, "mjcf": str(MJCF),
        "mjcf_sha256": digest(MJCF), "urdf": str(URDF), "urdf_sha256": digest(URDF),
        "policy_joint_order": prototype.joint_names, "root_translation_alignment": "world_z_heading_only",
        "solver": "mink_quadprog", "iterations": 4, "reference_fps": 30,
        "velocity_limits": "URDF_LIMITS_ACTIVE_HEAD_PASSIVE_FROZEN",
        "scope": "OPEN_CHAIN_KINEMATIC_REFERENCE_NOT_CLOSED_LOOP_DYNAMICS"})
    times = np.arange(31) / 30
    neutral = BodyReference(times, list(BODY_MAP), np.repeat(prototype.rest_position[None], len(times), 0),
                            np.repeat(Rotation.from_matrix(prototype.rest_rotation).as_quat(scalar_first=True)[None], len(times), 0),
                            prototype.rest_position, Rotation.from_matrix(prototype.rest_rotation).as_quat(scalar_first=True),
                            list(BODY_MAP), "SYNTHETIC_A3_NEUTRAL")
    neutral_dir = DATA / "mocap/synthetic_neutral"
    neutral_dir.mkdir(parents=True, exist_ok=True)
    neutral_result = prototype.solve(neutral, 1, neutral_dir)
    if neutral_result["ik_position_max_m"] > 1e-4:
        raise ValueError("Synthetic neutral mapping failed")
    positions, rotations, expected = [], [], []
    joint = prototype.joint_names.index("left_knee_joint")
    for value in .1 * np.sin(np.pi * times):
        q = prototype.data.qpos.copy()
        q[prototype.q_indices[joint]] += value
        prototype.data.qpos[:] = q
        mujoco.mj_forward(prototype.model, prototype.data)
        positions.append(prototype.data.xpos[prototype.bodies].copy())
        rotations.append(Rotation.from_matrix(prototype.data.xmat[prototype.bodies].reshape(-1, 3, 3)).as_quat(scalar_first=True))
        expected.append(q[prototype.q_indices].copy())
        prototype.data.qpos[:] = prototype.model.key_qpos[0]
    single = BodyReference(times, list(BODY_MAP), positions, rotations,
                           neutral.calibration_positions, neutral.calibration_orientations,
                           list(BODY_MAP), "SYNTHETIC_SINGLE_KNEE")
    single_dir = DATA / "mocap/synthetic_single_joint"
    single_dir.mkdir(parents=True, exist_ok=True)
    single_result = A3Retargeter().solve(single, 1, single_dir)
    with np.load(single_dir / "kinematic_trace.npz", allow_pickle=False) as arrays:
        single_result["expected_active_joint_max_abs_error_rad"] = float(np.max(np.abs(
            arrays["qpos"][:, prototype.q_indices] - np.asarray(expected))))
    write_json(single_dir / "retarget_summary.json", single_result)
    if single_result["expected_active_joint_max_abs_error_rad"] > .02:
        raise ValueError("Synthetic single-joint mapping failed")
    outcomes = []
    for name in SOURCES:
        directory = DATA / "mocap" / name
        source_path = directory / "body_reference.npz"
        if not source_path.exists():
            outcomes.append({"name": name, "status": "deferred", "reason": "input_normalization_unavailable"})
            continue
        try:
            source = BodyReference.load(source_path, list(BODY_MAP))
            solver = A3Retargeter()
            lengths = []
            robot_lengths = []
            for side in ("left", "right"):
                ids = [solver.names.index(f"{side}_{part}") for part in ("hip", "knee", "ankle")]
                lengths.append(np.median(np.linalg.norm(np.diff(source.positions[:, ids], axis=1), axis=-1).sum(axis=-1)))
                robot_lengths.append(np.linalg.norm(np.diff(solver.rest_position[ids], axis=0), axis=-1).sum())
            scale = float(np.mean(robot_lengths) / np.mean(lengths))
            if not .1 < scale < 2:
                raise ValueError("Implausible anthropometry")
            result = solver.solve(source, scale, directory)
            outcomes.append({"name": name, "status": "passed_kinematics_only", **result})
        except Exception as error:
            outcomes.append({"name": name, "status": "deferred", "reason": f"{type(error).__name__}: {error}"})
        write_json(DATA / "retarget_summary.json", outcomes)
    mark("retarget", "passed" if all(item["status"] == "passed_kinematics_only" for item in outcomes) else "deferred",
         evidence=str(DATA / "retarget_summary.json"), scope="kinematics_only_not_pose_fidelity_or_balance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "retarget"))
    args = parser.parse_args()
    if args.action == "audit":
        audit()
    else:
        retarget()


if __name__ == "__main__":
    main()
