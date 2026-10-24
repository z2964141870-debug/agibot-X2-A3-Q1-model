#!/usr/bin/env python3
"""Convert admitted BONES G1 candidates to Q1 without changing their native time."""

import argparse
import collections
import json
from pathlib import Path
import sys
import time
import traceback

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "q1", "supermimic_q1_phuma"):
    sys.path.insert(0, str(HERE.parent / directory))
from contact_protocol import ContactCorrectedAssets
from prepare_motion import export_q1, support_height
from q1_retarget import BodyReference, Q1Retargeter
from retarget import initialize_first_pose, source_feet
from supermimic.q1_assets import identity

from bones_g1_csv import JOINT_NAMES_29, decode_bones_g1, resample_native_motion
from intake_bones_archive import frozen_selection
from inventory_amass import health
from runtime_guard import StorageBudget, save


PROTOCOL = "BONES_G1_120HZ_TO_Q1_50HZ_NATIVE_TIME_V1"
BODY_MAP = json.loads((HERE / "bones_retarget_mapping.json").read_text())["body_map"]


def source_reference(model, motion, names):
    model_names = tuple(model.joint(j).name for j in range(1, model.njnt))
    if ((model.nq, model.nv, model.nu) != (36, 35, 29) or model_names != JOINT_NAMES_29
            or tuple(motion["joint_names"]) != model_names or motion["source_hz"] != 120.
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("BONES G1 native rate or named 29-joint source model differs")
    sampled = resample_native_motion(motion, target_hz=50.)
    states = sampled["qpos_wxyz"].copy()
    origin = states[0, :2].copy()
    states[:, :2] -= origin
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [model.body(BODY_MAP[name]).id for name in names]
    calibration_positions = data.xpos[ids].copy()
    calibration_orientations = data.xquat[ids].copy()
    floor = support_height(model, data, source_feet(model))
    calibration_positions[:, 2] -= floor
    positions, orientations = [], []
    for state in states:
        data.qpos[:] = state
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[ids].copy())
        orientations.append(data.xquat[ids].copy())
    reference = BodyReference(sampled["time_s"], names, positions, orientations,
                              calibration_positions, calibration_orientations,
                              names, "BONES_G1_120Hz_native_to_Q1_DECLARED_UNITREE_MODEL")
    return reference, {"source_hz": 120., "source_frames": motion["source_frames"],
                       "source_frame_start": int(motion["source_frame_ids"][0]),
                       "source_frame_end": int(motion["source_frame_ids"][-1]),
                       "source_duration_s": sampled["source_duration_s"],
                       "target_hz": 50., "target_frames": len(reference.times),
                       "target_duration_s": sampled["target_duration_s"],
                       "endpoint_truncation_s": sampled["endpoint_truncation_s"],
                       "source_calibration_floor_m": floor,
                       "source_horizontal_origin_m": origin.tolist(), "time_scale": 1.}


def grounded_solver(assets):
    solver = Q1Retargeter(HERE / "bones_retarget_control.json", HERE / "bones_retarget_mapping.json")
    floor = support_height(solver.sim.model, solver.sim.data, assets.foot_geoms)
    solver.sim.data.qpos[2] -= floor
    mujoco.mj_forward(solver.sim.model, solver.sim.data)
    solver.robot_positions = solver.sim.data.xpos[solver.bodies].copy()
    solver.robot_rotations = solver.sim.data.xmat[solver.bodies].reshape(-1, 3, 3).copy()
    solver.posture.set_target(solver.sim.data.qpos)
    return solver, floor


def displacement_scale(model, assets):
    source, target = mujoco.MjData(model), mujoco.MjData(assets.model)
    mujoco.mj_forward(model, source)
    target.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(assets.model, target)
    lengths = []
    for current, data in ((model, source), (assets.model, target)):
        lengths.append(np.linalg.norm(data.xpos[current.body("pelvis").id]
                                      - data.xpos[current.body("left_ankle_roll_link").id]))
    if not all(np.isfinite(lengths)) or min(lengths) <= .1:
        raise ValueError("Degenerate source or Q1 standing leg length")
    return float(lengths[1] / lengths[0])


def convert(args):
    if not 1 <= args.count <= 128 or args.offset < 0:
        raise ValueError("Use a finite batch of at most 128 frozen candidates")
    intake = json.loads(args.intake.read_text())
    if intake["status"] != "ARCHIVE_ACCEPTED_FROZEN_SELECTION_EXTRACTED":
        raise ValueError("A completed real archive intake is required")
    for record in intake["frozen"].values():
        if identity(record["path"]) != record:
            raise ValueError("Archive intake metadata or goal changed")
    summary, rows, _ = frozen_selection(Path(intake["frozen"]["metadata_summary"]["path"]))
    if summary["selected_candidate_groups"] != intake["selected_split_counts"]:
        raise ValueError("Frozen candidate group counts changed")
    selected = rows[args.offset:args.offset + args.count]
    if len(selected) != args.count:
        raise ValueError("Batch extends beyond the frozen candidate selection")
    comparison = json.loads(args.model_comparison.read_text())
    if comparison["status"] != "OFFICIAL_G1_KINEMATICS_EQUIVALENT":
        raise ValueError("Use the audited official-G1-compatible source candidate")
    xml = Path(comparison["candidate"]["path"])
    if identity(xml) != comparison["candidate"]:
        raise ValueError("Audited G1 source XML changed")
    model = mujoco.MjModel.from_xml_path(str(xml))
    assets = ContactCorrectedAssets(args.root)
    scale = displacement_scale(model, assets)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnostics").mkdir()
    storage = json.loads(Path(intake["frozen"]["goal_config"]["path"]).read_text())["storage"]
    budget = StorageBudget(args.output, Path(intake["storage_before"]["source"]["root"]),
                           storage["minimum_free_bytes"], storage["maximum_experiment_unique_allocated_bytes"],
                           storage["maximum_new_source_unique_allocated_bytes"])
    frozen = [identity(args.intake), identity(args.model_comparison), *intake["frozen"].values()]
    sources = [HERE / name for name in ("bones_retarget.py", "bones_g1_csv.py", "bones_retarget_control.json",
                                        "bones_retarget_mapping.json", "intake_bones_archive.py")]
    sources += [HERE.parent / "supermimic_q1_phuma/retarget.py", HERE.parent / "q1/q1_retarget.py",
                HERE.parent / "q1/q1_sim.py", HERE.parent / "supermimic_for_q1/prepare_motion.py"]
    version = [identity(path) for path in sources]
    started, motions = time.monotonic(), []
    manifest = {"status": "FAILED", "reference_protocol": PROTOCOL, "intake": frozen[0],
                "model_comparison": frozen[1], "source_model": comparison["candidate"],
                "bones_generation_model_identity_proven": False, "source_model_assumption": comparison["limits"],
                "q1_assets": assets.manifest(), "scale": scale, "offset": args.offset, "count": args.count,
                "motions": motions, "sources": version, "physics_steps": 0, "optimizer_updates": 0,
                "qualified_for_training": False, "storage_status": "LOCAL_ONLY"}
    try:
        budget.check()
        baseline = health(args.expected_oom_kill)
        for row in selected:
            clip = row["split"] + "_" + row["source_group"][:20]
            item = {"clip": clip, "bones_source": row, "split": row["split"],
                    "bucket": row["candidate_kind"], "qualified_for_training_pilot": False}
            try:
                source = intake["extraction"]["selected"][row["move_g1_path"]]
                path = Path(source["path"])
                if (not path.resolve().is_relative_to((args.intake.parent / "files").resolve())
                        or identity(path) != source):
                    raise ValueError("Extracted frozen BONES source changed")
                motion = decode_bones_g1(path.read_bytes(), source_hz=120.,
                                         expected_frames=row["move_duration_frames"])
                solver, floor = grounded_solver(assets)
                reference, details = source_reference(model, motion, solver.names)
                item.update(details, source_csv=source, target_calibration_floor_m=floor)
                item["first_pose_initialization"] = initialize_first_pose(solver, reference, scale)
                result = solver.retarget(reference, scale)
                trace = args.output / "diagnostics" / (clip + ".npz")
                np.savez_compressed(trace, time_s=reference.times, qpos_before_ground_origin=result[1],
                                    target_position_m_world=result[2], target_rotation_matrix=result[3],
                                    position_error_m=result[4], source_position_m_world=reference.positions,
                                    source_orientation_wxyz_world=reference.orientations,
                                    source_body_names=np.asarray(reference.names))
                exported = export_q1(assets, result[1], result[4], args.output / (clip + ".npz"))
                item.update(exported, kinematic_trace=identity(trace))
                item["converter_kinematic_candidate"] = item.pop("qualified_for_training_pilot")
                item["qualified_for_training_pilot"] = False
            except Exception as exc:
                item.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            motions.append(item)
            budget.check()
            health(baseline.get("oom_kill"))
            save(args.output / "progress.json", {"completed": len(motions), "requested": len(selected),
                                                "elapsed_s": time.monotonic() - started})
            print(json.dumps(item, allow_nan=False), flush=True)
        if any(identity(record["path"]) != record for record in frozen + version):
            raise ValueError("Frozen input or retarget code changed during conversion")
        manifest.update(status="BONES_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING",
                        converted_split_counts=dict(collections.Counter(m["split"] for m in motions if "output" in m)),
                        storage_after=budget.check(), health_after=health(baseline.get("oom_kill")))
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["seconds"] = time.monotonic() - started
        save(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "manifest": identity(args.output / "manifest.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--model-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    convert(parser.parse_args())
