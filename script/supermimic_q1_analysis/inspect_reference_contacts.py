#!/usr/bin/env python3
"""Recompute the three D04 peak frames and describe their active contacts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from inspect_reference_dynamics import checked, configured_model, container_guard, identity, inverse_with_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data"):
        raise ValueError("New project data output file required")
    baseline = container_guard(reserve=384 * 1024 ** 2)
    import mujoco as mj
    report_id, audit_id = identity(args.receipt), identity(args.audit)
    report = json.loads(checked(root, report_id).read_text())
    audit = json.loads(checked(root, audit_id).read_text())
    if (audit["diagnostic"] != report_id or audit["status"] != "REFERENCE_DYNAMICS_SAVED_SUMMARY_AUDIT_PASSED"
            or len(report["clips"]) != 3 or len(audit["clips"]) != 3
            or identity(Path(__file__).with_name("inspect_reference_dynamics.py")) != report["source"]):
        raise ValueError("Audited diagnostic or helper source changed")
    admission = json.loads(checked(root, report["admission"]).read_text())
    model, _, floor, _, mesh_ids = configured_model(root, admission, mj)
    results = []
    for clip, proof in zip(report["clips"], audit["clips"]):
        container_guard(baseline["oom_kill"])
        if clip["clip"] != proof["clip"]:
            raise ValueError("Clip order mismatch")
        peak = proof["peak_inverse_force_ratios"]["central3"][0]
        frame = peak["frame"]
        with np.load(checked(root, clip["reference"]), allow_pickle=False) as reference:
            qpos, qvel = reference["qpos"][frame].astype(float), reference["qvel"][frame].astype(float)
        with np.load(checked(root, clip["raw_arrays"]), allow_pickle=False) as raw:
            offsets = np.flatnonzero(raw["frame_index"] == frame)
            if len(offsets) != 1:
                raise ValueError("Peak frame missing")
            offset = offsets[0]
            acceleration = raw["central3_qacc"][offset].copy()
            expected = raw["central3_inverse"][offset].copy()
        data = mj.MjData(model)
        data.qpos[:], data.qvel[:] = qpos, qvel
        mj.mj_forward(model, data)
        force, decomposition_error = inverse_with_check(model, data, acceleration, mj)
        saved_difference = float(np.max(np.abs(force - expected)))
        if saved_difference > 1e-7:
            raise ValueError("Recomputed peak inverse force differs from saved trace")
        contacts = []
        for index, contact in enumerate(data.contact):
            local_force = np.zeros(6)
            mj.mj_contactForce(model, data, index, local_force)
            contacts.append({"geom1": model.geom(contact.geom1).name,
                             "geom2": model.geom(contact.geom2).name,
                             "body1": model.body(int(model.geom_bodyid[contact.geom1])).name,
                             "body2": model.body(int(model.geom_bodyid[contact.geom2])).name,
                             "floor_contact": bool(contact.geom1 == floor or contact.geom2 == floor),
                             "distance_m": float(contact.dist), "active": bool(contact.efc_address >= 0),
                             "local_force_6": local_force.tolist(),
                             "force_norm_N": float(np.linalg.norm(local_force[:3]))})
        contacts.sort(key=lambda row: row["force_norm_N"], reverse=True)
        joint = report["active_joint_names"].index(peak["joint"])
        dof = 6 + joint
        results.append({"clip": clip["clip"], "peak": peak, "saved_inverse_max_difference": saved_difference,
                        "decomposition_max_error": decomposition_error,
                        "peak_joint_bias_inertia_Nm": float(force[dof] + data.qfrc_passive[dof] + data.qfrc_constraint[dof]),
                        "peak_joint_passive_Nm": float(data.qfrc_passive[dof]),
                        "peak_joint_constraint_Nm": float(data.qfrc_constraint[dof]),
                        "constraint_type_counts": {str(mj.mjtConstraint(int(kind))): int(count)
                                                   for kind, count in zip(*np.unique(data.efc_type, return_counts=True))},
                        "contacts": contacts, "data_time_s": float(data.time)})
    for item in (report_id, audit_id, report["admission"], report["source"], *mesh_ids,
                 *(clip["reference"] for clip in report["clips"]),
                 *(clip["raw_arrays"] for clip in report["clips"])):
        checked(root, item)
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected Torch import")
    result = {"status": "REFERENCE_PEAK_CONTACT_DIAGNOSTIC_COMPLETED", "source": identity(__file__),
              "diagnostic": report_id, "summary_audit": audit_id, "frames": results,
              "health_before": baseline, "health_after": container_guard(baseline["oom_kill"]),
              "new_inverse_calls": len(results), "new_physics_steps": 0, "new_policy_inference": 0,
              "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
              "limits": "Three peak frames only; exact inverse/contact reproduction is not a whole-clip tracking test or causal training comparison."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": identity(output)}), flush=True)


if __name__ == "__main__":
    main()
