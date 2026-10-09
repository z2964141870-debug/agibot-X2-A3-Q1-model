#!/usr/bin/env python3
"""Verify saved tests and diagnose world-gravity leakage in root calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_input import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        parser.error("Preserve an existing manifest")
    suite_path = args.suite_dir / "suite_summary.json"
    suite = json.loads(suite_path.read_text())
    report = {"usage": "TEST_ONLY", "training_allowed": False,
              "suite": identity(suite_path), "collector": identity(__file__), "cases": [],
              "scope": "KINEMATIC_IK_AND_FIXED_PELVIS_PD_NOT_DYNAMIC_MOTION_TRACKING",
              "pose_fidelity_validated": False, "balance_verified": False,
              "policy_trained": False, "hardware_control": False, "baidu_backup_status": "LOCAL_ONLY"}
    for row in suite["cases"]:
        if row["status"] != "COMPLETE":
            report["cases"].append(row)
            continue
        summary_path = Path(row["summary"]["path"])
        if identity(summary_path)["sha256"] != row["summary"]["sha256"]:
            raise ValueError("Saved summary changed")
        summary = json.loads(summary_path.read_text())
        for artifact in summary["artifacts"]:
            if identity(summary_path.parent / artifact["name"])["sha256"] != artifact["sha256"]:
                raise ValueError("Saved result artifact changed")
        body_path = args.suite_dir / row["id"] / "body_reference.npz"
        with np.load(body_path, allow_pickle=False) as body:
            root = body["body_names"].tolist().index("pelvis")
            source_delta = body["position_m_world"][:, root] - body["calibration_position_m_world"][root]
            source_cal_rotation = Rotation.from_quat(body["calibration_orientation_wxyz_world"][root], scalar_first=True).as_matrix()
        with np.load(summary_path.parent / "kinematic_trace.npz", allow_pickle=False) as trace:
            qpos = trace["qpos"].copy()
        robot_cal_rotation = Rotation.from_quat(qpos[0, 3:7], scalar_first=True).as_matrix()
        alignment = robot_cal_rotation @ source_cal_rotation.T
        scale = summary["scaling"]["scale"]
        predicted = scale * source_delta @ alignment.T
        actual = qpos[:, :3] - qpos[0, :3]
        reproduction_error = float(np.max(np.abs(predicted - actual)))
        if reproduction_error > 1e-5:
            raise ValueError("Root trace does not reproduce the current calibration formula")
        horizontal_z = scale * source_delta[:, :2] @ alignment[2, :2]
        baseline_z = scale * source_delta[:, 2]
        diagnostic = {
            "method": "SAVED_TRACE_VS_FULL_SO3_FIRST_FRAME_CALIBRATION_FORMULA",
            "formula_reproduction_max_abs_error_m": reproduction_error,
            "alignment_matrix": alignment.tolist(),
            "source_world_z_range_scaled_m": float(np.ptp(baseline_z)),
            "q1_kinematic_root_z_range_m": float(np.ptp(actual[:, 2])),
            "world_horizontal_motion_contribution_to_q1_z_range_m": float(np.ptp(horizontal_z)),
            "world_up_alignment_angle_deg": float(np.rad2deg(np.arccos(np.clip(alignment[2, 2], -1, 1)))),
            "gravity_preserving_calibration_validated": False,
            "note": "Current full SO(3) root alignment can rotate horizontal translation into vertical motion; fixed support replay does not test this trajectory."}
        conversion = dict(summary["conversion"])
        if "betas" in conversion:
            shape_values = conversion.pop("betas")
            conversion["betas_json_sha256"] = hashlib.sha256(
                json.dumps(shape_values, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            conversion["shape_values_storage"] = "ORIGINAL_AND_CONVERSION_JSON_ON_SERVER_NOT_PUBLISHED"
        report["cases"].append({"id": row["id"], "kind": row["kind"], "status": row["status"],
                                "summary": row["summary"], "conversion": conversion,
                                "metrics": row["metrics"], "artifacts": summary["artifacts"],
                                "source_files": summary["source_files"],
                                "root_gravity_diagnostic": diagnostic})
        print(f"{row['id']}: root z {diagnostic['q1_kinematic_root_z_range_m']:.3f} m; "
              f"scaled source z {diagnostic['source_world_z_range_scaled_m']:.3f} m; "
              f"horizontal leakage {diagnostic['world_horizontal_motion_contribution_to_q1_z_range_m']:.3f} m", flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Manifest: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
