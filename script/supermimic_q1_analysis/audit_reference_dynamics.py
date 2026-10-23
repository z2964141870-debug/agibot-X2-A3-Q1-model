#!/usr/bin/env python3
"""Independently recompute D04 statistics and acceleration from saved arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_id(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def verified_file(root, expected):
    path = Path(expected["path"]).resolve()
    if not path.is_relative_to(root) or file_id(path) != expected:
        raise ValueError(f"File identity/scope mismatch: {path}")
    return path


def run(root, receipt_path, output):
    root, receipt_path, output = root.resolve(), receipt_path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(root / "data"):
        raise ValueError("New project data audit file required")
    report_id = file_id(receipt_path)
    report = json.loads(verified_file(root, report_id).read_text())
    if (report["status"] != "REFERENCE_INVERSE_CONTACT_DIAGNOSTIC_COMPLETED"
            or report["contact_protocol"] != "Q1_FOOT_PRIORITY_V2" or report["friction"] != 1.):
        raise ValueError("Unexpected diagnostic contract")
    caps = np.asarray(report["effective_effort_nm"])
    checks, max_difference, clips = 0, 0., []

    def equal(actual, expected):
        nonlocal checks, max_difference
        checks += 1
        if actual is None or expected is None:
            if actual is not expected:
                raise ValueError("Unexpected empty/nonempty statistic")
            return
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Statistic shape or finite mismatch")
        difference = float(np.max(np.abs(a - b)))
        max_difference = max(max_difference, difference)
        if not np.allclose(a, b, atol=1e-10, rtol=1e-12):
            raise ValueError(f"Statistic mismatch: {actual} versus {expected}")

    def check_stats(values, saved):
        flat = np.asarray(values).reshape(-1)
        equal(len(flat), saved["count"])
        for key, operation in (("mean", np.mean), ("max", np.max),
                               ("p50", lambda x: np.percentile(x, 50)),
                               ("p95", lambda x: np.percentile(x, 95))):
            equal(float(operation(flat)) if len(flat) else None, saved[key])

    for clip in report["clips"]:
        raw_path = verified_file(root, clip["raw_arrays"])
        reference_path = verified_file(root, clip["reference"])
        with np.load(raw_path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        with np.load(reference_path, allow_pickle=False) as saved:
            velocities = saved["qvel"].astype(float)
            poses = saved["qpos"].astype(float)
            times = saved["time_s"].astype(float)
        indices = arrays["frame_index"]
        equal(indices, np.arange(2, len(times) - 2))
        equal(arrays["time_s"], times[indices])
        equal(len(indices), clip["analyzed_frames"])
        equal(len(times), clip["source_frames"])
        equal(2 * len(indices), clip["inverse_calls"])
        equal(clip["excluded_boundary_frames"], [0, 1, len(times) - 2, len(times) - 1])
        gap, speed = arrays["foot_gap_m"], arrays["foot_tangent_speed_m_s"]
        equal(np.min(gap), clip["minimum_foot_gap_m"])
        equal(np.count_nonzero(arrays["active_foot_floor_contacts"] > 0) / len(indices),
              clip["foot_floor_contact_fraction"])
        check_stats(np.min(gap, axis=1), clip["lower_foot_gap_m"])
        for distance in (.001, .01, .03):
            mask = gap <= distance
            near_speeds = speed[mask]
            summary = clip["proximity"][str(distance)]
            equal(np.count_nonzero(np.any(mask, axis=1)) / len(indices), summary["frame_any_foot_fraction"])
            check_stats(near_speeds, summary["point_tangent_speed_m_s"])
            equal(np.count_nonzero(near_speeds > .05) / len(near_speeds) if len(near_speeds) else None,
                  summary["point_speed_above_0_05_fraction"])
        peaks = {}
        for label in ("central3", "central5"):
            dt = clip["dt_s"]
            if label == "central3":
                expected_acc = np.asarray([(velocities[i + 1] - velocities[i - 1]) / (2 * dt) for i in indices])
            else:
                expected_acc = np.asarray([(velocities[i - 2] - 8 * velocities[i - 1]
                                           + 8 * velocities[i + 1] - velocities[i + 2]) / (12 * dt)
                                          for i in indices])
            equal(expected_acc, arrays[label + "_qacc"])
            force = arrays[label + "_inverse"]
            if force.shape != (len(indices), 28):
                raise ValueError("Wrong saved generalized-force layout")
            ratio = np.abs(force[:, 6:]) / caps
            summary = clip["inverse"][label]
            norms = np.sqrt(np.sum(force[:, :3] ** 2, axis=1))
            check_stats(norms, summary["root_force_N"])
            check_stats(norms / clip["body_weight_N"], summary["root_force_body_weights"])
            check_stats(np.sqrt(np.sum(force[:, 3:6] ** 2, axis=1)), summary["root_torque_Nm"])
            check_stats(ratio, summary["active_effort_over_cap"])
            check_stats(np.abs(expected_acc[:, 6:]), summary["joint_acceleration_rad_s2"])
            equal(np.count_nonzero(np.any(ratio > 1, axis=1)) / len(indices),
                  summary["frames_any_active_effort_over_cap_fraction"])
            equal(np.count_nonzero(ratio > 1, axis=0) / len(indices), summary["per_joint_effort_over_cap_fraction"])
            equal(np.max(arrays[label + "_decomposition_error"]), summary["inverse_rne_decomposition_max_error"])
            peaks[label] = []
            for flat_index in np.argsort(ratio.reshape(-1))[-5:][::-1]:
                offset, joint = np.unravel_index(flat_index, ratio.shape)
                frame = int(indices[offset])
                peaks[label].append({"frame": frame, "time_s": float(times[frame]),
                                     "joint": report["active_joint_names"][joint],
                                     "effort_nm": float(force[offset, 6 + joint]), "cap_nm": float(caps[joint]),
                                     "ratio": float(ratio[offset, joint]), "qpos_rad": float(poses[frame, 7 + joint]),
                                     "qvel_rad_s": float(velocities[frame, 6 + joint]),
                                     "qacc_rad_s2": float(expected_acc[offset, 6 + joint])})
        verified_file(root, clip["raw_arrays"])
        verified_file(root, clip["reference"])
        clips.append({"clip": clip["clip"], "raw_arrays": clip["raw_arrays"],
                      "analyzed_frames": len(indices), "peak_inverse_force_ratios": peaks})
    verified_file(root, report_id)
    audit = {"status": "REFERENCE_DYNAMICS_SAVED_SUMMARY_AUDIT_PASSED", "source": file_id(__file__),
             "diagnostic": report_id, "checks": checks, "max_absolute_difference": max_difference,
             "clips": clips, "new_physics_steps": 0, "new_inverse_calls": 0,
             "new_policy_inference": 0, "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
             "limits": "Recomputes saved summaries and velocity derivatives; does not independently rerun MuJoCo inverse forces or prove tracking feasibility."}
    with output.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": audit["status"], "checks": checks,
                      "max_absolute_difference": max_difference, "audit": file_id(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.receipt, args.output)


if __name__ == "__main__":
    main()
