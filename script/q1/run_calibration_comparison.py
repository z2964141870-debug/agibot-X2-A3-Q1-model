#!/usr/bin/env python3
"""Compare preserved legacy, gravity-fixed and synthetic-standing Q1 calibration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from dataset_input import identity


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODES = {
    "legacy": ("full_rotation_legacy_diagnostic", "first-frame"),
    "gravity_fixed": ("gravity_preserving_heading", "first-frame"),
    "standing_pair": ("gravity_preserving_heading", "smplx-standing"),
}


def verify_summary(path):
    summary = json.loads(path.read_text())
    for item in summary["artifacts"]:
        if identity(path.parent / item["name"])["sha256"] != item["sha256"]:
            raise ValueError("Evaluation artifact changed")
    for item in summary["source_files"]:
        source = Path(item["path"])
        if not source.is_absolute():
            source = ROOT / source
        if identity(source)["sha256"] != item["sha256"]:
            raise ValueError("Completed evaluation used different code/config; preserve it")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, default=ROOT / "data/experiments/q1_datasets_20261009")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((HERE / "retarget_config.json").read_text())
    configs = {}
    for mode, (alignment, _) in MODES.items():
        config = dict(configuration, root_translation_alignment=alignment)
        path = args.output_dir / f"retarget_{mode}.json"
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError("Existing experiment config differs; use a new directory")
        else:
            path.write_text(json.dumps(config, indent=2) + "\n")
        configs[mode] = path
    report = {"usage": "TEST_ONLY", "training_allowed": False,
              "scope": "CALIBRATION_ABLATION_IK_FIXED_PELVIS_PD",
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "driver": identity(__file__), "cases": [], "policy_trained": False,
              "hardware_control": False, "backup_status": "LOCAL_ONLY"}
    cases = ["amass_stand", "amass_walk_turn", "amass_kick", "amass_swing_arms", "bones_jog"]
    for case in cases:
        for mode, (_, calibration) in MODES.items():
            if case == "bones_jog" and mode == "standing_pair":
                continue
            directory = args.output_dir / case / mode
            summary_path = directory / "summary.json"
            row = {"case": case, "mode": mode, "status": "RUNNING"}
            report["cases"].append(row)
            body = args.source_suite / case / "body_reference.npz"
            try:
                command = [sys.executable, str(HERE / "validate_egolocate.py"), "--input-format", "dataset-reference",
                           "--input", str(body), "--output-dir", str(directory),
                           "--retarget-config", str(configs[mode]), "--calibration", calibration]
                row["command"] = command
                if not summary_path.exists():
                    print(f"Starting {case}/{mode}", flush=True)
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    with (args.log_dir / f"{case}_{mode}.log").open("x") as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=ROOT, timeout=240, check=True)
                summary = verify_summary(summary_path)
                if summary["conversion"]["body_reference"]["sha256"] != identity(body)["sha256"]:
                    raise ValueError("Evaluation input identity differs")
                with np.load(body, allow_pickle=False) as source:
                    root = source["body_names"].tolist().index("pelvis")
                    source_z = summary["scaling"]["scale"] * source["position_m_world"][:, root, 2]
                with np.load(directory / "kinematic_trace.npz", allow_pickle=False) as trace:
                    qpos = trace["qpos"]
                    vertical_error = (qpos[:, 2] - qpos[0, 2]) - (source_z - source_z[0])
                    original = args.source_suite / case / "evaluation/kinematic_trace.npz"
                    if mode == "legacy":
                        with np.load(original, allow_pickle=False) as old:
                            old_error = float(np.max(np.abs(qpos - old["qpos"])))
                        if old_error > 1e-5:
                            raise ValueError(f"Legacy result failed baseline reproduction: {old_error}")
                        row["legacy_qpos_reproduction_max_abs_error"] = old_error
                row.update({"status": "COMPLETE", "summary": identity(summary_path),
                            "source_scaled_z_range_m": float(np.ptp(source_z)),
                            "q1_root_z_range_m": float(np.ptp(qpos[:, 2])),
                            "root_vertical_delta_max_error_m": float(np.max(np.abs(vertical_error))),
                            "ik_position_p95_m": summary["kinematic_body_position_error_p95_m"],
                            "q1_leg_geometry": summary["q1_leg_geometry"],
                            "source_leg_geometry": summary["conversion"]["source_leg_geometry"],
                            "joint_limit_violations": summary["reference_joint_limit_violations"],
                            "max_joint_speed_rad_s": summary["reference_max_joint_speed_rad_s"],
                            "supported_max_joint_rmse_rad": summary["supported_replay_max_joint_rmse_rad"],
                            "gravity_preserving_root_check_passed": bool(mode != "legacy" and np.max(np.abs(vertical_error)) < 1e-5),
                            "artifacts": summary["artifacts"], "source_files": summary["source_files"]})
                print(f"Complete {case}/{mode}: root Z {row['q1_root_z_range_m']:.4f} m, IK P95 {row['ik_position_p95_m']:.4f} m", flush=True)
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                print(f"Failed {case}/{mode}: {row['error']}", flush=True)
            (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    report["complete_count"] = sum(row["status"] == "COMPLETE" for row in report["cases"])
    report["failed_count"] = len(report["cases"]) - report["complete_count"]
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed {report['complete_count']}, failed {report['failed_count']}", flush=True)
    return bool(report["failed_count"])


if __name__ == "__main__":
    sys.exit(main())
