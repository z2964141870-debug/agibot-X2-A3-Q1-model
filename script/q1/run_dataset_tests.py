#!/usr/bin/env python3
"""Run bounded, resumable offline dataset conversions and Q1 supported replay."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from dataset_input import identity, load_dataset_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "dataset_test_cases_20261009.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    spec = json.loads(args.cases.read_text())
    if spec["usage"] != "TEST_ONLY" or spec["training_allowed"] is not False:
        raise ValueError("Test cases must declare TEST_ONLY")
    if len({c["id"] for c in spec["cases"]}) != len(spec["cases"]):
        raise ValueError("Duplicate test case IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    names = list(json.loads((HERE / "retarget_config.json").read_text())["body_map"])
    report = {"usage": "TEST_ONLY", "training_allowed": False,
              "cases_spec": identity(args.cases), "driver": identity(__file__), "cases": [],
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "policy_trained": False, "hardware_control": False, "baidu_backup_status": "LOCAL_ONLY"}
    for case in spec["cases"]:
        row = {"id": case["id"], "kind": case["kind"], "commands": [], "status": "RUNNING"}
        report["cases"].append(row)
        print(f"Starting {case['id']}", flush=True)
        directory = args.output_dir / case["id"]
        body = directory / "body_reference.npz"
        evaluation = directory / "evaluation"
        try:
            if body.exists():
                _, conversion = load_dataset_reference(body, names)
                if conversion["converter"]["sha256"] != identity(HERE / "dataset_input.py")["sha256"]:
                    raise ValueError("Existing conversion used another converter version; preserve it and use a new directory")
                row["conversion_resumed"] = True
            else:
                command = [sys.executable, str(HERE / "dataset_input.py"), "--kind", case["kind"],
                           "--input", case["input"], "--model", case["model"], "--seconds", str(args.seconds),
                           "--output", str(body)]
                row["commands"].append(command)
                with (args.log_dir / f"{case['id']}_convert.log").open("x") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
            if not (evaluation / "summary.json").exists():
                command = [sys.executable, str(HERE / "validate_egolocate.py"), "--input-format", "dataset-reference",
                           "--input", str(body), "--output-dir", str(evaluation)]
                if case["render"]:
                    command.extend(["--render", "--render-backend", "cpu"])
                row["commands"].append(command)
                with (args.log_dir / f"{case['id']}_evaluate.log").open("x") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
            summary = json.loads((evaluation / "summary.json").read_text())
            if summary["conversion"]["body_reference"]["sha256"] != identity(body)["sha256"]:
                raise ValueError("Evaluation input identity differs from current body reference")
            for artifact in summary["artifacts"]:
                if identity(evaluation / artifact["name"])["sha256"] != artifact["sha256"]:
                    raise ValueError("Evaluation artifact identity mismatch")
            for source in summary["source_files"]:
                source_path = Path(source["path"])
                if not source_path.is_absolute():
                    source_path = ROOT / source_path
                if identity(source_path)["sha256"] != source["sha256"]:
                    raise ValueError("Existing evaluation used another source version")
            row.update({"status": "COMPLETE", "summary": identity(evaluation / "summary.json"),
                        "body_conversion": identity(body.with_suffix(".json")),
                        "metrics": {k: summary[k] for k in ["kinematic_body_position_error_p95_m",
                            "reference_joint_limit_violations", "supported_replay_max_joint_rmse_rad",
                            "supported_replay_duration_s", "reference_max_joint_speed_rad_s",
                            "q1_leg_geometry", "kinematic_root_translation_range_m"]}})
            print(f"Complete {case['id']}: P95 {row['metrics']['kinematic_body_position_error_p95_m']:.4f} m", flush=True)
        except Exception as exc:
            row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            print(f"Failed {case['id']}: {row['error']}", flush=True)
        report["updated_unix_s"] = time.time()
        (args.output_dir / "suite_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["complete_count"] = sum(r["status"] == "COMPLETE" for r in report["cases"])
    report["failed_count"] = len(report["cases"]) - report["complete_count"]
    (args.output_dir / "suite_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Suite complete: {report['complete_count']} passed, {report['failed_count']} failed", flush=True)
    return bool(report["failed_count"])


if __name__ == "__main__":
    sys.exit(main())
