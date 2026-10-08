#!/usr/bin/env python3
"""Run the two TEST_ONLY clothing conversions and supported Q1 evaluations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ("G_0113_zj_train/183926_zj_walk",
             "G_0109_yy_train/193310_yy_oneLegStand")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--first-experiment", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", choices=("egl", "cpu"), default="egl")
    parser.add_argument("--archive", default="/media/yu/FAFF-E9771/data/fgp/CLOTHO_train_data/train_released.zip")
    parser.add_argument("--fgp-root", default="/home/yu/projects/x2-teleop/FGP-main")
    parser.add_argument("--velocity-root", default="/home/yu/projects/fgp/scripts/clotho_velocity")
    parser.add_argument("--velocity-checkpoint", default="/media/yu/FAFF-E9771/data/fgp/velocity_checkpoints/clotho_v2_displacement/last.pt")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.first_experiment < 1:
        parser.error("Use a simple run ID and a positive experiment number")
    directory = ROOT / "data/experiments" / args.run_id
    directory.mkdir(exist_ok=False)
    state = {"usage": "TEST_ONLY", "training_allowed": False, "run_id": args.run_id,
             "status": "RUNNING", "steps": [], "training_started": False,
             "hardware_control": False, "baidu_backup_status": "LOCAL_ONLY",
             "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = directory / "driver_summary.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "driver_summary.json")

    def run(name, script, arguments):
        command = [sys.executable, "-u", str(ROOT / "script/q1" / script), *arguments]
        log = ROOT / "logs" / f"{args.run_id}_{name}.log"
        step = {"name": name, "command": command, "log": str(log), "status": "RUNNING"}
        state["steps"].append(step)
        save()
        print(f"START {name}", flush=True)
        with log.open("x") as stream:
            process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        step["exit_code"] = process.returncode
        step["status"] = "PASS" if process.returncode == 0 else "FAIL"
        save()
        print(f"END {name} exit={process.returncode}", flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed; preserve its output and inspect {log}")

    save()
    try:
        run("input_tests", "test_egolocate_input.py", ["-v"])
        for offset, sequence in enumerate(SEQUENCES):
            number = args.first_experiment + offset
            inference = f"data/experiments/q1_cloth_fgp_20261008_E{number:03d}"
            evaluation = f"data/experiments/q1_cloth_q1_20261008_E{number:03d}"
            run(f"fgp_E{number:03d}", "infer_clotho_fgp.py", [
                "--archive", args.archive, "--sequence", sequence,
                "--fgp-root", args.fgp_root, "--velocity-root", args.velocity_root,
                "--velocity-checkpoint", args.velocity_checkpoint, "--device", args.device,
                "--output-dir", inference])
            summary = json.loads((ROOT / inference / "summary.json").read_text())
            if (summary["source_sequence"] != sequence
                    or not summary["velocity_valid_all_exported"]
                    or summary["usage"] != "TEST_ONLY" or summary["training_allowed"]):
                raise ValueError("Inference summary does not satisfy the test input contract")
            run(f"q1_E{number:03d}", "validate_egolocate.py", [
                "--input", f"{inference}/body_only.npz", "--output-dir", evaluation, "--render",
                "--render-backend", args.render_backend])
            result = json.loads((ROOT / evaluation / "summary.json").read_text())
            if (not result["input_decoded"] or not result["finite_kinematic_output"]
                    or not result["supported_replay_finite"]
                    or result["reference_joint_limit_violations"] != 0):
                raise ValueError("Q1 evaluation produced invalid references or replay")
        state["status"] = "COMPLETED_OFFLINE_TESTS_NOT_POSE_FIDELITY_OR_DYNAMIC_BALANCE_ACCEPTANCE"
        save()
    except Exception as error:
        state["status"] = "FAILED"
        state["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
