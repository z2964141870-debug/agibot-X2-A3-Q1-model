#!/usr/bin/env python3
"""Check the RF07 interface, H32 graph and actual evaluation entry on CPU."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent / "leg_control", HERE.parent,
                  HERE.parents[1] / "supermimic_q1_sources"):
    sys.path.insert(0, str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        parser.error("Use a new independent Q1 verification output")
    output.mkdir()
    os.environ["Q1_FEEDBACK_TEST_ROOT"] = str(root)
    started = time.monotonic()
    report = {"status": "FAILED", "optimizer_updates": 0, "fixture_only": True,
              "storage_status": "LOCAL_ONLY"}
    try:
        tests = unittest.defaultTestLoader.discover(str(HERE), pattern="test_feedback.py")
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        report.update(tests_run=result.testsRun, skipped=len(result.skipped),
                      tests_passed=result.wasSuccessful())
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError("The root-feedback contracts did not all run successfully")
        import torch
        from control import Leg50Assets
        from feedback import input_contract
        from relocated_io import migration_view
        from supermimic.policy.nets import Policy, PolicyCfg
        from supermimic.q1_assets import identity
        from supermimic.robot_spec import GOAL_DIM
        torch.set_num_threads(2)
        torch.manual_seed(20261026)
        assets = Leg50Assets(root)
        parent = root / "data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/run/admission.json"
        manifest = migration_view(root).rebase(json.loads(parent.read_text()))
        if manifest["assets"] != assets.manifest():
            raise ValueError("RF07 must preserve the actual RF06 physical control contract")
        cfg = PolicyCfg(encoder_hidden=(64, 32), decoder_hidden=(64, 32), goal_dim=GOAL_DIM + 9)
        policy = Policy(cfg).eval()
        manifest.update(policy=asdict(cfg), reference_feedback=input_contract(),
            initialization="CPU_INTERFACE_FIXTURE_RANDOM_SMALL_POLICY_NOT_RF07_WEIGHTS")
        fixture = output / "random_small_feedback_fixture.pt"
        torch.save({"step": 0, "policy_updates": 0, "manifest": manifest,
                    "policy": policy.state_dict(), "fixture_only": True}, fixture)
        phases = [
            ("evaluate", [str(HERE / "entry_root_feedback.py"), "evaluate", "--root", str(root),
                "--motions", str(root / "data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references"),
                "--checkpoint", str(fixture), "--output", str(output / "evaluation"),
                "--device", "cpu", "--count", "1", "--clips", "train_3419f09ba93c91d75fc1"]),
            ("audit", [str(HERE / "entry_root_feedback.py"), "audit", "--root", str(root),
                "--evaluation", str(output / "evaluation"), "--output", str(output / "evaluation/audit.json")]),
        ]
        records = []
        for label, command in phases:
            log = root / "logs" / (output.name + "_" + label + ".log")
            with log.open("x") as stream:
                completed = subprocess.run([sys.executable, "-u", *command], cwd=root,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=120, check=False,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
            records.append({"phase": label, "return_code": completed.returncode, "log": identity(log)})
            report["phases"] = records
            if completed.returncode:
                raise RuntimeError("Actual CPU interface phase failed: " + label)
        report.update(status="Q1_ROOT_FEEDBACK_CPU_CONTRACT_CHECK_COMPLETED",
            source=identity(__file__), input_contract=input_contract(), parent=identity(parent),
            fixture=identity(fixture), evaluation=identity(output / "evaluation/receipt.json"),
            evaluation_audit=identity(output / "evaluation/audit.json"),
            cuda_used=torch.cuda.is_initialized(),
            limits="Random narrow fixture and CPU H32 graph, no optimizer updates; proves interface consistency, not learned gradient accuracy or tracking improvement")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
