#!/usr/bin/env python3
"""CPU-only end-to-end RF06 adapter verification using a labelled RF04 fixture."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(here.parents[1] / "supermimic_q1_sources"))
    from control import Leg50Assets, augment_training_manifest
    from supermimic.q1_assets import identity
    from relocated_io import migration_view
    import numpy as np
    import torch
    torch.set_num_threads(2)
    root, output = args.root.resolve(strict=True), args.output.resolve()
    if not output.is_relative_to(root / "data/supermimic_for_q1") or output.exists():
        parser.error("Use a new independent CPU verification output")
    output.mkdir()
    started = time.monotonic()
    rf04 = root / "data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025"
    diagnostic = root / "data/supermimic_for_q1/RF04_LEG_D06B_20261026"
    inputs = [identity(rf04 / "run/final.pt"), identity(diagnostic / "receipt.json"),
              identity(diagnostic / "legs50_other25.npz")]
    parent = torch.load(inputs[0]["path"], map_location="cpu", weights_only=False)
    manifest = migration_view(root).rebase(parent["manifest"])
    manifest["assets"] = Leg50Assets(root).manifest()
    augment_training_manifest(manifest, here / "entry.py")
    manifest["initialization"] = "CPU_VERIFICATION_FIXTURE_FIXED_RF04_POLICY_NO_TRAINING"
    checkpoint_path = output / "RF04_policy_leg_contract_fixture.pt"
    torch.save({"step": parent["step"], "policy_updates": parent["policy_updates"],
                "manifest": manifest, "policy": parent["policy"], "fixture_only": True}, checkpoint_path)
    del parent
    entry = here / "entry.py"
    evaluation = output / "evaluation/receipt.json"
    library = root / "data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references"
    phases = [
        ("evaluate", [str(entry), "evaluate", "--root", str(root), "--motions", str(library),
            "--checkpoint", str(checkpoint_path), "--output", str(output / "evaluation"),
            "--device", "cpu", "--count", "1", "--clips", "train_3419f09ba93c91d75fc1"]),
        ("audit", [str(entry), "audit", "--root", str(root), "--evaluation", str(output / "evaluation"),
            "--output", str(output / "evaluation/audit.json")]),
        ("fullbody", [str(entry), "fullbody", "analyze", "--root", str(root),
            "--evaluation", "initial", str(evaluation), "--evaluation", "final", str(evaluation),
            "--output", str(output / "fullbody_fixture.json")]),
        ("legs", [str(here / "analyze.py"), "--root", str(root), "--evaluation", str(evaluation),
            "--output", str(output / "leg_analysis")]),
        ("render", [str(entry), "render", "--root", str(root), "--evaluation", str(evaluation),
            "--clip", "train_3419f09ba93c91d75fc1", "--output", str(output / "render_fixture"),
            "--max-seconds", "0.08"]),
    ]
    results = []
    for label, command in phases:
        log_path = root / "logs" / (output.name + "_" + label + ".log")
        with log_path.open("x") as stream:
            result = subprocess.run([sys.executable, "-u", *command], cwd=root,
                stdout=stream, stderr=subprocess.STDOUT, timeout=90, check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        results.append({"phase": label, "return_code": result.returncode,
                        "command": [sys.executable, "-u", *command], "log": identity(log_path)})
        if result.returncode:
            (output / "failure.json").write_text(json.dumps({"status": "CPU_CONTRACT_CHECK_FAILED",
                "fixture_only": True, "phases": results}, indent=2) + "\n")
            raise RuntimeError("CPU contract phase failed: " + label)
    evaluated = json.loads(evaluation.read_text())
    row, = evaluated["rows"]
    with np.load(row["trace"]["path"], allow_pickle=False) as actual, \
            np.load(inputs[2]["path"], allow_pickle=False) as expected:
        state_error = float(np.max(np.abs(actual["state"] - expected["state"])))
        action_error = float(np.max(np.abs(actual["actions"] - expected["actions"])))
    saved_legs = json.loads((output / "leg_analysis/receipt.json").read_text())["rows"][0]
    diagnostic_row, = [item for item in json.loads(Path(inputs[1]["path"]).read_text())["trials"]
                       if item["trial"] == "legs50_other25"]
    if (state_error > 2e-5 or action_error > 2e-5
            or saved_legs["joint_saturation_fraction"] != diagnostic_row["joint_saturation_fraction"]
            or saved_legs["double_support_boundary_fraction"] != diagnostic_row["double_support_boundary_fraction"]):
        raise ValueError("The real evaluation/analysis entries disagree with the independent closed-loop probe")
    if inputs != [identity(record["path"]) for record in inputs]:
        raise ValueError("Preserved input evidence changed during verification")
    rendered = json.loads((output / "render_fixture/receipt.json").read_text())
    if rendered["pixel_std_min"] < 1. or rendered["assets"] != manifest["assets"]:
        raise ValueError("The real render entry did not use the evaluated leg control contract")
    result = {"status": "Q1_LEG_ADAPTER_CPU_CONTRACT_CHECK_COMPLETED", "source": identity(__file__),
        "fixture_only": True, "fixture": identity(checkpoint_path), "inputs": inputs,
        "phases": results, "assets": manifest["assets"], "state_max_abs_error": state_error,
        "action_max_abs_error": action_error, "support_analysis_matches_independent_probe": True,
        "evaluated_steps": row["executed_steps"], "rendered_frames": rendered["rendered_frames"],
        "pixel_std_min": rendered["pixel_std_min"], "new_optimizer_updates": 0,
        "cuda_used": torch.cuda.is_initialized(), "elapsed_seconds": time.monotonic() - started,
        "limits": "Fixed RF04-policy fixture under RF06 controls; no RF06 training or performance claim. Duplicate initial/final labels exercise the analysis contract only.",
        "storage_status": "LOCAL_ONLY"}
    (output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "state_max_abs_error",
        "action_max_abs_error", "support_analysis_matches_independent_probe", "evaluated_steps",
        "rendered_frames", "pixel_std_min", "new_optimizer_updates", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
