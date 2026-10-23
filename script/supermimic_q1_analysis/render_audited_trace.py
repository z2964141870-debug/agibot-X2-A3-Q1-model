#!/usr/bin/env python3
"""Bind audited Q1 saved-trace rendering to its corrected contact assets."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_file(record):
    if identity(record["path"]) != record:
        raise ValueError("Saved rendering input differs from its recorded identity")
    return Path(record["path"])


def prepare(root, evaluation_path, audit_path, clip):
    root = Path(root).resolve()
    evaluation_path, audit_path = Path(evaluation_path), Path(audit_path)
    evaluation = json.loads(evaluation_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED"
            or audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
            or audit["verified"] is not True
            or audit["receipt"] != identity(evaluation_path)):
        raise ValueError("Rendering requires the matching successful trace audit")
    rows = {row["clip"]: row for row in evaluation["rows"]}
    audited = {row["clip"]: row for row in audit["clips"]}
    if (not rows or len(rows) != len(evaluation["rows"])
            or len(audited) != len(audit["clips"]) or rows.keys() != audited.keys()
            or not all(row["passed"] is True for row in audited.values())):
        raise ValueError("Audited clip coverage differs from the evaluation")
    row = rows[clip]
    expected, executed = row["expected_steps"], row["executed_steps"]
    if (type(expected) is not int or type(executed) is not int
            or not 0 < executed <= expected
            or type(row["failed"]) is not bool
            or (executed < expected and not row["failed"])):
        raise ValueError("Invalid full-clip denominator or unmarked early failure")
    if any(audited[clip]["recomputed"][key] != row[key]
           for key in ("expected_steps", "executed_steps", "failed")):
        raise ValueError("Rendered clip counts differ from the independent audit")
    checkpoint = checked_file(evaluation["checkpoint"])
    if not checkpoint.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Expected a checkpoint under the Q1 experiment root")
    admission_path = checkpoint.parent / "admission.json"
    admission = json.loads(admission_path.read_text())
    assets = admission["assets"]
    if assets.get("contact_protocol", {}).get("version") != "Q1_FOOT_PRIORITY_V2":
        raise ValueError("Rendering entry requires the corrected Q1 contact protocol")
    for record in (assets["mjcf"], assets["adapter"], assets["contact_protocol"]["source"],
                   row["reference"], row["trace"]):
        checked_file(record)
    driver = root / "script/supermimic_for_q1/render_trace.py"
    registered = {item["path"]: item for item in admission["source_files"]}
    if registered.get(str(driver)) != identity(driver):
        raise ValueError("Preserved renderer differs from the admitted source")
    return {"status": "AUDITED_TRACE_RENDER_PREFLIGHT_PASSED", "source": identity(__file__),
            "evaluation": identity(evaluation_path), "audit": identity(audit_path),
            "checkpoint": evaluation["checkpoint"], "admission": identity(admission_path),
            "preserved_renderer": identity(driver), "assets": assets,
            "clip": clip, "trace": row["trace"], "reference": row["reference"],
            "expected_steps": expected, "executed_steps": executed, "failed": row["failed"],
            "rendered_frames_expected": (executed + 1) // 2, "fps": 25,
            "max_seconds": expected / 50. + 1.,
            "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
            "storage_status": "LOCAL_ONLY",
            "limits": "Full executed trace, including early-failure disclosure. Preflight performs no rendering and establishes no policy improvement."}


def render(root, output, prepared):
    core = Path(root).resolve() / "script/supermimic_for_q1"
    sys.path.insert(0, str(core))
    driver = importlib.import_module("render_trace")
    from contact_protocol import ContactCorrectedAssets
    from supermimic.q1_assets import Q1Assets

    if (Path(driver.__file__).resolve() != core / "render_trace.py"
            or driver.Q1Assets is not Q1Assets
            or ContactCorrectedAssets(root).manifest() != prepared["assets"]):
        raise ValueError("Rendering constructor or corrected asset contract differs")
    driver.Q1Assets = ContactCorrectedAssets
    original_argv = sys.argv
    try:
        sys.argv = [str(driver.__file__), "--root", str(root),
                    "--evaluation", prepared["evaluation"]["path"],
                    "--clip", prepared["clip"], "--output", str(output),
                    "--max-seconds", str(prepared["max_seconds"])]
        driver.main()
    finally:
        driver.Q1Assets = Q1Assets
        sys.argv = original_argv
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "SAVED_TRACE_RENDER_COMPLETED"
            or receipt["assets"] != prepared["assets"]
            or receipt["evaluation"] != prepared["evaluation"]
            or receipt["trace"] != prepared["trace"]
            or receipt["reference"] != prepared["reference"]
            or receipt["rendered_frames"] != prepared["rendered_frames_expected"]):
        raise ValueError("Rendered output differs from the audited full-trace contract")
    for item in receipt["artifacts"]:
        checked_file(item)
    return {**prepared, "status": "AUDITED_SAVED_TRACE_RENDER_COMPLETED",
            "render_receipt": identity(receipt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.check_only and args.preflight_output is None:
        parser.error("--check-only requires --preflight-output")
    if args.preflight_output is not None and args.preflight_output.exists():
        raise FileExistsError(args.preflight_output)
    prepared = prepare(args.root, args.evaluation, args.audit, args.clip)
    if args.preflight_output is not None:
        with args.preflight_output.open("x") as stream:
            json.dump(prepared, stream, indent=2, allow_nan=False)
            stream.write("\n")
    result = prepared if args.check_only else render(args.root, args.output, prepared)
    if not args.check_only:
        with (args.output / "audited_render.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}, allow_nan=False))


if __name__ == "__main__":
    main()
