#!/usr/bin/env python3
"""Render the saved physical trace with the reproduction contact assets."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "supermimic_for_q1"
sys.path.insert(0, str(CORE))

from contact_protocol import ContactCorrectedAssets
from supermimic.q1_assets import identity


def renderer_metadata(value):
    metadata = json.loads(value)
    for row in metadata.get("rows", []):
        row["failed"] = row["early_terminated"]
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=12.)
    args, _ = parser.parse_known_args()
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
        raise ValueError("Render an actual completed physical evaluation")
    if ContactCorrectedAssets(args.root).manifest() != evaluation["assets"]:
        raise ValueError("Render asset/control identity differs from saved physical evaluation")
    spec = importlib.util.spec_from_file_location("q1_saved_trace_renderer", CORE / "render_trace.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.Q1Assets = ContactCorrectedAssets
    # The legacy renderer's "failed" label means early termination on screen.
    renderer.json = SimpleNamespace(loads=renderer_metadata, dump=json.dump, dumps=json.dumps)
    renderer.main()
    receipt_path = args.output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    row, = [row for row in evaluation["rows"] if row["clip"] == args.clip]
    receipt.update(reproduction_entry=identity(__file__),
                   physical_failed=row["failed"],
                   physical_early_termination=row["early_terminated"],
                   physical_executed_steps=row["executed_steps"],
                   physical_expected_steps=row["expected_steps"],
                   physical_root_xy_displacement_m=row["root_xy_displacement_m"],
                   physical_root_xy_path_length_m=row["root_xy_path_length_m"])
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
