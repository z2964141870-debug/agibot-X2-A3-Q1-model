#!/usr/bin/env python3
"""Inspect the four supplied Q1 ZIPs without extracting or executing them."""

import argparse
import importlib.util
import json
from pathlib import Path


NAMES = ("q1_v3.zip", "sdk_q1-v1.0.0.0.zip",
         "docker_x86_cross_arm-v1.0.0.zip", "docker_x86_native_x86.zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    args = parser.parse_args()
    helper = Path(__file__).parents[1] / "q1/inspect_resources.py"
    spec = importlib.util.spec_from_file_location("q1_inspection", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inspected = [module.inspect_archive(args.source / name) for name in NAMES]
    if any(item["unsafe_paths"] for item in inspected):
        raise ValueError("Unsafe ZIP path")
    full = {"status": "STATIC_ONLY", "archives": inspected}
    summary = {"date": "2026-10-23", "stage": "SM-Q1-P01",
               "source": str(args.source.resolve()), "storage_status": "LOCAL_ONLY",
               "runtime_validated": False, "archives": []}
    for item in inspected:
        result = {key: item[key] for key in
                  ("filename", "bytes", "sha256", "file_count", "uncompressed_bytes",
                   "zip_comment", "model_files", "unsafe_paths")}
        result["assets"] = [{key: asset[key] for key in
                             ("member", "format", "mass_kg")}
                            | {"joint_count": asset.get("movable_joint_count", asset.get("joint_count")),
                               "missing_mesh_references": sum(m["status"] == "missing"
                                                              for m in asset["mesh_references"])}
                            for asset in item.get("assets", [])]
        summary["archives"].append(result)
    for destination, value in ((args.private_report, full), (args.output, summary)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
