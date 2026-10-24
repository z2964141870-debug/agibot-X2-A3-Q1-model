#!/usr/bin/env python3
"""Receive pinned public HumanML provenance metadata, without downloading motions."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


COMMIT = "9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23"
FILES = ("index.csv", "LICENSE", "README.md", "raw_pose_processing.ipynb")


def identity(path):
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/references"):
        raise ValueError("Use a new metadata directory under the Q1 references tree")

    def health():
        base = Path("/sys/fs/cgroup/memory")
        oom = int(dict(line.split() for line in (base / "memory.oom_control").read_text().splitlines())["oom_kill"])
        memory = int((base / "memory.usage_in_bytes").read_text())
        if oom != args.expected_oom_kill or memory >= int(28.5 * 1024 ** 3):
            raise RuntimeError("Metadata intake memory or OOM reserve reached")
        if shutil.disk_usage(output.parent).free < 100 * 1024 ** 3:
            raise RuntimeError("Metadata intake disk reserve reached")
        return {"shared_memory_bytes": memory, "oom_kill": oom}

    before = health()
    output.mkdir()
    files = []
    for name in FILES:
        health()
        url = f"https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{COMMIT}/{name}"
        with urlopen(url, timeout=30) as response:
            payload = response.read(2 * 1024 ** 2 + 1)
        if not payload or len(payload) > 2 * 1024 ** 2:
            raise ValueError("Missing or oversized public metadata response")
        if name == "LICENSE" and not payload.startswith(b"MIT License"):
            raise ValueError("Pinned repository license differs from the inspected MIT document")
        path = output / name
        with path.open("xb") as stream:
            stream.write(payload)
        files.append({**identity(path), "url": url})
    reader = csv.DictReader(io.StringIO((output / "index.csv").read_text()))
    rows = list(reader)
    report = {"status": "PINNED_PUBLIC_HUMANML_METADATA_RECEIVED", "captured_utc": datetime.now(timezone.utc).isoformat(),
              "repository": "https://github.com/EricGuo5513/HumanML3D", "commit": COMMIT, "files": files,
              "index_columns": reader.fieldnames, "index_rows": len(rows), "index_examples": rows[:3],
              "source": identity(Path(__file__).resolve()), "health_before": before, "health_after": health(),
              "downloaded_motion_files": 0, "executed_upstream_code": False,
              "new_q1_references": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY", "limits": "Public source metadata and license only; no AMASS or HumanML motions, source-identity reconciliation or split changes. Repository MIT license does not replace dataset/body-model terms."}
    with (output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
