#!/usr/bin/env python3
"""Fetch pinned PHUMA metadata and G1 source assets with Git blob verification."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import xml.etree.ElementTree as ET


REPOSITORY = "DAVIAN-Robotics/PHUMA"
COMMIT = "2faf15b52a3498bfcc3352ecb443dc0b5cd642ac"
MODEL = "asset/humanoid_model/g1"
METADATA = (
    "README.md", "LICENSE", "download_phuma.sh",
    "data/split/phuma_train.txt", "data/split/phuma_test.txt",
    "src/curation/preprocess_g1_folder.py", "src/utils/visualize.py",
    MODEL + "/config.yaml", MODEL + "/custom.xml", MODEL + "/scene.xml",
)


def file_id(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "YUANQI-Q1-PHUMA"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def acquire(output):
    output = output.resolve()
    receipt = output / "source_receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["commit"] != COMMIT or any(file_id(x["path"]) != x for x in saved["files"]):
            raise ValueError("Existing source acquisition has changed")
        return saved
    output.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(fetch(tree_url))
    if tree.get("truncated"):
        raise ValueError("Incomplete upstream Git tree")
    entries = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"}
    records = []

    def download(name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name not in entries:
            raise ValueError("Unexpected upstream file")
        entry = entries[name]
        if entry.get("size", 0) > 16 * 1024 ** 2:
            raise ValueError("Unexpected source asset size")
        path = output / name
        data = path.read_bytes() if path.exists() else fetch(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{name}")
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != entry["size"] or blob != entry["sha"]:
            raise ValueError(f"Upstream blob mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
        records.append(file_id(path))
        print(json.dumps({"source_acquired": name, "bytes": len(data)}), flush=True)
        return path

    for name in METADATA:
        download(name)
    if "Apache License" not in (output / "LICENSE").read_text():
        raise ValueError("Review changed source license before proceeding")
    xml = ET.parse(output / MODEL / "custom.xml")
    compiler = xml.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    files = sorted({str(PurePosixPath(MODEL) / meshdir / mesh.attrib["file"])
                    for mesh in xml.findall("asset/mesh")})
    for name in files:
        download(name)
    saved = {"status": "PINNED_PHUMA_G1_SOURCE_ACQUIRED", "repository": REPOSITORY,
             "commit": COMMIT, "tree_identity": tree["sha"], "files": records,
             "code_executed_from_upstream": False, "new_physics_steps": 0,
             "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    with receipt.open("x") as stream:
        json.dump(saved, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = acquire(args.output)
    print(json.dumps({"status": saved["status"], "files": len(saved["files"]),
                      "receipt": file_id(args.output / "source_receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
