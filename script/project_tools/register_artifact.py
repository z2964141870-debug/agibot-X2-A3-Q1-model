#!/usr/bin/env python3
"""Hash one data artifact and register LOCAL_ONLY metadata; never uploads."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=["checkpoint", "export", "dataset", "recording", "evaluation", "reference"], required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.experiment_id):
        parser.error("experiment-id must use ASCII letters, digits, dots, underscores or hyphens")
    root = Path(__file__).resolve().parents[2]
    path = (root / args.path).resolve() if not args.path.is_absolute() else args.path.resolve()
    data_root = (root / "data").resolve()
    if not path.is_file() or data_root not in path.parents:
        parser.error("artifact must be an existing regular file within this project's data directory")
    relative = path.relative_to(root).as_posix()
    if relative.startswith("data/manifests/"):
        parser.error("register the data artifact, not a manifest")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        parser.error("artifact changed while hashing; retry when the writer has finished")
    sha = digest.hexdigest()
    path_id = hashlib.sha256(relative.encode()).hexdigest()[:8]
    manifest = data_root / "manifests" / f"artifact_{args.experiment_id}_{sha[:16]}_{path_id}.json"
    record = {
        "schema_version": 1,
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "kind": args.kind,
        "relative_path": relative,
        "bytes": after.st_size,
        "sha256": sha,
        "source": args.source,
        "backup": {
            "provider": "baidu_netdisk", "status": "LOCAL_ONLY",
            "proposed_remote_path": f"/YUANQI/{args.kind}/{args.experiment_id}/{path.name}",
            "actual_remote_path": None, "uploaded_at_utc": None,
            "remote_bytes": None, "downloaded_sha256": None,
            "verified_at_utc": None, "evidence_path": None,
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("sha256") != sha or existing.get("relative_path") != relative or existing.get("bytes") != after.st_size:
            parser.error("existing manifest conflicts; refusing to overwrite")
        print(f"Existing manifest preserved: {manifest.relative_to(root)}")
        return
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Registered LOCAL_ONLY: {manifest.relative_to(root)}")
    print(f"bytes={after.st_size} sha256={sha}")


if __name__ == "__main__":
    main()
