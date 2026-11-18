"""Pinned public assets. Explicit bootstrap only; inference never downloads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]
UPSTREAM_COMMIT = "770394b4a0cad9effbf807a6c03435d1699733d0"
RELEASE_URL = "https://github.com/lzyang2000/morphmimic/releases/download/v0.1/morphmimic_policy.zip"
RELEASE_BYTES = 63372873
RELEASE_SHA = "be874424beeda23cbccda08ed65b3fa146e926d2f338b8f898d28baa21d1b99f"
MODEL_SHA = "cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512"


def home():
    return Path(os.environ.get("X2_MORPH_HOME", PROJECT / "data/runtime/x2_morph_human")).resolve()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def manifest():
    return json.loads((PACKAGE / "assets_manifest.json").read_text())


def check(path, row):
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        raise ValueError(f"asset missing or different: {path}; run setup/bootstrap")


def verify(root=None, meshes=False):
    root = Path(root or home())
    rows = manifest()["files"]
    selected = [r for r in rows if meshes or not r["path"].startswith("assets/x2_source/")]
    for row in selected:
        check(root/row["path"], row)
    return {"files": len(selected), "bytes": sum(r["bytes"] for r in selected),
            "source_commit": UPSTREAM_COMMIT, "policy_sha256": MODEL_SHA}


def _download(url, target, row):
    if target.exists():
        check(target, row)  # Do not silently overwrite different source/weights.
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name+".download")
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "xb") as f:
        while True:
            b = response.read(1024*1024)
            if not b:
                break
            total += len(b)
            if total > row["bytes"]:
                raise ValueError("download exceeded pinned size")
            f.write(b)
    check(tmp, row)
    tmp.replace(target)


def bootstrap(root=None, meshes=False):
    root = Path(root or home())
    rows = manifest()["files"]
    selected = [r for r in rows if meshes or not r["path"].startswith("assets/x2_source/")]
    release = [r for r in selected if r.get("zip_member")]
    if any(not (root/r["path"]).exists() for r in release):
        archive = root/"assets/morphmimic_policy.zip"
        _download(RELEASE_URL, archive, dict(bytes=RELEASE_BYTES, sha256=RELEASE_SHA))
        with zipfile.ZipFile(archive) as z:
            for row in release:
                p = root/row["path"]
                if p.exists():
                    check(p, row)
                else:
                    data = z.read(row["zip_member"])
                    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                        raise ValueError("release member mismatch")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "xb") as f:
                        f.write(data)
    for row in selected:
        if row.get("url"):
            _download(row["url"], root/row["path"], row)
    return verify(root, meshes)


def activate(root=None):
    root = Path(root or home())
    verify(root)
    source = (root/"upstream").resolve()
    if "deploy" in sys.modules:
        paths = [Path(p).resolve() for p in sys.modules["deploy"].__path__]
        if not any(source in p.parents for p in paths):
            raise RuntimeError("another deploy package is loaded; use an isolated process")
    sys.path.insert(0, str(source))
    return root


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--meshes", action="store_true")
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    print(json.dumps(verify(meshes=a.meshes) if a.offline else bootstrap(meshes=a.meshes), indent=2))
