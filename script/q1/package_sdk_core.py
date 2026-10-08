#!/usr/bin/env python3
"""Package supplied Q1 SDK interfaces/examples without bulk dependencies."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from inspect_resources import digest


SDK_ROOT = "sdk_q1-v1.0.0.0/"
EXCLUDED = tuple(SDK_ROOT + "examples/" + name + "/"
                 for name in ("opencv", "ruckig_for_primebot"))


def add_file(archive, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.source.resolve(), args.manifest.resolve()):
        parser.error("Output, source and manifest must be distinct files")
    if args.source.resolve() == args.manifest.resolve():
        parser.error("Manifest must not overwrite source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(args.source) as source:
        entries = [entry for entry in source.infolist()
                   if not entry.is_dir() and entry.filename.startswith(SDK_ROOT)
                   and not entry.filename.startswith(EXCLUDED)]
        if not entries:
            parser.error("No expected SDK core files found")
        # Validate all selected paths before writing a transferable archive.
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\n" in entry.filename:
                parser.error("Unsafe archive member")
            if entry.file_size > 2 * 1024 * 1024:
                parser.error("Unexpected large SDK core member")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                parser.error("Symbolic links are not SDK core files")
        with tarfile.open(args.output, "w:gz") as target:
            for entry in sorted(entries, key=lambda item: item.filename):
                content = source.read(entry)
                checksum = hashlib.sha256(content).hexdigest()
                add_file(target, entry.filename, content)
                files.append({"path": entry.filename, "bytes": len(content),
                              "sha256": checksum})
            sums = "".join(item["sha256"] + "  " + item["path"] + "\n"
                           for item in files)
            add_file(target, "SHA256SUMS", sums.encode("utf-8"))
    payload = {
        "scope": "SDK_CORE_ONLY_EXCLUDES_OPENCV_RUCKIG",
        "source_filename": args.source.name,
        "source_sha256": digest(args.source),
        "server_directory": "data/references/q1/sdk_core",
        "bundle_filename": args.output.name,
        "bundle_bytes": args.output.stat().st_size,
        "bundle_sha256": digest(args.output),
        "transfer_status": "PACKAGED_LOCAL",
        "baidu_backup_status": "LOCAL_ONLY",
        "files": files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"SDK core: {len(files)} files, {sum(f['bytes'] for f in files)} bytes")
    print(f"Bundle: {payload['bundle_bytes']} bytes, sha256={payload['bundle_sha256']}")


if __name__ == "__main__":
    main()
