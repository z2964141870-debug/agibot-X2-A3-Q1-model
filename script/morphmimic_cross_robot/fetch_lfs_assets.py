"""Resolve only pinned SOMA assets needed by the X2/T2 test."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "cbdb2b494a2af82137bf9b6f15122d60167281c4"
BASE = "https://media.githubusercontent.com/media/NVIDIA/soma-retargeter/" + COMMIT + "/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    assets = root / "soma_retargeter/assets"
    candidates = list((assets / "robotics/agibot/x2").rglob("*"))
    candidates += [assets / "soma/soma_zero_frame0.bvh"]
    rows = []
    for path in sorted(set(candidates)):
        if not path.is_file() or path.stat().st_size > 200:
            continue
        pointer = path.read_text()
        if not pointer.startswith("version https://git-lfs.github.com/spec/v1\n"):
            continue
        fields = dict(line.split(" ", 1) for line in pointer.splitlines())
        sha = fields["oid"].removeprefix("sha256:")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid LFS identity")
        rows.append(dict(path=str(path.relative_to(root)), bytes=int(fields["size"]),
                         sha256=sha, pointer=pointer, status="PENDING"))
    if args.output.exists():
        raise FileExistsError("Retain each download attempt in its own receipt")
    result = dict(commit=COMMIT, storage_status="SERVER_LOCAL_ONLY", assets=rows)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    for row in rows:
        path = root / row["path"]
        temporary = path.with_name(path.name + ".lfs_download")
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(BASE + row["path"], timeout=60) as response:
                with temporary.open("xb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(block)
                        if size > row["bytes"]:
                            raise ValueError("LFS response exceeds pointer size")
                        digest.update(block)
                        stream.write(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("LFS bytes/hash mismatch")
            temporary.replace(path)
            row["status"] = "VERIFIED"
            print("VERIFIED " + row["path"], flush=True)
        except BaseException as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()
    result["status"] = "COMPLETED"
    save()


if __name__ == "__main__":
    main()
