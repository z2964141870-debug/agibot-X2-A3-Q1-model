#!/usr/bin/env python3
"""Flush this run's completed checkpoints and advise release of their page cache."""

import argparse
import json
import os
from pathlib import Path


def memory_usage():
    return int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve(strict=True)
    if not (run / "admission.json").is_file() or args.run.is_symlink():
        raise ValueError("An admitted run directory is required")
    before = memory_usage()
    rows = []
    for path in sorted(run.glob("*.pt")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Only regular checkpoint files are allowed: {path}")
        with path.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            os.fsync(stream.fileno())
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        rows.append({"path": str(path), "bytes": metadata.st_size, "inode": metadata.st_ino})
    print(json.dumps({"action": "SYNC_AND_ADVISE_OWN_CHECKPOINT_CACHE", "files": rows,
                      "before_cgroup_usage": before, "after_cgroup_usage": memory_usage(),
                      "files_deleted": 0, "advisory_only": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
