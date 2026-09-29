#!/usr/bin/env python3
"""Check staged Git blobs before publishing compact YUANQI records."""

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT_FILES = {"README.md", "AGENTS.md", ".gitignore"}
MODEL_SUFFIXES = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".pkl", ".pickle", ".npy", ".npz", ".bag", ".mcap", ".mp4", ".mov", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar")
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"(?i)(?:BDUSS|STOKEN)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def allowed_path(name):
    path = PurePosixPath(name)
    if name in ROOT_FILES:
        return True
    if path.parts[0] in {"script", "reports"}:
        return True
    if name in {"data/README.md", "data/manifests/README.md", "logs/README.md"}:
        return True
    if len(path.parts) == 3 and path.parts[:2] == ("data", "manifests"):
        return path.suffix in {".json", ".jsonl", ".csv", ".tsv"}
    if len(path.parts) == 3 and path.parts[:2] == ("logs", "session_records"):
        return path.suffix in {".log", ".txt", ".json", ".md"}
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-mib", type=float, default=10.0)
    args = parser.parse_args()
    if args.max_mib <= 0:
        parser.error("--max-mib must be positive")
    root = Path(__file__).resolve().parents[2]
    try:
        raw = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    except subprocess.CalledProcessError:
        return 2
    names = [item.decode("utf-8", errors="surrogateescape") for item in raw.split(b"\0") if item]
    errors = []
    total = 0
    for name in names:
        size = int(git(root, "cat-file", "-s", ":" + name))
        total += size
        if not allowed_path(name):
            errors.append((name, "outside approved record/code locations"))
        if name.lower().endswith(MODEL_SUFFIXES):
            errors.append((name, "model/data/archive payload belongs in data and Baidu Netdisk"))
        if size > args.max_mib * 1024 * 1024:
            errors.append((name, "exceeds staged single-file size limit"))
            continue
        basename = PurePosixPath(name).name.lower()
        if basename == ".env" or basename.startswith((".env.", "id_rsa", "id_ed25519")) or any(word in basename for word in ("credentials", "cookies")):
            errors.append((name, "credential-like filename"))
        content = git(root, "show", ":" + name)
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append((name, "possible secret; inspect locally, do not print its value"))
    for name, reason in errors:
        print(f"FAIL {name}: {reason}", file=sys.stderr)
    print(f"{'FAIL' if errors else 'PASS'}: {len(names)} staged files, {total} bytes; size limit {args.max_mib:g} MiB/file")
    print("Pattern checks are limited; manually review logs and diff before publishing.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
