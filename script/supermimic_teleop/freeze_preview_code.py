"""Export published evaluation Python identities from the authoritative Mac Git.

Run after push and independent remote verification. The server has no .git; it
checks this file's explicit SHA and all listed source bytes before every phase.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from preview_review import CODE_ROOTS, REQUIRED_CODE


def published_manifest(repo, commit):
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Require the independently verified full published commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != commit:
        raise ValueError("Authoritative worktree HEAD differs from the published commit")
    raw = subprocess.check_output(["git", "ls-tree", "-rz", "--name-only", commit, "--", *CODE_ROOTS], cwd=repo)
    names = [row.decode("utf-8") for row in raw.split(b"\0") if row and row.endswith(b".py")]
    if not set(REQUIRED_CODE) <= set(names) or len(names) > 256:
        raise ValueError("Missing published preview modules or oversized source manifest")
    rows = []
    for relative in names:
        path = repo / relative
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Published source must be a real file without symlink parents")
        blob = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=repo)
        if path.read_bytes() != blob:
            raise ValueError("Local source differs from published Git: " + relative)
        rows.append(dict(relative=relative, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    return dict(stage="PUBLISHED_PREVIEW_EVALUATION_CODE", evaluation_commit=commit, files=rows,
                total_bytes=sum(row["bytes"] for row in rows), original_training_inputs_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output
    if not output.is_absolute() or output.suffix != ".json" or output.exists():
        parser.error("Use a new absolute JSON manifest path")
    manifest = published_manifest(repo, args.evaluation_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps(dict(path=str(output), bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest(),
                          evaluation_commit=args.evaluation_commit, files=len(manifest["files"]))))


if __name__ == "__main__":
    main()
