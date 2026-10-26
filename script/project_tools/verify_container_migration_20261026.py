#!/usr/bin/env python3
"""Verify CM01 copied bytes while the independent source checksum stream grows."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def save(path, row):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(path)


def inventory(path):
    with gzip.open(path, "rb") as stream:
        records = stream.read().split(b"\0")
    expected = {}
    for row in records:
        if row.startswith(b"F\t"):
            name, mode, uid, gid, size = row[2:].rsplit(b"\t", 4)
            expected[os.fsdecode(name)] = int(size)
    return expected


def checksum_row(line):
    escaped = line.startswith(b"\\")
    line = line[1:] if escaped else line
    if len(line) < 67 or line[64:66] not in (b"  ", b" *"):
        raise ValueError("Invalid GNU checksum record")
    digest = line[:64].decode("ascii")
    if any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid SHA-256")
    name = line[66:]
    if escaped:
        output = bytearray()
        position = 0
        while position < len(name):
            char = name[position]
            if char == 92:
                position += 1
                char = {92: 92, 110: 10, 114: 13}[name[position]]
            output.append(char)
            position += 1
        name = bytes(output)
    return os.fsdecode(name), digest


def identity(st):
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def check_file(path, size, digest):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError("Destination size/type not ready")
        result = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
        if result is None:
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                hasher.update(block)
            result = hasher.hexdigest()
        after = os.fstat(stream.fileno())
    if identity(before) != identity(after) or result != digest:
        raise ValueError("Destination changed or checksum does not match yet")
    return identity(after)


def verify(args):
    expected = inventory(args.output / "source_tree.find.gz")
    status = args.output / "stream_verification.json"
    started = time.time()
    seen = set()
    total = 0
    row = {"status": "VERIFYING", "started_unix": started, "files": 0, "bytes": 0,
           "expected_files": len(expected), "expected_bytes": sum(expected.values())}
    save(status, row)
    last_save = started
    with (args.output / "source.sha256").open("rb") as source, gzip.open(
            args.output / "verified_files.jsonl.gz", "xt") as journal:
        pending = b""
        while True:
            if time.time() - started > 10800:
                raise TimeoutError("Migration verification exceeded three hours")
            block = source.readline()
            if not block:
                if (args.output / "source_hash.completed.json").exists():
                    if pending or seen != set(expected):
                        raise ValueError("Completed checksum stream does not cover the source inventory")
                    break
                time.sleep(1)
                continue
            pending += block
            if not pending.endswith(b"\n"):
                continue
            name, digest = checksum_row(pending[:-1])
            pending = b""
            path = Path(name)
            relative = str(path.relative_to(args.root))
            if ".." in path.parts or relative not in expected or relative in seen:
                raise ValueError("Unexpected, duplicate, or unsafe checksum path")
            waiting = time.monotonic()
            while True:
                try:
                    signature = check_file(path, expected[relative], digest)
                    break
                except (FileNotFoundError, ValueError):
                    if (args.output / "copy.completed.json").exists() or time.monotonic() - waiting > 1800:
                        raise
                    time.sleep(2)
            journal.write(json.dumps({"path": name, "sha256": digest, "identity": signature}) + "\n")
            seen.add(relative)
            total += expected[relative]
            if time.time() - last_save >= 20:
                row.update(files=len(seen), bytes=total, observed_unix=time.time())
                save(status, row)
                journal.flush()
                last_save = time.time()
    row.update(status="BYTES_VERIFIED_PENDING_FINAL_METADATA_CHECK", files=len(seen), bytes=total,
               ended_unix=time.time(), elapsed_seconds=time.time() - started)
    save(status, row)
    print(json.dumps(row), flush=True)


def finalize(args):
    if not (args.output / "copy.completed.json").exists():
        raise ValueError("Full copy has not completed")
    stream = json.loads((args.output / "stream_verification.json").read_text())
    if stream["status"] != "BYTES_VERIFIED_PENDING_FINAL_METADATA_CHECK":
        raise ValueError("The independent byte check has not completed")
    count = 0
    rehashed = 0
    with gzip.open(args.output / "verified_files.jsonl.gz", "rt") as journal:
        for line in journal:
            row = json.loads(line)
            path = Path(row["path"])
            if identity(path.lstat()) != row["identity"]:
                check_file(path, row["identity"][1], row["sha256"])
                rehashed += 1
            count += 1
    if count != stream["expected_files"]:
        raise ValueError("Final checked-file count differs")
    result = {"status": "ALL_SOURCE_FILE_BYTES_VERIFIED", "files": count,
              "bytes": stream["bytes"], "metadata_changed_files_rehashed": rehashed,
              "ended_unix": time.time(), "training_runs_started": 0}
    save(args.output / "final_byte_verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stream", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (verify if args.mode == "stream" else finalize)(args)


if __name__ == "__main__":
    main()
