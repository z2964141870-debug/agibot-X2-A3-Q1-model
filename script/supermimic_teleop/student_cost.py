"""Durable single-GPU reservations; an unacknowledged run keeps its full charge."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time


LIMIT = 14400
STAGES = {"r1": 2700, "r2": 4500, "r3": 7200}


def checked(value):
    if (value.get("schema") != "q1_low_latency_gpu_cost_v1"
            or value.get("limit_seconds") != LIMIT or not isinstance(value.get("runs"), list)
            or len(value["runs"]) > 64):
        raise ValueError("Invalid cumulative training cost ledger")
    names = set()
    for row in value["runs"]:
        if (row.get("run_id") in names or row.get("stage") not in STAGES
                or type(row.get("reserved_seconds")) is not int
                or not 0 < row["reserved_seconds"] <= STAGES[row["stage"]]
                or row.get("state") not in ("RESERVED", "SETTLED")
                or row.get("gpu_count") != 1):
            raise ValueError("Invalid or duplicate cost reservation")
        names.add(row["run_id"])
        if row["state"] == "SETTLED" and (type(row.get("charged_seconds")) is not int
                or row["charged_seconds"] < 0):
            raise ValueError("Missing nonnegative settled cost")
    return value


def charged(value):
    checked(value)
    return sum(row["reserved_seconds"] if row["state"] == "RESERVED"
               else row["charged_seconds"] for row in value["runs"])


@contextmanager
def locked(path):
    path = Path(path)
    if path.is_symlink() or not path.parent.is_dir():
        raise ValueError("Cost ledger needs a real existing parent")
    lock = path.with_suffix(path.suffix+".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            if path.stat().st_size > 65536:
                raise ValueError("Oversized training ledger")
            value = checked(json.loads(path.read_text()))
        else:
            value = dict(schema="q1_low_latency_gpu_cost_v1", limit_seconds=LIMIT, runs=[])
        yield value
        checked(value)
        temporary = path.with_suffix(path.suffix+".tmp")
        raw = (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()
        descriptor_out = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor_out, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)


def reserve(path, *, run_id, stage, owner, gpu, commit, started_ns=None):
    if (stage not in STAGES or not isinstance(run_id, str) or not Path(run_id).is_absolute()
            or not isinstance(owner, dict) or type(owner.get("pid")) is not int
            or owner["pid"] <= 1 or not str(owner.get("start_ticks", "")).isdigit()
            or not owner.get("boot") or not isinstance(gpu, str) or not gpu.startswith("GPU-")
            or not isinstance(commit, str) or len(commit) != 40
            or any(x not in "0123456789abcdef" for x in commit)):
        raise ValueError("Missing admitted run, owner, boot, GPU or published source identity")
    started_ns = time.monotonic_ns() if started_ns is None else started_ns
    if type(started_ns) is not int or started_ns < 0:
        raise ValueError("Invalid monotonic reservation timestamp")
    with locked(path) as value:
        if any(row["run_id"] == run_id for row in value["runs"]):
            raise ValueError("Single-use training run already has a reservation")
        if any(row["state"] == "RESERVED" for row in value["runs"]):
            raise RuntimeError("An earlier run remains reserved; resolve its receipt before dispatch")
        stage_charged = charged(dict(value, runs=[row for row in value["runs"] if row["stage"] == stage]))
        available = STAGES[stage]-stage_charged
        if charged(value)+max(1, available) > LIMIT:
            raise RuntimeError("Cumulative four-GPU-hour budget has insufficient unreserved capacity")
        if available <= 0:
            raise RuntimeError("Stage budget has insufficient unreserved capacity; do not reset its ledger")
        row = dict(run_id=run_id, stage=stage, state="RESERVED", gpu_count=1,
            reserved_seconds=available, owner=owner, gpu=gpu, source_commit=commit,
            started_monotonic_ns=started_ns, cost_scope="GPU-owning collection and training wall time")
        value["runs"].append(row)
    return row


def settle(path, *, run_id, owner, cleanup, ended_ns=None):
    if (cleanup.get("worker_reaped") is not True
            or not (type(cleanup.get("worker_return_code")) is int
                    or cleanup.get("worker_started") is False)):
        raise ValueError("Keep full reservation until the owned worker is actually reaped")
    ended_ns = time.monotonic_ns() if ended_ns is None else ended_ns
    with locked(path) as value:
        rows = [row for row in value["runs"] if row["run_id"] == run_id]
        if len(rows) != 1 or rows[0]["state"] != "RESERVED" or rows[0]["owner"] != owner:
            raise ValueError("Reservation identity or terminal state changed")
        row = rows[0]
        if type(ended_ns) is not int or ended_ns < row["started_monotonic_ns"]:
            raise ValueError("Cannot settle a different monotonic clock")
        seconds = math.ceil((ended_ns-row["started_monotonic_ns"])/1e9)
        row.update(state="SETTLED", ended_monotonic_ns=ended_ns, charged_seconds=seconds,
            over_reserved_budget=seconds > row["reserved_seconds"], cleanup=cleanup)
    return row
