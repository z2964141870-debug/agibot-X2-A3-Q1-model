"""Partition only owned inference tasks; never change system affinity or clocks."""
import hashlib
import json
import os
from pathlib import Path


PLAN = {"native": [0, 1], "gpu": [2, 3], "sender": [2, 3],
        "cpu": [4, 5], "policy": [6, 7]}
EXECUTION = ("__init__.py", "allocation.py", "worker.py", "run.py")


def validate_available(available):
    if not set(range(8)).issubset(available):
        raise ValueError("Allocation requires CPUs 0..7 in the inherited allowed set")


def bind(role):
    mask = set(PLAN[role])
    os.sched_setaffinity(0, mask)
    actual = set(os.sched_getaffinity(0))
    if actual != mask:
        raise RuntimeError("Owned task CPU allocation not applied")
    return sorted(actual)


def sources():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in EXECUTION}


def validate_prefix(directory):
    receipt = json.loads((directory/"allocation.json").read_text())
    worker = json.loads((directory/"affinity.json").read_text())
    if (receipt["variant"] != "bounded_affinity_r2" or receipt["mode"] != "prefix"
            or receipt["plan"] != PLAN or receipt["source_sha256"] != sources()
            or worker["source_sha256"] != sources() or worker["status"] != "WORKER_CLEAN_EXIT"
            or worker["plan"] != PLAN or worker["hardware_commands"] != 0):
        raise ValueError("Require this allocation's completed, unchanged prefix")
    roles = worker["roles"]
    if set(roles) != {"gpu", "cpu", "sender"}:
        raise ValueError("Missing execution-role affinity receipt")
    for role, row in roles.items():
        if row["cpus"] != PLAN[role]:
            raise ValueError("Role did not execute on the registered CPUs")
    if len({row["tid"] for row in roles.values()}) != 3:
        raise ValueError("Pipeline roles did not use separate owner threads")
