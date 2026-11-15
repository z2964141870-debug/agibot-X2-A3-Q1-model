import hashlib
import json
from pathlib import Path


PLANS = {
    "r2_profile": {"native": [0, 1], "gpu": [2, 3], "sender": [2, 3],
                   "cpu": [4, 5], "policy": [6, 7]},
    "dedicated_hosts": {"native": [1], "gpu": [2], "sender": [3],
                        "cpu": [4, 5], "policy": [6, 7]},
}
EXECUTION = ("__init__.py", "config.py", "profile.py", "worker.py", "run.py")


def settings(value):
    if set(value) != {"variant", "release"} or value["variant"] not in PLANS:
        raise ValueError("Unregistered allocation")
    release = value["release"]
    if len(release) != 40 or any(c not in "0123456789abcdef" for c in release):
        raise ValueError("Require a published full commit")
    return value


def sources():
    root = Path(__file__).parent
    return {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in EXECUTION}


def validate_prefix(directory, config):
    result = json.loads((directory/"opt_config.json").read_text())
    if (result["config"] != config or result["mode"] != "prefix"
            or result["source_sha256"] != sources() or not result["source_unchanged"]):
        raise ValueError("Require this exact candidate's new prefix")
    roles = [json.loads(x) for x in (directory/"kernel_binding_fgp.jsonl").read_text().splitlines()]
    if {r["role"] for r in roles} != {"gpu", "cpu", "sender"}:
        raise ValueError("Missing role binding")
    if len({r["tid"] for r in roles}) != 3:
        raise ValueError("Role owners must differ")
    for row in roles:
        if row["cpus"] != PLANS[config["variant"]][row["role"]]:
            raise ValueError("Wrong kernel CPU mask")
