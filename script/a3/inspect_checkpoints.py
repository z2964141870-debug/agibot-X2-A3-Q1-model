"""Compare trusted local SONIC checkpoints and register their hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # SONIC checkpoints contain trainer objects; only load our own trusted runs.
    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    result = {
        "steps": [int(first["state"].global_step), int(second["state"].global_step)],
        "backup_status": "LOCAL_ONLY",
        "artifacts": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [args.first, args.second]
        ],
        "networks": {},
    }
    for name in ["policy_state_dict", "value_state_dict"]:
        changes = {}
        finite = True
        for key, tensor in first[name].items():
            if not tensor.is_floating_point():
                continue
            updated = second[name][key]
            finite = finite and bool(torch.isfinite(updated).all())
            if key.endswith((".weight", ".bias")) or key == "std":
                changes[key] = float((tensor - updated).abs().max())
        result["networks"][name] = {
            "weight_or_bias_tensors": len(changes),
            "changed_weight_or_bias_tensors": sum(value > 0 for value in changes.values()),
            "max_weight_or_bias_change": max(changes.values(), default=0),
            "all_floating_tensors_finite": finite,
        }
        if not finite or not any(value > 0 for value in changes.values()):
            raise RuntimeError(f"Parameter update verification failed: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
