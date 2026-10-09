"""Durable SONIC checkpoint transactions; callers must supply trusted payloads."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_payload(payload):
    required = ["policy_state_dict", "value_state_dict", "optimizer_state_dict", "state"]
    if any(payload.get(key) is None for key in required):
        raise ValueError("Incomplete actor/critic/optimizer/trainer checkpoint")
    for name in required[:2]:
        for tensor in payload[name].values():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Nonfinite network tensor: {name}")
    for state in payload["optimizer_state_dict"]["state"].values():
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Nonfinite optimizer state")
    return int(payload["state"].global_step)


def atomic_json(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(directory, payload, milestone_interval=500):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    step = validate_payload(payload)
    destination = directory / f"model_step_{step:06d}.pt"
    if destination.exists():
        raise FileExistsError(f"Checkpoint step already exists: {destination}")
    previous = None
    if any(directory.glob("model_step_*.json")):
        try:
            previous, old_payload, _ = select_checkpoint(directory)
            del old_payload
        except RuntimeError:
            pass
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".incomplete", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checked = torch.load(temporary, map_location="cpu", weights_only=False)
        if validate_payload(checked) != step:
            raise ValueError("Checkpoint reload changed the step")
        del checked
        metadata = {
            "writer": "yuanqi_durable_v1", "step": step, "file": destination.name,
            "bytes": temporary.stat().st_size, "sha256": sha256(temporary),
            "backup_status": "LOCAL_ONLY",
        }
        os.replace(temporary, destination)
        fsync_directory(directory)
        atomic_json(destination.with_suffix(".json"), metadata)
        link = directory / ".last.pt.next"
        link.unlink(missing_ok=True)
        link.symlink_to(destination.name)
        os.replace(link, directory / "last.pt")
        fsync_directory(directory)
        # Only prune files published by this writer, after the new generation is durable.
        kept = {destination.name}
        if previous is not None:
            kept.add(previous.name)
        for sidecar in directory.glob("model_step_*.json"):
            try:
                record = json.loads(sidecar.read_text())
                old_step = int(record["step"])
            except (ValueError, KeyError):
                continue
            name = f"model_step_{old_step:06d}.pt"
            if record.get("writer") != "yuanqi_durable_v1" or record.get("file") != name:
                continue
            if name in kept:
                continue
            if milestone_interval > 0 and old_step % milestone_interval == 0:
                continue
            (directory / name).unlink(missing_ok=True)
            sidecar.unlink()
        fsync_directory(directory)
        return metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_checkpoint(directory):
    directory = Path(directory)
    candidates = sorted(directory.glob("model_step_*.json"), reverse=True)
    failures = []
    for sidecar in candidates:
        try:
            record = json.loads(sidecar.read_text())
            if record.get("writer") != "yuanqi_durable_v1":
                continue
            name = f"model_step_{int(record['step']):06d}.pt"
            if record["file"] != name or sidecar.stem != Path(name).stem:
                raise ValueError("Sidecar filename mismatch")
            path = directory / name
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError("Checkpoint size/hash mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if validate_payload(payload) != record["step"]:
                raise ValueError("Checkpoint counter mismatch")
            return path, payload, failures
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
            failures.append({"sidecar": str(sidecar), "error": str(error)})
    raise RuntimeError(f"No verified checkpoint available; rejected {failures}")
