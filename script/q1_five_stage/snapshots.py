"""Passive save metadata and zero-update parameter/RNG checks for fresh runs."""

import hashlib
import json
from pathlib import Path
import pickle
import random


MODULES = ("wm", "policy", "critic", "critic_kp", "obs_norm", "goal_norm",
           "human_norm", "heading_norm", "target_critic", "target_critic_kp")


def rng_sha():
    import numpy as np
    import torch
    digest = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=4))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for value in torch.cuda.get_rng_state_all():
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def model_sha(trainer):
    import torch
    digest = hashlib.sha256()
    for name in MODULES:
        module = getattr(trainer, name, None)
        if module is None:
            continue
        digest.update(name.encode("ascii"))
        for key, value in sorted(module.state_dict().items()):
            if isinstance(value, torch.Tensor):
                digest.update(key.encode("ascii"))
                digest.update(str((value.dtype, tuple(value.shape))).encode("ascii"))
                raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def install(trainer, plan, record):
    original_checkpoint, original_save = trainer.checkpoint, trainer.save_checkpoint

    def checkpoint(*args, **kwargs):
        value = original_checkpoint(*args, **kwargs)
        value["actual_training"] = dict(
            iteration=int(trainer.it),
            collected_environment_transitions=record["collected_environment_transitions"],
            optimizer_updates=dict(record["optimizer_updates"]),
            code_commit=plan["code_commit"], fresh_initialization=True,
            profile_weights_reused=False)
        return value

    def save(*args, **kwargs):
        if trainer.it != 0:
            return original_save(*args, **kwargs)
        if any(record["optimizer_updates"].values()) or record["collected_environment_transitions"]:
            raise RuntimeError("Initial snapshot must precede all collection and optimization")
        before = dict(parameters=model_sha(trainer), rng=rng_sha())
        path = original_save(*args, **kwargs)
        after = dict(parameters=model_sha(trainer), rng=rng_sha())
        audit = dict(status="PASSED" if before == after else "FAILED", iteration=0,
            before=before, after=after, parameters_unchanged=before["parameters"] == after["parameters"],
            rng_unchanged=before["rng"] == after["rng"],
            checkpoint=str(path), new_optimizer_updates=0, new_training_transitions=0)
        save_json(Path(plan["output"]) / ("zero_snapshot_" + record["robot"] + ".json"), audit)
        record["zero_snapshot_audit"] = audit
        if before != after:
            raise RuntimeError("Passive initial save changed parameters or RNG")
        return path

    trainer.checkpoint, trainer.save_checkpoint = checkpoint, save
