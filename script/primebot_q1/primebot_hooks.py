"""Passive optimizer counters and a verified zero-update checkpoint."""

import hashlib

from common import TARGET_TRANSITIONS, identity, require, save


def signature(value):
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def install_counts(algorithm):
    algorithm._q1_actual_optimizer_counts = {}
    algorithm._q1_count_handles = []
    for name in ("model_optimizer", "actor_optimizer", "critic_optimizer"):
        algorithm._q1_actual_optimizer_counts[name] = 0

        def step(optimizer, args, kwargs, key=name):
            algorithm._q1_actual_optimizer_counts[key] += 1

        algorithm._q1_count_handles.append(getattr(algorithm, name).register_step_post_hook(step))


def save_zero(checkpoint, modules, cfg, capture_rng):
    rng_before = signature(capture_rng(cfg.device))
    parameters_before = signature([m.state_dict() for m in modules])
    path = checkpoint()
    rng_after = signature(capture_rng(cfg.device))
    parameters_after = signature([m.state_dict() for m in modules])
    save(cfg.output_dir / "zero_snapshot_audit.json", dict(checkpoint=identity(path),
         optimizer_updates=0, rng_before=rng_before, rng_after=rng_after,
         parameters_before=parameters_before, parameters_after=parameters_after))
    require(rng_before == rng_after and parameters_before == parameters_after,
            "Zero-step checkpoint changed RNG or model state")


def exposure_snapshot(checkpoint, global_steps, cfg):
    marker = cfg.output_dir / "exposure_snapshot.json"
    if global_steps >= TARGET_TRANSITIONS and not marker.exists():
        from supermimic.checkpoint import capture_rng
        before = signature(capture_rng(cfg.device))
        path = checkpoint()
        after = signature(capture_rng(cfg.device))
        require(before == after, "Exposure snapshot changed RNG")
        require(abs(global_steps - TARGET_TRANSITIONS) <= TARGET_TRANSITIONS * .01,
                "Exposure checkpoint skipped the one-percent window")
        save(marker, dict(checkpoint=identity(path), global_env_steps=global_steps,
                          target=TARGET_TRANSITIONS, rng_before=before, rng_after=after))
