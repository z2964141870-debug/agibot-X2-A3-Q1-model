#!/usr/bin/env python3
"""Observe an unchanged upstream Trainer; outer runtime_guard bounds its process."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import signal
import time


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.train.trainer import TrainCfg, Trainer

    config = json.loads(args.config.read_text())
    if args.mode == "probe":
        config.update(num_envs=16, iterations=0,
                      run_name=config["run_name"] + "__probe")
    cfg = TrainCfg(**config)
    if cfg.robot != "g1" or cfg.robots is not None or cfg.wandb:
        raise ValueError("This reproduction entry is single G1 with local logging")
    if (Path(cfg.log_dir) / cfg.run_name).exists():
        raise FileExistsError("Original run output already exists; no overwrite or resume")
    record = {
        "status": "INITIALIZING",
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "optimizer_steps": {"wm": 0, "policy": 0, "critic": 0},
        "iteration_seconds": 0.0,
        "timing_definition": "Sum of synchronized original iteration calls; excludes initialization, warmup, logging and checkpoint writes. Outer guard limits total process wall time.",
    }
    progress = args.output / "progress.json"
    save_json(progress, record)
    trainer = None
    hooks = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        configure_torch_backends()
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        trainer = Trainer(cfg, "cuda:0")
        optimizers = {"wm": trainer.wm_opt, "policy": trainer.policy_opt,
                      "critic": trainer.critic_opt}

        def counted(name):
            def hook(optimizer, positional, keyword):
                record["optimizer_steps"][name] += 1
            return hook

        for name, optimizer in optimizers.items():
            hooks.append(optimizer.register_step_post_hook(counted(name)))

        record.update(status="RUNNING", run_dir=str(trainer.run_dir))
        if args.mode == "probe":
            trainer.collector.reset_all()
            stats = trainer.collector.step(40)
            torch.cuda.synchronize()
            for value in stats.values():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all()
            assert all(not opt.state for opt in optimizers.values())
            assert not any(record["optimizer_steps"].values())
            assert trainer.it == 0
            record["collected_transitions"] = int(stats["steps"])
            record["replay_rows"] = int(trainer.replay.num_valid_rows)
            record["checkpoint"] = str(trainer.save_checkpoint())
            record["status"] = "ZERO_OPTIMIZER_PROBE_COMPLETED"
        else:
            record["initial_checkpoint"] = str(trainer.save_checkpoint())
            original_iteration = trainer.iteration

            def observed_iteration():
                torch.cuda.synchronize()
                started = time.perf_counter()
                metrics = original_iteration()
                torch.cuda.synchronize()
                record["iteration_seconds"] += time.perf_counter() - started
                record.update(iteration=trainer.it,
                              gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                save_json(progress, record)
                return metrics

            trainer.iteration = observed_iteration
            trainer.train()
            if not all(record["optimizer_steps"].values()):
                raise RuntimeError("Run ended without all three optimizer updates")
            record.update(status="TRAINING_COMPLETED", iteration=trainer.it)
    except BaseException as exc:
        record.update(status="FAILED_OR_INTERRUPTED",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            for hook in hooks:
                hook.remove()
            if trainer is not None:
                trainer.close()
        finally:
            record["ended_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(progress, record)
            print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
