"""Prepare official data or observe an unchanged pinned MorphMimic Trainer."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def validate_train(config, mode):
    from scale_plan import make_plan, official_defaults
    defaults = official_defaults(Path(config["source"]) / "src/morphmimic/train/trainer.py")
    candidate = make_plan(defaults, config["train_cfg"]["num_envs"])["scaled"]
    actual = config["train_cfg"]
    for key, value in candidate.items():
        if mode == "profile" and key in {"iterations", "wm_only_iters", "ckpt_every"}:
            continue
        if key == "compile" and actual.get(key) == "none":
            if not config.get("eager_execution_resource_reason"):
                raise ValueError("Eager execution needs a recorded resource reason")
            continue
        comparable = tuple(actual[key]) if isinstance(value, tuple) else actual[key]
        if comparable != value:
            raise ValueError("Deviation from frozen scaling plan: " + key)
    if mode == "profile" and not (1 <= actual["iterations"] <= 8
                                  and actual["wm_only_iters"] == 0):
        raise ValueError("Resource profile must exercise complete updates in at most8 iterations")
    if actual.get("robots") or actual.get("robot") != "g1" or actual.get("wandb"):
        raise ValueError("This entry is an independent G1 baseline with local logging")
    return candidate


def prepare_data(config, output, progress):
    from morphmimic.data.manifest import build_subset, filtered, load_manifest
    from morphmimic.data.seed_convert import convert_all

    csv_root = Path(config["csv_root"])
    originals = sorted(csv_root.rglob("*.csv"))
    if len(originals) != config["expected_csv_count"]:
        raise ValueError("Raw CSV inventory differs from the frozen count")
    sources = [identity(path) for path in originals]
    save(output / "raw_sources.json", sources)
    save(progress, dict(status="CONVERTING", started_utc=stamp(), raw_csv_count=len(sources)))
    destination = output / "npz50"
    convert_all(str(csv_root), str(destination), config["metadata"],
                workers=config["workers"], robot="g1")
    frame = load_manifest(str(destination / "manifest.csv"))
    kept = filtered(frame)
    subsets = {}
    for name, hours in (("tiny", .1), ("1h", 1.), ("all", None)):
        target = destination / "subsets" / (name + ".yaml")
        clips = build_subset(kept, hours, 0, str(target))
        duration = float(kept.set_index("path").loc[clips, "seconds"].sum())
        subsets[name] = dict(path=str(target), clips=len(clips), seconds=duration,
                             identity=identity(target))
    import numpy as np
    artifacts = []
    for path in sorted(destination.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            if int(arrays["fps"]) != 50 or arrays["joint_pos"].shape[1] != 29:
                raise ValueError("Original conversion produced an unexpected G1 layout")
            for key in arrays.files:
                value = arrays[key]
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError("Nonfinite converted field: " + key)
        artifacts.append(identity(path))
    if [identity(path) for path in originals] != sources:
        raise RuntimeError("Raw CSV sources changed during conversion")
    receipt = dict(status="OFFICIAL_G1_DATA_PREPARED", finished_utc=stamp(),
        raw_csv_count=len(sources), converted_clips=len(frame),
        converted_seconds=float(frame.seconds.sum()), kept_clips=len(kept),
        kept_seconds=float(kept.seconds.sum()),
        name_rejections=dict(Counter(frame.name_reject)),
        feasibility_rejections=dict(Counter(frame.feas_reject)), subsets=subsets,
        source_manifest=identity(output / "raw_sources.json"),
        metadata=identity(Path(config["metadata"])), converted_artifacts=artifacts,
        storage="SERVER_LOCAL_ONLY", integration_steps=0, policy_calls=0, optimizer_updates=0,
        notes=["Original converter performs static CPU MuJoCo forward kinematics.",
               "Subset names do not establish duration; actual seconds are recorded."])
    save(output / "receipt.json", receipt)
    save(progress, receipt)


def run_training(config, output, progress, mode):
    validate_train(config, mode)
    import numpy as np
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.train.trainer import TrainCfg, Trainer

    cfg = TrainCfg(**config["train_cfg"])
    if (Path(cfg.log_dir) / cfg.run_name).exists():
        raise FileExistsError("Independent upstream output already exists")
    record = dict(status="INITIALIZING", mode=mode, started_utc=stamp(), config=asdict(cfg),
                  optimizer_steps={}, iteration_seconds=0., recent_iteration_seconds=[],
                  nonfinite_steps=0, minimum_learning_iterations=cfg.iterations,
                  old_gait_regression_stops=False, user_accepted_checkpoint=None)
    save(progress, record)
    trainer, hooks = None, []
    try:
        configure_torch_backends()
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        trainer = Trainer(cfg, "cuda:0")
        for name in ("wm_opt", "policy_opt", "critic_opt", "critic_kp_opt"):
            optimizer = getattr(trainer, name, None)
            if optimizer is None:
                continue
            record["optimizer_steps"][name] = 0
            def count(optimizer, args, kwargs, name=name):
                record["optimizer_steps"][name] += 1
            hooks.append(optimizer.register_step_post_hook(count))
        record["initial_checkpoint"] = str(trainer.save_checkpoint())
        original_iteration = trainer.iteration
        def observed_iteration():
            torch.cuda.synchronize()
            begin = time.perf_counter()
            metrics = original_iteration()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            record["iteration_seconds"] += elapsed
            recent = record["recent_iteration_seconds"]
            recent.append(elapsed)
            del recent[:-100]
            for key, value in metrics.items():
                if key.endswith("/nonfinite_steps"):
                    number = float(value)
                    if not math.isfinite(number) or number != 0:
                        record["nonfinite_steps"] += 1
                        raise FloatingPointError("Original learner reported " + key)
            record.update(status="RUNNING", iteration=trainer.it, updated_utc=stamp(),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save(progress, record)
            if trainer.it % cfg.log_every == 0 or trainer.it == cfg.iterations:
                scalars = {k: float(v) for k, v in metrics.items() if not k.startswith("_")}
                with (output / "scalar_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(iteration=trainer.it, **scalars), allow_nan=False) + "\n")
            if mode == "train" and trainer.it == 40000 and trainer.it % cfg.ckpt_every:
                trainer.save_checkpoint()
            return metrics
        trainer.iteration = observed_iteration
        trainer.train()
        if not record["optimizer_steps"] or not all(record["optimizer_steps"].values()):
            raise RuntimeError("Complete WM/motion/KP/critic updates were not observed")
        record.update(status="RESOURCE_PROFILE_COMPLETED" if mode == "profile" else "TRAINING_COMPLETED",
                      iteration=trainer.it, run_dir=str(trainer.run_dir), completed_utc=stamp())
        save(output / "receipt.json", record)
    except BaseException as error:
        record.update(status="FAILED_OR_INTERRUPTED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if trainer is not None:
            trainer.close()
        record["ended_utc"] = stamp()
        save(progress, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("data", "profile", "train"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}")
    signal.signal(signal.SIGTERM, interrupted)
    progress = args.output / "progress.json"
    if args.mode == "data":
        prepare_data(config, args.output, progress)
    else:
        run_training(config, args.output, progress, args.mode)


if __name__ == "__main__":
    main()
