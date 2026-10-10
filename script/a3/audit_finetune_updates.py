"""Read R05 optimizer and scalar evidence without resuming the trainer."""

import json
from pathlib import Path

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.regression_diagnostic import TRAINING, OUTPUT, LOGS
from script.a3.fullchain_support import VENDOR


def main():
    torch.set_num_threads(1)
    tags = ("Policy/approxkl_avg", "Policy/clipfrac_avg", "Objective/rewards",
            "Env/Metrics/motion/error_joint_pos", "Env/Metrics/motion/error_body_pos")
    runs = []
    for run in sorted(TRAINING.glob("R05_*")):
        if not (run / "tensorboard").is_dir():
            continue
        events = EventAccumulator(str(run / "tensorboard"), size_guidance={"scalars": 0})
        events.Reload()
        available = events.Tags()["scalars"]
        selected = list(tags) + [tag for tag in available if "lr" in tag.lower() or "learning_rate" in tag.lower()]
        metrics = {tag: [dict(step=e.step, value=e.value) for e in events.Scalars(tag)]
                   for tag in selected if tag in available}
        runs.append(dict(run=run.name, metrics=metrics,
                         event_sources=[dict(path=str(p), sha256=sha256(p))
                                        for p in (run / "tensorboard").glob("events*")]))
    checkpoints = []
    for directory, step in (("R05_001_s0", 2), ("R05_010_s120", 200)):
        path = TRAINING / directory / f"model_step_{step:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scheduler = payload.get("lr_scheduler_state_dict", {})
        checkpoints.append(dict(step=step, checkpoint=str(path),
                                args_learning_rate=float(payload["args"].learning_rate),
                                optimizer_learning_rates=[g["lr"] for g in payload["optimizer_state_dict"]["param_groups"]],
                                scheduler={k: scheduler[k] for k in ("base_lrs", "_last_lr", "last_epoch") if k in scheduler}))
        del payload
    config = yaml.safe_load((TRAINING / "R05_010_s120/config.yaml").read_text())
    # Saved Hydra config has nested agent parameters; locate the exact mapping structurally.
    def find_optimizer_config(value):
        if isinstance(value, dict):
            if "actor_learning_rate" in value and "desired_kl" in value:
                return {k: value.get(k) for k in ("actor_learning_rate", "critic_learning_rate", "schedule",
                                                "desired_kl", "adaptive_lr_min", "adaptive_lr_max",
                                                "num_learning_epochs", "num_mini_batches")}
            for child in value.values():
                found = find_optimizer_config(child)
                if found:
                    return found
        return None
    source = VENDOR / "gear_sonic/trl/trainer/ppo_trainer.py"
    receipt = dict(runs=runs, checkpoints=checkpoints, saved_config=find_optimizer_config(config),
                   source=dict(path=str(source), sha256=sha256(source),
                               gaussian_kl="old rollout policy to current policy, summed over action dimensions",
                               clipping_stat="fraction where clipped surrogate exceeds unclipped surrogate",
                               lr_update="KL handler writes all optimizer groups; scheduler.step then runs at batch end"),
                   scope="READ_ONLY_TRAINING_DIAGNOSTICS_NOT_CAUSAL_ATTRIBUTION")
    atomic_json(OUTPUT / "update_audit.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("checkpoints", "saved_config", "source")}, indent=2))


if __name__ == "__main__":
    main()
