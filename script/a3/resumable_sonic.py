"""Single-GPU resume adapters for pinned AgibotTech/sonic_for_a3.

Upstream source and checkpoint contract: Apache-2.0, commit fe6868b.
The PPO update and simulator implementation remain upstream code.
"""

import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from transformers import TrainerCallback

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
from script.a3.checkpoint_store import atomic_json, save_checkpoint


def optimizer_steps(state_dict):
    return sorted({int(value["step"]) for value in state_dict["state"].values() if "step" in value})


class ResumableTrainer(TRLPPOTrainer):
    def load_checkpoint(self, checkpoint_path, resume=False):
        checkpoint = super().load_checkpoint(checkpoint_path, resume=resume)
        if not resume:
            raise ValueError("The resume adapter requires resume=true")
        if self.accelerator.num_processes != 1:
            raise ValueError("Only single-GPU resume has been verified")
        # Upstream overwrites checkpoint optimizer LRs with the static config LR.
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.args.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        model = self.accelerator.unwrap_model(self.model)
        matches = {}
        for name, module in [("policy_state_dict", model.policy), ("value_state_dict", model.value_model)]:
            live = module.state_dict()
            source = checkpoint[name]
            matches[name] = live.keys() == source.keys() and all(
                torch.equal(live[key], source[key]) for key in live
            )
        if not all(matches.values()):
            raise RuntimeError("Loaded network differs from the source checkpoint")
        if checkpoint["state"].cur_episode_length.shape[0] != self.env.num_envs:
            raise ValueError("Resume must preserve num_envs")
        if self.state.global_step != checkpoint["state"].global_step:
            raise RuntimeError("Trainer counter was not restored")
        current_optimizer = self.optimizer.state_dict()
        if optimizer_steps(current_optimizer) != optimizer_steps(checkpoint["optimizer_state_dict"]):
            raise RuntimeError("Optimizer counter was not restored")
        # PhysX episodes restart: unfinished reward sums cannot cross that boundary.
        self.cur_reward_sum.zero_()
        self.cur_episode_length.zero_()
        self.state.cur_reward_sum = self.cur_reward_sum
        self.state.cur_episode_length = self.cur_episode_length
        rng = checkpoint.get("yuanqi_rng_state")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        receipt = {
            "source": str(checkpoint_path), "loaded_global_step": int(self.state.global_step),
            "optimizer_steps": optimizer_steps(current_optimizer),
            "optimizer_learning_rates": [float(group["lr"]) for group in current_optimizer["param_groups"]],
            "networks_match_source": matches,
            "simulator_episodes": "reset; not a bitwise trajectory continuation",
        }
        directory = Path(os.environ["YUANQI_RESUME_LOG_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "resume_loaded.json", receipt)
        print("[YUANQI Resume] " + json.dumps(receipt), flush=True)
        return checkpoint

    def train(self):
        target = int(self.config.num_learning_iterations)
        remaining = target - int(self.state.global_step)
        if remaining <= 0:
            raise ValueError(f"Target {target} already reached")
        self.args.num_total_batches = remaining
        self.args.total_episodes = remaining * self.args.batch_size
        print(f"[YUANQI Resume] target={target} remaining_updates={remaining}", flush=True)
        return super().train()


class DurableModelSaveCallback(TrainerCallback):
    def __init__(self, save_dir, save_frequency=25, save_last_frequency=25,
                 max_disk_usage=None, target_global_step=2000):
        self.save_dir = Path(save_dir)
        self.save_frequency = int(save_frequency)
        self.target_global_step = int(target_global_step)
        self.saved_step = None
        if self.save_frequency < 1:
            raise ValueError("save_frequency must be positive")

    def _save(self, args, state, **kwargs):
        if not state.is_world_process_zero or kwargs["env"].is_evaluating:
            return
        if self.saved_step == state.global_step:
            return
        model = kwargs["model"]
        snapshot = copy.copy(state)
        snapshot.__dict__ = {key: value for key, value in state.__dict__.items() if key != "log_history"}
        payload = {
            "policy_state_dict": model.policy.state_dict(),
            "value_state_dict": model.value_model.state_dict(),
            "optimizer_state_dict": kwargs["optimizer"].state_dict(),
            "lr_scheduler_state_dict": kwargs["lr_scheduler"].state_dict(),
            "state": copy.deepcopy(snapshot), "args": args,
            "env_state_dict": kwargs["env"].get_env_state_dict(),
            "yuanqi_rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            },
        }
        record = save_checkpoint(self.save_dir, payload)
        self.saved_step = state.global_step
        print("[YUANQI DurableCheckpoint] " + json.dumps(record), flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        state.max_steps = self.target_global_step
        self._save(args, state, **kwargs)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_frequency == 0 or state.global_step >= self.target_global_step:
            self._save(args, state, **kwargs)
        if state.global_step >= self.target_global_step:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._save(args, state, **kwargs)
        return control
