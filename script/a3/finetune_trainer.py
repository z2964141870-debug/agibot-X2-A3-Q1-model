"""Warm-start receipt plus the existing full-state, durable single-GPU adapter."""

import os
from pathlib import Path

import torch

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
from script.a3.checkpoint_store import atomic_json
from script.a3.resumable_sonic import ResumableTrainer


class FineTuneTrainer(ResumableTrainer):
    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            return super().load_checkpoint(checkpoint_path, resume=True)
        checkpoint = TRLPPOTrainer.load_checkpoint(self, checkpoint_path, resume=False)
        model = self.accelerator.unwrap_model(self.model)
        matches = {}
        for key, network in (("policy_state_dict", model.policy), ("value_state_dict", model.value_model)):
            live = network.state_dict()
            source = checkpoint[key]
            matches[key] = live.keys() == source.keys() and all(torch.equal(live[k], source[k]) for k in live)
        if not all(matches.values()) or self.state.global_step != 0 or self.optimizer.state:
            raise ValueError("Warm start must load exact actor/critic weights and reset optimizer/counter")
        receipt = {"mode": "weight_warm_start", "source": str(checkpoint_path),
                   "source_step": int(checkpoint["state"].global_step), "loaded_global_step": 0,
                   "networks_match_source": matches, "optimizer_state_entries": len(self.optimizer.state),
                   "optimizer_learning_rates": [group["lr"] for group in self.optimizer.param_groups]}
        atomic_json(Path(os.environ["YUANQI_RESUME_LOG_DIR"]) / "warm_start_loaded.json", receipt)
        return checkpoint
