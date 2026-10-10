"""Opt-in official auxiliary objective and auditable adaptive LR for new trials."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import torch

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from script.a3.checkpoint_store import atomic_json
from script.a3.finetune_trainer import FineTuneTrainer


class AdaptiveLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Count completed PPO batches without overwriting KL-controlled group LRs."""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def restore_adaptive_lr(trainer, checkpoint):
    saved = float(checkpoint["args"].learning_rate)
    if not math.isfinite(saved) or not trainer.adaptive_lr_min <= saved <= trainer.adaptive_lr_max:
        raise ValueError("Saved adaptive LR is nonfinite or outside the configured bounds")
    previous = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    trainer.args.learning_rate = saved
    for group in trainer.optimizer.param_groups:
        group["lr"] = saved
    trainer.lr_scheduler._last_lr = [saved] * len(previous)
    return dict(saved_args_lr=saved, before_repair_optimizer_lrs=previous,
                restored_optimizer_lrs=[group["lr"] for group in trainer.optimizer.param_groups],
                scheduler_last_lrs=trainer.lr_scheduler.get_last_lr(),
                lr_owner="KL_ADAPTIVE_CONTROLLER_SAVED_IN_ARGS")


class VerifiedFineTuneTrainer(FineTuneTrainer, TRLAuxLossPPOTrainer):
    """Explicit new-trial target; existing R05 entry and vendor stay unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.compute_aux_loss:
            raise ValueError("Verified official-objective entry requires compute_aux_loss=true")
        directory = Path(os.environ["YUANQI_RESUME_LOG_DIR"])
        self._install_optimizer_audit(directory / "actual_optimizer_steps.jsonl")

    def create_scheduler(self, num_training_steps, optimizer=None):
        kind = getattr(self.args.lr_scheduler_type, "value", self.args.lr_scheduler_type)
        if kind != "constant" or self.config.get("desired_kl") is None:
            raise ValueError("Verified entry currently requires constant scheduler and adaptive KL")
        if self.lr_scheduler is None:
            self.lr_scheduler = AdaptiveLRScheduler(optimizer or self.optimizer)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def load_checkpoint(self, checkpoint_path, resume=False):
        checkpoint = super().load_checkpoint(checkpoint_path, resume=resume)
        if resume:
            receipt = restore_adaptive_lr(self, checkpoint)
            receipt.update(source=str(checkpoint_path), loaded_global_step=int(self.state.global_step),
                           objective="OFFICIAL_TRLAuxLossPPOTrainer")
            atomic_json(Path(os.environ["YUANQI_RESUME_LOG_DIR"]) / "verified_resume_lr.json", receipt)
        return checkpoint

    def _install_optimizer_audit(self, path):
        self._step_log_path = Path(path)
        self._step_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._optimizer_step_ordinal = 0
        self._pending_step_lrs = None
        self._last_step_lrs = None
        optimizer = self.optimizer
        while not isinstance(optimizer, torch.optim.Optimizer) and hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Cannot attach LR audit to underlying PyTorch optimizer")
        self._lr_audit_handles = (
            optimizer.register_step_pre_hook(self._before_optimizer_step),
            optimizer.register_step_post_hook(self._after_optimizer_step),
        )

    def _before_optimizer_step(self, optimizer, args, kwargs):
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        expected = float(self.args.learning_rate)
        if not math.isfinite(expected) or any(
                not math.isfinite(rate) or not math.isclose(rate, expected, rel_tol=1e-12, abs_tol=0.0)
                for rate in rates):
            raise ValueError("Optimizer LR diverged from the adaptive controller before step")
        self._pending_step_lrs = rates

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self._optimizer_step_ordinal += 1
        self._last_step_lrs = self._pending_step_lrs
        record = dict(utc=datetime.now(timezone.utc).isoformat(),
                      trainer_global_step=int(self.state.global_step),
                      attempt_optimizer_step=self._optimizer_step_ordinal,
                      applied_group_lrs=self._last_step_lrs,
                      adaptive_args_lr=float(self.args.learning_rate),
                      scope="UNDERLYING_OPTIMIZER_STEP_RETURNED_NOT_POLICY_EFFECT")
        with self._step_log_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._pending_step_lrs = None

    def append_to_log_dict(self, log_dict):
        super().append_to_log_dict(log_dict)
        log_dict["Optimizer/applied_steps_this_attempt"] = self._optimizer_step_ordinal
        if self._last_step_lrs is not None:
            for index, rate in enumerate(self._last_step_lrs):
                log_dict[f"Optimizer/last_applied_group_{index}_lr"] = rate
