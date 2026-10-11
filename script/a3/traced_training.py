"""Opt-in diagnostic trainer; synchronized spans change timing, not objective."""

from script.a3.phase_trace import trace, set_trainer, wrap
from script.a3.verified_finetune import VerifiedFineTuneTrainer
from script.a3.resumable_sonic import DurableModelSaveCallback
from script.a3 import checkpoint_store
import torch

torch.load = wrap(torch.load, "TORCH_FILE_LOAD")
torch.save = wrap(torch.save, "TORCH_FILE_SAVE")
checkpoint_store.validate_payload = wrap(checkpoint_store.validate_payload, "CHECKPOINT_VALIDATE")
checkpoint_store.sha256 = wrap(checkpoint_store.sha256, "CHECKPOINT_SHA256", cuda_sync=False)


class TracedFineTuneTrainer(VerifiedFineTuneTrainer):
    def __init__(self, *args, **kwargs):
        set_trainer(self)
        with trace().span("TRAINER_INIT"):
            super().__init__(*args, **kwargs)
        self.accelerator.backward = wrap(self.accelerator.backward, "BACKWARD")
        self.optimizer.step = wrap(self.optimizer.step, "OPTIMIZER_STEP")
        self.optimizer.zero_grad = wrap(self.optimizer.zero_grad, "ZERO_GRAD")
        trace().emit("HOOKS_READY", "TRAINER", hooks=["BACKWARD", "OPTIMIZER_STEP", "ZERO_GRAD"])

    def train(self):
        with trace().span("TRAIN_LOOP"):
            return super().train()


for _method, _phase in {
    "load_checkpoint":"OFFICIAL_WEIGHT_LOAD", "_setup_storage":"ROLLOUT_STORAGE_CREATE",
    "_rollout_step":"ROLLOUT", "policy_step":"ROLLOUT_POLICY_FORWARD",
    "_get_rollout_data":"ROLLOUT_STATE_READ", "_get_mb_rollout_data":"MINIBATCH_READ",
    "_forward_model":"UPDATE_FORWARD", "_compute_loss":"LOSS",
    "_gradient_clipping":"GRADIENT_CLIP", "_compute_returns":"RETURNS",
    "sync_running_mean_std":"SYNC_RUNNING_STATS", "sync_adaptive_sampling":"SYNC_SAMPLING",
}.items():
    setattr(TracedFineTuneTrainer, _method, wrap(getattr(VerifiedFineTuneTrainer, _method), _phase))


class TracedSaveCallback(DurableModelSaveCallback):
    def _save(self, args, state, **kwargs):
        with trace().span("CHECKPOINT_SAVE", saved_global_step=int(state.global_step)):
            return super()._save(args, state, **kwargs)
