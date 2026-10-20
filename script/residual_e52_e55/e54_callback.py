"""E54 milestone callback: DurableModelSaveCallback with per-step milestone dirs.

Mirrors the FiniteCheckpointCallback._save wrapper (E53 precedent): each save lands
in milestone_{global_step}/ so the target-step checkpoint is never overwritten by
the end-of-training save. No budget constant is enforced (E54 pins 16 in the
registration instead).
"""

from e53_trainer import E53InjectedTrainer
from script.a3.resumable_sonic import DurableModelSaveCallback


class E54MilestoneCallback(DurableModelSaveCallback):
    def _save(self, args, state, **kwargs):
        directory = self.save_dir
        self.save_dir = directory / ("milestone_" + str(state.global_step))
        try:
            return super()._save(args, state, **kwargs)
        finally:
            self.save_dir = directory


class E54InjectedTrainer(E53InjectedTrainer):
    """E53InjectedTrainer + failure diagnostics on the t=0 anchor replay guard."""

    def load_checkpoint(self, *args, **kwargs):
        try:
            return super().load_checkpoint(*args, **kwargs)
        except ValueError as exc:
            if "warm start" not in str(exc):
                raise
            import os
            import torch
            try:
                rollout = torch.load(os.environ["YUANQI_CLOTH_ANCHOR_ROLLOUT"],
                                     map_location="cpu", weights_only=False)
                golden = torch.load(os.environ["YUANQI_CLOTH_ANCHOR_OUTPUTS"],
                                    map_location="cpu", weights_only=False)["initial"]["mean"].to(self.accelerator.device)
                policy = self.accelerator.unwrap_model(self.model).policy
                dev = self.accelerator.device
                obs = {k: v.to(dev) for k, v in rollout["all_obs_dict"].items()}
                with torch.no_grad():
                    means = torch.cat([policy.forward({k: v[i:i + 4] for k, v in obs.items()},
                                                      is_training=False)
                                       for i in range(0, golden.shape[0], 4)], dim=0)
                diff = (means - golden).abs()
                print("E54 DIAG anchor replay: max_abs=%.3e mean_abs=%.3e nan=%s golden_norm=%.6f means_norm=%.6f"
                      % (diff.max().item(), diff.mean().item(),
                         bool(torch.isnan(means).any().item()),
                         golden.norm().item(), means.norm().item()), flush=True)
            except Exception as diag_exc:
                print("E54 DIAG failed to recompute: %r" % (diag_exc,), flush=True)
            raise
