"""E54 milestone callback: DurableModelSaveCallback with per-step milestone dirs.

Mirrors the FiniteCheckpointCallback._save wrapper (E53 precedent): each save lands
in milestone_{global_step}/ so the target-step checkpoint is never overwritten by
the end-of-training save. No budget constant is enforced (E54 pins 16 in the
registration instead).
"""

from script.a3.resumable_sonic import DurableModelSaveCallback


class E54MilestoneCallback(DurableModelSaveCallback):
    def _save(self, args, state, **kwargs):
        directory = self.save_dir
        self.save_dir = directory / ("milestone_" + str(state.global_step))
        try:
            return super()._save(args, state, **kwargs)
        finally:
            self.save_dir = directory
