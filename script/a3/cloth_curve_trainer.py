"""E44 uses the E42 optimizer unchanged and records its sampled reference modes."""

from collections import Counter

from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import ClothCachedTokenTrainer
from script.a3.cloth_curve_protocol import CHECKPOINTS, UPDATES
from script.a3.resumable_sonic import DurableModelSaveCallback


class ClothCurveTrainer(ClothCachedTokenTrainer):
    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        self._curve_routes = Counter()
        self._curve_rollouts = []
        return result

    def _capture_tokens(self, module, arguments, output):
        super()._capture_tokens(module, arguments, output)
        if self._collect_sample_tokens:
            indices = module.parse_tokenizer_obs(arguments[0])['encoder_index']
            if indices.shape != (16, 1, 4):
                raise ValueError('Reference routing layout changed')
            for row in indices.detach().cpu().reshape(16, 4).tolist():
                key = tuple(row)
                if key not in ((1, 0, 0, 0), (1, 0, 0, 1)):
                    raise ValueError('Unexpected reference routing')
                self._curve_routes['a3_fast' if key[-1] else 'g1'] += 1

    def _rollout_step(self, *args, **kwargs):
        self._curve_routes = Counter()
        result = super()._rollout_step(*args, **kwargs)
        if sum(self._curve_routes.values()) != 16 * 24:
            raise ValueError('Reference routing coverage is incomplete')
        self._curve_rollouts.append(dict(update=len(self._curve_rollouts) + 1,
            trainer_global_step=int(self.state.global_step),
            routes=dict(self._curve_routes), transitions=16 * 24))
        atomic_json(self._cloth_directory / 'reference_route_coverage.json', dict(
            rollouts=self._curve_rollouts,
            interpretation='ACTUAL_SAMPLED_ROUTES_NOT_ASSUMED_FIFTY_FIFTY_COUNTS'))
        return result


class CurveCheckpointCallback(DurableModelSaveCallback):
    """One directory per milestone avoids pruning earlier curve checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.target_global_step != UPDATES:
            raise ValueError('The registered E44 training budget cannot change')

    def _save(self, args, state, **kwargs):
        original = self.save_dir
        self.save_dir = original / ('milestone_' + str(state.global_step))
        try:
            return super()._save(args, state, **kwargs)
        finally:
            self.save_dir = original

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in CHECKPOINTS:
            self._save(args, state, **kwargs)
        if state.global_step >= UPDATES:
            control.should_training_stop = True
        return control
