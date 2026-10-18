"""D03 keeps the D02 trainer and asserts its registered native sample route."""

import torch

from script.a3.humanplus_replay_trainer import ReplayLossTrainer, ReplayCheckpointCallback
from script.a3.humanplus_route_protocol import validate_backbone, validate_routes


class RouteTrainer(ReplayLossTrainer):
    def load_checkpoint(self, *args, **kwargs):
        validate_backbone(self.config['actor']['backbone'])
        return super().load_checkpoint(*args, **kwargs)

    def _capture_tokens(self, module, arguments, output):
        if self._collect_sample_tokens:
            routes = module.parse_tokenizer_obs(arguments[0])['encoder_index']
            expected = routes.new_tensor([1, 0, 0, 1]).expand(16, 1, 4)
            if routes.shape != expected.shape or not torch.equal(routes, expected):
                raise ValueError('Registered rollout must retain native a3_fast/G1 pairing')
        return super()._capture_tokens(module, arguments, output)

    def _rollout_step(self, *args, **kwargs):
        result = super()._rollout_step(*args, **kwargs)
        validate_routes(self._curve_rollouts, len(self._curve_rollouts))
        return result


class RouteCheckpointCallback(ReplayCheckpointCallback):
    pass
