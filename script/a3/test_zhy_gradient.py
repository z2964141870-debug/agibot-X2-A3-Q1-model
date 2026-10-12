"""Independent-loss, parameter/buffer, and no-update diagnostic contracts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from script.a3 import zhy_gradient as probe
from script.a3.zhy_gradient_trainer import ZHYGradientTrainer


class ZHYGradientTests(unittest.TestCase):
    def test_parameters_and_buffers_are_distinct(self):
        before = {'policy.encoder.weight': torch.tensor([1., 2.]),
            'value_model.running_mean': torch.tensor([0.]), 'empty': torch.empty(0)}
        after = {key: value.clone() for key, value in before.items()}
        after['policy.encoder.weight'][0] += .5
        after['value_model.running_mean'][0] += 1
        result = probe.state_delta(before, after, {'policy.encoder.weight'})
        self.assertEqual(result['changed_parameters'], ['policy.encoder.weight'])
        self.assertEqual(result['changed_buffers'], ['value_model.running_mean'])
        self.assertAlmostEqual(result['groups']['actor_encoder']['delta_l2'], .5)

    def test_incompatible_state_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            probe.state_delta({'p': torch.zeros(2)}, {'p': torch.zeros(3)}, {'p'})

    def test_buffer_names_are_preserved(self):
        with self.assertRaisesRegex(ValueError, 'names'):
            probe.buffer_changes({'first': torch.zeros(1)}, {'second': torch.zeros(1)})

    def test_loss_clipping_padding_and_auxiliary_coefficients(self):
        source = dict(advantages=torch.tensor([[[2.], [-1.]]]),
            logprobs=torch.zeros(1, 2), values=torch.zeros(1, 2, 1),
            returns=torch.ones(1, 2, 1), padding_mask=torch.tensor([[False, True]]),
            padding_mask_p1=torch.tensor([[False, True]]))
        row = dict(indices=torch.tensor([0]), value=torch.tensor([[[1.], [200.]]]),
            policy=dict(logprobs=torch.log(torch.tensor([[2., 1.]])),
                entropy=torch.tensor([[4., 900.]]), aux_losses={'reconstruction': torch.tensor(3.)},
                aux_loss_coef={'reconstruction': .5}))
        config = dict(cliprange=.2, cliprange_value=.2, vf_coef=2., entropy_coef=.1,
            ppo_loss_coef=.5, aux_loss_scale=2., multi_critic_advantage_weights=None)
        result = probe.recompute_terms(row, source, config)
        self.assertAlmostEqual(result['policy'], -1.2, places=6)
        self.assertAlmostEqual(result['value'], .64, places=6)
        self.assertAlmostEqual(result['entropy'], -.2, places=6)
        self.assertAlmostEqual(result['aux'], 3., places=6)

    def test_optimizer_step_is_forbidden_in_corrected_trainer(self):
        trainer = object.__new__(ZHYGradientTrainer)
        with self.assertRaisesRegex(RuntimeError, 'forbids'):
            trainer._before_optimizer_step(None, (), {})

    def test_existing_experiment_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(probe, 'OUT', Path(directory)):
                with self.assertRaisesRegex(FileExistsError, 'No repeat'):
                    probe.register()


if __name__ == '__main__':
    unittest.main()
