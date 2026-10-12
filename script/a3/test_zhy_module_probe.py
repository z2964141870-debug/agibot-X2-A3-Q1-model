"""Ablation scope and token-change conditioning contracts."""

import unittest

import torch

from script.a3.zhy_module_probe import drift, module_swap


class ModuleProbeTests(unittest.TestCase):
    def test_swap_changes_only_selected_module(self):
        base = {'actor_module.encoders.g1.weight': torch.tensor([1.]),
            'actor_module.decoders.g1_dyn.weight': torch.tensor([2.]), 'std': torch.tensor([3.])}
        updated = {name: value+10 for name, value in base.items()}
        for family, key in (('encoder', 'actor_module.encoders.g1.weight'),
                            ('decoder', 'actor_module.decoders.g1_dyn.weight'), ('std', 'std')):
            hybrid = module_swap(base, updated, family)
            for name in base:
                self.assertTrue(torch.equal(hybrid[name], updated[name] if name == key else base[name]))

    def test_swap_rejects_missing_module_or_state(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            module_swap({'std': torch.ones(1)}, {'std': torch.ones(1)}, 'encoder')
        with self.assertRaisesRegex(ValueError, 'names differ'):
            module_swap({'std': torch.ones(1)}, {}, 'std')

    def test_drift_conditions_on_changed_token_frames(self):
        base = dict(mean=torch.zeros(1, 2, 1), tokens=torch.zeros(1, 2, 2))
        updated = dict(mean=torch.tensor([[[2.], [0.]]]), tokens=torch.tensor([[[1., 0.], [0., 0.]]]))
        row = drift(base, updated)
        self.assertEqual(row['token_changed_coordinates'], 1)
        self.assertEqual(row['token_changed_frames'], 1)
        self.assertEqual(row['changed_token_frames_action_rmse'], 2.)
        self.assertEqual(row['unchanged_token_frames_action_rmse'], 0.)


if __name__ == '__main__':
    unittest.main()
