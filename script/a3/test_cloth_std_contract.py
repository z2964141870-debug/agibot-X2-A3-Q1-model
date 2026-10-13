"""Official runtime clamp changes only std and rejects nonfinite source values."""

import unittest

import torch

from script.a3.test_cloth_constrained import ClothContractTests
from script.a3.cloth_std_contract import canonical_policy_state


class StdBoundaryTests(unittest.TestCase):
    def test_only_std_is_canonicalized_without_mutating_source(self):
        std = torch.tensor([0., .01, .2, 1.])
        weight = torch.randn(2, 3)
        source = {'std': std, 'actor_module.decoders.g1_dyn.weight': weight}
        result = canonical_policy_state(source)
        self.assertTrue(torch.equal(result['std'], torch.tensor([.001, .01, .2, .5])))
        self.assertTrue(torch.equal(source['std'], torch.tensor([0., .01, .2, 1.])))
        self.assertIs(result['actor_module.decoders.g1_dyn.weight'], weight)
        self.assertEqual(result.keys(), source.keys())

    def test_nonfinite_std_rejected(self):
        with self.assertRaises(ValueError):
            canonical_policy_state({'std': torch.tensor([float('nan')])})


if __name__ == '__main__':
    unittest.main()
