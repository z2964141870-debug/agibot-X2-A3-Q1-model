"""Input evidence must reject shape errors and distinguish changed values."""

import unittest
import torch
from script.a3.zhy_wrapper_probe import input_identity


class InputTests(unittest.TestCase):
    def test_copied_input_and_mutation(self):
        expected = {'actor_obs':torch.zeros(4,24,3)}
        actual = {'actor_obs':expected['actor_obs'].clone()}
        self.assertTrue(input_identity(actual, expected)['actor_obs']['exact'])
        actual['actor_obs'][0,0,0] = 1
        row = input_identity(actual, expected)['actor_obs']
        self.assertFalse(row['exact'])
        self.assertEqual(row['max_abs'], 1)

    def test_different_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shapes differ'):
            input_identity({'actor_obs':torch.zeros(4,24,3)},
                           {'actor_obs':torch.zeros(24,4,3)})


if __name__ == '__main__':
    unittest.main()
