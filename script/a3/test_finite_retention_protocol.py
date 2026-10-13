"""Standard-library checks; no tensor runtime outside ZHY."""

import math
import unittest
from script.a3.finite_retention_protocol import accept, SCALES, LIMITS, validate_recipe


class ProtocolTests(unittest.TestCase):
    def test_gate_is_original_absolute_cap_not_e50_nonincrease(self):
        self.assertTrue(accept(1., dict(objective=.9, teacher_rmse=.049, teacher_max_abs=.49)))
        self.assertFalse(accept(1., dict(objective=.9, teacher_rmse=.0500001, teacher_max_abs=.49)))
        self.assertFalse(accept(1., dict(objective=.9, teacher_rmse=.04, teacher_max_abs=.500001)))
        self.assertFalse(accept(1., dict(objective=1.000001, teacher_rmse=.04, teacher_max_abs=.4)))
        with self.assertRaises(ValueError):
            accept(1., dict(objective=math.nan, teacher_rmse=.04, teacher_max_abs=.4))

    def test_bounded_first_passing_sequence(self):
        self.assertEqual(SCALES, tuple(2.**-i for i in range(8)))
        self.assertEqual(LIMITS['candidate_evaluations'], 256)
        self.assertEqual(LIMITS['transitions'], 8 * 16 * 24)
        self.assertEqual(LIMITS['effect_rollouts'], 10)

    def test_recipe_does_not_hide_reward_change(self):
        from copy import deepcopy
        a = dict(manager_env=dict(commands=dict(motion=dict(motion_lib_cfg=dict(motion_file='old'))),
                                 rewards=dict(a=1)), algo=dict(lr=1))
        b = deepcopy(a)
        b['manager_env']['commands']['motion']['motion_lib_cfg']['motion_file'] = 'new'
        validate_recipe(a, b)
        b['manager_env']['rewards']['a'] = 2
        with self.assertRaises(ValueError):
            validate_recipe(a, b)


if __name__ == '__main__':
    unittest.main()
