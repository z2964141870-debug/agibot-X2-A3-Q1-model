"""Focused failure contracts for the candidate converter's action boundary."""

from types import SimpleNamespace
import unittest

import numpy as np

from native_state_action_math import action_parity, isolated_converter, saved_prefix


class ActionContracts(unittest.TestCase):
    def test_continuous_prefix_is_bounded_and_copied(self):
        states, actions = np.zeros((51,79)), np.zeros((50,29))
        x,y = saved_prefix(states, actions, states.copy())
        self.assertEqual((x.shape,y.shape), ((32,79),(32,29)))
        x[0,0] = 1
        self.assertEqual(states[0,0], 0)

    def test_shifted_or_truncated_cpu_prefix_fails(self):
        states,actions = np.zeros((41,79)),np.zeros((40,29))
        shifted = states.copy(); shifted[1,0] = 1
        for candidate in (shifted, states[:31]):
            with self.assertRaises(ValueError):saved_prefix(states, actions, candidate)

    def test_action_tolerance_and_absent_slots(self):
        expected = np.ones(29); present = np.ones(29,dtype=bool)
        expected[3] = 0; present[3] = False
        candidate = expected.copy(); candidate[0] += 1e-6
        self.assertLess(action_parity(candidate, expected, present), 2e-5)
        for index,value in ((3,1e-9),(0,1.01),(0,float('nan'))):
            wrong = expected.copy(); wrong[index] = value
            with self.assertRaises(ValueError):action_parity(wrong, expected, present)

    def test_isolated_converter_restores_after_error(self):
        original = lambda *args:'original'
        module = SimpleNamespace(native_state=original)
        robot,spec = object(),object()
        with self.assertRaises(RuntimeError):
            with isolated_converter(module, robot, spec, lambda x:x+1):
                self.assertEqual(module.native_state(robot, 2, spec), 3)
                raise RuntimeError('Interrupted request')
        self.assertIs(module.native_state, original)

    def test_converter_rejects_other_model_or_spec(self):
        module = SimpleNamespace(native_state=lambda *args:None)
        robot,spec = object(),object()
        calls = []
        with isolated_converter(module,robot,spec,lambda x:calls.append(x)):
            with self.assertRaises(ValueError):module.native_state(object(),1,spec)
            with self.assertRaises(ValueError):module.native_state(robot,1,object())
        self.assertEqual(calls, [])


if __name__ == '__main__':unittest.main()
