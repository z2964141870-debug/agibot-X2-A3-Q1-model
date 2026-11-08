"""Detect future-state resets, severed gradients and invalid temporal support."""

import unittest

import numpy as np
import torch

from wm_multistep_math import common_support, directional_gradient, prefix_errors, rollout, translation_decomposition
from wm_multistep_worker import screen


class MultistepTests(unittest.TestCase):
    def test_open_loop_feeds_its_prediction_and_preserves_order(self):
        state = torch.tensor([[1.]])
        actions = torch.tensor([[[2.]], [[3.]], [[4.]]])
        result = rollout(lambda s,a,c:2*s+a, state, actions, torch.zeros(1, 1))
        torch.testing.assert_close(result, torch.tensor([[4.], [11.], [26.]]))
        actions[2] = 100
        altered = rollout(lambda s,a,c:2*s+a, state, actions, torch.zeros(1, 1))
        torch.testing.assert_close(result[:2], altered[:2])

    def test_gradient_reaches_first_action_through_all_model_states(self):
        actions = torch.zeros(4, 1, 2)
        check = directional_gradient(lambda s,a,c:2*s+a[:, :1], torch.ones(1, 1), actions,
            torch.zeros(1, 1), torch.tensor([True, False]), lambda p:p.sum())
        self.assertAlmostEqual(check['gradient_norm'], 8.)
        self.assertTrue(check['numerical_directional_gradient_passed'])
        self.assertFalse(check['physical_action_gradient_validated'])

    def test_fixed_parameters_receive_no_accumulated_gradient(self):
        weight = torch.nn.Parameter(torch.tensor(1.2), requires_grad=False)
        check = directional_gradient(lambda s,a,c:weight*s+a, torch.ones(1, 1), torch.zeros(2, 1, 1),
            torch.zeros(1, 1), torch.tensor([True]), lambda p:p.square().mean())
        self.assertTrue(check['numerical_directional_gradient_passed'])
        self.assertIsNone(weight.grad)

    def test_endpoint_cannot_replace_prefix_rms_or_pad_missing_support(self):
        result = prefix_errors({'known':np.array([3., 4.])}, 2)
        self.assertEqual(result['endpoint']['known'], 4.)
        self.assertAlmostEqual(result['prefix_rms']['known'], np.sqrt(12.5))
        for n in (0, 3, 1.5, True):
            with self.assertRaises(ValueError): prefix_errors({'known':np.array([1., 2.])}, n)

    def test_terminal_step_is_excluded_on_both_arms(self):
        rows = [dict(arm=arm, attempted_steps=3, valid_prefix_steps=2 if i == 0 else 3,
            failure_frame=13 if i == 0 else None, clip='known', anchor=10, split='validation')
            for i,arm in enumerate(('student_f0', 'privileged_teacher_h10'))]
        arrays = [dict(actual=np.zeros((4,79)), reference=np.zeros((4,79)),
            canonical_action=np.zeros((3,29)), load=np.zeros((4,2))) for _ in rows]
        self.assertEqual(common_support(rows, arrays), 2)
        rows[0]['valid_prefix_steps'] = 3
        with self.assertRaises(ValueError): common_support(rows, arrays)

    def test_arm_state_mismatch_or_test_source_is_rejected(self):
        rows = [dict(arm=arm, attempted_steps=2, valid_prefix_steps=2, failure_frame=None,
            clip='known', anchor=10, split='train') for arm in ('student_f0', 'privileged_teacher_h10')]
        arrays = [dict(actual=np.zeros((3,79)), reference=np.zeros((3,79)),
            canonical_action=np.zeros((2,29)), load=np.zeros((3,2))) for _ in rows]
        arrays[1]['actual'][0,0] = 1
        with self.assertRaises(ValueError): common_support(rows, arrays)
        arrays[1]['actual'][0,0] = 0
        for row in rows: row['split'] = 'test'
        with self.assertRaises(ValueError): common_support(rows, arrays)

    def test_root_relative_decomposition_keeps_negative_cross(self):
        actual, target = np.zeros((3,14,3)), np.zeros((3,14,3))
        actual[:, 0, 0] = 1
        values = translation_decomposition(actual, target, 0)
        np.testing.assert_allclose(values['world'], 1./14.)
        self.assertTrue((values['signed_cross'] < 0).all())

    def screen_fixture(self):
        horizons = (1,2,4,8,16,32,40)
        settings = dict(arms=['student_f0','privileged_teacher_h10'],
            validation_screen=dict(metrics=['body_position_m'],mean_identity_ratio_max=.95,
                individual_identity_ratio_max=1.25))
        rows = [dict(split='validation',arm=arm,steps=h,clip=clip,methods={
            'open_loop':dict(prefix_rms=dict(body_position_m=error)),
            'hold_initial':dict(prefix_rms=dict(body_position_m=1.))})
            for h in horizons for arm in settings['arms'] for clip,error in (('a',1.4),('b',.4))]
        gradients = [dict(steps=h,job_index=j,numerical_directional_gradient_passed=True)
            for h in horizons for j in (0,4)]
        return rows,gradients,settings

    def test_improved_mean_cannot_hide_one_physical_source_regression(self):
        value = screen(*self.screen_fixture())
        metric = value['per_horizon'][0]['metrics'][0]
        self.assertLess(metric['ratio'], .95)
        self.assertFalse(metric['passed'])
        self.assertFalse(value['all_requested_screens_passed'])

    def test_missing_numerical_gradient_is_not_a_vacuous_pass(self):
        rows,gradients,settings = self.screen_fixture()
        with self.assertRaises(RuntimeError): screen(rows,gradients[1:],settings)


if __name__ == '__main__':
    unittest.main()
