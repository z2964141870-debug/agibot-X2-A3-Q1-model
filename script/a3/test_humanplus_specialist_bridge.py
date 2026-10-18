"""Reduced-reference/full-robot mismatch controls with no tensors or simulator."""

import unittest

from script.a3.humanplus_specialist_bridge import correspondence


class BridgeControls(unittest.TestCase):
    def arguments(self):
        actions = ['joint_' + str(i) for i in range(29)]
        return dict(selected_body_names=['pelvis', 'left_wrist', 'right_wrist'],
            reference_body_count=3,
            robot_body_names=['pelvis', 'aux_a', 'aux_b', 'aux_c', 'left_wrist',
                              'aux_d', 'aux_e', 'aux_f', 'right_wrist'],
            reference_joint_names=['passive_a', *reversed(actions), 'passive_b'],
            reference_joint_count=31, robot_joint_names=[*actions, 'passive_a', 'passive_b'],
            action_joint_names=actions)

    def test_selected_reference_cannot_receive_full_robot_indices(self):
        mapping = correspondence(**self.arguments())
        self.assertEqual(mapping['reference_body_indices'], (0, 1, 2))
        self.assertEqual(mapping['robot_body_indices'], (0, 4, 8))
        reference = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
        self.assertEqual([reference[i] for i in mapping['reference_body_indices']], reference)

    def test_joint_order_uses_explicit_source_names_and_excludes_passive(self):
        args = self.arguments()
        mapping = correspondence(**args)
        reconstructed = [args['reference_joint_names'][i] for i in mapping['reference_joint_indices']]
        self.assertEqual(reconstructed, args['action_joint_names'])
        self.assertEqual(mapping['robot_joint_indices'], tuple(range(29)))

    def test_ambiguous_missing_or_wrong_shape_rejected_before_indexing(self):
        bad = self.arguments()
        for change in (dict(reference_body_count=9), dict(reference_joint_count=29),
                       dict(selected_body_names=['pelvis', 'pelvis', 'right_wrist']),
                       dict(robot_body_names=['pelvis']), dict(action_joint_names=['missing'] * 29)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                correspondence(**dict(bad, **change))


if __name__ == '__main__':
    unittest.main()
