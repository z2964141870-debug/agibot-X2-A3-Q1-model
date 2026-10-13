"""Controls for rollout exclusivity, first-batch selection and direction labels."""

import ast
from pathlib import Path
import unittest

from script.a3.fresh_batch_protocol import Budget, classify, require_batch_contract


class FreshBatchTests(unittest.TestCase):
    def test_no_second_rollout_candidate_or_extra_environment_step(self):
        budget = Budget()
        for name in ('fresh_rollouts', 'mathematical_candidates', 'environment_steps'):
            for _ in range(budget.LIMITS[name]):
                budget.take(name)
            with self.assertRaises(RuntimeError):
                budget.take(name)
            self.assertEqual(budget.counts[name], budget.LIMITS[name])

    def test_first_original_microbatch_contract(self):
        require_batch_contract(16, 24, 4, 16)
        for values in ((16, 25, 4, 16), (16, 24, 8, 16), (32, 24, 4, 32)):
            with self.assertRaises(ValueError):
                require_batch_contract(*values)

    def test_opposite_mechanisms_are_not_conflated(self):
        self.assertEqual(classify(-1, 1, 1), 'RAW_DESCENT_INWARD_ADAM_OUTWARD')
        self.assertEqual(classify(-1, -1, 1), 'FIRST_ORDER_INWARD_FINITE_OUTWARD')
        self.assertEqual(classify(1, -1, -1), 'FRESH_BATCH_RAW_DESCENT_OUTWARD')
        self.assertEqual(classify(-1, -1, -1), 'FRESH_BATCH_ALL_INWARD_FAILURE_NOT_REPRODUCED')
        self.assertEqual(classify(0, 0, 0), 'OTHER_OR_DEGENERATE_DIRECTION')

    def test_collector_cannot_enter_training_or_construct_optimizer(self):
        tree = ast.parse(Path(__file__).with_name('fresh_batch_collect.py').read_text())
        forbidden = {'train_agent', 'main', 'train', 'create_optimizer', 'load_checkpoint', '_init_trl'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)
                if node.func.attr == '__init__':
                    self.fail('Collector must not initialize the original trainer')
        permutations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == 'randperm']
        self.assertEqual(len(permutations), 1)


if __name__ == '__main__':
    unittest.main()
