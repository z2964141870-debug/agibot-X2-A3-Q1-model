from pathlib import Path
import unittest
from .prepare import transform


class BudgetPreparationTests(unittest.TestCase):
    def test_only_deadline_and_audit_metadata_change(self):
        original=(Path(__file__).parents[1]/'x2_morph_backend/cpp/native_controller.cpp').read_text()
        candidate=transform(original)
        self.assertEqual(candidate.count('received-sent>rpc_budget_ms/1000.'),2)
        self.assertNotIn('received-sent>.020',candidate)
        self.assertIn('received-f.oldest_receive_s>wc.feedback_age_s',candidate)
        self.assertIn('gate.Fault("rpc_or_feedback_deadline")',candidate)
        self.assertIn('rpc_budget_ms>30.',candidate)
    def test_unreviewed_source_rejected(self):
        with self.assertRaises(ValueError):transform('not the frozen source')
