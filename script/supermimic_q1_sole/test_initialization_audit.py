"""Offline counterexamples for initialization diagnostic records."""

import json
import unittest

import numpy as np

from initialization_audit import InitializationAudit, array_summary, compare_arrays


class InitializationAuditTest(unittest.TestCase):
    def test_float_tolerance_does_not_hide_changed_bytes(self):
        result = compare_arrays(np.array([1.]), np.array([1.000001]))
        self.assertTrue(result["matches"])
        self.assertEqual(result["changed_values"], 1)
        self.assertFalse(compare_arrays(np.array([1.]), np.array([1.001]))["matches"])

    def test_integer_and_rng_arrays_require_exact_equality(self):
        result = compare_arrays(np.array([32000]), np.array([32001]))
        self.assertFalse(result["matches"])
        self.assertEqual(result["outside_tolerance"], 1)

    def test_shape_dtype_nonfinite_and_empty_fields(self):
        self.assertFalse(compare_arrays(np.zeros(2), np.zeros((1, 2)))["matches"])
        self.assertFalse(compare_arrays(np.zeros(2), np.zeros(2, dtype=np.float32))["matches"])
        self.assertFalse(compare_arrays(np.array([np.nan]), np.array([np.nan]))["matches"])
        self.assertTrue(compare_arrays(np.empty((2, 0)), np.empty((2, 0)))["matches"])
        with self.assertRaises(ValueError):
            array_summary(np.array(["unsupported"]))

    def test_scalar_difference_and_nonfinite_receipts_remain_json_serializable(self):
        scalar = compare_arrays(np.asarray(0.), np.asarray(1.))
        self.assertEqual(scalar["maximum_difference_index"], [])
        invalid = compare_arrays(np.asarray(np.inf), np.asarray(1.))
        json.dumps([scalar, invalid], allow_nan=False)

    def test_first_action_difference_is_distinct_from_rng(self):
        receipt = {}
        audit = InitializationAudit(receipt)
        audit.begin("reference", {})
        audit.record("reset", {"state.s": np.zeros(2), "rng.cpu": np.array([1], dtype=np.uint8)})
        audit.record("action_0", {"policy.action": np.zeros(2)})
        audit.record("state_1", {"state.s": np.zeros(2)})
        audit.finish()
        audit.begin("branch", {})
        audit.record("reset", {"state.s": np.zeros(2), "rng.cpu": np.array([2], dtype=np.uint8)})
        audit.record("action_0", {"policy.action": np.ones(2) * .015625})
        audit.record("state_1", {"state.s": np.ones(2) * .07455075})
        audit.finish()
        branch = receipt["traces"][1]
        self.assertEqual(branch["first_rng_difference"], dict(stage="reset", field="rng.cpu"))
        self.assertEqual(branch["first_state_action_difference"], dict(stage="action_0", field="policy.action"))
        self.assertTrue(branch["complete"])
        json.dumps(receipt, allow_nan=False)

    def test_reference_is_copied_before_later_mutation(self):
        audit = InitializationAudit({})
        source = np.zeros(2)
        audit.begin("reference", {})
        audit.record("reset", {"state.s": source})
        source[:] = 9
        audit.begin("branch", {})
        audit.record("reset", {"state.s": np.zeros(2)})
        self.assertIsNone(audit.trace["first_state_action_difference"])

    def test_reset_solver_difference_and_missing_field_are_reported(self):
        audit = InitializationAudit({})
        audit.begin("reference", {})
        audit.record("reset", {"native.warmstart": np.zeros(2), "state.s": np.zeros(2)})
        audit.begin("branch", {})
        audit.record("reset", {"native.warmstart": np.ones(2)})
        self.assertEqual(audit.trace["first_state_action_difference"]["stage"], "reset")
        self.assertEqual(audit.trace["snapshots"][0]["comparison"]["state.s"]["reason"], "field_set_changed")

    def test_changed_stage_order_and_short_trace_are_visible(self):
        audit = InitializationAudit({})
        audit.begin("reference", {})
        audit.record("reset", {"state.s": np.zeros(2)})
        audit.record("action_0", {"policy.action": np.zeros(2)})
        audit.begin("branch", {})
        audit.record("action_0", {"policy.action": np.zeros(2)})
        audit.finish()
        self.assertFalse(audit.trace["complete"])
        self.assertEqual(audit.trace["first_state_action_difference"]["stage"], "action_0")


if __name__ == "__main__":
    unittest.main()
