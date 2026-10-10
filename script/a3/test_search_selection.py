"""Verify the acceptance gate rejects metric tradeoffs and additional falls."""

import copy
import unittest

from script.a3.search_selection import METRICS, gate


def baseline():
    summary = dict(falls=0, full_rollout={metric: 0.1 for metric in METRICS})
    return dict(all=copy.deepcopy(summary), groups={role: copy.deepcopy(summary) for role in ("train", "heldout")})


class SelectionTests(unittest.TestCase):
    def improved(self):
        row = baseline()
        row["all"]["full_rollout"]["joint_rmse_rad"] = 0.09
        row["groups"]["heldout"]["full_rollout"]["joint_rmse_rad"] = 0.09
        return row

    def test_improvement_is_diagnostic_only(self):
        result = gate(baseline(), self.improved())
        self.assertTrue(result["selected20_gate_passed"])
        self.assertFalse(result["general_acceptance"])

    def test_root_regression_rejects_lower_joint_error(self):
        candidate = self.improved()
        candidate["all"]["full_rollout"]["root_pos_error_m_mean"] = 0.11
        self.assertFalse(gate(baseline(), candidate)["selected20_gate_passed"])

    def test_additional_fall_and_identical_policy_rejected(self):
        candidate = self.improved()
        candidate["groups"]["heldout"]["falls"] = 1
        self.assertFalse(gate(baseline(), candidate)["selected20_gate_passed"])
        self.assertFalse(gate(baseline(), baseline())["selected20_gate_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
