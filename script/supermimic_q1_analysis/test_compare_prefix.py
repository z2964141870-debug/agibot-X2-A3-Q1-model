import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import compare_prefix as comparison


class PrefixComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = np.zeros((4, 29))
        self.target[:, 3] = 1.
        self.reference = self.root / "reference.npz"
        np.savez(self.reference, qpos=self.target, fps=np.array(50.))

    def evaluation(self, name, state, failed=False):
        trace = self.root / (name + ".npz")
        np.savez(trace, state=state)
        count = len(state)
        row = {"clip": "walk", "reference": comparison.identity(self.reference),
            "trace": comparison.identity(trace), "expected_steps": 4, "executed_steps": count,
            "failed": failed, "full_clip_completion": count / 4,
            "executed_joint_rmse_rad": comparison.errors(state, self.target[:count])["joint_rmse_rad"]}
        receipt = self.root / (name + "_receipt.json")
        receipt.write_text(json.dumps({"status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED", "rows": [row]}))
        audit = self.root / (name + "_audit.json")
        audit.write_text(json.dumps({"status": "EVALUATION_TRACE_AUDIT_PASSED", "verified": True,
                                     "receipt": comparison.identity(receipt)}))
        return receipt, audit

    def state(self, errors):
        state = np.zeros((len(errors), 57))
        state[:, 3] = 1.
        state[:, 7:29] = np.asarray(errors)[:, None]
        return state

    def test_prefix_comparison_does_not_confuse_truncation_with_improvement(self):
        initial = self.evaluation("initial", self.state([1., 1., 10., 10.]))
        final = self.evaluation("final", self.state([2., 2.]), failed=True)
        rows = comparison.compare({"initial": comparison.load_evaluation(*initial),
                                   "final": comparison.load_evaluation(*final)})
        self.assertEqual(rows[0]["common_prefix_steps"], 2)
        initial_metrics = rows[0]["evaluations"]["initial"]
        final_metrics = rows[0]["evaluations"]["final"]
        self.assertGreater(initial_metrics["executed_errors"]["joint_rmse_rad"],
                           final_metrics["executed_errors"]["joint_rmse_rad"])
        self.assertEqual(initial_metrics["common_prefix_errors"]["joint_rmse_rad"], 1.)
        self.assertEqual(final_metrics["common_prefix_errors"]["joint_rmse_rad"], 2.)
        self.assertEqual(final_metrics["full_clip_completion"], .5)
        self.assertTrue(final_metrics["failed"])

    def test_horizontal_norm_and_leg_group_have_known_answers(self):
        state = self.state([0., 0.])
        state[:, 0:2] = [3., 4.]
        state[:, 7:19] = 2.
        state[:, 21:29] = 3.
        metrics = comparison.errors(state, self.target[:2])
        self.assertEqual(metrics["root_horizontal_rmse_m"], 5.)
        self.assertEqual(metrics["leg_joint_rmse_rad"], 2.)
        self.assertEqual(metrics["arm_joint_rmse_rad"], 3.)
        self.assertAlmostEqual(metrics["joint_rmse_rad"], np.sqrt(120 / 22))

    def test_mismatched_audit_trace_and_reference_are_rejected(self):
        paths = self.evaluation("first", self.state([0., 0.]))
        loaded = comparison.load_evaluation(*paths)
        paths[0].write_text(paths[0].read_text() + "\n")
        with self.assertRaises(ValueError):
            comparison.load_evaluation(*paths)
        other = comparison.load_evaluation(*self.evaluation("other", self.state([0., 0.])))
        with Path(loaded["rows"]["walk"]["trace"]["path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            comparison.compare({"first": loaded, "other": other})
        other["rows"]["walk"]["reference"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            comparison.compare({"first": loaded, "other": other})


if __name__ == "__main__":
    unittest.main()
