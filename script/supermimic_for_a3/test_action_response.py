"""Scientific controls for sign, degeneracy and terminated sensitivity samples."""

import unittest

import numpy as np

from diagnose_action_response import analyze, summarize, variants, vector_comparison


class ActionResponseControls(unittest.TestCase):
    def fixture(self):
        native = np.zeros((17, 32, 83))
        model = native.copy()
        for i, item in enumerate(variants()):
            delta = item["magnitude"] * item["sign"]
            native[i, :, :3] = delta
            native[i, :, 7:36] = delta
            model[i] = -native[i]
        return {"native": native, "model": model, "lengths": np.full(17, 32),
                "terminated": np.zeros((17, 32), bool), "timeouts": np.zeros((17, 32), bool),
                "clipped": np.zeros((17, 29), bool), "jvp32": -np.ones((4, 3, 83)),
                "jvp64": -np.ones((4, 3, 83)), "control_fd64": -np.ones((2, 4, 3, 83))}

    def test_reversed_model_is_detected_and_not_hidden_by_magnitude(self):
        analysis = analyze(self.fixture(), range(29))
        result = summarize([{"analysis": analysis}])
        self.assertEqual(result["32"]["root"]["native_vs_model_fd"]["median_cosine"], -1.)
        self.assertEqual(result["32"]["active_joints"]["native_vs_model_fd"]["median_norm_ratio"], 1.)
        self.assertTrue(all(c["passed"] for row in analysis["numerical_controls"] for c in row["float64_jvp_vs_fd"]))

    def test_terminated_cases_stay_in_denominator(self):
        raw = self.fixture()
        raw["terminated"][1, 4] = True
        raw["lengths"][1] = 5
        analysis = analyze(raw, range(29))
        result = summarize([{"analysis": analysis}])
        self.assertEqual(result["1"]["root"]["valid"], 8)
        self.assertEqual(result["8"]["root"]["total"], 8)
        self.assertEqual(result["8"]["root"]["valid"], 7)
        self.assertEqual(result["32"]["root"]["valid"], 7)

    def test_zero_response_has_no_cosine(self):
        result = vector_comparison(np.zeros(3), np.ones(3), 1e-6)
        self.assertTrue(result["degenerate_reference"])
        self.assertIsNone(result["cosine"])
        self.assertIsNone(result["norm_ratio"])

    def test_nonfinite_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            vector_comparison(np.array([np.nan]), np.ones(1), 1e-6)


if __name__ == "__main__":
    unittest.main()
