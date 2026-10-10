"""Training acceptance must not be inferred from artifacts or forecast gains."""

import unittest

from script.a3.prepare_longtrain import evaluate_readiness


class ReadinessTests(unittest.TestCase):
    def test_infrastructure_does_not_imply_training_ready(self):
        checks = {name: dict(passed=True) for name in
                  ("official_bundle", "official_cpu_reload", "motion_split", "motionlib", "storage", "vendor")}
        result = evaluate_readiness({}, checks)
        self.assertTrue(result["infrastructure_ready"])
        self.assertFalse(result["long_training_ready"])
        self.assertIn("no_accepted_finetuned_policy", result["blockers"])
        self.assertFalse(result["launch_performed"])

    def test_failed_gate_cannot_be_hidden_by_ready_config(self):
        checks = {name: dict(passed=True) for name in ("official_bundle", "official_cpu_reload", "motion_split",
            "motionlib", "storage", "vendor", "runtime_likelihood", "candidate_policy", "full_causal_input", "fresh_validation")}
        config = dict(candidate_variable="fixed_test_variable", actor_update_scope="decoder", long_training_updates=1000)
        self.assertTrue(evaluate_readiness(config, checks)["long_training_ready"])
        checks["runtime_likelihood"]["passed"] = False
        self.assertFalse(evaluate_readiness(config, checks)["long_training_ready"])
        checks["storage"]["passed"] = False
        self.assertFalse(evaluate_readiness(config, checks)["infrastructure_ready"])


if __name__ == "__main__":
    unittest.main()
