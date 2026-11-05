"""Pinned native CPU forecast input equality; no simulation or optimization."""

import os
from pathlib import Path
import sys
import unittest


class ForecastPolicyRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("Q1_NATIVE_TEST_SOURCE"), "Requires pinned native server runtime and Q1 assets")
    def test_full_human_math_matches_all_causal_tail_methods(self):
        import torch
        root = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(root / "q1_five_stage"), str(root / "morphmimic_q1"),
                       str(Path(os.environ["Q1_NATIVE_TEST_SOURCE"]) / "src")]
        from bootstrap import install
        adapter = install("q1")
        from inference import NativePolicy
        from morphmimic.constants import HUMAN_FEAT_DIM, HUMAN_HEADING_DIM, PROPRIO_DIM
        from morphmimic.model.human import pack_human_goal
        from morphmimic.model.policy import Policy, PolicyCfg
        from morphmimic.train.normalizers import RunningNormalizer
        from causal_tail import forecast_window
        from forecast_policy import HumanForecastPolicy
        from snapshots import rng_sha
        from test_causal_tail import linear_window

        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())
        policy = Policy(PolicyCfg(enc_hidden=(32,), dec_hidden=(32,), morph_dim=154, film=True, human=True))
        policy.set_morphology(adapter["morphology_population"]()[0][1])
        policy.eval()
        norms = {name: RunningNormalizer(size).eval() for name, size in
                 (("obs_norm", PROPRIO_DIM), ("human_norm", HUMAN_FEAT_DIM), ("heading_norm", HUMAN_HEADING_DIM))}
        present = torch.tensor([value is not None for value in adapter["robot"].translator.perm])
        native = NativePolicy(policy, norms, present)
        actor = HumanForecastPolicy(native)
        history, torso = torch.zeros(1, 930), torch.tensor([[1., 0., 0., 0.]])
        before_rng = rng_sha()
        before_norms = {name: {key: value.clone() for key, value in norm.state_dict().items()}
                        for name, norm in norms.items()}
        comparisons = 0
        for horizon in (4, 5, 10):
            for method in ("hold", "cv", "damped_cv"):
                forecast = forecast_window(linear_window(horizon), method)
                action = actor(history, forecast, torso)
                position = torch.tensor([[pose.positions for pose in forecast.poses]])
                orientation = torch.tensor([[pose.orientations for pose in forecast.poses]])
                expected = native(history, dict(goal_kp=pack_human_goal(position, orientation, torso)), "human")
                self.assertTrue(torch.equal(action, expected))
                self.assertEqual(torch.count_nonzero(action[:, ~present]).item(), 0)
                comparisons += 1
        self.assertEqual(comparisons, 9)
        self.assertEqual(before_rng, rng_sha())
        for name, norm in norms.items():
            self.assertTrue(all(torch.equal(value, before_norms[name][key]) for key, value in norm.state_dict().items()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
