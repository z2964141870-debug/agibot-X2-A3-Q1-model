"""Pinned native-policy CPU equivalence; requires the server runtime and Q1 assets."""

import os
from pathlib import Path
import sys
import unittest


class PreviewPolicyRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("Q1_NATIVE_TEST_SOURCE"), "Requires pinned native server runtime and Q1 assets")
    def test_full_preview_equivalence_and_hidden_pose_heading_invariance(self):
        import torch
        root = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(root / "q1_five_stage"), str(root / "morphmimic_q1"),
                       str(Path(os.environ["Q1_NATIVE_TEST_SOURCE"]) / "src")]
        from bootstrap import install
        adapter = install("q1")
        from inference import NativePolicy
        from morphmimic.constants import HUMAN_FEAT_DIM, HUMAN_HEADING_DIM, PROPRIO_DIM
        from morphmimic.model.human import fixed_human_masks, pack_human_goal
        from morphmimic.model.policy import Policy, PolicyCfg
        from morphmimic.train.imagination import act_human
        from morphmimic.train.normalizers import RunningNormalizer
        from preview_policy import HumanPreviewPolicy
        from snapshots import rng_sha
        from stream_contract import PoseTimeline
        from test_stream_contract import SOURCE, sample, window

        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())
        morphology = adapter["morphology_population"]()[0][1]
        policy = Policy(PolicyCfg(enc_hidden=(32,), dec_hidden=(32,), morph_dim=154, film=True, human=True))
        policy.set_morphology(morphology)
        policy.eval()
        norms = {name: RunningNormalizer(size).eval() for name, size in
                 (("obs_norm", PROPRIO_DIM), ("human_norm", HUMAN_FEAT_DIM),
                  ("heading_norm", HUMAN_HEADING_DIM))}
        present = torch.tensor([value is not None for value in adapter["robot"].translator.perm])
        native = NativePolicy(policy, norms, present)
        preview = HumanPreviewPolicy(native)
        history = torch.zeros(1, 930)
        torso = torch.tensor([[1., 0., 0., 0.]])
        before_rng = rng_sha()
        before = {name: {key: value.clone() for key, value in norm.state_dict().items()}
                  for name, norm in norms.items()}
        for horizon in (10, 5, 4, 2, 1):
            timeline = PoseTimeline(SOURCE)
            for i in range(horizon+1):
                timeline.ingest(sample(i))
            visible = window(timeline, horizon)
            actual = preview(history, visible, torso)
            position = torch.tensor([[sample(i).pose.positions for i in range(1, 11)]])
            orientation = torch.tensor([[sample(i).pose.orientations for i in range(1, 11)]])
            if horizon < 10:
                position[:, horizon:] += 1000.
                orientation[:, horizon:, :, 0] = 0.
                orientation[:, horizon:, :, 3] = 1.
            raw = pack_human_goal(position, orientation, torso)
            with torch.inference_mode():
                expected = act_human(policy, norms["obs_norm"], norms["human_norm"],
                    norms["heading_norm"], history, raw, *fixed_human_masks(1, "full", horizon, "cpu", heading=True))
            self.assertTrue(torch.equal(actual, expected), "Masked pose or heading leaked into the native action")
            if horizon == 10:
                self.assertTrue(torch.equal(actual, native(history, {"goal_kp": raw}, "human")))
            self.assertEqual(torch.count_nonzero(actual[:, ~present]).item(), 0)
        self.assertEqual(before_rng, rng_sha())
        for name, norm in norms.items():
            self.assertTrue(all(torch.equal(value, before[name][key]) for key, value in norm.state_dict().items()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
