"""Run the actual small v0.1 native policy on CPU without simulation."""

import os
from pathlib import Path
import sys
import tempfile
import unittest


class RuntimeInferenceTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("Q1_NATIVE_TEST_SOURCE"), "Requires the fixed server source and Q1 assets")
    def test_native_motion_human_masks_and_frozen_statistics(self):
        import torch
        sys.path[:0] = [str(Path(os.environ["Q1_NATIVE_TEST_SOURCE"]) / "src"),
                       str(Path(__file__).resolve().parents[1] / "morphmimic_q1")]
        from bootstrap import install
        adapter = install("q1")
        from morphmimic.model.policy import Policy, PolicyCfg
        from morphmimic.train.normalizers import RunningNormalizer
        from morphmimic.constants import GOAL_DIM, HUMAN_FEAT_DIM, HUMAN_HEADING_DIM, PROPRIO_DIM
        from inference import load_policy
        from test_inference import checkpoint
        from snapshots import rng_sha

        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())
        value = checkpoint(0)
        value["cfg"].update(policy_hidden=[32], act="elu", latent=32, decoder_film=True)
        actor = adapter["morphology_population"]()[0][1]
        policy = Policy(PolicyCfg(enc_hidden=(32,), dec_hidden=(32,), morph_dim=154, film=True, human=True))
        policy.set_morphology(actor)
        value.update(policy=policy.state_dict(), morph=dict(actor=actor, film=True), imu_body="pelvis")
        for name, size in (("obs_norm", PROPRIO_DIM), ("goal_norm", GOAL_DIM),
                           ("human_norm", HUMAN_FEAT_DIM), ("heading_norm", HUMAN_HEADING_DIM)):
            value[name] = RunningNormalizer(size).state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.pt"
            torch.save(value, path)
            actor_fn, meta = load_policy(path, adapter["robot"], "a" * 40)
            history = torch.zeros(1, 930)
            human = torch.zeros(1, 10, 24, 7)
            human[..., 3] = 1.
            goals = dict(goal=torch.zeros(1, 670),
                         goal_kp=torch.cat((human.flatten(1), torch.tensor([[1., 0., 0., 0.]])), -1))
            before = rng_sha()
            for interface in ("motion", "human"):
                action = actor_fn(history, goals, interface)
                self.assertTrue(torch.isfinite(action).all().item())
                self.assertEqual(torch.count_nonzero(action[:, ~actor_fn.present]).item(), 0)
            self.assertEqual(before, rng_sha())
            self.assertTrue(all(norm.count.item() == 0 and not norm.training for norm in actor_fn.norms.values()))
            self.assertEqual(meta["iteration"], 0)
            self.assertFalse(torch.cuda.is_initialized())
            from morphmimic.eval.policy_loader import load_policy as original_load
            for interface in ("motion", "human"):
                reference_fn, _ = original_load(path, "cpu", interface=interface,
                                                human_points="full", kp_horizon=10, human_heading=True)
                expected = reference_fn(history, goals)
                self.assertTrue(torch.equal(expected, actor_fn(history, goals, interface)))
            self.assertTrue(all(norm.count.item() == 0 for norm in actor_fn.norms.values()))
            original_actor = value["morph"]["actor"].clone()
            original_decoder_actor = value["policy"]["decoder.morph_actor"].clone()
            value["morph"]["actor"] = original_actor.clone()
            value["policy"]["decoder.morph_actor"] = value["policy"]["decoder.morph_actor"].clone()
            value["morph"]["actor"][0] += .125
            value["policy"]["decoder.morph_actor"][0] += .125
            torch.save(value, path)
            with self.assertRaisesRegex(RuntimeError, "actual native assets"):
                load_policy(path, adapter["robot"], "a" * 40)
            value["morph"]["actor"] = original_actor
            value["policy"]["decoder.morph_actor"] = original_decoder_actor
            value["policy"]["decoder.morph_actor"][-1] = 1.
            torch.save(value, path)
            with self.assertRaises(RuntimeError):
                load_policy(path, adapter["robot"], "a" * 40)


if __name__ == "__main__":
    unittest.main()
