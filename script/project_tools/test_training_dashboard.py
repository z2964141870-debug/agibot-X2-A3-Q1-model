"""Check iteration windows and completed-episode accounting independently."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from plot_training_dashboard import episode_points, rolling_ratio
from train_q1_with_dashboard import EpisodeAccumulator, RecordingSim


class DashboardTests(unittest.TestCase):
    def test_iteration_window_is_not_row_count(self):
        points = [(25, 1., 1.), (500, 3., 1.), (525, 8., 2.), (1500, 7., 1.)]
        self.assertEqual(rolling_ratio(points, 500), [1., 2., 11. / 3., 7.])

    def test_completed_episodes_and_simultaneous_timeout(self):
        episodes = EpisodeAccumulator(2, .1)
        episodes.update([.1, .05], [False, True], [False, False])
        first = episodes.flush()
        self.assertEqual(first["completed_episodes"], 1)
        self.assertEqual(first["length_steps_sum"], 1)
        self.assertAlmostEqual(first["normalized_return_sum"], .5)
        episodes.update([0., .1], [False, True], [True, True])
        second = episodes.flush()
        self.assertEqual(second["completed_episodes"], 2)
        self.assertEqual(second["hard_terminations"], 1)
        self.assertEqual(second["length_steps_sum"], 3)
        self.assertAlmostEqual(second["normalized_return_sum"], 1.5)
        self.assertEqual(episodes.lengths, [0, 0])
        self.assertEqual(episodes.returns, [0., 0.])

    def test_zero_episode_windows_and_weighted_average(self):
        rows = [{"step": 25, "completed_episodes": 0, "hard_terminations": 0},
                {"step": 50, "completed_episodes": 2, "hard_terminations": 1},
                {"step": 75, "completed_episodes": 8, "hard_terminations": 0}]
        points = episode_points(rows, "hard_terminations")
        self.assertEqual(rolling_ratio(points, 500), [None, .5, .1])

    def test_incomplete_episode_is_carried_between_log_windows(self):
        episodes = EpisodeAccumulator(1, .1)
        episodes.update([.1], [False], [False])
        self.assertEqual(episodes.flush()["completed_episodes"], 0)
        episodes.update([.1], [False], [True])
        row = episodes.flush()
        self.assertEqual(row["length_steps_sum"], 2)
        self.assertAlmostEqual(row["normalized_return_sum"], 1.)

    @unittest.skipUnless(os.environ.get("YUANQI_DASHBOARD_TEST_ROOT"), "Optional real CPU MuJoCo control")
    def test_real_simulator_and_rng_are_unchanged(self):
        import sys
        root = Path(os.environ["YUANQI_DASHBOARD_TEST_ROOT"])
        sys.path.insert(0, str(root / "script/supermimic_q1_reproduction"))
        import train_reproduction as training
        import torch

        manifest = json.loads((root / "data/supermimic_for_q1/BONES_RF02_12000_20261025/run/admission.json").read_text())
        assets = training.Q1Assets(root)
        heads = assets.heads("cpu")
        library = training.MotionLibrary([Path(manifest["train_clips"][0]["path"])],
            training.TRACKED, assets.q0, device="cpu", active_indices=assets.active_indices)
        cfg = training.SimCfg(num_envs=2, threads=1, domain_randomization=False,
                             observation_noise=False, action_noise=0., episode_steps=2)
        observed = []
        with tempfile.TemporaryDirectory(dir=root / "data/supermimic_for_q1", prefix="dashboard_control_") as folder:
            for with_recording in (False, True):
                torch.manual_seed(20261025)
                sim = training.Q1Sim(assets, library, cfg, heads, 20261025)
                if with_recording:
                    sim = RecordingSim(sim, Path(folder))
                rows = []
                try:
                    for _ in range(5):
                        result = sim.step(torch.zeros(2, training.ACT_DIM))
                        rows.append({key: result[key].clone() for key in
                                     ("reward", "terminated", "timeout", "state", "frames")})
                    rng = torch.get_rng_state().clone()
                finally:
                    sim.close()
                observed.append((rows, rng))
            for baseline, recorded in zip(observed[0][0], observed[1][0]):
                for key in baseline:
                    self.assertTrue(torch.equal(baseline[key], recorded[key]), key)
            self.assertTrue(torch.equal(observed[0][1], observed[1][1]))
            logged = [json.loads(line) for line in (Path(folder) / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged[0]["step"], 5)
            self.assertGreaterEqual(logged[0]["completed_episodes"], 4)
            self.assertLessEqual(logged[0]["length_steps_sum"], 10)


if __name__ == "__main__":
    unittest.main()
