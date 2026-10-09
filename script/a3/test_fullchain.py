"""Regression tests for causal timing, source isolation and warm-start resets."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from script.a3.fullchain_support import DATA, ROOT
from script.a3.reference_contract import causal_window, validate_references
from script.a3.finetune_job import recovery_remaining, select_trial_checkpoint
from script.a3.evaluate_mujoco import rollout_windows
from script.a3.finetune_trainer import FineTuneTrainer, TRLPPOTrainer


class CausalWindowTests(unittest.TestCase):
    def test_cold_start_has_no_window(self):
        self.assertIsNone(causal_window(np.arange(20) / 30, .1))

    def test_30hz_arrivals_never_read_future(self):
        times = np.arange(100) / 30
        ages = []
        for arrival in np.arange(11, 150) / 50:
            window = causal_window(times, arrival)
            self.assertIsNotNone(window)
            self.assertLessEqual(times[window["upper"]].max(), arrival + 1e-9)
            self.assertEqual(len(window["requested"]), 10)
            ages.append(window["reference_age_s"])
        self.assertAlmostEqual(max(ages), .22)

    def test_missing_frames_hold_an_older_reference(self):
        times = np.array([0, .02, .04, .06, .08, .10, .12, .14, .16, .18, .2, .36])
        window = causal_window(times, .3)
        self.assertLessEqual(times[window["upper"]].max(), .3)
        self.assertGreater(window["reference_age_s"], .18)

    def test_exact_boundary_accepts_arrived_frame(self):
        window = causal_window(np.arange(20) / 50, .18)
        self.assertAlmostEqual(window["base"], 0)
        np.testing.assert_allclose(window["requested"], np.arange(10) / 50)

    def test_duplicate_time_is_rejected(self):
        with self.assertRaises(ValueError):
            causal_window([0, .02, .02], .3)

    def test_nonfinite_or_short_delay_is_rejected(self):
        for arrival, delay in ((float("nan"), .18), (.3, .1)):
            with self.assertRaises(ValueError):
                causal_window([0, .02, .04], arrival, delay)


class WarmStartTests(unittest.TestCase):
    def setup_trainer(self):
        policy = torch.nn.Linear(2, 2)
        critic = torch.nn.Linear(2, 1)
        payload = {"policy_state_dict": policy.state_dict(), "value_state_dict": critic.state_dict(),
                   "state": SimpleNamespace(global_step=200000)}
        model = SimpleNamespace(policy=policy, value_model=critic)
        trainer = SimpleNamespace(accelerator=SimpleNamespace(unwrap_model=lambda model: model), model=model,
                                  state=SimpleNamespace(global_step=0),
                                  optimizer=torch.optim.Adam(list(policy.parameters()) + list(critic.parameters())))
        return trainer, payload

    def test_official_counter_does_not_become_trial_counter(self):
        trainer, payload = self.setup_trainer()
        with TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"YUANQI_RESUME_LOG_DIR": directory}), patch.object(
                    TRLPPOTrainer, "load_checkpoint", return_value=payload):
                FineTuneTrainer.load_checkpoint(trainer, "released.pt", resume=False)
            receipt = json.loads((Path(directory) / "warm_start_loaded.json").read_text())
        self.assertEqual(receipt["source_step"], 200000)
        self.assertEqual(receipt["loaded_global_step"], 0)
        self.assertEqual(receipt["optimizer_state_entries"], 0)

    def test_restored_old_counter_is_rejected(self):
        trainer, payload = self.setup_trainer()
        trainer.state.global_step = 200000
        with patch.object(TRLPPOTrainer, "load_checkpoint", return_value=payload):
            with self.assertRaises(ValueError):
                FineTuneTrainer.load_checkpoint(trainer, "released.pt", resume=False)

    def test_populated_old_optimizer_is_rejected(self):
        trainer, payload = self.setup_trainer()
        for parameter in trainer.model.policy.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer.optimizer.step()
        with patch.object(TRLPPOTrainer, "load_checkpoint", return_value=payload):
            with self.assertRaises(ValueError):
                FineTuneTrainer.load_checkpoint(trainer, "released.pt", resume=False)

    def test_trial_lineage_rejects_legacy_high_step(self):
        payload = {"state": SimpleNamespace(global_step=2950, cur_episode_length=torch.zeros(16))}
        with patch("script.a3.finetune_job.select_checkpoint", return_value=(Path("model.pt"), payload, [])):
            with self.assertRaises(ValueError):
                select_trial_checkpoint([{"run_dir": str(ROOT / "data/training/a3_finetune_20261010/R04_001_s0")}])

    def test_other_lineage_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            select_trial_checkpoint([{"run_dir": str(ROOT / "data/training/a3_20261009/E008_cont_01_s2700")}])

    def test_interrupted_supervisor_cooldown_is_persisted(self):
        state = {"attempts": [{"boot_id": "same", "status": "running"}]}
        config = {"cooldown_seconds": 300}
        self.assertEqual(recovery_remaining(state, config, "same", 1000, 500), 300)
        self.assertEqual(recovery_remaining(state, config, "same", 1000, 800), 0)
        self.assertEqual(recovery_remaining(state, config, "new", 100, 800), 200)


class RolloutTests(unittest.TestCase):
    def test_before_fall_excludes_first_fallen_frame(self):
        trace = {"policy_tick": [0, 1, 2], "q_state_29": np.ones((3, 29)).tolist(),
                 "reference_q_29": np.zeros((3, 29)).tolist(),
                 "root_pos_error_m": [0, 1, 2], "root_quat_error_deg": [0, 1, 2],
                 "anchor_pos_error_m": [0, 1, 2]}
        summary = {"num_policy_steps": 3, "fall": True, "fall_tick": 2, "tracking": {"all_29_rmse": 1}}
        result = rollout_windows(summary, trace)
        self.assertEqual(result["before_first_fall"]["policy_steps"], 2)
        self.assertEqual(result["before_first_fall"]["root_pos_error_m_mean"], .5)
        self.assertEqual(result["full_rollout"]["root_pos_error_m_mean"], 1)
        trace["q_state_29"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            rollout_windows(summary, trace)


class ArtifactTests(unittest.TestCase):
    def test_whole_motion_split_is_disjoint(self):
        split = json.loads((DATA / "split.json").read_text())
        training = {row["sha256"] for row in split["train"]}
        heldout = {row["sha256"] for row in split["heldout"]}
        self.assertEqual((len(training), len(heldout)), (16, 4))
        self.assertFalse(training & heldout)

    def test_source_usage_remains_test_only(self):
        for directory in (DATA / "mocap").glob("*"):
            path = directory / "body_reference.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as source:
                    self.assertEqual(source["usage"].item(), "TEST_ONLY")
                    self.assertFalse(source["training_allowed"].item())

    def test_synthetic_joint_is_reconstructed(self):
        row = json.loads((DATA / "mocap/synthetic_single_joint/retarget_summary.json").read_text())
        self.assertLess(row["expected_active_joint_max_abs_error_rad"], .02)
        self.assertEqual(row["joint_limit_violations"], 0)

    def test_roundtrip_and_sampling_cover_all_references(self):
        rows = validate_references()
        self.assertEqual(len(rows), 26)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
