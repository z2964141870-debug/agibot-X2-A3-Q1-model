"""CPU tests for recovery limits, authorization, and checkpoint integrity."""

import fcntl
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from script.a3 import autoresume as recovery
from script.a3.checkpoint_store import atomic_json, save_checkpoint


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {"max_attempts": 3, "max_no_progress": 2, "target_step": 2000,
                       "expires_at": 10000, "num_envs": 64, "source_dirs": ["source"]}

    def tearDown(self):
        self.temporary.cleanup()

    def ledger(self, steps):
        return {"config_sha256": recovery.fingerprint(self.config), "status": "running",
                "attempts": [{"start_step": step, "run_dir": f"run{index}"}
                             for index, step in enumerate(steps)]}

    def test_budget_survives_serialization_and_progress(self):
        path = self.root / "state.json"
        atomic_json(path, self.ledger([600, 625, 650]))
        state = recovery.read_state(path, self.config)
        self.assertEqual(recovery.decision(self.config, state, 675, 100), "attempt_limit")

    def test_two_failures_without_saved_progress_stop(self):
        self.assertEqual(recovery.decision(self.config, self.ledger([600, 600]), 600, 100), "no_progress")

    def test_progress_clears_only_no_progress_count(self):
        self.assertEqual(recovery.decision(self.config, self.ledger([600, 600]), 625, 100), "run")
        self.assertEqual(len(self.ledger([600, 600])["attempts"]), 2)

    def test_expiration(self):
        self.assertEqual(recovery.decision(self.config, self.ledger([]), 600, 10000), "expired")

    def test_until_target_allows_many_progressing_attempts_without_expiry(self):
        config = dict(self.config, max_attempts=None, expires_at=None)
        state = self.ledger(list(range(600, 1000, 25)))
        self.assertEqual(recovery.decision(config, state, 1000, 1000000), "run")
        self.assertEqual(recovery.decision(config, state, 2000, 1000000), "complete")

    def test_until_target_retains_no_progress_stop(self):
        config = dict(self.config, max_attempts=None, expires_at=None)
        self.assertEqual(recovery.decision(config, self.ledger([600, 600]), 600, 1000000), "no_progress")

    def test_until_target_init_and_runtime_immutability(self):
        job = self.root / "data/training/until_target"
        argv = ["autoresume", str(job), "--init", "--until-target", "--target-step", "10000"]
        with patch.object(recovery, "ROOT", self.root), patch("sys.argv", argv):
            self.assertEqual(recovery.main(), 0)
        config = json.loads((job / "job.json").read_text())
        self.assertIsNone(config["max_attempts"])
        self.assertIsNone(config["expires_at"])
        self.assertEqual(config["max_no_progress"], 2)
        with patch("sys.argv", ["autoresume", str(job), "--until-target"]):
            with self.assertRaises(SystemExit):
                recovery.main()

    def test_target_precedes_attempt_limit(self):
        self.assertEqual(recovery.decision(self.config, self.ledger([600, 625, 650]), 2000, 100), "complete")

    def test_changed_config_rejected(self):
        path = self.root / "state.json"
        atomic_json(path, self.ledger([600]))
        with self.assertRaises(ValueError):
            recovery.read_state(path, dict(self.config, max_attempts=30))

    def test_missing_ledger_never_resets_budget(self):
        with self.assertRaises(FileNotFoundError):
            recovery.read_state(self.root / "missing.json", self.config)

    def test_corrupt_ledger_is_not_recreated(self):
        path = self.root / "state.json"
        path.write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            recovery.read_state(path, self.config)
        self.assertEqual(path.read_text(), "{")

    def test_init_does_not_reset_existing_job(self):
        job = self.root / "data/training/job"
        job.mkdir(parents=True)
        (job / "state.json").write_text('{"attempts": [1, 2, 3]}')
        with patch.object(recovery, "ROOT", self.root), patch("sys.argv", ["autoresume", str(job), "--init"]):
            with self.assertRaises(FileExistsError):
                recovery.main()
        self.assertEqual(json.loads((job / "state.json").read_text())["attempts"], [1, 2, 3])

    def test_new_job_target_does_not_modify_completed_ledger(self):
        old = self.root / "data/training/R01"
        old.mkdir(parents=True)
        atomic_json(old / "state.json", {"status": "complete", "attempts": [1, 2, 3]})
        previous = (old / "state.json").read_bytes()
        new = self.root / "data/training/R02"
        argv = ["autoresume", str(new), "--init", "--source-dir", "data/training/source2000",
                "--target-step", "10000", "--run-prefix", "E007_long"]
        with patch.object(recovery, "ROOT", self.root), patch("sys.argv", argv):
            self.assertEqual(recovery.main(), 0)
        config = json.loads((new / "job.json").read_text())
        self.assertEqual(config["target_step"], 10000)
        self.assertEqual(config["source_dirs"], ["data/training/source2000"])
        self.assertEqual(config["max_attempts"], 3)
        self.assertEqual(recovery.read_state(new / "state.json", config)["attempts"], [])
        self.assertEqual((old / "state.json").read_bytes(), previous)

    def test_job_target_cannot_be_changed_on_run(self):
        with patch("sys.argv", ["autoresume", str(self.root), "--target-step", "10000"]):
            with self.assertRaises(SystemExit):
                recovery.main()

    def test_new_job_rejects_escaping_source_or_run_prefix(self):
        for source, prefix in [("../outside", "valid"), ("data/training/source", "../outside")]:
            job = self.root / "data/training" / ("bad_source" if source.startswith("..") else "bad_prefix")
            argv = ["autoresume", str(job), "--init", "--source-dir", source, "--run-prefix", prefix]
            with patch.object(recovery, "ROOT", self.root), patch("sys.argv", argv):
                with self.assertRaises(SystemExit):
                    recovery.main()
            self.assertFalse((job / "job.json").exists())

    def test_second_supervisor_cannot_launch(self):
        job = self.root / "data/training/job"
        job.mkdir(parents=True)
        with (job / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(recovery, "ROOT", self.root), patch("sys.argv", ["autoresume", str(job)]), patch.object(recovery, "run_job") as launch:
                self.assertEqual(recovery.main(), 2)
                launch.assert_not_called()

    def test_controller_error_persists_block_without_resetting_budget(self):
        job = self.root / "data/training/job"
        job.mkdir(parents=True)
        atomic_json(job / "job.json", self.config)
        atomic_json(job / "state.json", self.ledger([600]))
        with patch.object(recovery, "ROOT", self.root), patch("sys.argv", ["autoresume", str(job)]), patch.object(recovery, "run_job", side_effect=OSError("injected diagnostic error")):
            self.assertEqual(recovery.main(), 2)
        saved = recovery.read_state(job / "state.json", self.config)
        self.assertEqual(saved["status"], "blocked")
        self.assertEqual(len(saved["attempts"]), 1)

    def checkpoint(self, directory, step, environments=64):
        save_checkpoint(self.root / directory, {
            "policy_state_dict": {"weight": torch.tensor([1.0])},
            "value_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {}},
            "state": SimpleNamespace(global_step=step, cur_episode_length=torch.zeros(environments)),
        })

    def test_corrupt_checkpoint_fallback_and_registered_lineage(self):
        self.checkpoint("source", 575)
        self.checkpoint("source", 600)
        (self.root / "source/model_step_000600.pt").write_bytes(b"corrupt")
        self.checkpoint("unrelated", 1900)
        with patch.object(recovery, "ROOT", self.root):
            step, path, rejected = recovery.best_checkpoint(self.config, self.ledger([]))
        self.assertEqual(step, 575)
        self.assertIn("source", path)
        self.assertEqual(len(rejected), 1)

    def test_latest_registered_attempt_is_selected(self):
        self.checkpoint("source", 600)
        self.checkpoint("run0", 625)
        with patch.object(recovery, "ROOT", self.root):
            self.assertEqual(recovery.best_checkpoint(self.config, self.ledger([600]))[0], 625)

    def test_environment_count_mismatch_blocks(self):
        self.checkpoint("source", 600, environments=16)
        with patch.object(recovery, "ROOT", self.root), self.assertRaises(ValueError):
            recovery.best_checkpoint(self.config, self.ledger([]))

    def test_normal_training_error_is_not_retried(self):
        state = self.ledger([600])
        self.assertEqual(recovery.finish_attempt(self.config, state, state["attempts"][0], 625, 1, None), 2)
        self.assertEqual(state["status"], "blocked")

    def test_killed_training_can_retry_with_budget_retained(self):
        state = self.ledger([600])
        self.assertEqual(recovery.finish_attempt(self.config, state, state["attempts"][0], 625, -9, None), 75)
        self.assertEqual(len(state["attempts"]), 1)

    def test_temperature_blocks_even_with_sigkill(self):
        state = self.ledger([600])
        self.assertEqual(recovery.finish_attempt(self.config, state, state["attempts"][0], 625, -9, "temperature_stop"), 2)

    def thermal_ledger(self, steps):
        state = self.ledger(steps)
        state.update(status="blocked", reason="temperature_stop", verified_step=1800)
        return state

    def test_explicit_rearm_preserves_attempts_and_config(self):
        state = self.thermal_ledger([600, 950])
        attempts = json.loads(json.dumps(state["attempts"]))
        saved_hash = state["config_sha256"]
        recovery.rearm_temperature(self.config, state, "User authorized reduced CPU quota", 100)
        self.assertEqual(state["status"], "armed")
        self.assertEqual(state["attempts"], attempts)
        self.assertEqual(state["config_sha256"], saved_hash)
        self.assertEqual(state["operator_authorizations"][0]["previous_reason"], "temperature_stop")
        with self.assertRaises(ValueError):
            recovery.rearm_temperature(self.config, state, "Repeated authorization", 101)

    def test_rearm_cannot_extend_budget_or_expiry(self):
        for steps, now in [([600, 950, 1800], 100), ([600, 950], 10000)]:
            state = self.thermal_ledger(steps)
            with self.assertRaises(ValueError):
                recovery.rearm_temperature(self.config, state, "Authorized", now)
            self.assertEqual(state["status"], "blocked")

    def test_rearm_rejects_other_stop_and_missing_authorization(self):
        state = self.thermal_ledger([600, 950])
        with self.assertRaises(ValueError):
            recovery.rearm_temperature(self.config, state, " ", 100)
        state["reason"] = "controller_error"
        with self.assertRaises(ValueError):
            recovery.rearm_temperature(self.config, state, "Authorized", 100)

    def test_rejected_cli_rearm_leaves_ledger_unchanged(self):
        job = self.root / "data/training/job"
        job.mkdir(parents=True)
        atomic_json(job / "job.json", self.config)
        atomic_json(job / "state.json", self.thermal_ledger([600, 950]))
        previous = (job / "state.json").read_bytes()
        with patch.object(recovery, "ROOT", self.root), patch("sys.argv", [
                "autoresume", str(job), "--rearm-temperature", "Authorized but expired"]):
            self.assertEqual(recovery.main(), 2)
        self.assertEqual((job / "state.json").read_bytes(), previous)

    def test_complete_requires_success_exit(self):
        state = self.ledger([600])
        self.assertEqual(recovery.finish_attempt(self.config, state, state["attempts"][0], 2000, 0, None), 0)
        self.assertEqual(state["status"], "complete")

    def test_expired_monitor_terminates_real_child(self):
        child = subprocess.Popen(["sleep", "30"], start_new_session=True)
        config = dict(self.config, expires_at=0, max_gpu_c=85, max_cpu_c=90, sample_seconds=0.01)
        health = {"gpus": [], "temperatures_c": {}, "gpu_query": {"returncode": 0}}
        with patch.object(recovery, "sample", return_value=health):
            code, reason = recovery.monitor_child(child, self.root, config)
        self.assertEqual(reason, "expired")
        self.assertEqual(code, -signal.SIGTERM)

    def test_monitor_exception_cleans_real_child(self):
        child = subprocess.Popen(["sleep", "30"], start_new_session=True)
        with patch.object(recovery, "sample", side_effect=OSError("injected telemetry failure")):
            with self.assertRaises(OSError):
                recovery.monitor_child(child, self.root, self.config)
        self.assertIsNotNone(child.poll())

    def test_until_target_monitor_retains_temperature_stop(self):
        child = subprocess.Popen(["sleep", "30"], start_new_session=True)
        config = dict(self.config, expires_at=None, max_attempts=None, max_gpu_c=85,
                      max_cpu_c=90, sample_seconds=0.01)
        health = {"gpus": [], "temperatures_c": {"x86_pkg_temp": 90},
                  "gpu_query": {"returncode": 0}}
        with patch.object(recovery, "sample", return_value=health):
            code, reason = recovery.monitor_child(child, self.root, config)
        self.assertEqual(reason, "temperature_stop")
        self.assertEqual(code, -signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
