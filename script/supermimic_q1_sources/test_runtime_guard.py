import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from runtime_guard import StorageBudget, StorageLimit, allocated_tree, guarded_run


class RuntimeGuardTests(unittest.TestCase):
    def test_hardlinks_are_unique_and_external_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            root.mkdir()
            payload = root / "payload"
            payload.write_bytes(b"x" * 8192)
            before = allocated_tree(root)
            os.link(payload, root / "hardlink")
            after = allocated_tree(root)
            self.assertEqual(before["unique_allocated_bytes"], after["unique_allocated_bytes"])
            self.assertEqual(before["unique_inodes"], after["unique_inodes"])
            external = Path(folder) / "outside"
            external.mkdir()
            (external / "large").write_bytes(b"x" * (1024 * 1024))
            (root / "reference").symlink_to(external, target_is_directory=True)
            final = allocated_tree(root)
            self.assertEqual(final["symlinks_not_followed"], 1)
            self.assertLess(final["unique_allocated_bytes"], 128 * 1024)

    def test_preflight_refuses_to_start_when_budget_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = StorageBudget(root, root, 0, 0, 0)
            (root / "occupied").write_bytes(b"x" * 4096)
            with self.assertRaises(StorageLimit):
                guarded_run([sys.executable, "-c", "raise RuntimeError('must not start')"],
                            root, budget, root / "status.json", root / "worker.log")
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "STOPPED_STORAGE")
            self.assertFalse(status["cleanup"]["worker_started"])
            self.assertFalse((root / "worker.log").exists())

    def test_free_reserve_and_source_cap_are_checked_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "occupied").write_bytes(b"x" * 4096)
            budget = StorageBudget(root, root, 10, 1024 ** 2, 1024 ** 2)
            with patch("runtime_guard.shutil.disk_usage", return_value=SimpleNamespace(free=5)):
                with self.assertRaisesRegex(StorageLimit, "Free space"):
                    budget.check()
            budget.minimum_free_bytes = 0
            budget.maximum_source_bytes = 0
            with self.assertRaisesRegex(StorageLimit, "Source storage"):
                budget.check()

    def test_exception_during_active_worker_terminates_and_reaps_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = Mock()
            budget.check.side_effect = [{"first": True}, StorageLimit("injected storage failure")]
            with self.assertRaises(StorageLimit):
                guarded_run([sys.executable, "-c", "import time; time.sleep(60)"], root,
                            budget, root / "status.json", root / "worker.log",
                            check_interval_s=.1, grace_s=.2)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "STOPPED_STORAGE")
            self.assertTrue(status["cleanup"]["worker_reaped"])
            with self.assertRaises(ProcessLookupError):
                os.kill(status["worker_pid"], 0)

    def test_broken_status_io_also_cleans_up_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = Mock()
            budget.check.return_value = {}
            children, original = [], subprocess.Popen

            def start(*args, **kwargs):
                child = original(*args, **kwargs)
                children.append(child)
                return child

            try:
                with patch("runtime_guard.subprocess.Popen", side_effect=start), patch(
                        "runtime_guard.save", side_effect=OSError("injected status I/O failure")):
                    with self.assertRaises(OSError):
                        guarded_run([sys.executable, "-c", "import time; time.sleep(60)"], root,
                                    budget, root / "status.json", root / "worker.log",
                                    check_interval_s=.1, grace_s=.2)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].returncode)
                with self.assertRaises(ProcessLookupError):
                    os.kill(children[0].pid, 0)
            finally:
                for child in children:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()

    def test_stalled_progress_terminates_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = Mock()
            budget.check.return_value = {}
            with self.assertRaisesRegex(RuntimeError, "no progress"):
                guarded_run([sys.executable, "-c", "import time; time.sleep(60)"], root,
                            budget, root / "status.json", root / "worker.log",
                            check_interval_s=.1, grace_s=.2,
                            progress_path=root / "absent_progress.json", maximum_idle_s=.15)
            status = json.loads((root / "status.json").read_text())
            self.assertTrue(status["cleanup"]["worker_reaped"])
            with self.assertRaises(ProcessLookupError):
                os.kill(status["worker_pid"], 0)

    def test_signal_to_guard_stops_and_reaps_owned_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            goal = {"storage": {"minimum_free_bytes": 0,
                               "maximum_experiment_unique_allocated_bytes": 1024 ** 2,
                               "maximum_new_source_unique_allocated_bytes": 1024 ** 2}}
            (root / "goal.json").write_text(json.dumps(goal))
            guard = subprocess.Popen([sys.executable, str(Path(__file__).with_name("runtime_guard.py")),
                                      "--goal-config", str(root / "goal.json"),
                                      "--experiment-root", str(root), "--source-root", str(root),
                                      "--workdir", str(root), "--status", str(root / "status.json"),
                                      "--log", str(root / "worker.log"), "--", sys.executable,
                                      "-c", "import time; time.sleep(60)"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while not (root / "status.json").exists() and time.monotonic() < deadline:
                    if guard.poll() is not None:
                        self.fail(guard.stderr.read().decode())
                    time.sleep(.03)
                self.assertTrue((root / "status.json").exists())
                guard.terminate()
                self.assertEqual(guard.wait(timeout=10), 1)
                status = json.loads((root / "status.json").read_text())
                self.assertEqual(status["status"], "STOPPED_SIGNAL")
                self.assertTrue(status["cleanup"]["worker_reaped"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(status["worker_pid"], 0)
            finally:
                if guard.poll() is None:
                    guard.kill()
                    guard.wait()
                guard.stderr.close()

    def test_nonzero_exit_and_single_use_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = StorageBudget(root, root, 0, 1024 ** 2, 1024 ** 2)
            result = guarded_run([sys.executable, "-c", "raise SystemExit(7)"], root,
                                 budget, root / "status.json", root / "worker.log")
            self.assertEqual(result["status"], "WORKER_FAILED")
            self.assertEqual(result["cleanup"]["worker_return_code"], 7)
            with self.assertRaises(FileExistsError):
                guarded_run([sys.executable, "-c", "pass"], root, budget,
                            root / "status.json", root / "worker.log")


if __name__ == "__main__":
    unittest.main()
