"""Exercise total-memory reclaim, ownership and unchanged training safeguards."""

from contextlib import contextmanager
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

from guard import Budget
from guard_phuma_batch import MemoryLimit
from memory_pressure import (GIB, TRIGGER, TARGET, PressureMemoryBudget, PressureReclaimer,
                             accounting, eligible_files, paused_descendants, supervisor_ancestors)
from memory_watch import process


@contextmanager
def no_pause(*args):
    yield [dict(pid=123, start_ticks="456")]


class PressureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.cgroup = self.root / "cgroup"
        self.cgroup.mkdir()
        (self.root / "data").mkdir()
        self.file = self.root / "data/frozen.npz"
        self.file.write_bytes(b"preserved reference" * 256)
        self.output = self.root / "reclaim.json"
        self.write_memory(29 * GIB)
        supervisors = patch("memory_pressure.supervisor_ancestors", return_value=[])
        supervisors.start()
        self.addCleanup(supervisors.stop)

    def write_memory(self, total, cache=20 * GIB, oom=0, under_oom=0, limit=32 * GIB):
        (self.cgroup / "memory.usage_in_bytes").write_text(str(total))
        (self.cgroup / "memory.limit_in_bytes").write_text(str(limit))
        (self.cgroup / "memory.stat").write_text(
            f"total_cache {cache}\ntotal_rss {max(0,total-cache)}\ntotal_shmem 0\n"
            "total_dirty 0\ntotal_writeback 0\n")
        (self.cgroup / "memory.oom_control").write_text(f"oom_kill {oom}\nunder_oom {under_oom}\n")

    def budget(self, **kwargs):
        return PressureMemoryBudget(Mock(), TRIGGER, 0, root=self.root, output=self.output,
                                    preferred=[self.file], cgroup=self.cgroup, **kwargs)

    def release_to(self, total):
        def release(*args):
            self.write_memory(total, cache=18 * GIB)
            return 4096, True
        return release

    def test_large_cache_below30_does_not_trigger_cleanup(self):
        self.write_memory(29 * GIB, cache=25 * GIB)
        with patch("memory_pressure.cache.release") as release:
            value = self.budget().memory()
        release.assert_not_called()
        self.assertEqual(value["accounting"]["immediate_headroom_bytes"], 3 * GIB)
        self.assertEqual(value["accounting"]["guard_headroom_bytes"], GIB)
        self.assertFalse(self.output.exists())

    def test_exact30_trigger_reclaims_and_preserves_file_contents(self):
        self.write_memory(TRIGGER)
        observer = Mock()
        before = self.file.read_bytes()
        with patch("memory_pressure.Path.cwd", return_value=self.root), \
             patch("memory_pressure.paused_descendants", no_pause), \
             patch("memory_pressure.cache.active_files", return_value=(set(), [], [])), \
             patch("memory_pressure.subtree", return_value=[]), \
             patch("memory_pressure.cache.release", side_effect=self.release_to(TARGET)) as release:
            value = self.budget(observer=observer).memory()
        observer.record.assert_called_once_with()
        release.assert_called_once()
        self.assertEqual(value["shared_bytes"], TARGET)
        self.assertEqual(value["pressure_reclaim_events"], 1)
        self.assertEqual(self.file.read_bytes(), before)
        receipt = json.loads(self.output.read_text())["events"][0]
        self.assertEqual(receipt["status"], "RECLAIMED_BELOW_ORIGINAL_LIMIT")
        self.assertEqual(receipt["observed_total_reduction_bytes"], 2 * GIB)
        self.assertFalse(receipt["global_drop_caches"])
        self.assertFalse(receipt["workers_restarted"])
        self.assertEqual(receipt["files_deleted"], 0)

    def test_memory_crossing_during_storage_check_also_reclaims(self):
        budget = self.budget()
        budget.storage.check.side_effect = lambda: (self.write_memory(TRIGGER) or {})
        with patch("memory_pressure.Path.cwd", return_value=self.root), \
             patch("memory_pressure.paused_descendants", no_pause), \
             patch("memory_pressure.cache.active_files", return_value=(set(), [], [])), \
             patch("memory_pressure.subtree", return_value=[]), \
             patch("memory_pressure.cache.release", side_effect=self.release_to(TARGET)):
            result = budget.check()
        self.assertEqual(result["shared_memory"]["shared_bytes"], TARGET)

    def test_later_resident_chunk_counts_the_file_once(self):
        self.write_memory(TRIGGER)
        with patch("memory_pressure.Path.cwd", return_value=self.root), \
             patch("memory_pressure.paused_descendants", no_pause), \
             patch("memory_pressure.cache.active_files", return_value=(set(), [], [])), \
             patch("memory_pressure.subtree", return_value=[]), \
             patch("memory_pressure.cache.CHUNK", 4096), \
             patch("memory_pressure.cache.release", side_effect=[(0, False), (4096, True)]):
            with self.assertRaises(MemoryLimit):
                self.budget().memory()
        row = json.loads(self.output.read_text())["events"][0]
        self.assertEqual(row["files_advised"], 1)

    def test_failed_final_measurement_preserves_the_original_error(self):
        self.write_memory(TRIGGER)
        reclaimer = self.budget().reclaimer
        before = accounting(self.cgroup)
        with patch("memory_pressure.Path.cwd", return_value=self.root), \
             patch("memory_pressure.paused_descendants", side_effect=RuntimeError("pause fixture")), \
             patch("memory_pressure.accounting", side_effect=[before, OSError("measurement fixture")]):
            with self.assertRaisesRegex(RuntimeError, "pause fixture"):
                reclaimer.reclaim()
        row = json.loads(self.output.read_text())["events"][0]
        self.assertIn("pause fixture", row["error"])
        self.assertIn("measurement fixture", row["after_measurement_error"])
        self.assertNotIn("observed_total_reduction_bytes", row)

    def test_ineffective_cleanup_still_stops_at_original_limit(self):
        self.write_memory(TRIGGER)
        events = []

        @contextmanager
        def paused(*args):
            events.append("paused")
            try:
                yield []
            finally:
                events.append("resumed")

        with patch("memory_pressure.Path.cwd", return_value=self.root), \
             patch("memory_pressure.paused_descendants", paused), \
             patch("memory_pressure.cache.active_files", return_value=(set(), [], [])), \
             patch("memory_pressure.subtree", return_value=[]), \
             patch("memory_pressure.cache.release", return_value=(4096, True)):
            with self.assertRaisesRegex(MemoryLimit, "remains at30GiB"):
                self.budget().memory()
        self.assertEqual(events, ["paused", "resumed"])
        self.assertEqual(json.loads(self.output.read_text())["events"][0]["status"], "FAILED")

    def test_oom_change_never_attempts_cleanup(self):
        for oom, under in ((1, 0), (0, 1)):
            with self.subTest(oom=oom, under=under):
                self.write_memory(TRIGGER, oom=oom, under_oom=under)
                budget = self.budget()
                with patch.object(budget.reclaimer, "reclaim") as reclaim:
                    with self.assertRaisesRegex(MemoryLimit, "OOM state changed"):
                        budget.memory()
                reclaim.assert_not_called()

    def test_short_repeated_pressure_does_not_create_a_cleanup_loop(self):
        budget = self.budget()
        budget.reclaimer.last_attempt = time.monotonic()
        self.write_memory(TRIGGER)
        with patch("memory_pressure.paused_descendants") as pause:
            with self.assertRaisesRegex(MemoryLimit, "within30s"):
                budget.memory()
        pause.assert_not_called()

    def test_foreign_project_compute_stops_before_any_advice(self):
        self.write_memory(TRIGGER)
        with patch("memory_pressure.Path.cwd", return_value=self.root), \
             patch("memory_pressure.paused_descendants", no_pause), \
             patch("memory_pressure.cache.active_files", return_value=(set(), [dict(pid=999)], [])), \
             patch("memory_pressure.subtree", return_value=[]), \
             patch("memory_pressure.cache.release") as release:
            with self.assertRaisesRegex(MemoryLimit, "Unowned project compute"):
                self.budget().memory()
        release.assert_not_called()

    def test_wrong_root_stops_before_pause(self):
        self.write_memory(TRIGGER)
        with patch("memory_pressure.Path.cwd", return_value=self.root / "other"), \
             patch("memory_pressure.paused_descendants") as pause:
            with self.assertRaisesRegex(MemoryLimit, "unique physical"):
                self.budget().memory()
        pause.assert_not_called()

    def test_existing_receipts_and_original_limits_cannot_be_overwritten(self):
        self.output.write_text("original")
        with self.assertRaises(FileExistsError):
            self.budget()
        self.assertEqual(self.output.read_text(), "original")
        for maximum, oom in ((TRIGGER + 1, 0), (TRIGGER, 1)):
            with self.assertRaisesRegex(ValueError, "preserve"):
                PressureMemoryBudget(Mock(), maximum, oom, root=self.root, output=self.root / "new")

    def test_accounting_requires_real_capacity_and_keeps_cache_separate(self):
        value = accounting(self.cgroup)
        self.assertEqual(value["total_bytes"], 29 * GIB)
        self.assertEqual(value["immediate_headroom_bytes"], 3 * GIB)
        self.write_memory(TRIGGER, limit=TRIGGER)
        with self.assertRaisesRegex(MemoryLimit, "measured container capacity"):
            accounting(self.cgroup)

    def test_preferred_candidates_skip_symlinks_protected_and_outside(self):
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        link = self.root / "data/link"
        link.symlink_to(outside)
        item = self.file.stat()
        protected = {(item.st_dev, item.st_ino)}
        self.assertEqual(list(eligible_files(self.root, [outside, link, self.file], protected,
                                             time.monotonic() + 5)), [])
        self.assertEqual(list(eligible_files(self.root, [self.file], set(), time.monotonic() - 1)), [])

    def test_device_failure_precedes_pressure_cleanup(self):
        config = dict(adapted_source=str(self.root), memory_pressure_reclaim=True,
                      expected_boot="boot", expected_gpu="expected")
        with patch("guard.StorageBudget"), patch("memory_pressure.PressureMemoryBudget") as inner, \
             patch("guard.Path.read_text", return_value="boot"), \
             patch("guard.subprocess.check_output", return_value="wrong,25,None,1"):
            budget = Budget(config, self.root, 60, GIB)
            with self.assertRaisesRegex(RuntimeError, "GPU identity"):
                budget.check()
            inner.return_value.check.assert_not_called()

    def test_device_health_is_checked_again_after_cleanup(self):
        config = dict(adapted_source=str(self.root), memory_pressure_reclaim=True,
                      expected_boot="boot", expected_gpu="expected")
        with patch("guard.StorageBudget"), patch("memory_pressure.PressureMemoryBudget") as inner, \
             patch("guard.Path.read_text", return_value="boot"), \
             patch("guard.subprocess.check_output", side_effect=["expected,25,None,1", "expected,25,Reset,1"]):
            inner.return_value.reclaimer.events = 0
            def cleaned():
                inner.return_value.reclaimer.events = 1
                return {}
            inner.return_value.check.side_effect = cleaned
            budget = Budget(config, self.root, 60, GIB)
            with self.assertRaisesRegex(RuntimeError, "GPU identity or health"):
                budget.check()

    def test_v04_has_total_policy_and_no_cache_admission(self):
        config = json.loads((Path(__file__).parent / "conditional_shared_v04_20261104.json").read_text())
        self.assertIs(config["memory_pressure_reclaim"], True)
        self.assertNotIn("profile_maximum_initial_cache_bytes", config)
        self.assertFalse(config["automatic_formal_retry"])
        self.assertTrue(config["record_memory_breakdown"])


class PauseTests(unittest.TestCase):
    def test_failed_process_read_closes_the_pidfd_without_signalling(self):
        owner = dict(pid=10, start_ticks="100", state="S")
        child = dict(pid=11, start_ticks="200", state="S")
        with patch("memory_pressure.os.pidfd_open", return_value=123, create=True), \
             patch("memory_pressure.signal.pidfd_send_signal", create=True) as send, \
             patch("memory_pressure.os.close") as close, \
             patch("memory_pressure.process", side_effect=[owner, owner, OSError("proc fixture")]), \
             patch("memory_pressure.subtree", return_value=[owner, child]):
            with self.assertRaisesRegex(OSError, "proc fixture"):
                with paused_descendants(Path("/proc"), 10, time.monotonic() + 2):
                    self.fail("Must not enter the cleanup body")
        send.assert_not_called()
        close.assert_called_once_with(123)

    def test_pid_reuse_never_sends_stop_or_continue(self):
        owner = dict(pid=10, start_ticks="100", state="S")
        child = dict(pid=11, start_ticks="200", state="S")
        replacement = dict(child, start_ticks="201")
        with patch("memory_pressure.os.pidfd_open", return_value=123, create=True), \
             patch("memory_pressure.signal.pidfd_send_signal", create=True) as send, \
             patch("memory_pressure.os.close") as close, \
             patch("memory_pressure.process", side_effect=[owner, owner, replacement]), \
             patch("memory_pressure.subtree", return_value=[owner, child]):
            with self.assertRaisesRegex(MemoryLimit, "PID changed before pause"):
                with paused_descendants(Path("/proc"), 10, time.monotonic() + 2):
                    self.fail("Must not enter the cleanup body")
        send.assert_not_called()
        close.assert_called_once_with(123)

    def test_previously_stopped_child_is_not_resumed(self):
        owner = dict(pid=10, start_ticks="100", state="S")
        child = dict(pid=11, start_ticks="200", state="T")
        with patch("memory_pressure.os.pidfd_open", return_value=123, create=True), \
             patch("memory_pressure.signal.pidfd_send_signal", create=True) as send, \
             patch("memory_pressure.os.close"), \
             patch("memory_pressure.process", side_effect=[owner, owner, child]), \
             patch("memory_pressure.subtree", return_value=[owner, child]):
            with paused_descendants(Path("/proc"), 10, time.monotonic() + 2) as paused:
                self.assertEqual(paused, [])
        send.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open")
                         and hasattr(signal, "pidfd_send_signal"), "Linux pidfd integration runs on the server")
    def test_real_owned_child_resumes_after_cleanup_exception(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with paused_descendants(Path("/proc"), os.getpid(), time.monotonic() + 2) as paused:
                    self.assertIn(child.pid, [row["pid"] for row in paused])
                    self.assertIn(process(Path("/proc"), child.pid)["state"], {"T", "t"})
                    raise RuntimeError("fixture failure")
            deadline = time.monotonic() + 2
            while process(Path("/proc"), child.pid)["state"] in {"T", "t"} and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertNotIn(process(Path("/proc"), child.pid)["state"], {"T", "t"})
        finally:
            child.terminate()
            child.wait(timeout=5)


class SupervisorTests(unittest.TestCase):
    def rows(self):
        return (dict(pid=10, ppid=9, start_ticks="100"),
                dict(pid=9, ppid=1, start_ticks="99"))

    def test_waiting_controller_is_recognized_with_start_tick(self):
        root = Path("/opt/data/private/ZHY")
        owner, parent = self.rows()
        for script in ("script/q1_five_stage/controller.py", "script/q1_five_stage/shared_review.py"):
            mode = "\0run" if script.endswith("controller.py") else ""
            command = f"python\0-u\0{script}{mode}\0".encode()
            with self.subTest(script=script), \
                 patch("memory_pressure.process", side_effect=[owner, parent, parent]), \
                 patch("memory_pressure.Path.read_bytes", return_value=command), \
                 patch("memory_pressure.os.readlink", return_value=str(root)):
                rows = supervisor_ancestors(Path("/proc"), 10, root)
            self.assertEqual(rows, [dict(pid=9, start_ticks="99", script=str(root / script))])

    def test_unrelated_python_or_audit_is_not_a_supervisor(self):
        root = Path("/opt/data/private/ZHY")
        owner, parent = self.rows()
        for command in (b"python\0-u\0script/morphmimic_q1/train_native.py\0",
                        b"python\0-u\0script/q1_five_stage/controller.py\0audit\0"):
            with self.subTest(command=command), patch("memory_pressure.process", side_effect=[owner, parent]), \
                 patch("memory_pressure.Path.read_bytes", return_value=command), \
                 patch("memory_pressure.os.readlink", return_value=str(root)):
                self.assertEqual(supervisor_ancestors(Path("/proc"), 10, root), [])

    def test_waiting_controller_reuse_is_rejected(self):
        owner, parent = self.rows()
        with patch("memory_pressure.process", side_effect=[owner, parent, dict(parent, start_ticks="101")]), \
             patch("memory_pressure.Path.read_bytes", return_value=b"python\0-u\0script/q1_five_stage/controller.py\0run\0"), \
             patch("memory_pressure.os.readlink", return_value="/opt/data/private/ZHY"):
            with self.assertRaisesRegex(MemoryLimit, "controller PID identity"):
                supervisor_ancestors(Path("/proc"), 10, Path("/opt/data/private/ZHY"))


if __name__ == "__main__":
    unittest.main()
