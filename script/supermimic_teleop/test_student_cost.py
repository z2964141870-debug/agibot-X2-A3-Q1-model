"""Crash/unknown reservations cannot be silently released or double dispatched."""

import json
from pathlib import Path
import tempfile
import unittest

from student_cost import charged, reserve, settle


OWNER = dict(pid=123, start_ticks="456", boot="fixture_boot")


class CostTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/"cost.json"

    def admit(self, run="/fixture/r1", stage="r1", started=0):
        return reserve(self.path, run_id=run, stage=stage, owner=OWNER,
                       gpu="GPU-fixture", commit="a"*40, started_ns=started)

    def test_unknown_run_keeps_full_charge_and_blocks_another_dispatch(self):
        self.admit()
        value = json.loads(self.path.read_text())
        self.assertEqual(charged(value), 2700)
        with self.assertRaisesRegex(RuntimeError, "earlier run remains reserved"):
            self.admit("/fixture/r2", "r2")
        self.assertEqual(json.loads(self.path.read_text()), value)

    def test_only_reaped_worker_releases_unused_budget_and_ceil_cost_is_preserved(self):
        self.admit()
        with self.assertRaises(ValueError):
            settle(self.path, run_id="/fixture/r1", owner=OWNER,
                   cleanup=dict(worker_reaped=False, worker_return_code=-15), ended_ns=5_000_000_001)
        result = settle(self.path, run_id="/fixture/r1", owner=OWNER,
            cleanup=dict(worker_reaped=True, worker_return_code=-15), ended_ns=5_000_000_001)
        self.assertEqual(result["charged_seconds"], 6)
        self.assertEqual(charged(json.loads(self.path.read_text())), 6)
        self.admit("/fixture/r2", "r2", started=10_000_000_000)
        self.assertEqual(charged(json.loads(self.path.read_text())), 4506)

    def test_ended_run_is_single_use_and_pid_identity_cannot_change(self):
        self.admit()
        with self.assertRaises(ValueError):
            settle(self.path, run_id="/fixture/r1", owner=dict(OWNER, start_ticks="789"),
                   cleanup=dict(worker_reaped=True, worker_return_code=0), ended_ns=1)
        settle(self.path, run_id="/fixture/r1", owner=OWNER,
               cleanup=dict(worker_reaped=True, worker_return_code=0), ended_ns=1)
        with self.assertRaisesRegex(ValueError, "Single-use"):
            self.admit()

    def test_overrun_is_charged_honestly_and_prevents_overspending(self):
        self.admit(stage="r3")
        row = settle(self.path, run_id="/fixture/r1", owner=OWNER,
            cleanup=dict(worker_reaped=True, worker_return_code=-9), ended_ns=14_400_000_000_001)
        self.assertTrue(row["over_reserved_budget"])
        self.assertEqual(row["charged_seconds"], 14401)
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            self.admit("/fixture/new")

    def test_stage_cap_cannot_be_reset_by_creating_a_second_run(self):
        self.admit()
        settle(self.path, run_id="/fixture/r1", owner=OWNER,
            cleanup=dict(worker_reaped=True, worker_return_code=0), ended_ns=2_700_000_000_000)
        with self.assertRaisesRegex(RuntimeError, "Stage budget"):
            self.admit("/fixture/r1_again")

    def test_a_separate_followup_can_only_reserve_the_remaining_stage_budget(self):
        self.admit()
        settle(self.path, run_id="/fixture/r1", owner=OWNER,
            cleanup=dict(worker_reaped=True, worker_return_code=1), ended_ns=10_000_000_000)
        row = self.admit("/fixture/r1_followup", started=11_000_000_000)
        self.assertEqual(row["reserved_seconds"], 2690)
        self.assertEqual(charged(json.loads(self.path.read_text())), 2700)


if __name__ == "__main__":
    unittest.main()
