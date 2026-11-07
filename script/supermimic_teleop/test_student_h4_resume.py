"""Regression gates for actual H4 media setup and preservation of prior physics."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from student_h4_review_resume import ResumeBudget,execute,remaining_phases,spent_seconds
from common import FIXED


class H4ResumeTests(unittest.TestCase):
    def test_actual_media_branch_sets_runtime_paths_before_rendering(self):
        args = SimpleNamespace(child="media",config=Path("config"),runtime_config=Path("runtime"),
            output=Path("out"),event_kind="nominal",variant="student",horizon=4,clip=FIXED[0])
        calls = []
        runtime = dict(primebot_python="/primebot/python")
        with patch("student_h4_review_resume.read",side_effect=[{},runtime]), \
                patch("student_h4_review_resume.runtime_paths",side_effect=lambda r:calls.append(("paths",r))), \
                patch("student_h4_review_resume.media",side_effect=lambda *a:calls.append(("media",a[2]))):
            execute(args)
        self.assertEqual(calls,[("paths",runtime),("media",runtime)])

    def test_only_six_new_replays_and_all_seven_media_phases_remain(self):
        phases = remaining_phases()
        self.assertEqual(sum(p == "replay" for c,p in phases),6)
        self.assertEqual(sum(p == "media" for c,p in phases),7)
        self.assertNotIn((("nominal",FIXED[0]),"replay"),phases)

    def test_failed_charge_cannot_be_reset_or_unrelated_failure_reused(self):
        first = "nominal_student_h4_"+FIXED[0]
        value = dict(status="FAILED",stage="H4_SPECIALIST01_MATCHED_CLOSED_LOOP",phase=first+"_media",cases=[],
            error="RuntimeError: H4 review phase failed: "+first+"_media",
            cost_settlement=dict(state="SETTLED",charged_seconds=101),
            started_utc="2026-11-08T04:00:00+08:00",ended_utc="2026-11-08T04:01:40+08:00")
        log = "ModuleNotFoundError: No module named 'video'"
        self.assertEqual(spent_seconds(value,log),101)
        for bad in (dict(value,status="RUNNING"),dict(value,cost_settlement=dict(state="RESERVED")),
                dict(value,ended_utc="2026-11-08T04:40:00+08:00")):
            with self.assertRaises(RuntimeError):
                spent_seconds(bad,log)
        with self.assertRaises(RuntimeError):
            spent_seconds(value,"another error")

    def test_original1800_second_wall_budget_remains_cumulative(self):
        budget = object.__new__(ResumeBudget)
        budget.spent,budget.started = 101.,100.
        with patch("student_h4_review_resume.time.monotonic",return_value=1800.):
            with self.assertRaisesRegex(RuntimeError,"Cumulative H4"):
                budget.check()


if __name__ == "__main__":
    unittest.main()
