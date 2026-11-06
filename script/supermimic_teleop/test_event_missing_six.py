"""The follow-up must preserve78 and never reschedule a completed case."""

import copy
import unittest

from event_missing_six import MISSING, cases, partition


def preserved():
    keys = [key for key in cases() if key not in MISSING]
    return dict(stage="PARTIAL_SAVED_EVENT_ANALYSIS", source_campaign_status="FAILED",
        source_final_gate_passed=False, partial_source_reaudit="PASSED",
        rows=[dict(event=e, method=m, horizon_frames=h, clip=c) for e,m,h,c in keys],
        missing=[dict(event=e, method=m, horizon_frames=h, clip=c) for e,m,h,c in MISSING])


class MissingSixTests(unittest.TestCase):
    def test_only_the_exact_six_are_scheduled(self):
        reused, scheduled = partition(preserved())
        self.assertEqual(len(reused), 78)
        self.assertEqual(scheduled, MISSING)
        self.assertFalse(set(reused)&set(scheduled))
        self.assertEqual(set(reused)|set(scheduled), set(cases()))

    def test_duplicate_or_other_missing_cases_are_rejected(self):
        value = preserved()
        value["rows"][1] = copy.deepcopy(value["rows"][0])
        with self.assertRaises(RuntimeError):
            partition(value)
        value = preserved()
        value["missing"][0]["event"] = "stop"
        with self.assertRaises(RuntimeError):
            partition(value)

    def test_original_failure_cannot_be_relabelled_completed(self):
        value = preserved()
        value["source_campaign_status"] = "COMPLETED"
        with self.assertRaises(RuntimeError):
            partition(value)
        value = preserved()
        value["source_final_gate_passed"] = True
        with self.assertRaises(RuntimeError):
            partition(value)


if __name__ == "__main__":
    unittest.main()
