from decimal import Decimal
from unittest import TestCase

from .audit_sonic_handoff import audit, pose_source
from .sonic_reference_provenance import SOURCE_CODES


class HandoffAuditTests(TestCase):
    def review(self, samples, requests):
        return audit({Decimal(t): {"policy_head": head} for t, head in samples},
                     [(Decimal(t), mode) for t, mode in requests],
                     max_gap_ms=Decimal("35"), max_confirmation_ms=Decimal("50"))

    def test_confirms_actual_head_on_entry_and_exit(self):
        result = self.review([("0", "dual_pose"), (".020", "dual_smpl"),
                              (".040", "dual_smpl"), (".060", "dual_pose")],
                             [(".010", "tracking"), (".050", "planner")])
        self.assertEqual(result["outcomes"], {"confirmed": 2})
        self.assertEqual([event["confirmation_ms"] for event in result["events"]], [10., 10.])
        self.assertEqual(result["unmatched_observed_head_changes"], [])

    def test_no_head_switch_by_deadline_with_continuous_coverage(self):
        result = self.review([(t, "pose") for t in ("0", ".020", ".040", ".060", ".080")],
                             [(".010", "tracking")])
        self.assertEqual(result["outcomes"], {"confirmation_timeout": 1})

    def test_cycle_on_deadline_is_enough_to_rule_out_a_switch(self):
        result = self.review([(t, "pose") for t in ("0", ".020", ".040", ".060")],
                             [(".010", "tracking")])
        self.assertEqual(result["outcomes"], {"confirmation_timeout": 1})

    def test_unknown_head_and_missing_ticks_cannot_confirm(self):
        for middle in ([], [(".020", "unrecorded")]):
            with self.subTest(middle=middle):
                result = self.review([("0", "pose"), *middle, (".040", "token")],
                                     [(".010", "tracking")])
                self.assertEqual(result["outcomes"], {"head_coverage_gap": 1})

    def test_early_end_does_not_imply_timeout(self):
        result = self.review([("0", "pose"), (".020", "pose")], [(".010", "tracking")])
        self.assertEqual(result["outcomes"], {"insufficient_tail": 1})

    def test_already_active_and_unsolicited_switch_stay_distinct(self):
        result = self.review([("0", "pose"), (".020", "token"), (".040", "token")],
                             [(".030", "tracking")])
        self.assertEqual(result["outcomes"], {"already_active": 1})
        self.assertEqual(result["unmatched_observed_head_changes"][0]["t"], "0.020")

    def test_equal_clock_values_cannot_establish_order(self):
        result = self.review([("0", "pose"), (".020", "token")], [(".020", "tracking")])
        self.assertEqual(result["outcomes"], {"timestamp_tie": 1})

    def test_new_request_cannot_confirm_superseded_request(self):
        result = self.review([("0", "pose"), (".020", "pose"), (".040", "token")],
                             [(".010", "tracking"), (".030", "planner")])
        self.assertEqual(result["outcomes"], {"superseded": 1, "already_active": 1})
        self.assertEqual(len(result["unmatched_observed_head_changes"]), 1)

    def test_planner_head_confirmation_distinguishes_fallback_and_unknown(self):
        samples = {Decimal("0"): {"policy_head": "dual_smpl"},
                   Decimal(".020"): {"policy_head": "dual_pose"}}
        request = [(Decimal(".010"), "planner")]
        settings = dict(max_gap_ms=Decimal("35"), max_confirmation_ms=Decimal("50"))
        result = audit(samples, request, **settings)
        self.assertEqual(result["outcomes"], {"confirmed": 1})
        self.assertEqual(result["confirmed_planner_side_source_kinds"], {"unknown": 1})
        for i in range(10):
            samples[Decimal(".020")][f"watchdog_provenance_{i}"] = str((8 << 3) | SOURCE_CODES["cold_idle"])
        result = audit(samples, request, **settings)
        self.assertEqual(result["outcomes"], {"confirmed": 1})
        self.assertEqual(result["confirmed_planner_side_source_kinds"], {"cold_idle": 1})

    def test_incomplete_provenance_cannot_establish_upstream(self):
        stamp = str((8 << 3) | SOURCE_CODES["upstream"])
        self.assertEqual(pose_source({"watchdog_provenance_0": stamp, "watchdog_provenance_1": "invalid"}),
                         {"source_kind": "partial", "slot_counts": {"upstream": 1, "unknown": 9}})
