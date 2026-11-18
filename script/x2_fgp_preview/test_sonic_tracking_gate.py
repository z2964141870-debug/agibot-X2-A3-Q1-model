from dataclasses import replace
from unittest import TestCase

from .sonic_tracking_gate import Cycle, JointEnvelope, TrackingGate


class TrackingGateTests(TestCase):
    def setUp(self):
        self.gate = TrackingGate(ready_for_s=1., max_input_age_s=.12)

    def step(self, i, **changes):
        t = i * .02
        cycle = Cycle(epoch="first", now=t, planner_sequence=i, planner_received=t,
                      feedback_received=t, planner_source="upstream", planner_head=True,
                      steady=True, human_ready=True, operator_requests_tracking=False,
                      target_envelope_ok=True)
        return self.gate.update(replace(cycle, **changes))

    def prime(self):
        for i in range(51):
            self.step(i)

    def test_cold_start_with_stale_engage_file_requires_new_edge(self):
        for i in range(60):
            result = self.step(i, operator_requests_tracking=True)
            self.assertFalse(result.tracking_requested)
        self.assertTrue(result.planner_ready)
        self.step(60)
        self.assertEqual(self.step(61, operator_requests_tracking=True).event, "enter:planner_ready")

    def test_early_request_is_consumed_instead_of_queued(self):
        self.step(0)
        self.assertEqual(self.step(1, operator_requests_tracking=True).event,
                         "reject:planner_or_human_unready")
        for i in range(2, 60):
            self.assertFalse(self.step(i, operator_requests_tracking=True).tracking_requested)

    def test_watchdog_fallback_breaks_readiness(self):
        for i in range(60):
            self.assertFalse(self.step(i, planner_source="cold_idle").planner_ready)
        for i in range(60, 100):
            self.step(i)
        self.assertEqual(self.step(100, operator_requests_tracking=True).event,
                         "reject:planner_or_human_unready")

    def test_duplicate_sequence_cannot_be_kept_fresh_by_a_new_clock(self):
        self.step(0)
        for i in range(1, 65):
            decision = self.step(i, planner_sequence=0)
        self.assertFalse(decision.planner_ready)

    def test_request_has_to_wait_for_latest_feedback_and_human(self):
        self.prime()
        self.assertEqual(self.step(51, human_ready=False, operator_requests_tracking=True).event,
                         "reject:planner_or_human_unready")
        self.step(52)
        result = self.step(53, feedback_received=.5, operator_requests_tracking=True)
        self.assertFalse(result.tracking_requested)
        self.assertFalse(result.planner_ready)

    def test_operator_release_requires_actual_planner_head_before_reentry(self):
        self.prime()
        self.step(51, operator_requests_tracking=True)
        self.assertEqual(self.step(52, planner_head=False).event, "release:operator")
        for i in range(53, 112):
            self.assertFalse(self.step(i, planner_head=False).planner_ready)
        for i in range(112, 165):
            self.step(i)
        self.assertTrue(self.step(165, operator_requests_tracking=True).tracking_requested)

    def test_disconnect_releases_and_does_not_arm_on_reconnect(self):
        self.prime()
        self.step(51, operator_requests_tracking=True)
        result = self.step(52, human_ready=False, operator_requests_tracking=True)
        self.assertEqual(result.event, "release:input_unavailable")
        for i in range(53, 120):
            self.assertFalse(self.step(i, operator_requests_tracking=True).tracking_requested)

    def test_epoch_and_cycle_discontinuity_force_release(self):
        self.prime()
        self.step(51, operator_requests_tracking=True)
        self.assertEqual(self.step(52, epoch="new", operator_requests_tracking=True).event, "release:epoch_changed")
        for i in range(53, 110):
            self.step(i, epoch="new")
        self.assertEqual(self.step(110, epoch="new", operator_requests_tracking=True).event, "enter:planner_ready")
        self.assertEqual(self.step(120, epoch="new", operator_requests_tracking=True).event, "release:cycle_gap")

    def test_bad_target_releases_even_when_transport_and_feedback_are_fresh(self):
        self.prime()
        self.assertFalse(self.step(51, target_envelope_ok=False,
                                   operator_requests_tracking=True).tracking_requested)
        for i in range(52, 104):
            self.step(i)
        self.assertTrue(self.step(104, operator_requests_tracking=True).tracking_requested)
        self.assertEqual(self.step(105, target_envelope_ok=False,
                                   operator_requests_tracking=True).event, "release:target_envelope")

    def test_position_clamp_alone_allows_a_large_step(self):
        gate = JointEnvelope(-.45, .45, .5, .45)
        faults = gate.violations(target=.45, measured=-.24, previous=-.45, dt=.01999)
        self.assertEqual(faults, ("rate", "feedback_error"))
        self.assertEqual(gate.violations(target=0., measured=0., previous=0., dt=.02), ())
