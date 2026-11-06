"""Event visibility must not fill packet losses with stored future labels."""

import unittest

import numpy as np

from stream_events import DT_NS, EventSpec, EventTimeline, derived_poses


class EventTests(unittest.TestCase):
    def test_nominal_prefix_and_transport_are_separate_from_preview(self):
        nominal = EventTimeline(200,EventSpec("nominal"))
        delayed = EventTimeline(200,EventSpec("nominal",transport_frames=2))
        self.assertEqual(nominal.visibility(10,4)["continuous_future_prefix"],(11,12,13,14))
        self.assertEqual(delayed.visibility(10,4)["continuous_future_prefix"],(11,12))
        self.assertEqual(delayed.visibility(10,4)["newest_capture_age_ns"],2*DT_NS)

    def test_loss_stays_missing_after_recovery_without_future_substitution(self):
        event = EventTimeline(200,EventSpec("dropout",first_frame=50,duration_frames=10))
        for frame in range(199):
            value = event.visibility(frame,4)
            self.assertFalse(set(range(50,60)) & set(value["visible_sequences"]))
            self.assertTrue(all(i*DT_NS<=value["simulated_control_ns"] for i in value["visible_sequences"]))
        value = event.visibility(49,4)
        self.assertEqual(value["continuous_future_prefix"],())
        self.assertTrue(value["physical_fallback_required"])
        self.assertEqual(event.visibility(59,4)["continuous_future_prefix"],(60,61,62,63))

    def test_delayed_packets_arrive_only_at_declared_burst(self):
        event = EventTimeline(200,EventSpec("late_burst",first_frame=50,duration_frames=10))
        self.assertFalse(any(50<=p.sequence<60 for p in event.visible(59*DT_NS)))
        recovered = event.visible(60*DT_NS)
        self.assertEqual(tuple(p.sequence for p in recovered[-11:]),tuple(range(50,61)))
        self.assertTrue(all(p.arrival_ns==60*DT_NS for p in recovered[-11:]))

    def test_stop_and_reverse_derive_past_source_without_fake_fresh_measurement(self):
        for kind in ("stop","reverse"):
            timeline = EventTimeline(200,EventSpec(kind,first_frame=50))
            self.assertTrue(all(source<=capture for capture,source in enumerate(timeline.indices)))
            self.assertTrue(all(p.provenance=="synthetic_offline:"+kind for p in timeline.packets))
            self.assertEqual(timeline.indices[50],50)
        self.assertEqual(EventTimeline(200,EventSpec("stop",first_frame=50)).indices[51],50)
        self.assertEqual(EventTimeline(200,EventSpec("reverse",first_frame=50)).indices[51],49)

    def test_derivative_is_recomputed_after_stop_and_reversal(self):
        count = 120
        positions = np.zeros((count,24,3))
        positions[:,:,0] = np.arange(count)[:,None]*.02
        quaternions = np.zeros((count,24,4))
        angle = np.arange(count)*.02
        quaternions[:,:,0] = np.cos(angle[:,None]/2)
        quaternions[:,:,3] = np.sin(angle[:,None]/2)
        for kind,sign in (("stop",0),("reverse",-1)):
            timeline = EventTimeline(count,EventSpec(kind,first_frame=50))
            result = derived_poses(positions,quaternions,timeline)
            np.testing.assert_allclose(result["linear_velocity_w"][51,:,0],sign,atol=1e-12)
            np.testing.assert_allclose(result["angular_velocity_w"][51,:,2],sign,atol=1e-12)
            np.testing.assert_allclose(result["angular_velocity_w"][:,:,:2],0,atol=1e-12)
            self.assertTrue(result["first_frame_velocity_is_unknown"])
            self.assertIsNone(result["measured_robot_latency_ns"])
        np.testing.assert_allclose(positions[51,:,0],1.02)

    def test_invalid_clock_or_unbounded_event_rejected(self):
        with self.assertRaises(ValueError): EventSpec("dropout",duration_frames=51)
        with self.assertRaises(ValueError): EventTimeline(100,EventSpec("reverse",first_frame=99))
        with self.assertRaises(ValueError): EventTimeline(100,EventSpec("dropout",first_frame=90,duration_frames=10))
        with self.assertRaises(ValueError): EventTimeline(200,EventSpec("nominal")).visible(-1)
        with self.assertRaises(ValueError): EventTimeline(200,EventSpec("nominal")).visibility(10,0)


if __name__ == "__main__":
    unittest.main()
