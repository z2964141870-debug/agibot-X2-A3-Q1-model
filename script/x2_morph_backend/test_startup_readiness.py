import unittest
from .control_session import StableReadiness


def sample(requests=0, rejected=0, **kwargs):
    return dict(stage="STANDBY", ready_target=True, mc_capture_complete=True,
                transport_failed=False, requests=requests, rejected=rejected, **kwargs)


class StableReadinessTests(unittest.TestCase):
    def test_transient_and_frozen_snapshot_cannot_admit(self):
        gate = StableReadiness()
        self.assertFalse(gate.update(sample(), 0))
        self.assertFalse(gate.update(sample(), 2))
        self.assertTrue(gate.update(sample(requests=50), 2.1))

    def test_unobserved_rejection_between_polls_resets_window(self):
        gate = StableReadiness()
        gate.update(sample(), 0)
        self.assertFalse(gate.update(sample(requests=60, rejected=1), 1.2))
        self.assertFalse(gate.update(sample(requests=61, rejected=1), 1.3))
        self.assertTrue(gate.update(sample(requests=110, rejected=1), 2.4))

    def test_missing_target_or_transport_or_capture_cannot_admit(self):
        for field in ("ready_target", "mc_capture_complete", "transport_failed"):
            gate = StableReadiness()
            gate.update(sample(), 0)
            row = sample(requests=100)
            row[field] = field == "transport_failed"
            self.assertFalse(gate.update(row, 2))
