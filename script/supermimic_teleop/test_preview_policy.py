"""Input-boundary checks for the independent short-preview policy adapter."""

from dataclasses import replace
import unittest

from preview_policy import validate_window
from stream_contract import PoseTimeline
from test_stream_contract import DT_NS, SOURCE, sample, window


class PreviewPolicyContractTests(unittest.TestCase):
    def setUp(self):
        timeline = PoseTimeline(SOURCE)
        for i in range(5):
            timeline.ingest(sample(i))
        self.valid = window(timeline, 4)

    def test_rejects_unavailable_stale_and_inconsistent_masks(self):
        self.assertEqual(validate_window(self.valid), 4)
        for bad in (replace(self.valid, can_step=False),
                    replace(self.valid, policy_frame_mask=(True, False)*5),
                    replace(self.valid, measured_mask=(True,)*10),
                    replace(self.valid, poses=(None,)*10),
                    replace(self.valid, capture_cutoff_ns=5*DT_NS),
                    replace(self.valid, used_sequences=(1,)),
                    replace(self.valid, target_ns=tuple(i*DT_NS for i in range(10)))):
            with self.subTest(bad=bad.status), self.assertRaises(ValueError):
                validate_window(bad)

    def test_only_declared_offline_terminal_padding_is_enabled(self):
        offline = replace(SOURCE, preprocessing="offline_unverified")
        timeline = PoseTimeline(offline)
        for i in range(4):
            timeline.ingest(replace(sample(i), source=offline, dependency_end_ns=None, terminal=i == 3))
        padded = window(timeline, 4)
        self.assertEqual(validate_window(padded), 4)
        self.assertEqual(padded.provenance[3], "declared_clip_terminal_hold")
        with self.assertRaisesRegex(ValueError, "cannot claim live"):
            validate_window(replace(padded, source=SOURCE))


if __name__ == "__main__":
    unittest.main()
