import unittest
from .timing import reference_ages, statistics


class TimingTests(unittest.TestCase):
    def reply(self):
        slots = [100_000_000+i*20_000_000 for i in range(10)]
        return dict(ready=True, timing=dict(capture_clock="same", slot_capture_ns=slots,
                                            raw_support_end_ns=slots))

    def test_preview_is_counted_once(self):
        result = reference_ages(self.reply(), 320_000_000, "same")
        self.assertEqual(result["oldest_reference_age_ms"], 220)
        self.assertEqual(result["newest_slot_age_ms"], 40)

    def test_clock_mismatch_refused(self):
        with self.assertRaises(ValueError):
            reference_ages(self.reply(), 320_000_000, "different")

    def test_future_support_refused(self):
        reply = self.reply()
        reply["timing"]["raw_support_end_ns"][-1] = 400_000_000
        with self.assertRaises(ValueError):
            reference_ages(reply, 320_000_000, "same")

    def test_empty_and_boundary(self):
        self.assertIsNone(statistics([]))
        self.assertEqual(statistics([20., 21.], 20.)["over_budget"], 1)
        self.assertIsNone(reference_ages(dict(ready=False), 0, "same"))


if __name__ == "__main__":
    unittest.main()
