"""Budget boundaries must hold while evaluation pauses the training clock."""

import unittest

from supervise import active_training_seconds


class TrainingCapTests(unittest.TestCase):
    def clock(self, **values):
        return dict(absolute_maximum_seconds=1200, current_cap_seconds=1200,
                    inherited_training_seconds=0, accumulated_training_seconds=300,
                    active_started_monotonic=None, **values)

    def test_paused_evaluation_excluded(self):
        self.assertEqual(active_training_seconds(self.clock(), 5000), 300)

    def test_active_interval_counted(self):
        clock = self.clock()
        clock["active_started_monotonic"] = 100
        self.assertEqual(active_training_seconds(clock, 150), 350)

    def test_extended_inherited_or_invalid_clocks_rejected(self):
        for key, value in (("absolute_maximum_seconds", 1201), ("current_cap_seconds", 2400),
                           ("inherited_training_seconds", 1), ("accumulated_training_seconds", -1),
                           ("accumulated_training_seconds", float("nan")),
                           ("active_started_monotonic", 201), ("active_started_monotonic", float("inf"))):
            clock = self.clock()
            clock[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                active_training_seconds(clock, 200)


if __name__ == "__main__":
    unittest.main()
