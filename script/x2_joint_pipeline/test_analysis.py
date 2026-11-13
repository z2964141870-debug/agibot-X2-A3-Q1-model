import unittest

from .analyze import interval_overlap


class Contract(unittest.TestCase):
    def test_disjoint_and_touching_intervals(self):
        self.assertEqual(interval_overlap([(0, 2), (4, 6)], [(2, 4)]), 0)

    def test_one_stage_spans_multiple_other_frames(self):
        self.assertEqual(interval_overlap([(0, 10)], [(1, 3), (6, 8)]), 4)

    def test_unsorted_and_empty(self):
        self.assertEqual(interval_overlap([(8, 10), (0, 2)], [(1, 9)]), 2)
        self.assertEqual(interval_overlap([], [(0, 1)]), 0)


if __name__ == "__main__":
    unittest.main()
