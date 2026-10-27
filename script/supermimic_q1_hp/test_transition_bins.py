"""Check every legal transition is covered exactly once across clip boundaries."""

import unittest

from transition_bins import transition_bin_layout


class TransitionBinTests(unittest.TestCase):
    def check_coverage(self, lengths, width=50):
        layout = transition_bin_layout(lengths, width)
        offset = 0
        expected = []
        for length in lengths:
            expected.append(set(range(offset, offset + length - 1)))
            offset += length
        self.assertEqual(len(layout["frame_bin"]), offset)
        covered = [set() for _ in lengths]
        ends = []
        offset = 0
        for length in lengths:
            ends.append(offset + length - 1)
            offset += length
        for index, (clip, start) in enumerate(zip(layout["bin_clip"], layout["bin_start"])):
            stop = min(start + width, ends[clip])
            self.assertLess(start, stop)
            frames = set(range(start, stop))
            self.assertFalse(frames & covered[clip])
            for frame in frames:
                self.assertEqual(layout["frame_bin"][frame], index)
                self.assertLessEqual(frame + 1, ends[clip])
            covered[clip].update(frames)
        self.assertEqual(covered, expected)
        for end in ends:
            self.assertEqual(layout["frame_bin"][end], layout["frame_bin"][end - 1])

    def test_all_lengths_through_eight_bins(self):
        self.check_coverage(list(range(2, 403)))

    def test_exact_bin_and_one_frame_tail(self):
        lengths = [2, 49, 50, 51, 52, 100, 101, 102]
        self.check_coverage(lengths)
        layout = transition_bin_layout([51, 101])
        self.assertEqual(layout["bin_clip"], [0, 1, 1])
        self.assertEqual(layout["bin_start"], [0, 51, 101])

    def test_other_bin_widths(self):
        for width in (1, 2, 17, 1000):
            self.check_coverage([2, 3, 18, 51, 101], width)

    def test_invalid_transition_inputs(self):
        for lengths, width in (([], 50), ([1], 50), ([True], 50), ([3], 0), ([3], True)):
            with self.assertRaises(ValueError):
                transition_bin_layout(lengths, width)


if __name__ == "__main__":
    unittest.main()
