"""A policy must preview t+1 and hold the final reference, never cross clips."""

import unittest

from q1_native_review import reference_window_indices


class ReferenceWindowTests(unittest.TestCase):
    def test_reference_lead_and_tail_hold(self):
        self.assertEqual(reference_window_indices(0, 20), list(range(1, 11)))
        self.assertEqual(reference_window_indices(17, 20), [18] + [19] * 9)
        self.assertEqual(reference_window_indices(18, 20), [19] * 10)

    def test_no_policy_after_the_final_transition(self):
        for frame, length in ((-1, 20), (19, 20), (0, 1)):
            with self.subTest(frame=frame), self.assertRaises(RuntimeError):
                reference_window_indices(frame, length)


if __name__ == "__main__":
    unittest.main()
