"""Keep stricter compiled-run admission independent of the original30GiB guard."""

import unittest

from run_native import check_initial_memory


class InitialMemoryTests(unittest.TestCase):
    def test_default_keeps_the_original_admission(self):
        check_initial_memory(24 * 1024**3, 64 * 1024**3)
        with self.assertRaisesRegex(RuntimeError, "admission"):
            check_initial_memory(24 * 1024**3 + 1, 64 * 1024**3)

    def test_compiled_pair_requires_extra_headroom(self):
        check_initial_memory(12 * 1024**3, 64 * 1024**3, 12)
        with self.assertRaisesRegex(RuntimeError, "admission"):
            check_initial_memory(12 * 1024**3 + 1, 64 * 1024**3, 12)

    def test_original_container_reserve_cannot_be_reduced(self):
        with self.assertRaisesRegex(RuntimeError, "admission"):
            check_initial_memory(1, 30 * 1024**3, 12)

    def test_nonfinite_or_looser_admission_is_rejected(self):
        for value in (0, -1, 25, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_initial_memory(1, 64 * 1024**3, value)


if __name__ == "__main__":
    unittest.main()
