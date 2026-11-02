"""Reject the 32-bit compatibility libraries in a mixed NVIDIA archive."""

import unittest

from prepare_egl import is_x86_64


class ArchitectureTests(unittest.TestCase):
    def header(self, bits=2, machine=62, endian=1):
        data = bytearray(20)
        data[:4] = b"\x7fELF"
        data[4], data[5] = bits, endian
        data[18:20] = machine.to_bytes(2, "little")
        return bytes(data)

    def test_native_library(self):
        self.assertTrue(is_x86_64(self.header()))

    def test_compatibility_and_other_architectures(self):
        for data in (self.header(1, 3), self.header(2, 183), self.header(2, 62, 2)):
            with self.subTest(data=data):
                self.assertFalse(is_x86_64(data))

    def test_truncated_or_non_elf(self):
        for data in (b"", b"\x7fELF", b"not a library", b"x" * 20):
            with self.subTest(data=data):
                self.assertFalse(is_x86_64(data))


if __name__ == "__main__":
    unittest.main()
