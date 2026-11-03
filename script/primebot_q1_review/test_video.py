"""Decoded frame counts and timebase must match the original trajectory."""

import unittest

from video import decoded_metadata


class VideoContract(unittest.TestCase):
    def fixture(self):
        return ("#tb 0: 1/50\n#dimensions 0: 1280x392\n" +
                "\n".join(f"0, {i}, {i}, 1, 1505280, " + "a" * 64 for i in range(2)))

    def test_full_decode_accepts_expected_video(self):
        self.assertEqual(decoded_metadata(self.fixture(), 2)["duration"], .04)

    def test_wrong_frame_count_or_timestamp_refuses(self):
        with self.assertRaises(RuntimeError):
            decoded_metadata(self.fixture(), 3)
        with self.assertRaises(RuntimeError):
            decoded_metadata(self.fixture().replace("0, 1, 1,", "0, 2, 2,"), 2)

    def test_wrong_dimensions_or_rate_refuses(self):
        for before, after in (("1/50", "1/25"), ("1280x392", "640x392")):
            with self.assertRaises(RuntimeError):
                decoded_metadata(self.fixture().replace(before, after), 2)


if __name__ == "__main__":
    unittest.main()
