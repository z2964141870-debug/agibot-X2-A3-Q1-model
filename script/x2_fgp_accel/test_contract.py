import unittest

import numpy as np

from .probe import differences
from .trt_session import validate_feed


class ContractTests(unittest.TestCase):
    def feed(self):
        return {name: np.zeros((1, 30, 10, 9), dtype=np.float32) for name in ("x_t", "x_cond")}

    def test_valid(self):
        result = validate_feed(self.feed())
        self.assertEqual(result["x_t"].nbytes, 10800)

    def test_missing_input(self):
        with self.assertRaises(ValueError):
            validate_feed({"x_t": self.feed()["x_t"]})

    def test_shape(self):
        value = self.feed()
        value["x_t"] = value["x_t"][:, :1]
        with self.assertRaises(ValueError):
            validate_feed(value)

    def test_dtype(self):
        value = self.feed()
        value["x_t"] = value["x_t"].astype(np.float16)
        with self.assertRaises(ValueError):
            validate_feed(value)

    def test_nonfinite(self):
        value = self.feed()
        value["x_cond"][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_feed(value)

    def test_geodesic_and_translation(self):
        identity = np.eye(3)[None, None]
        angle = np.deg2rad(2)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])[None, None]
        result = differences(rotation, np.array([[.01, 0, 0]]), identity, np.zeros((1, 3)))
        self.assertAlmostEqual(result["pose_max_deg"], 2, places=8)
        self.assertAlmostEqual(result["root_max_m"], .01)


if __name__ == "__main__":
    unittest.main()
