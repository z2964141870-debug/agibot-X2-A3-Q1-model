import unittest

import numpy as np

from .probe import differences, LIMITS
from .build import PRECISION_FLAGS
from .trt_session import validate_feed
from .precision_plan import make_plan, specifications


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

    def test_mixed_constraints_and_original_limits(self):
        self.assertEqual(LIMITS["mixed"], LIMITS["fp16"])
        self.assertIn("--precisionConstraints=obey", PRECISION_FLAGS["mixed"])
        self.assertTrue(any("*LayerNormalization:fp32" in value for value in PRECISION_FLAGS["mixed"]))

    def test_float_compute_plan_preserves_non_matrix_outputs(self):
        rows = [dict(name="/linear/MatMul", type="MATRIX_MULTIPLY", floating_data_outputs=True),
                dict(name="/residual/Add", type="ELEMENTWISE", floating_data_outputs=True),
                dict(name="/shape/Add", type="ELEMENTWISE", floating_data_outputs=False)]
        plan = make_plan(rows, "source")
        self.assertEqual(specifications(plan, "source"),
                         ("/linear/MatMul:fp16,/residual/Add:fp32", "/linear/MatMul:fp32,/residual/Add:fp32"))
        with self.assertRaises(ValueError):
            specifications(plan, "changed source")
        plan["layers"][1]["precision"] = "fp16"
        with self.assertRaises(ValueError):
            specifications(plan, "source")

    def test_precision_plan_rejects_duplicates_and_wildcards(self):
        row = dict(name="/linear/MatMul", type="MATRIX_MULTIPLY", floating_data_outputs=True)
        with self.assertRaises(ValueError):
            make_plan([row, row], "source")
        with self.assertRaises(ValueError):
            make_plan([dict(row, name="*:fp16")], "source")
        internal = dict(name="(Unnamed Layer* 265) [ElementWise]", type="ELEMENTWISE", floating_data_outputs=True)
        self.assertEqual(len(make_plan([row, internal], "source")["layers"]), 2)
        with self.assertRaises(ValueError):
            make_plan([row, internal, dict(internal, name="(Unnamed Layer 265) [ElementWise]")], "source")


if __name__ == "__main__":
    unittest.main()
