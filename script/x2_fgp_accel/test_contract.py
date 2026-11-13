import unittest
from types import SimpleNamespace

import numpy as np

from .probe import differences, LIMITS
from .build import PRECISION_FLAGS
from .trt_session import TrtSession, validate_feed, validate_gp_feed, GP_INPUTS, GP_OUTPUTS
from .precision_plan import make_plan, specifications
from .feature_cache import ExactFeatureCache


class ContractTests(unittest.TestCase):
    def test_feature_cache_copies_inputs_and_outputs(self):
        class Tensor:
            def __init__(self, value):
                self.value = value
            def clone(self):
                return Tensor(self.value.copy())
        cache = ExactFeatureCache(lambda *values: tuple(Tensor(v.copy()) for v in values))
        arrays = [np.zeros((2, 3), np.float32) for _ in range(3)]
        first = cache(*arrays)
        first[0].value.fill(9)
        second = cache(*arrays)
        self.assertEqual(second[0].value.flat[0], 0)
        second[0].value.fill(8)
        self.assertEqual(cache(*arrays)[0].value.flat[0], 0)
        arrays[0].fill(1)
        self.assertEqual(cache(*arrays)[0].value.flat[0], 1)
        self.assertEqual((cache.hits, cache.misses), (2, 2))

    def test_feature_cache_rejects_inexact_keys_and_nan_hits(self):
        class Tensor:
            def clone(self):
                return Tensor()
        cache = ExactFeatureCache(lambda *values: (Tensor(), Tensor(), Tensor()))
        a = [np.zeros((1,), np.float32) for _ in range(3)]
        cache(*a)
        cache(a[0].astype(np.float64), *a[1:])
        a[0].fill(-0.)
        cache(*a)
        a[0].fill(0.)
        cache(*a)
        a[2][0] = np.nan
        cache(*a)
        cache(*a)
        self.assertEqual((cache.hits, cache.misses), (0, 6))

    def test_explicit_gp_state_contract(self):
        feed = {name: np.zeros(shape, np.float32) for name, shape in GP_INPUTS.items()}
        self.assertEqual(set(validate_gp_feed(feed)), set(GP_INPUTS))
        with self.assertRaises(ValueError):
            validate_gp_feed({"features": feed["features"]})
        feed["h"][0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_gp_feed(feed)

    def test_gp_state_output_order_and_recording_ownership(self):
        session = TrtSession.__new__(TrtSession)
        session.gp, session.closed, session.recorded = True, False, []
        session.outputs = {name: np.full(shape, i+1, np.float32)
                           for i, (name, shape) in enumerate(GP_OUTPUTS.items())}
        session.pointers = dict.fromkeys((*GP_INPUTS, *GP_OUTPUTS), None)
        session.stream = None
        session.cuda = SimpleNamespace(cudaMemcpyAsync=lambda *args: 0,
                                       cudaStreamSynchronize=lambda *args: 0)
        session.enqueue = lambda: None
        feed = {name: np.zeros(shape, np.float32) for name, shape in GP_INPUTS.items()}
        prediction, h, c = session.run(None, feed)
        self.assertEqual((prediction.flat[0], h.flat[0], c.flat[0]), (1, 2, 3))
        feed["h"].fill(9)
        self.assertEqual(session.recorded[0][0]["h"].flat[0], 0)
        session.outputs["h_out"].fill(8)
        self.assertEqual(h.flat[0], 2)
        with self.assertRaises(RuntimeError):
            session.run(["c_out", "h_out", "prediction"], feed)

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
