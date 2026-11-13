"""Numerical and execution-boundary checks for independent performance adapters."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from script.x2_morph_human.core import Calibration, SmplAdapter, axis_angle
from .adapter import BatchedSmplAdapter
from .run import select_worker


class Contract(unittest.TestCase):
    def test_random_full_tree_agrees(self):
        rng = np.random.default_rng(711)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"skeleton.json"
            path.write_text(json.dumps(dict(bones=rng.normal(size=(24, 3)).tolist())))
            calibration = Calibration("batched_fk_contract", translation_scale=1.1,
                                      world_offset_m=(.2, -.3, .4))
            original = SmplAdapter(path, calibration)
            fast = BatchedSmplAdapter(path, calibration)
            for _ in range(128):
                q = axis_angle(rng.normal(size=(24, 3))*3)
                root = rng.normal(size=3)
                a, b = original._fk_from_local_quats(q, root)
                c, d = fast._fk_from_local_quats(q, root)
                np.testing.assert_allclose(a, c, atol=1e-12, rtol=0)
                np.testing.assert_allclose(b, d, atol=1e-12, rtol=0)

    def test_native_command_unchanged(self):
        command = ["native", "--mode", "observe", "--endpoint", "tcp://127.0.0.1:1234"]
        self.assertEqual(select_worker(command), command)

    def test_only_owned_workers(self):
        self.assertEqual(select_worker(["python", "-m", "script.x2_joint_latency.worker", "fgp"]),
                         ["python", "-m", "script.x2_joint_perf.worker", "fgp"])
        with self.assertRaises(ValueError):
            select_worker(["python", "-m", "script.x2_joint_latency.worker", "unregistered"])


if __name__ == "__main__":
    unittest.main()
