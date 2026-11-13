import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile

from .pipeline import BoundedPipeline, PreparedDenoiser
from .run import select_worker, validate_parity


class Contract(unittest.TestCase):
    def test_order_bound_and_stage_ownership(self):
        cpu, sent, threads = [], [], set()
        def process(x):
            threads.add(threading.get_ident())
            cpu.append(x)
            time.sleep(.001)
            return x*2
        def send(x):
            threads.add(threading.get_ident())
            sent.append(x)
            time.sleep(.001)
        p = BoundedPipeline(process, send)
        for x in range(32):
            p.reserve()
            p.submit(x)
        p.close()
        self.assertEqual(cpu, list(range(32)))
        self.assertEqual(sent, list(range(0, 64, 2)))
        self.assertEqual((p.maximum, p.active, p.completed), (2, 0, 32))
        self.assertEqual(len(threads), 2)

    def test_stage_failure_propagates_and_cleans_up(self):
        def fail(x):
            raise ValueError("deliberate stage failure")
        p = BoundedPipeline(fail, lambda x: None)
        p.reserve()
        p.submit(1)
        with self.assertRaises(ValueError):
            p.close()
        self.assertEqual(p.active, 0)
        with self.assertRaises(RuntimeError):
            p.reserve()

    def test_sender_failure_propagates(self):
        def fail(x):
            raise ValueError("deliberate sender failure")
        p = BoundedPipeline(lambda x: x, fail)
        p.reserve()
        p.submit(1)
        with self.assertRaises(ValueError):
            p.close()
        self.assertEqual(p.active, 0)

    def test_prepared_is_single_use(self):
        d = PreparedDenoiser()
        with self.assertRaises(RuntimeError):
            d.step(None, None)
        d.bind(1, 2)
        with self.assertRaises(RuntimeError):
            d.bind(3, 4)
        self.assertEqual(d.step(None, None), (1, 2))
        self.assertEqual(len(d.acc_buffer), 30)
        with self.assertRaises(RuntimeError):
            d.reset()

    def test_native_command_is_preserved(self):
        command = ["native", "--mode", "observe"]
        self.assertEqual(select_worker(command), command)
        self.assertEqual(select_worker(["python", "-m", "script.x2_joint_latency.worker", "fgp"])[2],
                         "script.x2_joint_pipeline.worker")
        with self.assertRaises(ValueError):
            select_worker(["python", "-m", "script.x2_joint_latency.worker", "other"])

    def test_two_inflight_failures_release_both_credits(self):
        gate = threading.Event()
        def process(x):
            gate.wait(1)
            raise ValueError("both fail")
        p = BoundedPipeline(process, lambda x: None)
        p.reserve(); p.submit(1)
        p.reserve(); p.submit(2)
        gate.set()
        with self.assertRaises(ValueError):
            p.close()
        self.assertEqual(p.active, 0)

    def test_incomplete_or_changed_prefix_gate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = dict(status="PIPELINE_PREFIX_EQUIVALENCE_PASSED", outputs=64,
                fixture_sha256="same", engine_sha256="same", reference="serial",
                reference_timing_sha256="same", reference_pose_sha256="same")
            (root/"parity.json").write_text(json.dumps(receipt))
            guard = dict(status=receipt["status"], child_returncode=1)
            (root/"guard.json").write_text(json.dumps(guard))
            with patch("script.x2_joint_pipeline.run.file_digest", return_value="same"):
                with self.assertRaises(ValueError):
                    validate_parity(root, Path("fixture"))
                guard["child_returncode"] = 0
                (root/"guard.json").write_text(json.dumps(guard))
                self.assertEqual(validate_parity(root, Path("fixture")), receipt)
            with patch("script.x2_joint_pipeline.run.file_digest", return_value="changed"):
                with self.assertRaises(ValueError):
                    validate_parity(root, Path("fixture"))


if __name__ == "__main__":
    unittest.main()
