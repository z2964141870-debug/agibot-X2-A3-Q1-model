"""Paused-state admission, exclusive GPU ownership and bounded campaign contracts."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paused_runtime import assert_owner, own_gpu, paused_state
from paused_preview import COMMIT, FIXED, ITERATION, cases, execute, identity
from preview_review import validate_stream
from test_preview_review import telemetry


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PausedAdmissionTests(unittest.TestCase):
    def test_pause_identity_and_absent_original_pids_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proc, training = root / "proc", root / "training"
            boot = proc / "sys/kernel/random/boot_id"
            boot.parent.mkdir(parents=True)
            boot.write_text("expected-boot\n")
            pipeline = training / "pipeline.json"
            write(pipeline, dict(status="FAILED", error="RuntimeError: Controller stopped after a phase; do not advance"))
            write(training / "e1024/train/guard.json", dict(status="STOPPED_SIGNAL"))
            config = dict(output=str(training), expected_boot="expected-boot", expected_gpu="expected-gpu")
            receipt = dict(pause_verified=True, training_commit=COMMIT, boot="expected-boot",
                           gpu="expected-gpu", training_root=str(training))
            def check(record=receipt):
                return paused_state(config, record, proc=proc, training=training, root=root)
            with patch("paused_runtime.Path.cwd", return_value=root), patch("paused_runtime.process", return_value=None):
                check()
                for record in (dict(receipt, pause_verified=False), dict(receipt, boot="another"),
                               dict(receipt, gpu="another"), dict(receipt, training_commit="a"*40)):
                    with self.assertRaisesRegex(RuntimeError, "pause receipt"):
                        check(record)
                with patch("paused_runtime.process", return_value=dict(start_ticks="reused")):
                    with self.assertRaisesRegex(RuntimeError, "training PID"):
                        check()
                write(pipeline, dict(status="TRAINING", error=""))
                with self.assertRaisesRegex(RuntimeError, "stopped state"):
                    check()

    def test_owner_must_match_pid_start_tick_and_exact_lock_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "gpu.lock"
            lock.touch()
            stat = lock.stat()
            device = f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}"
            row = dict(pid=123, start_ticks="456")
            def line(inode=stat.st_ino, pid=123):
                (root / "locks").write_text(f"1: FLOCK ADVISORY WRITE {pid} {device}:{inode} 0 EOF\n")
            with patch("paused_runtime.process", return_value=row):
                line()
                self.assertEqual(assert_owner(123, "456", lock=lock, proc=root), row)
                with self.assertRaisesRegex(RuntimeError, "PID was reused"):
                    assert_owner(123, "457", lock=lock, proc=root)
                for inode, pid in ((stat.st_ino+1, 123), (stat.st_ino, 124)):
                    line(inode, pid)
                    with self.assertRaisesRegex(RuntimeError, "exclusive GPU lock"):
                        assert_owner(123, "456", lock=lock, proc=root)

    def test_real_nonblocking_conflict_preserves_existing_holder_and_releases_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "gpu.lock"
            with lock.open("a") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Another job"):
                    with own_gpu(lock):
                        self.fail("A held GPU lock was acquired")
                fcntl.flock(holder, fcntl.LOCK_UN)
            with patch("paused_runtime.compute_pids", return_value={999}):
                with self.assertRaisesRegex(RuntimeError, "existing compute"):
                    with own_gpu(lock):
                        self.fail("An occupied GPU was admitted")
            with lock.open("a") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_gpu_timing_cannot_be_reported_as_cpu_or_end_to_end_latency(self):
        cpu = telemetry()
        with self.assertRaisesRegex(RuntimeError, "truthful stream timing"):
            validate_stream(cpu, 4, 0, policy_device="cuda")
        gpu = dict(cpu, policy_device="cuda", gpu_policy_wall_compute_ns=123)
        with self.assertRaisesRegex(RuntimeError, "labeled CPU"):
            validate_stream(gpu, 4, 0, policy_device="cuda")
        del gpu["cpu_policy_wall_compute_ns"]
        validate_stream(gpu, 4, 0, policy_device="cuda")
        with self.assertRaises(RuntimeError):
            validate_stream(dict(gpu, measured_end_to_end_latency_ns=80_000_000), 4, 0, policy_device="cuda")


class PausedControllerTests(unittest.TestCase):
    def fixture(self, root, fail=None):
        source = root / "source.py"
        source.write_text("# independent code fixture\n")
        record = identity(source)
        args = SimpleNamespace(child=None, config=source, runtime_config=source,
            pause_receipt=source, audit_root=root, code_manifest=source,
            manifest_sha256=record["sha256"], evaluation_commit="a"*40, output=root / "output")
        runtime = dict(primebot_python=sys.executable, morph_python=sys.executable,
                       graphics_dir="/graphics", ffmpeg="/ffmpeg")
        owner = dict(pid=123, start_ticks="456")
        labels = []
        @contextmanager
        def lock():
            yield owner
        def run(command, workdir, budget, status, log, **kwargs):
            label = status.name.removesuffix(".guard.json")
            labels.append(label)
            if label == "comparison":
                write(args.output / "comparison.json", dict(status="COMPLETED"))
            value = dict(status="WORKER_FAILED" if fail == label else "WORKER_COMPLETED",
                         cleanup=dict(worker_reaped=True, worker_return_code=1 if fail == label else 0))
            write(status, value)
            return value
        with patch("paused_preview.preflight", return_value=({}, runtime, {}, {}, {}, [record], {} , [record])), \
             patch("paused_preview.identity", return_value=record), \
             patch("paused_preview.validate_code_manifest", return_value=[record]), \
             patch("paused_preview.own_gpu", side_effect=lock), \
             patch("paused_preview.Budget"), patch("paused_preview.guarded_run", side_effect=run):
            try:
                execute(args)
                error = None
            except RuntimeError as caught:
                error = caught
        return labels, json.loads((args.output / "result.json").read_text()), error

    def test_32k_27_distinct_cases_and_equivalence_precedes_physics(self):
        self.assertEqual(ITERATION, 32000)
        self.assertEqual(len(cases()), 27)
        self.assertEqual(len(set(cases())), 27)
        self.assertEqual(sum(method == "masked" for method, _, _ in cases()), 15)
        self.assertEqual({clip for _, _, clip in cases()}, set(FIXED))
        before = dict(os.environ)
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        with tempfile.TemporaryDirectory() as temporary:
            labels, result, error = self.fixture(Path(temporary))
        self.assertIsNone(error)
        self.assertEqual(labels[0], "equivalence")
        self.assertEqual(labels[-1], "comparison")
        self.assertEqual(len(labels), 56)
        for replay, media in zip(labels[1:-1:2], labels[2:-1:2]):
            self.assertEqual(replay.removesuffix("_replay"), media.removesuffix("_media"))
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["new_training_updates"], 0)
        self.assertFalse(result["original_training_restarted"])
        self.assertEqual(dict(os.environ), before)
        self.assertTrue(all(signal.getsignal(s) == value for s, value in handlers.items()))

    def test_failed_equivalence_stops_without_physics_or_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels, result, error = self.fixture(Path(temporary), fail="equivalence")
        self.assertIsNotNone(error)
        self.assertEqual(labels, ["equivalence"])
        self.assertEqual(result["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
