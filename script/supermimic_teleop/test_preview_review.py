"""Fail-closed preview orchestration and published-code/telemetry contracts."""

import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from preview_review import (HORIZONS, REQUIRED_CODE, TRAINING, execute, identity,
                            validate_code_manifest, validate_stream)
from freeze_preview_code import published_manifest


def telemetry(horizon=4, frame=0):
    return dict(requested_preview_ns=horizon*20_000_000,
        scheduled_origin_to_control_ns=horizon*20_000_000, policy_prefix_frames=horizon,
        source_preprocessing="offline_unverified", measured_end_to_end_latency_ns=None,
        used_sequences=list(range(frame+1, frame+horizon+1)),
        provenance=["measured_exact"]*horizon + ["masked_hold_placeholder"]*(10-horizon),
        measured_frames=horizon, observed_prefix_frames=horizon, cpu_policy_wall_compute_ns=1000)


class PreviewReviewContractTests(unittest.TestCase):
    def test_manifest_export_checks_git_bytes_and_full_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for relative in REQUIRED_CODE:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"# fixture\n")

            def git(command, **kwargs):
                if command[1] == "rev-parse":
                    return "a"*40 + "\n"
                if command[1] == "ls-tree":
                    return b"\0".join(name.encode() for name in REQUIRED_CODE) + b"\0"
                return b"# fixture\n"

            with patch("freeze_preview_code.subprocess.check_output", side_effect=git):
                self.assertEqual(len(published_manifest(root, "a"*40)["files"]), len(REQUIRED_CODE))
                with self.assertRaisesRegex(ValueError, "HEAD differs"):
                    published_manifest(root, "b"*40)
                (root / REQUIRED_CODE[0]).write_text("# modified\n")
                with self.assertRaisesRegex(ValueError, "differs from published"):
                    published_manifest(root, "a"*40)

    def test_rejects_wrong_or_modified_published_code_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = []
            for relative in REQUIRED_CODE:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# independent preview test fixture\n")
                row = identity(path)
                files.append(dict(relative=relative, bytes=row["bytes"], sha256=row["sha256"]))
            manifest = root / "manifest.json"

            def write(value):
                manifest.write_text(json.dumps(value))
                return identity(manifest)["sha256"]

            content = dict(evaluation_commit="a"*40, files=files)
            digest = write(content)
            self.assertEqual(len(validate_code_manifest(manifest, digest, "a"*40, root)), len(REQUIRED_CODE))
            with self.assertRaisesRegex(RuntimeError, "SHA or published commit"):
                validate_code_manifest(manifest, "b"*64, "a"*40, root)
            changed = root / REQUIRED_CODE[-1]
            changed.write_text("# different code\n")
            with self.assertRaisesRegex(RuntimeError, "differs from published"):
                validate_code_manifest(manifest, digest, "a"*40, root)
            changed.write_text("# independent preview test fixture\n")
            for invalid in (dict(content, evaluation_commit="b"*40),
                            dict(content, files=files[:-1]),
                            dict(content, files=files+[files[0]]),
                            dict(content, files=[dict(files[0], relative="script/supermimic_teleop/../x.py")])):
                with self.subTest(invalid=invalid.get("evaluation_commit")), self.assertRaises(RuntimeError):
                    validate_code_manifest(manifest, write(invalid), "a"*40, root)
            digest = write(content)
            changed.unlink()
            changed.symlink_to(root / REQUIRED_CODE[0])
            with self.assertRaisesRegex(RuntimeError, "symlinks"):
                validate_code_manifest(manifest, digest, "a"*40, root)

    def test_runtime_telemetry_rejects_future_identity_or_latency_claims(self):
        for horizon in HORIZONS:
            validate_stream(telemetry(horizon), horizon, 0)
        value = telemetry()
        terminal = dict(value, used_sequences=[1, 2, 2, 2], measured_frames=2, observed_prefix_frames=2,
            provenance=["measured_exact"]*2 + ["declared_clip_terminal_hold"]*2 + ["masked_hold_placeholder"]*6)
        validate_stream(terminal, 4, 0)
        for bad in (dict(value, measured_end_to_end_latency_ns=80_000_000),
                    dict(value, used_sequences=[1, 2, 3, 5]),
                    dict(value, used_sequences=[1, 2, 2, 4]),
                    dict(value, observed_prefix_frames=3),
                    dict(terminal, used_sequences=[1, 2, 1, 2]),
                    dict(terminal, provenance=["declared_clip_terminal_hold"]*4 + ["masked_hold_placeholder"]*6),
                    dict(value, provenance=["measured_exact"]*10)):
            with self.subTest(bad=bad["used_sequences"]), self.assertRaises(RuntimeError):
                validate_stream(bad, 4, 0)


class PreviewControllerTests(unittest.TestCase):
    def run_fixture(self, folder, runtime_text="Ran 1 test in 0.010s\n\nOK\n", fail_label=None):
        source = folder / "source.py"
        source.write_text("# original source fixture\n")
        source_id = identity(source)
        plan_path = folder / "plan.json"
        plan_path.write_text("{}")
        args = SimpleNamespace(child=None, output=folder / "preview", reuse_audit=source,
            code_manifest=source, manifest_sha256=source_id["sha256"], evaluation_commit="a"*40,
            config=source, runtime_config=source)
        runtime = dict(morph_python=sys.executable, primebot_python=sys.executable,
            ffmpeg="/existing/ffmpeg", graphics_dir="/existing/graphics")
        plan = dict(source="/frozen/morph")
        calls = []
        actual_read = lambda path: json.loads(Path(path).read_text())

        def read(path):
            if path == TRAINING / "e1024/train/status.json":
                return dict(inputs=[source_id])
            return actual_read(path)

        def fake_guard(command, workdir, budget, status_path, log_path, **kwargs):
            label = status_path.name.removesuffix(".guard.json")
            calls.append(label)
            phase = dict(status="WORKER_FAILED" if label == fail_label else "WORKER_COMPLETED",
                         cleanup=dict(worker_reaped=True, worker_return_code=1 if label == fail_label else 0))
            status_path.write_text(json.dumps(phase))
            log_path.write_text(runtime_text if label == "runtime" else "phase fixture\n")
            if label == "equivalence":
                (args.output / "equivalence.json").write_text(json.dumps(dict(status="PASSED")))
            if label == "comparison":
                (args.output / "comparison.json").write_text("{}")
            return phase

        with patch("preview_review.validate_code_manifest", return_value=[source_id]), \
                patch("preview_review.preflight", return_value=({}, runtime, plan, {}, [source_id])), \
                patch("preview_review.reusable_audit", return_value=dict(status="PASSED", iteration=16000)), \
                patch("preview_review.read", side_effect=read), \
                patch("preview_review.identity", side_effect=lambda path: source_id if path ==
                      TRAINING / "e1024_train_plan.json" else identity(path)), \
                patch("preview_review.ReviewBudget"), \
                patch("preview_review.guarded_run", side_effect=fake_guard):
            try:
                result = execute(args)
                return calls, result, None
            except RuntimeError as error:
                return calls, actual_read(args.output / "result.json"), error

    def test_runtime_and_actual_h10_equivalence_precede_all_15_physical_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            before = dict(os.environ)
            handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
            calls, result, error = self.run_fixture(Path(temporary))
            self.assertIsNone(error)
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(calls[:2], ["runtime", "equivalence"])
            self.assertEqual(calls[-1], "comparison")
            self.assertEqual(sum(name.endswith("_replay") for name in calls), 15)
            self.assertEqual(sum(name.endswith("_media") for name in calls), 15)
            for first, second in zip(calls[2:-1:2], calls[3:-1:2]):
                self.assertEqual(first.removesuffix("_replay"), second.removesuffix("_media"))
            self.assertFalse(result["final_acceptance"])
            self.assertEqual(dict(os.environ), before)
            self.assertTrue(all(signal.getsignal(number) == handler for number, handler in handlers.items()))

    def test_skipped_native_test_or_failed_phase_prevents_any_later_dispatch(self):
        for runtime_text, fail, expected in (
                ("Ran 1 test in 0.001s\nOK (skipped=1)\n", None, ["runtime"]),
                ("Ran 0 tests in 0.001s\nOK\n", None, ["runtime"]),
                ("Ran 1 test in 0.001s\n\nOK\n", "equivalence", ["runtime", "equivalence"])):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                calls, result, error = self.run_fixture(Path(temporary), runtime_text, fail)
                self.assertIsNotNone(error)
                self.assertEqual(calls, expected)
                self.assertEqual(result["status"], "FAILED")
                self.assertFalse(result["final_acceptance"])


if __name__ == "__main__":
    unittest.main()
