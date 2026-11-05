"""Campaign sequencing and exit behavior; fixtures never run robot physics."""

import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preview_campaign as campaign
from common import identity


class PreviewCampaignTests(unittest.TestCase):
    def fixture(self, folder, fail_method=None, baseline=None, mutate=False):
        source = folder / "published_source.py"
        source.write_text("# controlled fixture\n")
        source_id = identity(source)
        args = SimpleNamespace(config=source, runtime_config=source, reuse_audit=source,
            code_manifest=source, manifest_sha256=source_id["sha256"],
            evaluation_commit="a" * 40, output=folder / "campaign",
            baseline_output=baseline, tail_methods=campaign.DEFAULT_METHODS)
        calls = []
        normal_read = campaign.read

        def read(path):
            return dict(inputs=[source_id]) if path == campaign.TRAINING / \
                "e1024/train/status.json" else normal_read(path)

        def run(child, budget_factory):
            calls.append((child.tail_method, child.output, child.baseline_output))
            child.output.mkdir()
            (child.output / "result.json").write_text("{}")
            (child.output / "comparison.json").write_text("{}")
            if child.tail_method == fail_method:
                raise RuntimeError("preserved study failure")
            if mutate:
                source.write_text("# modified fixture\n")
            count = 15 if child.tail_method == "masked" else 6
            return dict(status="COMPLETED", source_integrity="PASSED", total_cases=count,
                        media_videos=count, comparison=identity(child.output / "comparison.json"))

        with patch("preview_campaign.review.validate_code_manifest", return_value=[source_id]), \
                patch("preview_campaign.review.preflight", return_value=({}, {}, {}, {}, [source_id])), \
                patch("preview_campaign.read", side_effect=read), \
                patch("preview_campaign.review.execute", side_effect=run):
            try:
                value = campaign.execute(args)
                return calls, value, None
            except RuntimeError as error:
                value = json.loads((args.output / "result.json").read_text())
                return calls, value, error

    def test_masked_then_three_methods_reuse_one_completed_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
            calls, value, error = self.fixture(Path(temporary))
            self.assertIsNone(error)
            self.assertEqual([row[0] for row in calls], ["masked", "hold", "damped_cv", "cv"])
            self.assertTrue(all(row[2] == calls[0][1] for row in calls[1:]))
            self.assertEqual(value["total_new_cases"], 33)
            self.assertEqual(value["total_videos"], 33)
            self.assertEqual(value["status"], "COMPLETED")
            self.assertFalse(value["final_acceptance"])
            self.assertTrue(all(signal.getsignal(number) == handler for number, handler in before.items()))

    def test_existing_masked_baseline_does_not_dispatch_its_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            baseline = folder / "already_completed"
            calls, value, error = self.fixture(folder, baseline=baseline)
            self.assertIsNone(error)
            self.assertEqual([row[0] for row in calls], ["hold", "damped_cv", "cv"])
            self.assertTrue(all(row[2] == baseline for row in calls))
            self.assertEqual(value["total_new_cases"], 18)

    def test_failed_study_or_changed_config_prevents_later_methods(self):
        for failure, mutate, expected in (("masked", False, ["masked"]),
                ("damped_cv", False, ["masked", "hold", "damped_cv"]),
                (None, True, ["masked"])):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                calls, value, error = self.fixture(Path(temporary), failure, mutate=mutate)
                self.assertIsNotNone(error)
                self.assertEqual([row[0] for row in calls], expected)
                self.assertEqual(value["status"], "FAILED")

    def test_global_budget_rejects_deadline_and_combined_storage_before_child_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            budget = campaign.CampaignBudget({}, Path(temporary), Path(temporary), 100.)
            for clock, allocated, message in ((101., 0, "two-hour"),
                    (99., 3 * campaign.GIB, "combined output")):
                with self.subTest(message=message), \
                        patch("preview_campaign.time.monotonic", return_value=clock), \
                        patch("preview_campaign.allocated_tree", return_value=dict(unique_allocated_bytes=allocated)), \
                        patch.object(campaign.ReviewBudget, "check") as inner:
                    with self.assertRaisesRegex(RuntimeError, message):
                        budget.check()
                    inner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
