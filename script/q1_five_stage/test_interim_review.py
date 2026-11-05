"""Intermediate resource boundaries and the unchanged final-completion gate."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import FIXED, identity
from interim_review import GIB, ReviewBudget, copy_saved_cases, fixed_rows, resource_limits, validate_saved_case
from native_review import ready_training
import runtime_guard


class IntermediateTests(unittest.TestCase):
    def setUp(self):
        self.memory = dict(container_limit_bytes=34292629504, total_bytes=21 * GIB,
                           oom_kill=0, under_oom=0)
        self.gpu = dict(used_mib=16177, temperature_c=52, recovery="None")

    def test_reserves_exit_below_original_training_pressure(self):
        resource_limits(self.memory, self.gpu, initial=True)
        for key, value in (("total_bytes", 24 * GIB), ("container_limit_bytes", 30 * GIB),
                           ("oom_kill", 1), ("under_oom", 1)):
            bad = dict(self.memory, **{key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                resource_limits(bad, self.gpu)
        with self.assertRaises(RuntimeError):
            resource_limits(dict(self.memory, total_bytes=22 * GIB), self.gpu, initial=True)

    def test_graphics_cannot_consume_training_reserve(self):
        for key, value in (("used_mib", 18 * 1024), ("temperature_c", 80), ("recovery", "Reset")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                resource_limits(self.memory, dict(self.gpu, **{key: value}))

    def test_shared_mapping_rss_is_diagnostic_and_total_remains_binding(self):
        with tempfile.TemporaryDirectory() as directory, patch("interim_review.live"), \
                patch("interim_review.memory", return_value=self.memory), \
                patch("interim_review.process", return_value={}), \
                patch("interim_review.subtree", return_value=[dict(pid=1, VmRSS_bytes=6 * GIB)]), \
                patch("interim_review.gpu_snapshot", return_value=self.gpu), \
                patch("interim_review.allocated_tree", return_value=dict(unique_allocated_bytes=0)), \
                patch("interim_review.shutil.disk_usage") as disk:
            disk.return_value.free = 200 * GIB
            budget = ReviewBudget({}, Path(directory))
            self.assertEqual(budget.check()["review_subtree_rss_bytes"], 6 * GIB)
            self.memory["total_bytes"] = 24 * GIB
            with self.assertRaisesRegex(RuntimeError, "memory reserve"):
                budget.check()
            saved = json.loads((Path(directory) / "resources.json").read_text())
            self.assertEqual(saved["maximum_total_bytes"], 24 * GIB)

    def test_selects_exact_three_fixed_clips_in_order(self):
        frozen = dict(rows=[dict(q1_primebot=dict(path="/data/" + name + ".npz"))
                            for name in (*reversed(FIXED), "unrelated")])
        self.assertEqual([row["q1_primebot"]["path"] for row in fixed_rows(frozen)],
                         ["/data/" + name + ".npz" for name in FIXED])
        with self.assertRaises(RuntimeError):
            fixed_rows(dict(rows=frozen["rows"][1:]))

    def test_final_cli_still_rejects_live_training(self):
        plan = dict(purpose="train", robots=["g1", "q1"], code_commit="a" * 40,
                    cfg=dict(iterations=160000), output="/data/train")
        pipeline = dict(status="TRAINING", formal_training_completed=False)
        for iteration in (0, 16000):
            audit = dict(status="PASSED", cuda_initialized=False, iteration=iteration)
            with self.subTest(iteration=iteration), self.assertRaises(RuntimeError):
                ready_training(copy.deepcopy(plan), pipeline, {}, audit, "q1", {})

    def test_saved_case_requires_the_complete_exact_reference(self):
        directory = Path("/review")
        item = {key: dict(path="/data/clip.npz", sha256=str(i) * 64)
                for i, key in enumerate(("q1_primebot", "q1_morph", "human"))}
        row = dict(clip="clip", seed=0, interface="motion", evaluation_randomization=False,
                   complete=True, failure_frame=None, valid_steps=10, attempted_steps=10,
                   reference_steps=10, trace="/review/rollout.npz", reference_sha256="0" * 64,
                   native_reference_sha256="1" * 64, human_sha256="2" * 64)
        validate_saved_case(row, item, directory)
        for key, value in (("human_sha256", "3" * 64), ("seed", 1), ("complete", False),
                           ("attempted_steps", 9), ("trace", "/another/rollout.npz")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_saved_case(dict(row, **{key: value}), item, directory)

    def test_saved_trajectory_copy_preserves_source_and_checks_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.npz"
            original.write_bytes(b"existing physical trajectory")
            before = identity(original)
            case = dict(case=dict(clip=FIXED[0], trace=str(original)), files=[before],
                        source_authorization={}, source_guard={})
            copy_saved_cases([case], root / "new")
            self.assertEqual(identity(original), before)
            self.assertEqual((root / "new/review" / (FIXED[0] + "_s0/original.npz")).read_bytes(),
                             original.read_bytes())
            original.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed before copy"):
                copy_saved_cases([case], root / "rejected")

    def test_resource_abort_reaps_only_its_own_process_group(self):
        class StopBudget:
            calls = 0

            def check(self):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("review reserve reached")
                return {}

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(runtime_guard, "signal_group", wraps=runtime_guard.signal_group) as signals:
                with self.assertRaisesRegex(RuntimeError, "review reserve"):
                    runtime_guard.guarded_run([sys.executable, "-c", "import time;time.sleep(30)"],
                        folder, StopBudget(), folder / "guard.json", folder / "stdout.log",
                        check_interval_s=.01, grace_s=.1)
            result = json.loads((folder / "guard.json").read_text())
            self.assertTrue(result["cleanup"]["worker_reaped"])
            self.assertNotEqual(result["cleanup"]["worker_return_code"], 0)
            self.assertTrue(signals.call_args_list)
            self.assertEqual({call.args[0].pid for call in signals.call_args_list},
                             {result["worker_pid"]})


if __name__ == "__main__":
    unittest.main()
