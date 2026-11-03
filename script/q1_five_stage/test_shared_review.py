"""Serial coverage, resource bounds, and failed formal/phase rejection without CUDA."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared_review import (comparison_command, comparisons, execute, graphics, jobs,
                           phase_budget, phase_passed, replay_command)
from common import FIXED


class SharedReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(g1=dict(path=name + ".npz"), frames=100) for name in (*FIXED, "extra")]

    def test_matrix_separates_native_robots_common_q1_and_interfaces(self):
        matrix = jobs()
        self.assertEqual(len(matrix), 24)
        self.assertEqual(len({row["name"] for row in matrix}), 24)
        self.assertEqual(sum(row["mode"] == "fixed" for row in matrix), 12)
        self.assertTrue(all(row["robot"] == "q1" for row in matrix if row["plant"] == "common"))
        self.assertEqual({(row["robot"], row["interface"], row["stage"]) for row in matrix if row["plant"] == "native"},
                         {(r, i, s) for r in ("g1", "q1") for i in ("motion", "human") for s in ("initial", "final")})
        self.assertEqual(sum(phase_budget(row, self.rows)["videos"] for row in matrix), 36)
        expected = 12 * 3 * 3 * 99 + 12 * 4 * 99
        self.assertEqual(sum(phase_budget(row, self.rows)["maximum_physical_transitions"] for row in matrix), expected)
        self.assertEqual(len(comparisons()), 14)

    def test_budget_counts_repeats_and_full_raw_media_not_just_guard_folder(self):
        job = jobs()[0]
        budget = phase_budget(job, self.rows)
        self.assertEqual(budget["maximum_physical_transitions"], 3 * 3 * 99)
        self.assertGreater(budget["maximum_bytes"], 300 * 1280 * 392 * 3)
        for rows in (self.rows[1:], [dict(g1=dict(path=name + ".npz"), frames=1) for name in FIXED]):
            with self.assertRaises(RuntimeError):
                phase_budget(job, rows)

    def test_failed_formal_preflight_never_creates_output_or_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data/supermimic_for_q1/review"
            with patch("shared_review.ROOT", root), patch("shared_review.read", return_value={}), \
                 patch("shared_review.preflight", side_effect=RuntimeError("formal training incomplete")), \
                 patch("shared_review.Runner") as runner:
                with self.assertRaisesRegex(RuntimeError, "formal training incomplete"):
                    execute(root / "config", root / "pipeline", root / "runtime", output, "a" * 40)
                runner.assert_not_called()
            self.assertFalse(output.exists())

    def synthetic_suite(self, directory, failed=False):
        root = Path(directory)
        output = root / "data/supermimic_for_q1/evaluation"
        names = [*FIXED, *(f"clip_{i}" for i in range(390))]
        rows = [dict(g1=dict(path=name + ".npz"), frames=10) for name in names]
        plan = dict(source="/source", output=str(root / "train"), cfg=dict(iterations=160000, num_envs=1024, wm_only_iters=2000))
        config = dict(output="unused", primebot_review="/primebot", frozen_data="frozen", morph_python="python",
                      primebot_pipeline="primebot_pipeline", delivery_manifest="manifest", mac_receipt="mac",
                      expected_boot="boot", expected_gpu="gpu")
        runtime = dict(morph_source="/source", morph_python="python", primebot_python="primebot_python", adapted_source="/primebot_source",
                       graphics_dir="/graphics", ffmpeg="encoder", expected_boot="boot", expected_gpu="gpu")
        audits = dict(initial=Path("initial_audit"), final=Path("final_audit"))
        for robot in ("g1", "q1"):
            events = root / "train/run" / robot / "events.out.tfevents.synthetic"
            events.parent.mkdir(parents=True)
            events.touch()
        calls = []
        matrix = {row["name"]: row for row in jobs()}

        def read(path):
            path = str(path)
            if path == "config":
                return config
            if path == "runtime":
                return runtime
            if path == "/primebot/checkpoint_audit.json":
                return dict(final=dict(checkpoint=dict(path="checkpoint", sha256="a", bytes=1)))
            if path == "/primebot/exposure_comparison_checkpoint.json":
                return dict(status="UNAVAILABLE_EARLY_STOP_BEFORE_TARGET")
            if path.endswith("source_receipt.json"):
                return dict(log_rows=2)
            if path.endswith("comparison.json"):
                return dict(status="MISSING_COMPARABLE_PREFIX")
            phase = str(Path(path).parent.parent.relative_to(output))
            row = matrix[phase]
            selected, seeds = (names, (0,)) if row["mode"] == "library" else (FIXED, (0, 1, 2))
            return dict(status="PASSED", cases=[dict(clip=name, seed=seed, interface=row["interface"], complete=False,
                failure_frame=1, valid_steps=0, attempted_steps=1, reference_steps=9, evaluation_randomization=False,
                reference_sha256="a", trace="server.npz") for name in selected for seed in seeds])

        def phase(name, command, *args):
            calls.append(name)
            if failed:
                return dict(status="FAILED"), "CUDA out of memory"
            if name in matrix:
                self.assertEqual(Path(command[command.index("--output") + 1]), output / name / "review")
                self.assertEqual(Path(command[command.index("--pipeline") + 1]), output / "training_pipeline.json")
            return dict(status="COMPLETED", inputs_unchanged=True, cgroup_oom_kill_delta=0, gpu_recovery_action="None",
                guard=dict(status="WORKER_COMPLETED", cleanup=dict(worker_reaped=True, worker_return_code=0))), ""

        with patch("shared_review.ROOT", root), patch("shared_review.read", side_effect=read), \
             patch("shared_review.preflight", return_value=(Path("plan"), plan, dict(rows=rows), names, audits, [])), \
             patch("shared_review.identity", return_value=dict(path="checkpoint", sha256="a", bytes=1)), \
             patch("shared_review.shutil.copyfile"), \
             patch("shared_review.shutil.disk_usage", return_value=type("Disk", (), dict(free=1000 * 1024**3))()), \
             patch("shared_review.Runner") as runner, patch("shared_delivery.write_conclusions"), \
             patch("shared_delivery.prepare", return_value=dict(artifacts=[], media=[{}] * 36)) as prepare:
            runner.return_value.phase.side_effect = phase
            if failed:
                with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                    execute(Path("config"), Path("pipeline"), Path("runtime"), output, "a" * 40)
                self.assertEqual(calls, [jobs()[0]["name"]])
                prepare.assert_not_called()
                import json
                self.assertEqual(json.loads((output / "evaluation.json").read_text())["status"], "FAILED")
            else:
                value = execute(Path("config"), Path("pipeline"), Path("runtime"), output, "a" * 40)
                self.assertEqual(value["status"], "SHARED_EVALUATION_COMPLETED_PENDING_MAC_DELIVERY")
                self.assertFalse(value["goal_completed"])
                self.assertEqual(value["videos"], 36)
                self.assertEqual(calls, [row["name"] for row in jobs() + comparisons()] + ["curves/g1", "curves/q1"])
                prepare.assert_called_once()

    def test_serial_suite_finishes_all_replays_comparisons_curves_before_mac(self):
        with tempfile.TemporaryDirectory() as directory:
            self.synthetic_suite(directory)

    def test_replay_failure_keeps_evidence_and_never_retries_or_compares(self):
        with tempfile.TemporaryDirectory() as directory:
            self.synthetic_suite(directory, failed=True)

    def test_phase_requires_success_reaping_unchanged_sources_and_device(self):
        success = dict(status="COMPLETED", inputs_unchanged=True, cgroup_oom_kill_delta=0, gpu_recovery_action="None",
            guard=dict(status="WORKER_COMPLETED", cleanup=dict(worker_reaped=True, worker_return_code=0)))
        phase_passed(success)
        for mutation in (lambda x: x.update(status="FAILED"), lambda x: x.update(inputs_unchanged=False),
                         lambda x: x.update(cgroup_oom_kill_delta=1), lambda x: x.update(gpu_recovery_action="Reset"),
                         lambda x: x["guard"]["cleanup"].update(worker_reaped=False),
                         lambda x: x["guard"]["cleanup"].update(worker_return_code=-15)):
            value = copy.deepcopy(success)
            mutation(value)
            with self.assertRaises(RuntimeError):
                phase_passed(value)

    def test_each_replay_output_lives_inside_its_guarded_phase(self):
        root = Path("/server/review")
        plan = dict(output="/train", cfg=dict(iterations=160000))
        config = dict(morph_python="python", frozen_data="frozen")
        runtime = dict(adapted_source="primebot", primebot_python="primebot_python")
        audits = dict(initial="audit0", final="auditfinal")
        for row in jobs():
            output = root / row["name"] / "review"
            command = replay_command(row, config, runtime, Path("plan"), plan, Path("pipeline"), audits,
                                     output, "primebot_checkpoint", "a" * 40)
            self.assertEqual(command[command.index("--output") + 1], str(output))
            self.assertEqual("--render" in command, row["mode"] == "fixed")
            self.assertEqual("--library" in command, row["mode"] == "library")
            self.assertIn("--pipeline", command)
            self.assertIn("--audit", command)
            self.assertEqual("--primebot-source" in command, row["plant"] == "common")
            self.assertEqual(command[0], "primebot_python" if row["plant"] == "common" else "python")
            if row["plant"] == "common":
                self.assertEqual(command[command.index("--plant-python") + 1], "primebot_python")
                self.assertEqual(command[command.index("--morph-python") + 1], "python")

    def test_direct_comparisons_use_primebot_final_and_shared_motion_only(self):
        config = dict(morph_python="python", frozen_data="frozen", primebot_review="/primebot")
        for row in comparisons():
            command, target, extra = comparison_command(row, config, Path("/shared"))
            self.assertTrue(target.is_relative_to(Path("/shared") / row["name"]))
            self.assertEqual(command[command.index("--right") + 1], str(extra[1]))
            if row["direct"]:
                self.assertEqual(row["interface"], "motion")
                self.assertEqual(str(extra[0]), f"/primebot/final_{row['mode']}/result.json")
                self.assertNotIn("--left-interface", command)
            elif row["plant"] == "native":
                self.assertIn("--robot", command)

    def test_graphics_runtime_restores_parent_environment_after_failure(self):
        import os
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "parent", "EGL_PLATFORM": "parent-platform"}):
            before = dict(os.environ)
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with graphics(dict(graphics_dir="/graphics", ffmpeg="/encoder")):
                    self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/graphics/lib:parent")
                    self.assertEqual(os.environ["Q1_REVIEW_FFMPEG"], "/encoder")
                    raise RuntimeError("stop")
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
