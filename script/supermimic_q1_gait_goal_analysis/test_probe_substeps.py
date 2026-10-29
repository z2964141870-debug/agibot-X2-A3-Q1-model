"""Check fixed observation scope and authorization without constructing physics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from probe_substeps import authorize, case_spec, compare_initial, identity, WALKS, SEEDS
import probe_substeps


class ProbeTests(unittest.TestCase):
    def specification(self):
        checkpoint = dict(path="server-only.pt", bytes=1, sha256="a" * 64)
        selected = {key: checkpoint for key in ("initial", "stage_01", "stage_02")}
        return dict(maximum_controls_per_line=3180, maximum_physical_substeps_per_line=12720,
            maximum_policy_batches_per_line=6360, maximum_worlds_at_once=1,
            control_dt_s=.02, physics_dt_s=.005, optimizer_updates=0, automatic_retry=False,
            maximum_whole_wall_seconds_per_line=1800, grace_seconds=10,
            lines={line: dict(cases=case_spec(line, selected)) for line in ("base", "ppo")})

    def test_independent_lines_have_exact_single_clip_bounds_and_paired_seeds(self):
        for line, spec in self.specification()["lines"].items():
            cases = spec["cases"]
            self.assertEqual(len(cases), 12)
            self.assertEqual(sum(case["maximum_controls"] for case in cases), 3180)
            self.assertEqual(sum(case["maximum_physical_substeps"] for case in cases), 12720)
            for clip in WALKS:
                subset = [row for row in cases if row["clip"] == clip]
                self.assertEqual([row["seed"] for row in subset], list(SEEDS) * 2)
            if line == "ppo":
                self.assertEqual(cases[0]["checkpoint"], cases[-1]["checkpoint"])
        with self.assertRaises(ValueError):
            case_spec("combined", {})

    def test_new_exact_plan_authorization_and_zero_update_bounds_are_required(self):
        plan = self.specification()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            path.write_text(json.dumps(plan))
            record = dict(status="USER_AUTHORIZED_ZERO_OPTIMIZER_SUBSTEP_OBSERVATION",
                plan_sha256=identity(path)["sha256"], allowed_lines=["base", "ppo"],
                maximum_controls_per_line=3180, optimizer_updates=0)
            authorize(plan, path, record, "base")
            authorize(plan, path, record, "ppo")
            for key, wrong in (("status", "OLD_TRAINING_AUTHORIZED"), ("plan_sha256", "b" * 64),
                    ("allowed_lines", []), ("maximum_controls_per_line", 6360), ("optimizer_updates", 1)):
                changed = dict(record, **{key: wrong})
                with self.assertRaises(ValueError):
                    authorize(plan, path, changed, "base")
            for key, wrong in (("maximum_worlds_at_once", 2), ("physics_dt_s", .01),
                    ("automatic_retry", True), ("maximum_physical_substeps_per_line", 12724)):
                changed = dict(plan, **{key: wrong})
                with self.assertRaises(ValueError):
                    authorize(changed, path, record, "base")
            changed = copy.deepcopy(plan)
            changed["lines"]["base"]["cases"][0]["seed"] = 0
            with self.assertRaises(ValueError):
                authorize(changed, path, record, "base")
            changed = copy.deepcopy(plan)
            changed["lines"]["base"]["cases"][0] = dict(changed["lines"]["base"]["cases"][1])
            with self.assertRaises(ValueError):
                authorize(changed, path, record, "base")

    def test_initial_equality_is_separate_from_unproven_full_solver_replay(self):
        first = dict(history=np.ones((1, 10, 80)), native_qpos=np.ones((1, 29)))
        same = {key: value.copy() for key, value in first.items()}
        result = compare_initial(first, same)
        self.assertTrue(result["all_recorded_inputs_exact"])
        self.assertFalse(result["complete_solver_state_equivalence_established"])
        self.assertFalse(result["historical_rollout_replay_established"])
        same["native_qpos"][0, 0] += .1
        result = compare_initial(first, same)
        self.assertFalse(result["all_recorded_inputs_exact"])
        self.assertAlmostEqual(result["fields"]["native_qpos"]["maximum_absolute_difference"], .1)
        with self.assertRaises(ValueError):
            compare_initial(first, {})
        unsigned = compare_initial(dict(rng=np.array([1], np.uint8)), dict(rng=np.array([2], np.uint8)))
        self.assertEqual(unsigned["fields"]["rng"]["maximum_absolute_difference"], 1.)

    def test_readonly_preparation_freezes_runtime_files_and_rejects_changed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, tools, site = (root / name for name in ("runtime", "script/tools", "site"))
            for path in (runtime, tools, site):
                path.mkdir(parents=True)
            config = runtime / "q1_config.yaml"
            config.write_text("synthetic_config: true\n")
            (tools / "observer.py").write_text("# Synthetic source inventory\n")
            (site / "backend.py").write_text("# No environment or solver\n")
            helper = root / "script/supermimic_q1_gait_goal/supervise.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("# Synthetic supervisor identity\n")
            checkpoint, calibration, library = (root / name for name in ("saved.pt", "calibration.json", "library.json"))
            checkpoint.write_bytes(b"Not a loaded checkpoint")
            calibration.write_text("{}")
            rows = []
            for index, name in enumerate(WALKS):
                path = root / (str(index) + ".csv")
                path.write_text("synthetic_reference\n")
                path.with_suffix(".json").write_text("{}")
                rows.append(dict(name=name, path=str(path), sha256=identity(path)["sha256"]))
            library.write_text(json.dumps(dict(splits=dict(train=rows))))
            arguments = ["--runtime", str(runtime), "--library", str(library), "--checkpoint", str(checkpoint), "--calibration", str(calibration)]
            frozen = [identity(config), identity(library), identity(checkpoint), identity(calibration)]
            for line in ("base", "ppo"):
                trial = root / line
                (trial / "run").mkdir(parents=True)
                (trial / "plan.json").write_text(json.dumps(dict(source=frozen, phases=[dict(command=arguments)])))
                candidates = [dict(label=name, checkpoint=identity(checkpoint)) for name in ("initial", "stage_01", "stage_02")]
                (trial / "run/checkpoint_selection.json").write_text(json.dumps(dict(candidates=candidates)))
            args = SimpleNamespace(base=root / "base", ppo=root / "ppo", output=root / "preparation", source_base_commit="f" * 40)
            with patch.multiple(probe_substeps, ROOT=root, HERE=tools, SITE=site, BACKEND_FILES=("backend.py",), SCRIPT_DIRS=()):
                probe_substeps.prepare(args)
                result = json.loads((args.output / "observation_plan.json").read_text())
                self.assertEqual(result["physical_substeps_executed"], 0)
                self.assertEqual(result["policy_calls_executed"], 0)
                self.assertIn(str(config), [row["path"] for row in result["source"]])
                self.assertNotIn(str(runtime), [row["path"] for row in result["source"]])
                with self.assertRaises(FileExistsError):
                    probe_substeps.prepare(args)
                args.output = root / "reject_changed_source"
                config.write_text("synthetic_config: changed\n")
                with self.assertRaises(ValueError):
                    probe_substeps.prepare(args)
                self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
