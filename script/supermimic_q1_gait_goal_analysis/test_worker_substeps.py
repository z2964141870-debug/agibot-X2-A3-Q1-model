"""Exercise actual worker bookkeeping on arithmetic-only one-control stand-ins."""

import csv
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import probe_substeps
from substep_observer import SubstepObserver


class Teacher:
    def __init__(self):
        self.hook = None

    def requires_grad_(self, enabled):
        return self

    def eval(self):
        return self

    def register_forward_pre_hook(self, function):
        self.hook = function
        return SimpleNamespace(remove=lambda: setattr(self, "hook", None))

    def emit_input_only(self):
        self.hook(self, (torch.zeros(1, 10, 80), torch.zeros(1, 590)))


class Backend:
    def __init__(self):
        self.native_calls = self.closes = 0

    def _execute_device_steps(self, count):
        self.native_calls += count

    def close(self):
        self.closes += 1


class Sampler:
    def __init__(self, env):
        self.records, env.sampler = [], self
        self.coefficients = dict(actuator_forcerange=np.tile([-22., 22.], (1, 22, 1)))
        self.actuation = dict(actuator_ids=np.arange(22), force_range=np.tile([-22., 22.], (22, 1)))

    def before(self, control, substep):
        return None

    def after(self, control, substep, before):
        self.records.append({"pre_" + key: np.zeros((1, count)) for key, count in
            (("qpos", 29), ("qvel", 28), ("qacc_warmstart", 28), ("qacc", 28),
             ("time", 1), ("qfrc_applied", 28), ("xfrc_applied", 6))})

    def archive(self):
        count = len(self.records)
        return dict(control_index=np.zeros(count, dtype=np.int64), solve_actuator_force=np.zeros((count, 1, 22)))

    def contract(self):
        return dict(synthetic_only=True)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    def tearDown(self):
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])

    def fixture(self, root):
        output = root / "output"
        output.mkdir()
        library, calibration = root / "library.json", root / "calibration.json"
        library.write_text("{}")
        calibration.write_text(json.dumps(dict(thresholds=[1., 1., 1., 1.])))
        env = SimpleNamespace(backend=Backend(), device="cpu", closes=0,
            supermimic_mdp=SimpleNamespace(motion=SimpleNamespace(clip_len=torch.tensor([2]))))
        env.close = lambda: setattr(env, "closes", env.closes + 1)
        def rollout(env, teacher, head, thresholds, reference, **options):
            teacher.emit_input_only()
            env.backend._execute_device_steps(4)
            options["progress"](0)
            return dict(steps=np.array([1]), initial=np.zeros((1, 57)))
        def write_csv(path, rows):
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        loaded = SimpleNamespace(prepare_seeded=lambda *args: (None, env, None), policy_digest=lambda teacher: "fixed_teacher",
            configure=lambda *args: None, rollout=rollout, SubstepObserver=SubstepObserver, WarpSampler=Sampler,
            process_identity=lambda: dict(synthetic_only=True), load_policy=lambda *args: Teacher(),
            Q1_JOINT_NAMES=[f"joint{i}" for i in range(22)], Head=None, PhaseSoleReference=lambda *args: None,
            clip_metrics=lambda *args: dict(synthetic_only=True), gait_clip=lambda *args: dict(synthetic_only=True), write_csv=write_csv)
        case = dict(condition="initial", checkpoint=dict(path="unused.pt"), clip="synthetic_clip", repeat=0, seed=32000, maximum_controls=1)
        plan = dict(source=[], lines=dict(base=dict(inputs=dict(runtime=str(root / "runtime"), library=str(library),
            checkpoint="unused.pt", calibration=str(calibration)), cases=[case], rows=[dict(name="synthetic_clip")])))
        return SimpleNamespace(line="base", output=output), plan, loaded, env

    def run_worker(self, args, plan, loaded):
        with patch.object(probe_substeps, "components", return_value=loaded), patch.object(torch, "load", return_value={}), \
                patch.object(torch.cuda, "get_rng_state", return_value=torch.zeros(4, dtype=torch.uint8)), \
                patch.object(torch.cuda, "manual_seed_all"), patch.object(torch.cuda, "empty_cache"):
            probe_substeps.worker(args, plan)

    def test_success_counts_each_synthetic_control_once_and_closes_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, loaded, env = self.fixture(Path(directory))
            self.run_worker(args, plan, loaded)
            result = json.loads((args.output / "results.json").read_text())
            self.assertEqual(result["controlled_transitions"], 1)
            self.assertEqual(result["completed_physical_substeps"], 4)
            self.assertEqual(result["attempted_native_substeps"], 4)
            self.assertEqual(result["returned_native_substeps"], 4)
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(env.backend.native_calls, 4)
            self.assertEqual((env.closes, env.backend.closes), (1, 1))
            self.assertEqual(result["cases"][0]["policy_batches"], dict(teacher=1, head=0))

    def test_failed_status_save_does_not_double_count_the_finished_case(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, loaded, env = self.fixture(Path(directory))
            original_save, injected = probe_substeps.save, False
            def fail_once(path, record):
                nonlocal injected
                if (path == args.output / "run/status.json" and record.get("status") == "RUNNING"
                        and record.get("controlled_transitions") == 1 and not injected):
                    injected = True
                    raise OSError("Synthetic completed-case status write failure")
                original_save(path, record)
            with patch.object(probe_substeps, "save", side_effect=fail_once):
                with self.assertRaises(OSError):
                    self.run_worker(args, plan, loaded)
            status = json.loads((args.output / "run/status.json").read_text())
            self.assertTrue(injected)
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["total_completed_substeps_lower_bound"], 4)
            self.assertEqual(status["total_control_calls"], 1)
            self.assertEqual(status["total_attempted_native_substeps"], 4)
            self.assertEqual(status["total_returned_native_substeps"], 4)
            self.assertFalse(status["failed_case"]["possible_partial_native_launch"])
            self.assertEqual(env.backend.native_calls, 4)
            self.assertEqual((env.closes, env.backend.closes), (1, 1))
            self.assertFalse((args.output / "results.json").exists())

    def test_partial_second_case_counts_prior_verified_substeps_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, loaded, env = self.fixture(Path(directory))
            cases = plan["lines"]["base"]["cases"]
            cases.append(dict(cases[0], repeat=1, seed=32001))
            original_after, returned = Sampler.after, 0
            def fail_second_substep(sampler, control, substep, before):
                nonlocal returned
                returned += 1
                if returned == 6:
                    raise RuntimeError("Synthetic asynchronous read failure")
                original_after(sampler, control, substep, before)
            with patch.object(Sampler, "after", fail_second_substep):
                with self.assertRaises(RuntimeError):
                    self.run_worker(args, plan, loaded)
            status = json.loads((args.output / "run/status.json").read_text())
            self.assertEqual(status["completed_physical_substeps"], 4)
            self.assertEqual(status["failed_case"]["completed_substeps_lower_bound"], 1)
            self.assertEqual(status["total_completed_substeps_lower_bound"], 5)
            self.assertEqual(status["total_control_calls"], 2)
            self.assertEqual(status["total_attempted_native_substeps"], 6)
            self.assertEqual(status["total_returned_native_substeps"], 6)
            self.assertTrue(status["failed_case"]["possible_partial_native_launch"])
            self.assertEqual(env.backend.native_calls, 6)
            self.assertEqual((env.closes, env.backend.closes), (2, 2))
            self.assertFalse((args.output / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
