from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from control import ContactCorrectedAssets, Leg50Assets, VARIANT, augment_training_manifest, leg_mask
import run_single_motion
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import TRACKED
from supermimic.q1_sim import Q1Sim, SimCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "supermimic_q1_expanded"))


@unittest.skipUnless(os.environ.get("Q1_CONTROL_TEST_ROOT"), "requires the actual Q1 model/reference")
class LegControlContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(os.environ["Q1_CONTROL_TEST_ROOT"])
        cls.base, cls.legs = ContactCorrectedAssets(cls.root), Leg50Assets(cls.root)

    def test_only_twelve_force_caps_change(self):
        selected = leg_mask(self.base.names)
        np.testing.assert_array_equal(self.legs.effort[selected], self.base.effort[selected] * 2.)
        np.testing.assert_array_equal(self.legs.effort[~selected], self.base.effort[~selected])
        for name in ("scale", "kp", "kd", "q0", "soft_limits", "body_ids", "foot_geoms"):
            np.testing.assert_array_equal(getattr(self.legs, name), getattr(self.base, name))
        self.assertEqual(asdict(self.base.heads().reward_cfg), asdict(self.legs.heads().reward_cfg))
        manifest = self.legs.manifest()
        self.assertEqual(manifest, json.loads(json.dumps(manifest)))
        np.testing.assert_array_equal(manifest["torque_fraction_of_model"], np.where(selected, .5, .25))
        self.assertEqual(manifest["contact_protocol"], self.base.manifest()["contact_protocol"])

    def test_native_and_slotted_models_apply_the_same_variant(self):
        from slotted_sim import SlottedQ1Sim
        reference = self.root / "data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references/train_3419f09ba93c91d75fc1.npz"
        library = MotionLibrary([reference], TRACKED, self.legs.q0, device="cpu",
                                active_indices=self.legs.active_indices)
        cfg = SimCfg(num_envs=2, threads=1, domain_randomization=False,
                     observation_noise=False, action_noise=0.)
        native = Q1Sim(self.legs, library, cfg, self.legs.heads(), seed=0)
        slotted = SlottedQ1Sim(self.legs, library, cfg, self.legs.heads(), seed=0, slots=1)
        try:
            for sim in (native, slotted):
                sim.reset(sim.ids, torch.tensor([0, 0]))
                for model in sim.models:
                    np.testing.assert_array_equal(model.actuator_forcerange[self.legs.aids],
                        np.stack([-self.legs.effort, self.legs.effort], axis=1))
                    np.testing.assert_array_equal(model.actuator_gainprm[self.legs.aids, 0], self.legs.kp)
            action = torch.linspace(-1., 1., 22)[None].repeat(2, 1)
            for _ in range(4):
                actual = native.step(action, auto_reset=False)
                other = slotted.step(action, auto_reset=False)
                for key in ("state", "reward", "joint_rmse", "terminated"):
                    torch.testing.assert_close(actual[key], other[key], rtol=0., atol=0.)
        finally:
            native.close()
            slotted.close()

    def test_checkpoint_contract_round_trip_and_truthful_declaration(self):
        manifest = {"assets": self.legs.manifest(), "training_entry": {"path": "/base.py"},
                    "paper_deviations": ["CPU MuJoCo backend; Q1 simulation PD50/2 and 25% model torque bounds"]}
        augment_training_manifest(manifest, Path(__file__).parent / "entry.py")
        payload = io.BytesIO()
        torch.save({"manifest": manifest}, payload)
        payload.seek(0)
        restored = torch.load(payload, map_location="cpu", weights_only=False)["manifest"]
        self.assertEqual(restored["assets"], self.legs.manifest())
        self.assertIn("50% model leg", restored["paper_deviations"][0])
        self.assertEqual(restored["assets"]["control_variant"]["version"], VARIANT)
        self.assertNotEqual(restored["assets"], self.base.manifest())
        manifest["assets"] = self.base.manifest()
        with self.assertRaises(ValueError):
            augment_training_manifest(manifest, __file__)

    def test_training_dispatch_records_the_live_leg_contract(self):
        from unittest.mock import patch
        import contact_protocol
        import entry
        import train_reproduction
        import train_q1_with_dashboard
        original_assets, original_write = train_reproduction.Q1Assets, train_reproduction.atomic_json
        original_contact = contact_protocol.ContactCorrectedAssets
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "admission.json"

            def fake_loop():
                manifest = {"assets": train_reproduction.Q1Assets(self.root).manifest(),
                    "training_entry": {"path": "/base.py"},
                    "paper_deviations": ["CPU MuJoCo backend; Q1 simulation PD50/2 and 25% model torque bounds"]}
                train_reproduction.atomic_json(path, manifest)

            try:
                with patch.object(sys, "argv", [entry.__file__, "train"]), \
                        patch.object(train_q1_with_dashboard, "main", fake_loop):
                    entry.main()
                recorded = json.loads(path.read_text())
                self.assertEqual(recorded["assets"], self.legs.manifest())
                self.assertEqual(recorded["main_variable"], "LEG_ACTUATOR_FORCE_CAPS_AND_USER_SHORTENED_TIME_BUDGET")
                self.assertEqual(recorded["user_training_budget"]["maximum_training_seconds"], 3600.)
            finally:
                train_reproduction.Q1Assets, train_reproduction.atomic_json = original_assets, original_write
                contact_protocol.ContactCorrectedAssets = original_contact


class PipelineIsolationTests(unittest.TestCase):
    def test_every_control_aware_phase_uses_variant_entry(self):
        root = Path("/project")
        args = SimpleNamespace(root=root, output=root / "data/supermimic_for_q1/rf06",
            dashboard_label="RF06", steps=12000, num_envs=4096, threads=8, model_slots=8,
            wm_batch=2048, batch=4096, code_commit="a" * 40, expected_oom_kill=0,
            training_seconds=3000.)
        phases, _, _ = run_single_motion.variant_commands(args, run_single_motion.baseline.commands)
        tasks = {item["name"]: item["command"] for item in phases}
        for name, task in (("train", "train"), ("evaluate_initial", "evaluate"),
                           ("evaluate_final", "evaluate"), ("audit_initial", "audit"),
                           ("audit_final", "audit"), ("fullbody", "fullbody"),
                           ("render_initial", "render"), ("render_final", "render")):
            self.assertEqual(Path(tasks[name][2]).name, "entry.py")
            self.assertEqual(tasks[name][3], task)
        self.assertEqual(Path(tasks["save_check"][2]).name, "audit_q1_checkpoint.py")
        self.assertEqual(Path(tasks["legs"][2]).name, "analyze.py")
        train = tasks["train"]
        self.assertEqual(train[train.index("--wm-batch") + 1], "2048")
        self.assertEqual(train[train.index("--batch") + 1], "4096")
        self.assertEqual(train[train.index("--training-seconds") + 1], "3000.0")

    def test_phase_deadline_stops_without_extending_the_trial(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "guard.json"

            def fake_runner(*values, **options):
                self.assertTrue(options["stop_requested"]())
                return {"status": "STOPPED_SIGNAL", "error": None}

            command = ["python", "-u", str(run_single_motion.HERE / "entry.py"), "train"]
            with patch.object(run_single_motion.time, "monotonic", side_effect=[0., 3500.1]):
                result = run_single_motion.limited_guard(fake_runner, command,
                    Path(temporary), None, status, Path(temporary) / "stdout.log")
            self.assertTrue(result["training_phase_time_limit_reached"])
            self.assertIn("one-hour", result["error"])

    def test_over_one_hour_receipt_is_not_accepted(self):
        def fake_validator(*values):
            return {"training_seconds": 3600.01}
        with self.assertRaises(ValueError):
            run_single_motion.completed_trial(Path("/run"), 4096, 12000, 3000., fake_validator)

    def test_running_or_under_budget_baseline_cannot_start_rf06(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "data/supermimic_for_q1" / run_single_motion.RF05
            (run / "run").mkdir(parents=True)

            def write(name, value):
                (run / name).write_text(json.dumps(value))

            write("status.json", {"status": "RUNNING"})
            write("train_guard.json", {"status": "WORKER_COMPLETED", "cleanup": {
                "worker_reaped": True, "worker_return_code": 0}})
            with self.assertRaises(ValueError):
                run_single_motion.require_baseline(root)
            write("status.json", {"status": "COMPLETED"})
            write("run/admission.json", {"train_clips": [{"path": "/" + run_single_motion.baseline.CLIP + ".npz"}]})
            write("run/receipt.json", {"status": "TRAINING_FINISHED_PENDING_EVALUATION", "step": 1000,
                "env_steps": 4096000, "training_seconds": 7199.9, "error": None})
            with self.assertRaises(ValueError):
                run_single_motion.require_baseline(root)


if __name__ == "__main__":
    unittest.main()
