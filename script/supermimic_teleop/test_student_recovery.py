"""Restoration, fixed source ownership and unchanged first-action contracts."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from delivery import read
from student_proprio_review_worker import ProprioReviewSession
from student_recovery_state import TENSORS, checked_state, fixed_jobs, restore, snapshot
from student_recovery_worker import RecoverySession

try:
    import mujoco
except ImportError:
    mujoco = None


def actor():
    value = RecoverySession.__new__(RecoverySession)
    value.args = SimpleNamespace(deadline_ns=time.monotonic_ns()+60_000_000_000)
    value.job = dict(anchor=64)
    value.arm, value.torch, value.forwards = "student_f0", torch, 0
    value.saved = dict(actual=np.zeros((65, 79)), canonical_action=np.ones((65, 29)))
    return value


class RecoveryTests(unittest.TestCase):
    def test_fixed_anchors_and_split_ownership_do_not_use_outcomes(self):
        source = read(Path(__file__).with_name("student_onpolicy_plan_20261108.json"))["jobs"]
        plan = read(Path(__file__).with_name("student_recovery_plan_20261108.json"))
        self.assertEqual(fixed_jobs(source), plan["jobs"])
        self.assertEqual([r["anchor"] for r in plan["jobs"]], [64, 96]*4)
        bad = copy.deepcopy(source); bad[8]["split"] = "test"
        with self.assertRaises(RuntimeError):
            fixed_jobs(bad)

    def test_saved_state_gate_rejects_divergence_nonfinite_and_wrong_shape(self):
        state = np.zeros(79)
        self.assertEqual(checked_state(state, state.copy()), 0.)
        for other in (np.full(79, .001), np.full(79, np.nan), np.zeros(78)):
            with self.subTest(shape=other.shape), self.assertRaises(RuntimeError):
                checked_state(state, other)

    def test_student_uses_inherited_action_without_querying_teacher(self):
        value = actor()
        reply = dict(canonical_action=[1.]*29, native_action=[1.]*22, stream=dict(input_ready=True))
        with patch.object(ProprioReviewSession, "action", return_value=copy.deepcopy(reply)) as inherited:
            result = value.action(dict(frame=64, state=[0.]*79))
        inherited.assert_called_once()
        self.assertEqual(result["canonical_action"], reply["canonical_action"])
        self.assertFalse(result["training_only_privileged_reference"])
        self.assertEqual(result["cumulative_model_forwards"], 1)

    def test_first_action_and_exact_anchor_must_match_the_saved_trace(self):
        reply = dict(canonical_action=[1.01]*29, native_action=[1.]*22, stream=dict(input_ready=True))
        with patch.object(ProprioReviewSession, "action", return_value=reply), self.assertRaisesRegex(RuntimeError, "reproduce"):
            actor().action(dict(frame=64, state=[0.]*79))
        reply["canonical_action"] = [1.]*29
        with patch.object(ProprioReviewSession, "action", return_value=reply), self.assertRaisesRegex(RuntimeError, "exactly"):
            actor().action(dict(frame=64, state=[.001]*79))

    def test_policy_and_physical_horizon_limits_precede_new_inference(self):
        for forwards, frame in ((1280, 64), (0, 144)):
            value = actor(); value.forwards = forwards
            with patch.object(ProprioReviewSession, "action") as inherited, self.assertRaisesRegex(RuntimeError, "bound"):
                value.action(dict(frame=frame, state=[0.]*79))
            inherited.assert_not_called()

    @unittest.skipUnless(mujoco is not None, "Pinned server MuJoCo3.6 required for actual MjData copy binding")
    def test_full_mjdata_and_controller_rng_restore_reproduce_next_step(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j" type="hinge"/><geom type="sphere" size=".1"/></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>')
        data = mujoco.MjData(model)
        class Plant:
            device, num_envs = "cpu", 1
            cpu_models, cpu_data = [model], [data]
            def _sync_cpu(self):
                self.qpos = torch.tensor(data.qpos)
                self.qvel = torch.tensor(data.qvel)
            def state(self):
                return torch.cat((self.qpos, self.qvel, self.last_action))
        plant = Plant()
        for name in TENSORS:
            setattr(plant, name, torch.ones(1))
        data.ctrl[0] = .3
        for _ in range(5):
            mujoco.mj_step(model, data)
        plant._sync_cpu()
        value = snapshot(plant)
        next_random = torch.rand(2)
        mujoco.mj_step(model, data); plant._sync_cpu(); expected = plant.state().clone()
        plant.last_action.fill_(9.)
        restore(plant, value)
        self.assertTrue(torch.equal(torch.rand(2), next_random))
        mujoco.mj_step(model, data); plant._sync_cpu()
        self.assertTrue(torch.equal(expected, plant.state()))
        self.assertEqual(data.time, value["data"].time+model.opt.timestep)


if __name__ == "__main__":
    unittest.main()
