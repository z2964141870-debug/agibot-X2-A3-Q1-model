"""Reject command-tail fake gradients, behavior substitution and boundary leakage."""

import copy
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from wm_probe_data import nominal_context, physical_errors, selected_transitions, summarize
from wm_probe_math import StepCounter, gradient_probe


class ToyPhysicalModel:
    def __init__(self, physical=True, broken_gradient=False):
        self.physical, self.broken_gradient = physical, broken_gradient
        self.present = torch.arange(29) < 22

    def step(self, state, action, context):
        projected = torch.where(self.present, action, 0.)
        actual = projected.detach() if self.broken_gradient else projected
        delta = torch.nn.functional.pad(actual*.2, (0, 211)) if self.physical else torch.zeros_like(state[:, :240])
        return torch.cat((state[:, :240]+delta, projected), -1)


class WorldModelProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def probe(self, model):
        state, action, ctx = torch.zeros(1, 269), torch.ones(1, 29), torch.zeros(1, 62)
        counter = StepCounter(model, time.monotonic_ns()+30_000_000_000)
        result = gradient_probe(counter, state, action, ctx, model.present, lambda x:x,
            epsilons=(.01, .001))
        self.assertEqual(counter.calls, 91)
        return result

    def test_command_tail_identity_does_not_pass_as_physical_action_gradient(self):
        result = self.probe(ToyPhysicalModel(physical=False))
        self.assertFalse(result["nonzero_physical_action_response"])
        self.assertEqual(result["physical_gradient_max"], 0.)

    def test_real_physical_gradient_matches_finite_difference_and_masks_absent_joints(self):
        result = self.probe(ToyPhysicalModel())
        self.assertTrue(result["finite_difference_passed"])
        self.assertTrue(result["nonzero_physical_action_response"])
        self.assertTrue(result["absent_joint_gradient_exact_zero"])
        self.assertEqual(result["absent_joint_physical_response_max"], 0.)

    def test_numerical_action_response_with_detached_autograd_is_rejected(self):
        result = self.probe(ToyPhysicalModel(broken_gradient=True))
        self.assertFalse(result["finite_difference_passed"])

    def test_deadline_and_call_limit_stop_before_an_extra_forward(self):
        counter = StepCounter(ToyPhysicalModel(), time.monotonic_ns()+10_000_000_000, maximum=1)
        values = torch.zeros(1, 269), torch.zeros(1, 29), torch.zeros(1, 62)
        counter(*values)
        with self.assertRaises(RuntimeError):
            counter(*values)
        self.assertEqual(counter.calls, 1)

    def test_selection_uses_executed_mixture_and_excludes_terminal_transition(self):
        actions = np.ones((8, 29), dtype=np.float32)*.25
        states = np.arange(9*79, dtype=np.float32).reshape(9, 79)
        teacher = np.ones_like(actions)
        data = dict(frame=np.arange(8), clip_id=np.zeros(8, dtype=np.int64),
            actual_state=states[:-1], teacher_action=teacher, behavior_action=actions)
        traces = {0:dict(actual=states, canonical_action=actions, teacher_action=teacher)}
        selected, audit = selected_transitions(data, traces, {0:7}, aggregate=True, quota=3)
        self.assertEqual([r["frame"] for r in selected], [0, 2, 5])
        self.assertTrue(np.array_equal(selected[0]["executed"], actions[0]))
        self.assertEqual(len(audit["terminal_rows_excluded"]), 2)
        bad = copy.deepcopy(data)
        bad["behavior_action"] = teacher
        with self.assertRaisesRegex(ValueError, "executed behavior"):
            selected_transitions(bad, traces, {0:None}, aggregate=True)

    def test_context_reads_xml_friction_and_refuses_unknown_delay(self):
        model = SimpleNamespace(ngeom=2, geom_friction=np.array([[.8, .01, .001], [.8, .01, .001]]),
            geom=lambda i:SimpleNamespace(name=("left_foot1_collision", "right_foot1_collision")[i]))
        robot, spec = SimpleNamespace(model=model), SimpleNamespace(foot_geom_regex=r".*foot1_collision")
        context, receipt = nominal_context(robot, spec, dict(action_scale=.5, modeled_action_delay=False))
        self.assertEqual(context.shape, (62,))
        self.assertAlmostEqual(float(context[0]), .8)
        self.assertIn("NOT_RECORDED", receipt["kind"])
        with self.assertRaises(ValueError):
            nominal_context(robot, spec, dict(action_scale=.5, modeled_action_delay=True))

    def test_quaternion_sign_and_command_memory_do_not_change_physical_metrics(self):
        present = np.arange(29) < 22
        target = np.zeros((1, 269))
        target[:, 100:156].reshape(1, 14, 4)[..., 0] = 1
        value = target.copy()
        value[:, 100:156] *= -1
        value[:, 240:] += 3
        result = physical_errors(value, target, present)
        self.assertEqual(float(result["body_orientation_rad"][0]), 0.)
        self.assertEqual(float(result["joint_position_rad"][0]), 0.)
        self.assertEqual(float(result["command_memory_rms"][0]), 3.)

    def test_validation_gate_keeps_worst_source_and_ignores_command_tail(self):
        rows = []
        for origin in ("original", "mixed"):
            for split in ("train", "validation"):
                for clip, error in ((0, .1), (1, 1.3)):
                    rows.append(dict(origin=origin, split=split, clip_id=clip,
                        wm=dict(joint_position_rad=error, body_position_m=error, command_memory_rms=0.),
                        identity=dict(joint_position_rad=1., body_position_m=1., command_memory_rms=100.)))
        result = summarize(rows)
        self.assertLess(result["groups"][1]["ratios"]["joint_position_rad"], .95)
        self.assertFalse(result["calibration_screen_passed"])
        self.assertTrue(result["screening_rule"]["command_memory_excluded"])


if __name__ == "__main__":
    unittest.main()
