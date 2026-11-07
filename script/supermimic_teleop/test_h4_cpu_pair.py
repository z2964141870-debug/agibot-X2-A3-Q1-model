"""Reject frame/visibility drift and preserve all calls in the H4 CPU comparison."""

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from h4_cpu_pair import CASES, FRAMES, compare_pair, saved_prefix, timing


class H4PairContracts(unittest.TestCase):
    def rows(self):
        return [dict(frame=i,canonical_action=[0.]*29,stream=dict(used_sequences=[i+1],input_ready=True,
            fallback=None,max_used_dependency_ns=(i+1)*20_000_000,observed_prefix_frames=1,
            policy_prefix_frames=1,requested_preview_ns=80_000_000,gpu_policy_wall_compute_ns=1_000_000,
            worker_action_wall_ns=2_000_000,isolated_request_worker_wall_ns=3_000_000,
            local_pipe_rpc_ns=4_000_000)) for i in range(FRAMES)]

    def test_prefix_has_no_shift_or_frame_selection(self):
        actual,actions = np.arange(161*79).reshape(161,79),np.zeros((160,29))
        a,b = saved_prefix(actual,actions)
        self.assertEqual((a.shape,b.shape),((128,79),(128,29)))
        self.assertTrue(np.array_equal(a,actual[:128]))
        a[0,0] = -1
        self.assertEqual(actual[0,0],0)
        with self.assertRaises(RuntimeError):
            saved_prefix(actual[:128],actions[:127])

    def test_visibility_or_sequence_shift_is_rejected(self):
        a = self.rows()
        for key,value in (("frame",2),("used_sequences",[999]),("input_ready",False),("fallback","previous_action_hold")):
            b = copy.deepcopy(a)
            if key == "frame":
                b[1][key] = value
            else:
                b[1]["stream"][key] = value
            with self.assertRaises(RuntimeError):
                compare_pair(a,b,[True]*29)

    def test_actions_and_missing_slots_retain_the_original_gate(self):
        a = self.rows()
        b = copy.deepcopy(a)
        b[4]["canonical_action"][0] = 1e-6
        self.assertEqual(compare_pair(a,b,[True]*29)["calls_per_converter"],128)
        for present in ([True]*28+[False],[True]*29):
            b = copy.deepcopy(a)
            b[3]["canonical_action"][-1] = 1e-9 if not present[-1] else .1
            with self.assertRaises(ValueError):
                compare_pair(a,b,present)

    def test_timing_keeps_initial_spikes_and_correct_capture_clock(self):
        rows = self.rows()
        rows[0]["stream"]["worker_action_wall_ns"] = 500_000_000
        rows[2]["stream"]["worker_action_wall_ns"] = 30_000_000
        report = timing(rows)
        self.assertEqual(report["all"]["worker_action_wall_ns"]["calls_over_20ms"],2)
        self.assertEqual(report["without_first_two"]["worker_action_wall_ns"]["calls_over_20ms"],1)
        self.assertEqual(report["synthetic_capture_50_60"]["worker_action_wall_ns"]["samples"],10)
        with self.assertRaises(RuntimeError):
            timing(rows[1:])

    def test_design_balances_order_and_caps_real_calls(self):
        self.assertEqual(CASES,(("nominal","original"),("nominal","kinematic"),
            ("dropout","kinematic"),("dropout","original")))
        self.assertEqual(len(CASES)*FRAMES,512)

    def test_real_request_wrapper_restores_converter_after_an_error(self):
        from h4_cpu_pair_worker import PairedSession, StudentReviewSession
        session = object.__new__(PairedSession)
        session.robot,session.spec = object(),object()
        session.args = SimpleNamespace(cpu_converter="kinematic")
        session.converter = lambda value:value+1
        original = lambda *args:"original"
        module = SimpleNamespace(native_state=original)
        mujoco = SimpleNamespace(**{k:(lambda *a:None) for k in
            ("mj_step","mj_step1","mj_step2","mj_forward","mj_inverse")})

        def failing_action(self,request):
            self_test.assertEqual(module.native_state(self.robot,2,self.spec),3)
            with self_test.assertRaises(AssertionError):
                mujoco.mj_step(None,None)
            raise RuntimeError("Interrupted original action")

        self_test = self
        with patch.dict(sys.modules,{"mujoco":mujoco,"q1_native_review":module}), \
                patch.object(StudentReviewSession,"action",failing_action):
            with self.assertRaisesRegex(RuntimeError,"Interrupted original action"):
                session.action({})
        self.assertIs(module.native_state,original)


if __name__ == "__main__":
    unittest.main()
