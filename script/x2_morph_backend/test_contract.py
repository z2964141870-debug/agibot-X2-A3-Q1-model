"""Backend boundary tests, independent of ROS, robot hardware and ONNX."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from .contract import Backend, Contract, GROUPS, MODEL_SHA


def contract():
    return Contract(tuple(f"joint_{i}" for i in range(31)), tuple(100.+i for i in range(31)),
                    (2.,)*31, (0.,)*31, ("pelvis", "torso_link"))


def measurement(c=None, sequence=0):
    c = c or contract()
    return dict(kind="x2_measurement", version=1, sequence=sequence, joint_names=list(c.names),
                q_rad=[i*.01 for i in range(31)], dq_rad_s=[i*.02 for i in range(31)],
                imu_quat_wxyz=[1., 0., 0., 0.], imu_gyro_rad_s=[0., 0., 0.],
                imu_body="pelvis", body_from_sensor_wxyz=[1., 0., 0., 0.],
                source_receive_ages_s={k:.001 for k in (*GROUPS, "imu")})


def advisory(c=None, sequence=0):
    c = c or contract()
    return dict(kind="x2_morph_advisory_target", version=1, ready=True, publish_allowed=False,
                robot_profile="morphmimic_v01_x2_ultra", policy_sha256=MODEL_SHA,
                state_sequence=sequence, joint_names=list(c.names),
                q_target_rad=[.11+i*.037 for i in range(31)], canonical_action=[.123]*29)


class Client:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.requests = reply or advisory(), fail, []

    def request(self, message):
        self.requests.append(deepcopy(message))
        if self.fail:
            raise TimeoutError("fixture timeout")
        return deepcopy(self.reply)


class ContractTests(unittest.TestCase):
    def test_native_targets_and_model_gains_copied_exactly(self):
        c, reply = contract(), advisory()
        result = Backend(c, Client(reply)).step(measurement())
        self.assertTrue(result["ready"])
        self.assertEqual(result["q_target_rad"], reply["q_target_rad"])
        flat = sum((result["groups"][k]["position_rad"] for k in GROUPS), [])
        self.assertEqual(flat, reply["q_target_rad"])
        self.assertEqual(result["groups"]["waist"]["model_kp"], list(c.kp[12:15]))
        self.assertEqual(result["groups"]["arm"]["model_kd"], list(c.kd[15:29]))
        self.assertFalse(result["publish_allowed"])
        self.assertFalse(result["hardware_gains_validated"])

    def test_feedback_is_mapped_by_name(self):
        state = measurement()
        for key in ("joint_names", "q_rad", "dq_rad_s"):
            state[key].reverse()
        client = Client()
        self.assertTrue(Backend(contract(), client).step(state)["ready"])
        self.assertEqual(client.requests[0]["q_rad"], measurement()["q_rad"])
        self.assertEqual(client.requests[0]["dq_rad_s"], measurement()["dq_rad_s"])

    def test_feedback_incomplete_nonfinite_or_wrong_extrinsic_never_requested(self):
        cases = []
        for key, value in (("joint_names", ["duplicate"]*31), ("q_rad", [0.]*30),
                           ("dq_rad_s", [float("nan")]*31), ("imu_body", "unknown"),
                           ("body_from_sensor_wxyz", [0., 0., 0., 0.]),
                           ("imu_quat_wxyz", [2., 0., 0., 0.]), ("sequence", True)):
            state = measurement(); state[key] = value; cases.append(state)
        for state in cases:
            client = Client()
            with self.subTest(state=state):
                result = Backend(contract(), client).step(state)
                self.assertFalse(result["ready"])
                self.assertNotIn("q_target_rad", result)
                self.assertEqual(client.requests, [])

    def test_all_five_sources_need_fresh_local_receive_ages(self):
        for ages in ({}, {"leg":0.}, {k:.121 for k in (*GROUPS,"imu")},
                     {k:-.001 for k in (*GROUPS,"imu")}, {k:True for k in (*GROUPS,"imu")}):
            state = measurement(); state["source_receive_ages_s"] = ages
            client = Client()
            self.assertFalse(Backend(contract(), client).step(state)["ready"])
            self.assertFalse(client.requests)

    def test_model_identity_order_sequence_and_nonfinite_targets_rejected(self):
        cases = (("policy_sha256", "bad"), ("robot_profile", "g1"),
                 ("joint_names", list(reversed(contract().names))), ("state_sequence", 1),
                 ("state_sequence", 0.), ("publish_allowed", True), ("ready", 1),
                 ("q_target_rad", [float("inf")]*31), ("canonical_action", [0.]*28))
        for key, value in cases:
            reply = advisory(); reply[key] = value
            result = Backend(contract(), Client(reply)).step(measurement())
            with self.subTest(field=key):
                self.assertFalse(result["ready"])
                self.assertNotIn("q_target_rad", result)
                self.assertNotIn("source_advisory", result)

    def test_waiting_has_no_stale_or_zero_target(self):
        client = Client()
        backend = Backend(contract(), client)
        self.assertTrue(backend.step(measurement())["ready"])
        client.reply = dict(ready=False, publish_allowed=False, reason="human stream missing")
        result = backend.step(measurement(sequence=1))
        self.assertFalse(result["ready"])
        self.assertNotIn("q_target_rad", result)
        self.assertNotIn("groups", result)

    def test_duplicate_measurement_is_not_sent_twice(self):
        client = Client(); backend = Backend(contract(), client)
        backend.step(measurement())
        self.assertFalse(backend.step(measurement())["ready"])
        self.assertEqual(len(client.requests), 1)

    def test_timeout_latches_and_does_not_retry_or_reset_shared_service(self):
        client = Client(fail=True); backend = Backend(contract(), client)
        self.assertEqual(backend.step(measurement())["reason"], "policy_transport_failed")
        client.fail = False
        self.assertFalse(backend.step(measurement(sequence=1))["ready"])
        self.assertEqual(len(client.requests), 1)
        self.assertTrue(backend.transport_failed)

    def test_late_result_is_discarded(self):
        with patch("script.x2_morph_backend.contract.time.perf_counter_ns", side_effect=[0, 21_000_000]):
            result = Backend(contract(), Client()).step(measurement())
        self.assertEqual(result["reason"], "policy_rpc_budget_exceeded")
        self.assertNotIn("q_target_rad", result)

    def test_feedback_can_expire_during_rpc(self):
        state = measurement()
        state["source_receive_ages_s"]["arm"] = .115
        with patch("script.x2_morph_backend.contract.time.perf_counter_ns", side_effect=[0, 6_000_000]):
            result = Backend(contract(), Client()).step(state)
        self.assertEqual(result["reason"], "feedback_expired_during_rpc")

    def test_backend_has_no_command_publish_or_mc_api(self):
        root = Path(__file__).parent
        banned = {"create_publisher", "create_client", "PublishCommand", "kill", "killpg"}
        for path in (root/"__main__.py", root/"contract.py"):
            tree = ast.parse(path.read_text())
            called = {node.func.attr for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            self.assertFalse(called & banned)


if __name__ == "__main__":
    unittest.main()
