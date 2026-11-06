"""Visibility, paired velocities and standalone event entry-point regressions."""

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from event_inputs import EventReceiver, paired_reference, validate_event_timing
from stream_contract import DT_NS, HumanPose, PoseFrame, PoseTimeline, SourceIdentity
from stream_events import EventSpec, EventTimeline
from preview_worker import preview_goal


def poses(frames=120):
    position = np.zeros((frames, 24, 3))
    position[:, :, 0] = np.arange(frames)[:, None]*.01
    position[:, :, 1] = np.sin(np.arange(frames)[:, None]*.07)
    quat = np.zeros((frames, 24, 4))
    quat[:, :, 0] = 1.
    return position, quat


def gpu_timing(goal):
    return dict(goal.timing, policy_device="cuda", gpu_policy_wall_compute_ns=123)


class EventInputTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "Actual action-path CPU contract requires Torch")
    def test_actual_action_path_merges_existing_gpu_compute_scope(self):
        import torch
        from event_gpu_worker import EventSession

        class CpuValue:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, item):
                return CpuValue(self.value[item])

            def to(self, device):
                return self.value

        modules = {name: ModuleType(name) for name in (
            "morphmimic", "morphmimic.constants", "morphmimic.model", "morphmimic.model.human",
            "morphmimic.model.rewards_torch", "morphmimic.model.state", "morphmimic.train",
            "morphmimic.train.imagination", "q1_native_review")}
        modules["morphmimic.constants"].ANCHOR_IDX = 0
        modules["morphmimic.model.human"].fixed_human_masks = lambda *args, **kwargs: (None,)*4
        modules["morphmimic.model.human"].pack_human_goal = lambda *args: torch.zeros(1, 1)
        modules["morphmimic.model.rewards_torch"].analytic_obs = lambda *args: torch.zeros(1, 93)
        modules["morphmimic.model.state"].split_state = lambda value: SimpleNamespace(
            body_quat=CpuValue(torch.zeros(1, 1, 4)))
        modules["morphmimic.train.imagination"].act_human = lambda *args: torch.zeros(1, 29)
        modules["q1_native_review"].native_state = lambda *args: torch.zeros(1, 250)
        session = EventSession.__new__(EventSession)
        operations = ("float32", "isfinite", "zeros", "where", "cat", "count_nonzero", "inference_mode")
        session.torch = SimpleNamespace(**{name: getattr(torch, name) for name in operations},
            tensor=lambda value, **kwargs: torch.tensor(value, **{k:v for k,v in kwargs.items() if k != "device"}))
        session.frame, session.clip, session.robot, session.default = 0, "fixture", None, torch.zeros(29)
        session.args = SimpleNamespace(preview_frames=4)
        session.tail_method = "hold"
        session.actor = SimpleNamespace(present=torch.ones(29, dtype=torch.bool))
        session.spec = SimpleNamespace(imu_slot=0,
            translator=SimpleNamespace(to_native_delta=lambda value: value[:, :22]))
        session.history = SimpleNamespace(reset=lambda *args: None, push=lambda *args: None,
            flat=lambda: CpuValue(torch.zeros(1, 6)))
        native = SimpleNamespace(policy=None, norms={name:None for name in ("obs_norm", "human_norm", "heading_norm")})
        session.preview = SimpleNamespace(policy=SimpleNamespace(native=native))
        position, quat = poses()
        session.events = EventReceiver(position, quat, EventSpec("nominal", first_frame=50), "a"*64, "test")
        session.compute_record = lambda started: dict(policy_device="cuda", gpu_policy_wall_compute_ns=123,
            compute_scope="existing base GPU scope")
        with patch.dict(sys.modules, modules):
            reply = session.action(dict(frame=0, state=[0.]*79))
        self.assertEqual(len(reply["native_action"]), 22)
        self.assertEqual(len(reply["canonical_action"]), 29)
        self.assertEqual(reply["stream"]["gpu_policy_wall_compute_ns"], 123)
        self.assertIn("event visibility", reply["stream"]["compute_scope"])
        validate_event_timing(reply["stream"], 0, 4, "hold", session.events.events)
        self.assertEqual(session.frame, 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_nominal_poses_and_tails_equal_the_published_adapter_including_terminal(self):
        position, quat = poses()
        for method, horizon in (("masked", 10), ("masked", 4), ("hold", 5), ("damped_cv", 4)):
            receiver = EventReceiver(position, quat, EventSpec("nominal", first_frame=50), "a"*64, "test")
            timeline = PoseTimeline(SourceIdentity("source", "test", "clock", "smpl24_world_m_wxyz",
                                                   "offline_unverified", "a"*64))
            admitted = -1
            for frame in range(len(position)-1):
                for i in range(admitted+1, min(frame+horizon, len(position)-1)+1):
                    timestamp = i*DT_NS
                    timeline.ingest(PoseFrame(timeline.source, i, timestamp, None, timestamp, timestamp,
                        HumanPose(position[i], quat[i]), terminal=i == len(position)-1))
                    admitted = i
                window = timeline.window(origin_ns=frame*DT_NS, control_ns=(frame+horizon)*DT_NS,
                    preview_frames=horizon, max_capture_age_ns=10*DT_NS)
                expected, timing = preview_goal(window, method)
                actual = receiver.step(frame, horizon, method)
                self.assertEqual(actual.poses, expected.poses)
                self.assertEqual(actual.timing["measured_mask"], timing["measured_mask"])
                self.assertEqual(actual.timing["generated_mask"], timing["generated_mask"])
                validate_event_timing(gpu_timing(actual), frame, horizon, method, receiver.events)

    def test_lost_payload_changes_cannot_change_any_goal(self):
        position, quat = poses()
        changed = position.copy()
        changed[50:60] += 1000.
        event = EventSpec("dropout", first_frame=50, duration_frames=10)
        left, right = (EventReceiver(p, quat, event, "a"*64, "test") for p in (position, changed))
        for frame in range(len(position)-1):
            a, b = left.step(frame, 4, "damped_cv"), right.step(frame, 4, "damped_cv")
            self.assertEqual(a.poses, b.poses)
            self.assertEqual(a.timing, b.timing)
        np.testing.assert_array_equal(position[50, :, 0], .5)

    def test_burst_payload_is_invisible_until_arrival_then_is_admitted(self):
        position, quat = poses()
        changed = position.copy()
        changed[50:60] += 10.
        event = EventSpec("late_burst", first_frame=50, duration_frames=10)
        left, right = (EventReceiver(p, quat, event, "a"*64, "test") for p in (position, changed))
        for frame in range(56):
            self.assertEqual(left.step(frame, 4, "hold").poses, right.step(frame, 4, "hold").poses)
        self.assertNotEqual(left.step(56, 4, "hold").poses, right.step(56, 4, "hold").poses)

    def test_future_capture_changes_do_not_change_earlier_goal(self):
        position, quat = poses()
        changed = position.copy()
        changed[5:] += 123.
        receivers = [EventReceiver(p, quat, EventSpec("nominal", first_frame=50), "a"*64, "test")
                     for p in (position, changed)]
        self.assertEqual(receivers[0].step(0, 4, "damped_cv").poses,
                         receivers[1].step(0, 4, "damped_cv").poses)

    def test_empty_prefix_hold_and_recovery_are_explicit_and_no_later_slot_is_substituted(self):
        position, quat = poses()
        receiver = EventReceiver(position, quat, EventSpec("dropout", first_frame=50), "a"*64, "test")
        values = [receiver.step(i, 4, "damped_cv") for i in range(61)]
        self.assertEqual(values[48].timing["effective_tail_method"], "hold")
        self.assertIsNone(values[48].timing["fallback"])
        for frame in range(49, 59):
            goal = values[frame]
            self.assertEqual(goal.timing["fallback"], "visible_past_pose_hold")
            self.assertEqual(goal.timing["used_packets"][0]["sequence"], 49)
            self.assertFalse(any(goal.timing["measured_mask"]))
            self.assertTrue(all(goal.timing["generated_mask"]))
            self.assertEqual(goal.policy_frames, 10)
            validate_event_timing(gpu_timing(goal), frame, 4, "damped_cv", receiver.events)
        self.assertIsNone(values[59].timing["fallback"])
        self.assertEqual(values[59].timing["used_packets"][0]["sequence"], 60)

    def test_transport_is_separate_from_preview_and_no_visible_pose_fails_closed(self):
        position, quat = poses()
        receiver = EventReceiver(position, quat, EventSpec("nominal", first_frame=50, transport_frames=2), "a"*64, "test")
        goal = receiver.step(0, 4, "masked")
        self.assertEqual(goal.policy_frames, 2)
        self.assertEqual(goal.timing["requested_preview_ns"], 4*DT_NS)
        self.assertEqual(goal.timing["latest_capture_age_ns"], 2*DT_NS)
        validate_event_timing(gpu_timing(goal), 0, 4, "masked", receiver.events)
        missing = EventReceiver(position, quat, EventSpec("nominal", first_frame=50, transport_frames=10), "a"*64, "test")
        with self.assertRaisesRegex(ValueError, "No visible past"):
            missing.step(0, 4, "hold")

    def test_parent_rejects_changed_packets_masks_or_latency_claims(self):
        position, quat = poses()
        receiver = EventReceiver(position, quat, EventSpec("nominal", first_frame=50), "a"*64, "test")
        goal = receiver.step(0, 4, "masked")
        value = gpu_timing(goal)
        variants = [dict(value, measured_end_to_end_latency_ns=80_000_000),
            dict(value, cpu_policy_wall_compute_ns=123), dict(value, observed_prefix_frames=0),
            dict(value, used_packets=[dict(value["used_packets"][0], arrival_ns=999999999)])]
        for changed in variants:
            with self.assertRaises(ValueError):
                validate_event_timing(changed, 0, 4, "masked", receiver.events)
        with self.assertRaises(ValueError):
            replace(goal, timing=dict(goal.timing, used_packets=[dict(goal.timing["used_packets"][0], capture_ns=999999999)])).validate()

    def test_reference_packet_events_preserve_the_frozen_states_and_inputs(self):
        reference = np.zeros((120, 79), dtype=np.float32)
        reference[:, 3] = 1.
        reference[:, 29:57] = 123.
        for kind in ("dropout", "late_burst"):
            value, record = paired_reference(reference, EventSpec(kind, first_frame=50))
            np.testing.assert_array_equal(reference, value)
            self.assertFalse(record["input_reference_changed"])
        bad = reference.copy()
        bad[:, 57] = 1.
        with self.assertRaises(ValueError): paired_reference(bad, EventSpec("stop", first_frame=50))

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Pinned MuJoCo source-layout test needs its independent runtime")
    def test_real_mujoco_event_reference_world_angular_and_joint_velocity_conventions(self):
        import mujoco

        self.assertEqual(mujoco.__version__, "3.6.0", "Geometry convention must be checked in the pinned plant version")
        children = "".join(f'<body pos="0 0 .1"><joint name="j{i}" type="hinge"/><geom size=".01" mass="1"/></body>'
                           for i in range(22))
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".01" mass="1"/>'
                                             +children+'</body></worldbody></mujoco>')
        time = np.arange(120)*.02
        reference = np.zeros((120, 79), dtype=np.float64)
        reference[:, 0] = time*2
        c, s, half = np.cos(time/2), np.sin(time/2), np.sqrt(.5)
        reference[:, 3:7] = np.stack((c*half, c*half, s*half, s*half), axis=1)
        reference[:, 7:29] = time[:, None]
        reference[:, 29] = 2.
        reference[:, 34:57] = 1.
        for kind, sign in (("stop", 0), ("reverse", -1)):
            value, mapping = paired_reference(reference, EventSpec(kind, first_frame=50), model)
            np.testing.assert_array_equal(value[:51], reference[:51])
            np.testing.assert_allclose(value[51, 29:35], [2*sign, 0, 0, 0, 0, sign], atol=1e-12)
            np.testing.assert_allclose(value[51, 35:57], sign, atol=1e-12)
            np.testing.assert_array_equal(reference[:, 0], time*2)
            self.assertFalse(mapping["physical_feasibility_validated"])

    def test_standalone_entrypoints_work_without_inherited_pythonpath(self):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("event_gpu_worker.py", "event_review.py"):
                result = subprocess.run([sys.executable, str(Path(__file__).resolve().with_name(name)), "--help"],
                    cwd=temporary, env=environment, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--gpu-owner-pid", result.stdout)


class EventControllerTests(unittest.TestCase):
    def test_schedule_and_nominal_gate_precede_gpu_lock(self):
        from event_review import HERE, NEW_FILES, cases, execute

        self.assertEqual(len(cases()), 84)
        self.assertEqual(len(set(cases())), 84)
        args = SimpleNamespace(child=None)
        code = [dict(path=str(HERE / name)) for name in NEW_FILES]
        with patch("event_review.preflight", return_value=({}, {}, {}, {}, {}, [], {}, code)), \
                patch("event_review.nominal_evidence", side_effect=RuntimeError("nominal not completed")) as nominal, \
                patch("event_review.own_gpu") as lock:
            with self.assertRaisesRegex(RuntimeError, "nominal not completed"): execute(args)
            nominal.assert_called_once()
            lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
