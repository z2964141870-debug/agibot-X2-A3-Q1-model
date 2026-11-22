"""Exercise the ROS bridge with real model files, fake time and an in-memory publisher."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


def check(candidate, baseline):
    sys.path.insert(0, str(candidate / "code"))
    import q1_ros2_lowlevel_bridge as bridge

    class Clock:
        seconds = 1.0

        def monotonic(self):
            return self.seconds

        def monotonic_ns(self):
            return round(self.seconds * 1e9)

        def sleep(self, seconds):
            self.seconds += seconds

    fake = Clock()
    with (
        patch.object(bridge, "time", fake),
        patch.object(bridge.Node, "__init__", return_value=None),
        patch.object(bridge.Node, "get_logger", return_value=Mock()),
        patch.object(bridge.Node, "create_subscription"),
        patch.object(bridge.Node, "create_publisher", side_effect=lambda *_: Mock()),
    ):
        node = bridge.Q1LowLevelBridge(candidate, enable_recovery=True, recovery_offline=True)
        q = node.default_q + 0.03
        upright = np.eye(3, dtype=np.float32)
        down = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        omega = np.zeros(3, np.float32)
        qd = np.zeros(22, np.float32)
        fallen = down, omega, q, qd, None
        node._base_step = Mock(side_effect=AssertionError("base invoked during offline rise"))
        node._infer_once(fallen)
        assert node.recovery.owner == "recovery" and node.recovery.phase == "rise"
        assert not node._base_step.called

        desired = q + 0.81
        node.desired_target = desired.copy()
        node._decode_state = lambda *_: fallen
        node._snapshot = lambda: (object(), object(), fake.seconds)
        node._command = lambda target, *_: target.copy()
        samples = []
        previous = []

        def infer(_):
            previous.append(node.prev_action.copy())

        def publish(target):
            if samples:
                expected_previous = (samples[-1][1] - node.default_q) / node.action_scale
                np.testing.assert_allclose(node.prev_action, expected_previous)
            samples.append((fake.seconds, target))
            if len(samples) == 11:
                node.stop_event.set()

        node._infer_once = infer
        node.publisher.publish.side_effect = publish
        node._control_loop()
        times, targets = zip(*samples)
        np.testing.assert_allclose(np.diff(times), 0.002)
        np.testing.assert_allclose(np.asarray(targets[:4])[:, 0] - q[0], [0.2, 0.4, 0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(targets[4], desired)
        np.testing.assert_allclose(previous[1], (targets[9] - node.default_q) / node.action_scale)

        online = bridge.Q1LowLevelBridge(
            candidate, enable_recovery=True, human_smpl=True, fgp_url="http://127.0.0.1:1"
        )
        online._base_step = Mock(side_effect=AssertionError("planner invoked during recovery"))
        queued = None
        online.fgp.get_latest = lambda: (queued, 0)
        online._infer_once(fallen)
        assert online.recovery.owner == "recovery"
        # Inject standing feedback and the rise module's standing result to isolate the bridge.
        online.recovery.rise.step = lambda *_: (online.default_q.copy(), qd, 0.0)
        standing = upright, omega, online.default_q.copy(), qd, None
        for _ in range(26):
            fake.seconds += 0.02
            online._infer_once(standing)
        assert online.recovery.standing_confirmed
        tick = SimpleNamespace(segment=4, root_rotation=upright, root_xy=np.zeros(2, np.float32),
                               feature=np.zeros(430, np.float32))
        for age_ms in (150, 101, -5):
            fake.seconds += 0.02
            queued = SimpleNamespace(tick=tick, arrival_ns=fake.monotonic_ns() - age_ms * 1_000_000)
            online._infer_once(standing)
            assert online.recovery.owner == "recovery" and online.fgp_segment is None

        online._reset_fgp_segment = Mock(wraps=online._reset_fgp_segment)
        online.tracker.reset = Mock(wraps=online.tracker.reset)
        sent_before = online.target.copy()
        action_before = online.prev_action.copy()
        fake.seconds += 0.02
        queued = SimpleNamespace(tick=tick, arrival_ns=fake.monotonic_ns() - 95_000_000)
        online._infer_once(standing)
        assert online.recovery.owner == "teleop" and online.fgp_segment == 4
        assert online._reset_fgp_segment.call_count == online.tracker.reset.call_count == 1
        np.testing.assert_array_equal(online.prev_action, action_before)
        np.testing.assert_array_equal(online.target, sent_before)
        fake.seconds += 0.02
        queued = SimpleNamespace(tick=tick, arrival_ns=fake.monotonic_ns())
        online._infer_once(standing)
        assert online._reset_fgp_segment.call_count == 1

        for root in (baseline, candidate):
            planner = bridge.Q1LowLevelBridge(root, enable_recovery=False)
            assert planner.recovery is None and planner.cfg["recovery"]["enabled_by_default"] is False
            planner._base_step = Mock(return_value=(q + 1.0, qd, {}))
            planner._infer_once(fallen)
            np.testing.assert_allclose(planner.target, q + 0.2)
        old = bridge.Q1LowLevelBridge(baseline, enable_recovery=True)
        assert not old.recovery_v2 and type(old.recovery) is bridge.Q1RecoveryBranch

    return {
        "transport": "mock publisher and deterministic clock; no ROS initialization",
        "offline_fall_takeover": True,
        "sender_samples": len(samples),
        "nominal_sender_period_ms": 2,
        "max_increment_rad": round(float(np.max(np.abs(np.diff(np.asarray(targets), axis=0)))), 6),
        "inference_reads_last_published_action": True,
        "fgp_rejected_ages_ms": [150, 101, -5],
        "fgp_handback_age_ms": 95,
        "fgp_handback_reference_resets": 1,
        "fgp_handback_tracker_resets": 1,
        "fgp_handback_preserves_last_sent": True,
        "candidate_and_baseline_planner_default_recovery_off": True,
        "baseline_recovery_branch": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.candidate.resolve(), args.baseline.resolve()), indent=2))
