"""Deterministic ownership and clock regression checks; no simulator required."""

import numpy as np
from q1_autonomous_recovery import Q1AutonomousRecovery


def check(tracker, expert, xml):
    branch = Q1AutonomousRecovery(tracker, expert, xml)
    q = tracker.default_q.copy()
    zero = np.zeros_like(q)
    w = np.zeros(3, dtype=np.float32)
    upright = np.eye(3, dtype=np.float32)
    down = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    calls = []

    def live():
        calls.append("teleop")
        return q.copy(), zero.copy(), {}

    def anchor():
        calls.append("anchor")

    def tick(t, stamp, R=upright, rate=w):
        return branch.step(
            R, rate, q, zero, zero, live, anchor, sample_time_s=t, teleop_stamp_s=stamp
        )[2]

    assert tick(0.0, 0.0) == 0.0
    assert tick(0.02, 0.02, down) == 0.0 and tick(0.04, 0.04, down) == 0.0
    assert tick(0.06, 0.06, down) == 1.0
    calls.clear()
    # Isolate the supervisor: supply a settled private policy target on every tick.
    original_step = branch.rise.step
    branch.rise.step = lambda *args: (q.copy(), zero.copy(), 0.0)
    for k in range(4, 45):
        assert tick(k * 0.02, None) == 1.0
    assert branch.standing_confirmed and not calls
    assert tick(0.9, 0.6) == 1.0 and tick(0.92, 0.94) == 1.0
    assert tick(0.94, 0.94, rate=np.ones(3, np.float32)) == 1.0
    assert not branch.standing_confirmed
    for k in range(48, 65):
        assert tick(k * 0.02, k * 0.02) == 1.0
    assert tick(2.0, 2.0) == 1.0  # missing samples break dwell
    for k in range(101, 125):
        assert tick(k * 0.02, k * 0.02) == 1.0
    assert tick(2.5, 2.5) == 0.0
    assert calls == ["anchor", "teleop"] and branch.owner == "teleop"
    assert tick(2.52, 2.52) == 0.0
    for k in (127, 128):
        assert tick(k * 0.02, k * 0.02, down) == 0.0
    assert tick(2.58, 2.58, down) == 1.0 and branch.owner == "recovery"
    try:
        tick(2.58, 2.58)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate timestamp accepted")
    branch.reset()
    assert branch.owner == "teleop" and not branch.standing_confirmed
    assert tick(0.0, None, down) == 1.0
    branch.rise.step = original_step
    calls.clear()
    for k in range(1, 1501):
        assert tick(k * 0.02, k * 0.02, down) == 1.0
    assert not calls and branch.owner == "recovery"
    return {
        "persistent_down_no_timeout_s": 30,
        "automatic_takeover": True,
        "offline_hold": True,
        "stale_and_future_stamp_rejected": True,
        "instability_restarts_dwell": True,
        "sample_gap_restarts_dwell": True,
        "anchor_before_teleop": True,
        "second_fall": True,
        "timestamp_reset": True,
    }
