"""Offline admission and release contract for planner-to-human tracking."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Cycle:
    epoch: str
    now: float
    planner_sequence: int
    planner_received: float
    planner_source: str
    feedback_received: float
    planner_head: bool
    steady: bool
    human_ready: bool
    operator_requests_tracking: bool
    target_envelope_ok: bool


@dataclass(frozen=True)
class Decision:
    tracking_requested: bool
    planner_ready: bool
    event: str | None = None


@dataclass(frozen=True)
class JointEnvelope:
    minimum: float
    maximum: float
    max_speed: float
    max_error: float

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.minimum, self.maximum, self.max_speed, self.max_error))
                or not self.minimum < self.maximum or min(self.max_speed, self.max_error) <= 0):
            raise ValueError("invalid joint envelope")

    def violations(self, *, target, measured, previous, dt):
        if not all(math.isfinite(v) for v in (target, measured, previous, dt)) or dt <= 0:
            return ("invalid_target_or_clock",)
        problems = []
        if not self.minimum <= target <= self.maximum:
            problems.append("position")
        if abs(target - previous) > self.max_speed * dt:
            problems.append("rate")
        if abs(target - measured) > self.max_error:
            problems.append("feedback_error")
        return tuple(problems)


class TrackingGate:
    """Return an advisory decision only. No robot or network dependencies."""

    def __init__(self, *, ready_for_s, max_input_age_s):
        if not all(math.isfinite(v) and v > 0 for v in (ready_for_s, max_input_age_s)):
            raise ValueError("readiness and age limits must be positive and finite")
        self.ready_for_s = ready_for_s
        self.max_input_age_s = max_input_age_s
        self.epoch = None
        self._clear()

    def _clear(self):
        self.last_cycle = None
        self.last_planner_sequence = -1
        self.last_planner_advance = None
        self.stable_since = None
        self.released = False
        self.tracking = False

    def update(self, cycle: Cycle) -> Decision:
        if not cycle.epoch or not math.isfinite(cycle.now):
            return self._invalidate("invalid_cycle")
        if cycle.epoch != self.epoch:
            had_tracking = self.tracking
            self.epoch = cycle.epoch
            self._clear()
            if had_tracking:
                return Decision(False, False, "release:epoch_changed")
        if self.last_cycle is not None:
            elapsed = cycle.now - self.last_cycle
            if elapsed <= 0:
                return self._invalidate("clock_regressed")
            if elapsed > self.max_input_age_s:
                self.last_cycle = cycle.now
                return self._invalidate("cycle_gap")
        self.last_cycle = cycle.now
        if cycle.planner_sequence < self.last_planner_sequence:
            return self._invalidate("planner_sequence_regressed")

        if (cycle.planner_source == "upstream" and cycle.planner_sequence >= 0
                and self._fresh(cycle.now, cycle.planner_received)
                and cycle.planner_sequence > self.last_planner_sequence):
            self.last_planner_sequence = cycle.planner_sequence
            self.last_planner_advance = cycle.now
        planner_live = (cycle.planner_source == "upstream"
                        and cycle.planner_sequence >= 0
                        and self._fresh(cycle.now, cycle.planner_received)
                        and self._fresh(cycle.now, self.last_planner_advance))
        feedback_live = self._fresh(cycle.now, cycle.feedback_received)
        edge = self.released and cycle.operator_requests_tracking
        self.released = not cycle.operator_requests_tracking
        if self.tracking:
            if not cycle.operator_requests_tracking:
                return self._invalidate("operator")
            if not cycle.target_envelope_ok:
                return self._invalidate("target_envelope")
            if not planner_live or not feedback_live or not cycle.human_ready:
                return self._invalidate("input_unavailable")
            return Decision(True, False)

        if not (planner_live and feedback_live and cycle.planner_head and cycle.steady
                and cycle.target_envelope_ok):
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = cycle.now
        ready = self.stable_since is not None and cycle.now - self.stable_since >= self.ready_for_s
        if edge and ready and cycle.human_ready:
            self.tracking = True
            self.stable_since = None
            return Decision(True, True, "enter:planner_ready")
        if edge:
            return Decision(False, ready, "reject:planner_or_human_unready")
        return Decision(False, ready)

    def _fresh(self, now, received):
        return received is not None and 0 <= now - received <= self.max_input_age_s

    def _invalidate(self, reason):
        was_tracking = self.tracking
        self.tracking = False
        self.stable_since = None
        self.released = reason == "operator"
        event = ("release:" if was_tracking else "reject:") + reason
        return Decision(False, False, event)
