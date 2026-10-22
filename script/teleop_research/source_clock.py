"""Offline reference-velocity baseline; not a robot safety controller.

source_time_s is in one declared source timeline. arrival_time_s is diagnostic
only, in one receiver timeline; never subtract across clocks without sync.
No prediction, extrapolation, interpolation or hidden sample clipping.
"""
import math


class SourceClockDerivative:
    def __init__(self, dimension, max_gap_s, max_abs_velocity, clock_id):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError('positive dimension required')
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError('explicit source clock identity required')
        for x in (max_gap_s, max_abs_velocity):
            if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
                raise ValueError('positive finite bounds required')
        self.dimension = dimension
        self.max_gap_s = max_gap_s
        self.max_abs_velocity = max_abs_velocity
        self.clock_id = clock_id
        self.previous = None

    def reset(self):
        self.previous = None

    def step(self, q, source_time_s, arrival_time_s, clock_id):
        if clock_id != self.clock_id:
            raise ValueError('source clock changed: explicit reset/reconfiguration required')
        q = tuple(q)
        if len(q) != self.dimension or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in q):
            raise ValueError('invalid position vector')
        if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in (source_time_s, arrival_time_s)):
            raise ValueError('invalid timestamp')
        current = (q, source_time_s, arrival_time_s)
        if self.previous is None:
            self.previous = current
            return dict(status='FIRST_FRAME_NO_VELOCITY', velocity=None, source_dt_s=None, arrival_dt_s=None)
        prev_q, prev_t, prev_arrival = self.previous
        dt = source_time_s - prev_t
        if dt <= 0:
            # Reject without changing accepted state; do not replay stale source frame.
            return dict(status='REJECT_NONINCREASING_SOURCE_TIME', velocity=None, source_dt_s=dt, arrival_dt_s=None)
        arrival_dt = arrival_time_s - prev_arrival
        if arrival_dt < 0:
            return dict(status='REJECT_ARRIVAL_CLOCK_REVERSAL', velocity=None, source_dt_s=dt, arrival_dt_s=arrival_dt)
        if dt > self.max_gap_s:
            self.previous = current
            return dict(status='GAP_REPRIME_NO_VELOCITY', velocity=None, source_dt_s=dt, arrival_dt_s=arrival_dt)
        velocity = tuple((a-b)/dt for a,b in zip(q, prev_q))
        if any(abs(v) > self.max_abs_velocity for v in velocity):
            self.previous = current
            return dict(status='REJECT_IMPLAUSIBLE_VELOCITY_REPRIME', velocity=None, source_dt_s=dt, arrival_dt_s=arrival_dt)
        self.previous = current
        return dict(status='OK', velocity=velocity, source_dt_s=dt, arrival_dt_s=arrival_dt)
