"""Upgraded threaded FGP HTTP client with paced FIFO consumption and safe fallback transition guard."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import Callable
from urllib.request import urlopen

import numpy as np

from q1_fgp_smpl_v2 import FGPSMPLStreamV2, SMPLTick


@dataclass(frozen=True)
class FGPQueuedTick:
    tick: SMPLTick
    arrival_ns: int


def _decay_tick_motion(tick: SMPLTick, hold_step: int, decay: float = 0.5) -> SMPLTick:
    """Shift causal 10-frame history forward with exponential velocity/angular-rate decay for brief hold ticks."""
    scale = float(decay ** max(hold_step, 1))
    hist = tick.feature.reshape(10, 43).astype(np.float32, copy=True)
    # Shift window by 1 tick so temporal velocity consistency is preserved while damping motion to 0
    latest = hist[-1].copy()
    latest[0:3] *= scale   # p_delta relative displacement
    latest[10:16] *= scale # v_local & w_local
    shifted = np.concatenate((hist[1:], latest[None, :]), axis=0).reshape(430)
    return SMPLTick(
        segment=tick.segment,
        stamp_ns=tick.stamp_ns,
        feature=shifted,
        root_xy=tick.root_xy.copy(),
        root_rotation=tick.root_rotation.copy(),
    )


class FGPHttpSourceV2:
    """Poll `/api/smpl`, validate through FGPSMPLStreamV2, and serve smoothly paced 50 Hz ticks."""

    def __init__(
        self,
        url: str,
        dt: float = 0.02,
        *,
        standing_height_m: float | None = None,
        poll_period_s: float = 0.01,
        timeout_s: float = 0.15,
        fresh_ms: float = 220.0,
        hold_grace_ms: float = 220.0,
        max_buffer_ticks: int = 3,
        queue_size: int = 32,
        on_sample: Callable[[dict], None] | None = None,
        **stream_kwargs,
    ):
        if not url.startswith(("http://", "https://")):
            raise ValueError("FGP URL must start with http:// or https://")
        if poll_period_s <= 0 or timeout_s <= 0 or fresh_ms <= 0 or queue_size < 2:
            raise ValueError("FGP polling parameters must be positive")
        self.url = url.rstrip("/") + "/api/smpl"
        self.poll_period_s = float(poll_period_s)
        self.timeout_s = float(timeout_s)
        self.fresh_ns = int(float(fresh_ms) * 1e6)
        self.hold_grace_ns = int(float(hold_grace_ms) * 1e6)
        self.max_buffer_ticks = max(1, int(max_buffer_ticks))
        self.stream = FGPSMPLStreamV2(
            dt,
            standing_height_m=standing_height_m,
            outage_grace_ms=fresh_ms,
            **stream_kwargs,
        )
        self.queue: deque[FGPQueuedTick] = deque(maxlen=int(queue_size))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_arrival_ns: int | None = None
        self.last_tick: FGPQueuedTick | None = None
        self.hold_steps = 0
        self.dropped_ticks = 0
        self.held_ticks = 0
        self.http_errors = 0
        self.responses = 0
        self.on_sample = on_sample

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="q1_fgp_http_v2", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def ingest(self, response: dict, arrival_ns: int | None = None) -> int:
        """Ingest one decoded `/api/smpl` response. Returns number of emitted 50 Hz ticks."""
        arrival = time.monotonic_ns() if arrival_ns is None else int(arrival_ns)
        with self.lock:
            self.responses += 1
            self.last_arrival_ns = arrival
            accepted_before = self.stream.events["accepted_frames"]
            segment = self.stream.segment
            ticks = self.stream.push_response(response, arrival)
            accepted_after = self.stream.events["accepted_frames"]
            if self.stream.previous is None or self.stream.segment != segment:
                self.queue.clear()
                self.last_tick = None
                self.hold_steps = 0
            for tick in ticks:
                if len(self.queue) == self.queue.maxlen:
                    self.queue.popleft()
                    self.dropped_ticks += 1
                self.queue.append(FGPQueuedTick(tick=tick, arrival_ns=arrival))
            # Keep bounded FIFO backlog so latency stays <= max_buffer_ticks * 20 ms
            while len(self.queue) > self.max_buffer_ticks + 1:
                self.queue.popleft()
                self.dropped_ticks += 1
        if accepted_after > accepted_before and self.on_sample is not None:
            self.on_sample(response)
        return len(ticks)

    def pop_paced(self, now_ns: int | None = None) -> tuple[FGPQueuedTick | None, int, bool]:
        """Consume 1 smoothly paced 50 Hz tick (FIFO up to max_buffer_ticks) or a decayed hold tick on short gaps.

        Returns `(queued_tick, dropped_count, is_hold_tick)`.
        """
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self.lock:
            dropped = 0
            while len(self.queue) > self.max_buffer_ticks:
                self.queue.popleft()
                dropped += 1
            self.dropped_ticks += dropped
            if self.queue:
                queued = self.queue.popleft()
                self.last_tick = queued
                self.hold_steps = 0
                return queued, dropped, False
            if self.last_tick is not None and 0 <= now - self.last_tick.arrival_ns <= self.hold_grace_ns:
                self.hold_steps += 1
                self.held_ticks += 1
                decayed = FGPQueuedTick(
                    tick=_decay_tick_motion(self.last_tick.tick, self.hold_steps),
                    arrival_ns=self.last_tick.arrival_ns,
                )
                return decayed, dropped, True
            return None, dropped, False

    def get_latest(self) -> tuple[FGPQueuedTick | None, int]:
        """Drop-in replacement for FGPHttpSource.get_latest(), backed by smoothed paced consumption."""
        queued, dropped, _ = self.pop_paced()
        return queued, dropped

    def status(self) -> dict:
        now = time.monotonic_ns()
        with self.lock:
            queued = self.queue[-1] if self.queue else self.last_tick
            queue_len = len(self.queue)
            dropped = self.dropped_ticks
            held = self.held_ticks
            responses = self.responses
            http_errors = self.http_errors
            last_arrival = self.last_arrival_ns
            stream_events = dict(self.stream.events)
            source_segment = self.stream.segment
        age_ms = None if queued is None else (now - queued.arrival_ns) / 1e6
        return {
            "url": self.url,
            "thread_alive": bool(self.thread is not None and self.thread.is_alive()),
            "queue_len": queue_len,
            "responses": responses,
            "http_errors": http_errors,
            "dropped_ticks": dropped,
            "held_ticks": held,
            "last_arrival_age_ms": None if last_arrival is None else (now - last_arrival) / 1e6,
            "last_tick_age_ms": age_ms,
            "fresh": age_ms is not None and age_ms <= self.fresh_ns / 1e6,
            "stream_events": stream_events,
            "source_segment": source_segment,
        }

    def _run(self) -> None:
        next_poll = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_poll:
                self.stop_event.wait(next_poll - now)
                continue
            next_poll = max(next_poll + self.poll_period_s, time.monotonic())
            try:
                with urlopen(self.url, timeout=self.timeout_s) as response:
                    payload = json.load(response)
                self.ingest(payload)
            except (OSError, ValueError, TypeError, OverflowError):
                with self.lock:
                    self.http_errors += 1
                    self.stream.invalidate("http_error")
                    self.queue.clear()
                    self.last_tick = None
                    self.hold_steps = 0


class Q1FGPFallbackGuard:
    """Tracks whether `human_smpl` or `planner` last owned CGFA/R_ref so fallback NEVER uses stale planner yaw/XY."""

    def __init__(self) -> None:
        self.last_step_kind = "planner"

    def reset(self, initial_kind: str = "planner") -> None:
        self.last_step_kind = str(initial_kind)

    def resolve_branch(self, active_mode: str, live_fgp: bool) -> tuple[str, bool]:
        """Return `(branch_to_run, need_reanchor_to_planner)` where branch is `'human_smpl'` or `'planner'`.

        Fixes the Q1_deploy `fgp=null` crash (`t=42.65s -> 58.7 deg`) where falling back to `_base_step(neutral=True)`
        while `active_mode == 'human_smpl'` skipped `_reanchor_reference` and mixed stale planner `yaw_plan`/`p_plan_xy`
        with live CGFA state.
        """
        if active_mode == "human_smpl" and live_fgp:
            self.last_step_kind = "human_smpl"
            return "human_smpl", False
        need_reanchor = self.last_step_kind == "human_smpl"
        self.last_step_kind = "planner"
        return "planner", need_reanchor
