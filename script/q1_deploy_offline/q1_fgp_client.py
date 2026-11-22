"""Threaded FGP HTTP client for the Q1 human_smpl adapter."""

from __future__ import annotations

from collections import deque
import json
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.request import urlopen

from q1_fgp_smpl import FGPSMPLStream, SMPLTick


@dataclass(frozen=True)
class FGPQueuedTick:
    tick: SMPLTick
    arrival_ns: int


class FGPHttpSource:
    """Poll `/api/smpl`, validate through FGPSMPLStream, and queue 50 Hz ticks."""

    def __init__(
        self,
        url: str,
        dt: float,
        *,
        standing_height_m: float | None = None,
        poll_period_s: float = 0.01,
        timeout_s: float = 0.15,
        fresh_ms: float = 150.0,
        queue_size: int = 32,
        on_sample: Callable[[dict], None] | None = None,
    ):
        if not url.startswith(("http://", "https://")):
            raise ValueError("FGP URL must start with http:// or https://")
        if poll_period_s <= 0 or timeout_s <= 0 or fresh_ms <= 0 or queue_size < 2:
            raise ValueError("FGP polling parameters must be positive")
        self.url = url.rstrip("/") + "/api/smpl"
        self.poll_period_s = float(poll_period_s)
        self.timeout_s = float(timeout_s)
        self.fresh_ns = int(float(fresh_ms) * 1e6)
        self.stream = FGPSMPLStream(dt, standing_height_m=standing_height_m)
        self.queue = deque(maxlen=int(queue_size))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_arrival_ns: int | None = None
        self.last_tick: FGPQueuedTick | None = None
        self.dropped_ticks = 0
        self.http_errors = 0
        self.responses = 0
        self.on_sample = on_sample

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="q1_fgp_http", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def ingest(self, response: dict, arrival_ns: int | None = None) -> int:
        """Ingest one decoded `/api/smpl` response. Returns the number of new ticks."""
        arrival = time.monotonic_ns() if arrival_ns is None else int(arrival_ns)
        accepted_before = 0
        accepted_after = 0
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
            for tick in ticks:
                if len(self.queue) == self.queue.maxlen:
                    self.queue.popleft()
                    self.dropped_ticks += 1
                self.queue.append(FGPQueuedTick(tick=tick, arrival_ns=arrival))
        if accepted_after > accepted_before and self.on_sample is not None:
            self.on_sample(response)
        return len(ticks)

    def get_latest(self) -> tuple[FGPQueuedTick | None, int]:
        """Return the newest queued tick and discard older ticks in the same batch."""
        with self.lock:
            if not self.queue:
                queued = self.last_tick
                dropped = 0
            else:
                queued = self.queue[-1]
                dropped = len(self.queue) - 1
                self.queue.clear()
                self.last_tick = queued
                self.dropped_ticks += dropped
            return queued, dropped

    def status(self) -> dict:
        now = time.monotonic_ns()
        with self.lock:
            queued = self.last_tick
            queue_len = len(self.queue)
            dropped = self.dropped_ticks
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
