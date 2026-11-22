"""Asynchronous Q1 runtime recorder for synchronized deployment traces."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class Q1RuntimeRecorder:
    """Write deployment streams without blocking the 500 Hz control loop."""

    STREAMS = ("fgp", "state_500hz", "command_500hz", "policy_50hz")

    def __init__(self, directory: Path, metadata: dict[str, Any], *, queue_size: int = 50000):
        self.directory = directory
        self.metadata = _json_value(metadata)
        self.queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=queue_size)
        self.dropped = Counter()
        self.written = Counter()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.files: dict[str, Any] = {}

    def start(self) -> None:
        if self.thread is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "session.json").write_text(
            json.dumps(
                {
                    **self.metadata,
                    "started_unix_ns": time.time_ns(),
                    "streams": {name: f"{name}.jsonl" for name in self.STREAMS},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        self.thread = threading.Thread(target=self._run, name="q1_runtime_recorder", daemon=True)
        self.thread.start()

    def record(self, stream: str, payload: dict[str, Any]) -> None:
        if stream not in self.STREAMS or self.thread is None:
            return
        item = {"recorded_monotonic_ns": time.monotonic_ns(), **_json_value(payload)}
        try:
            self.queue.put_nowait((stream, item))
        except queue.Full:
            self.dropped[stream] += 1

    def close(self) -> None:
        if self.thread is None:
            return
        self.queue.put(None)
        self.thread.join(timeout=10.0)
        summary = {
            "closed_unix_ns": time.time_ns(),
            "written": dict(self.written),
            "dropped": dict(self.dropped),
            "queue_unfinished_tasks": self.queue.unfinished_tasks,
        }
        (self.directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        self.thread = None

    def _run(self) -> None:
        self.files = {
            name: (self.directory / f"{name}.jsonl").open("w", encoding="utf-8", buffering=1 << 20)
            for name in self.STREAMS
        }
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        return
                    stream, payload = item
                    self.files[stream].write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    self.written[stream] += 1
                finally:
                    self.queue.task_done()
        finally:
            for handle in self.files.values():
                handle.close()
            self.files = {}
