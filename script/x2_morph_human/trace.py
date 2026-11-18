"""Bounded, opt-in recording of the advisory service for offline diagnosis."""
from __future__ import annotations

import base64
import json
import os
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread

import numpy as np


class SessionCapture:
    """Snapshot the arrays at the ONNX Runtime call boundary."""

    def __init__(self, session, trace):
        self.session = session
        self.trace = trace

    def run(self, names, feeds):
        self.trace["onnx_inputs"] = {name: value.copy() for name, value in feeds.items()}
        result = self.session.run(names, feeds)
        self.trace["onnx_outputs"] = {name: value.copy() for name, value in zip(names, result)}
        return result


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape),
                    data_b64=base64.b64encode(value.tobytes(order="C")).decode("ascii"))
    if isinstance(value, bytes):
        try:
            return dict(json_text=value.decode("utf-8"))
        except UnicodeDecodeError:
            return dict(data_b64=base64.b64encode(value).decode("ascii"))
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


class TraceRecorder:
    def __init__(self, path, max_bytes, metadata):
        self.max_bytes = max_bytes
        self.queue = Queue(maxsize=64)
        self.done = Event()
        self.error = None
        self.submitted = self.written = 0
        self.file = self._open(path)
        self.thread = Thread(target=self._write, args=(metadata,), name="morph-trace")
        self.thread.start()

    @staticmethod
    def _open(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        return os.fdopen(fd, "wb")

    @property
    def active(self):
        return self.error is None and not self.done.is_set()

    def _fail(self, reason):
        if self.error is None:
            self.error = reason
            print(f"Morph trace incomplete: {reason}", file=sys.stderr, flush=True)

    def submit(self, event):
        if self.active:
            self.submitted += 1
            try:
                self.queue.put_nowait(dict(event, request_index=self.submitted))
            except Full:
                self._fail("writer queue full")

    def _write(self, metadata):
        def line(event):
            return (json.dumps(event, default=encode, allow_nan=False,
                               separators=(",", ":")) + "\n").encode("utf-8")

        try:
            with self.file as out:
                header = line(dict(metadata, kind="trace_start", version=1))
                out.write(header)
                size = len(header)
                while not (self.done.is_set() and self.queue.empty()):
                    try:
                        event = self.queue.get(timeout=.1)
                    except Empty:
                        continue
                    payload = line(event)
                    if size + len(payload) + 4096 > self.max_bytes:
                        self._fail("file size limit reached")
                        break
                    out.write(payload)
                    size += len(payload)
                    self.written += 1
                complete = self.error is None and self.written == self.submitted
                out.write(line(dict(kind="trace_end", complete=complete, written=self.written,
                                    submitted=self.submitted, error=self.error)))
        except (OSError, ValueError, TypeError) as error:
            self._fail(f"{type(error).__name__}: {error}")

    def close(self):
        self.done.set()
        self.thread.join()
