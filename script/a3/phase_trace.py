"""Durable nested diagnostic spans without importing torch or Isaac."""

from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import sys
import threading
import time


class PhaseTrace:
    def __init__(self, directory, sync=None, context=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"phases.{os.getpid()}.jsonl"
        self.boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.sync = sync or synchronize_if_initialized
        self.context = context or (lambda: {})
        self.ordinal = 0
        self.lock = threading.RLock()
        self.local = threading.local()

    def emit(self, event, phase, call_id=None, **details):
        with self.lock:
            record = {**details, **self.context(), "at_utc":datetime.now(timezone.utc).isoformat(),
                      "monotonic_ns":time.monotonic_ns(), "pid":os.getpid(),
                      "thread_id":threading.get_ident(), "boot_id":self.boot,
                      "event":event, "phase":phase, "call_id":call_id}
            line = json.dumps(record, allow_nan=False) + "\n"
            # O_APPEND + fsync each boundary; independent per-process files.
            existed = self.path.exists()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                data = line.encode()
                while data:
                    data = data[os.write(fd, data):]
                os.fsync(fd)
            finally:
                os.close(fd)
            if not existed:
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

    @contextmanager
    def span(self, phase, cuda_sync=True, **details):
        with self.lock:
            self.ordinal += 1
            call_id = f"{os.getpid()}:{self.ordinal}"
        stack = getattr(self.local, "stack", [])
        parent = stack[-1] if stack else None
        self.local.stack = [*stack, call_id]
        self.emit("BEGIN", phase, call_id, parent_call_id=parent, **details)
        try:
            if cuda_sync:
                self.emit("PRE_SYNC_BEGIN", phase, call_id)
                synced = self.sync()
                self.emit("PRE_SYNC_END", phase, call_id, cuda_synchronized=synced)
            yield
            self.emit("CALL_RETURNED", phase, call_id)
            if cuda_sync:
                self.emit("POST_SYNC_BEGIN", phase, call_id)
                synced = self.sync()
                self.emit("POST_SYNC_END", phase, call_id, cuda_synchronized=synced)
            self.emit("END", phase, call_id)
        except BaseException as error:
            self.emit("ERROR", phase, call_id, error_type=type(error).__name__, error=str(error)[:1000])
            raise
        finally:
            self.local.stack = stack


def synchronize_if_initialized():
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        return True
    return False


_trace = None
_trainer = None


def trainer_context():
    state = getattr(_trainer, "state", None)
    return dict(trainer_global_step=getattr(state, "global_step", None),
                attempt_optimizer_step=getattr(_trainer, "_optimizer_step_ordinal", None))


def trace():
    global _trace
    if _trace is None:
        _trace = PhaseTrace(os.environ["YUANQI_PHASE_TRACE_DIR"], context=trainer_context)
    return _trace


def set_trainer(trainer):
    global _trainer
    _trainer = trainer


def wrap(function, phase, cuda_sync=True):
    if getattr(function, "_yuanqi_phase_wrapped", False):
        return function
    @functools.wraps(function)
    def traced(*args, **kwargs):
        with trace().span(phase, cuda_sync=cuda_sync):
            return function(*args, **kwargs)
    traced._yuanqi_phase_wrapped = True
    return traced


def decorate(phase, cuda_sync=True):
    return lambda function: wrap(function, phase, cuda_sync)


def summarize_files(directory):
    rows, partial_lines = [], []
    for path in sorted(Path(directory).glob("phases.*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                partial_lines.append(dict(file=path.name, line=number))
    rows.sort(key=lambda row: (row["boot_id"], row["monotonic_ns"], row["pid"]))
    pending = {}
    last_end = None
    errors = []
    for row in rows:
        key = (row["boot_id"], row["call_id"])
        if row["event"] == "BEGIN":
            pending[key] = dict(begin=row, last_event=row)
        elif row["event"] == "END":
            pending.pop(key, None)
            last_end = row
        elif key in pending:
            pending[key]["last_event"] = row
        if row["event"] == "ERROR":
            errors.append(row)
    return dict(records=len(rows), last_completed=last_end,
                incomplete=list(pending.values()), errors=errors[-10:], partial_lines=partial_lines,
                scope="LAST_DURABLE_BOUNDARIES_NOT_ROOT_CAUSE_PROOF")
