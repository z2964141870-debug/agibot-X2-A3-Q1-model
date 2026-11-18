"""Record an opt-in gesture check without changing the installed FGP or policy."""
import argparse
import base64
import json
from pathlib import Path
from queue import Full, Queue
import signal
import sys
from threading import Thread
import time
from unittest.mock import patch


class Recorder:
    def __init__(self, destination):
        self.file = destination.open("x", encoding="utf-8")
        self.queue = Queue(maxsize=64)
        self.written = self.dropped = 0
        self.error = None
        self.thread = Thread(target=self.write, name="gesture-trace")
        self.thread.start()

    def submit(self, row):
        try:
            self.queue.put_nowait(row)
        except Full:
            self.dropped += 1

    def write(self):
        remaining = 48 * 1024 * 1024
        while (row := self.queue.get()) is not None:
            if self.error:
                continue
            try:
                window = row.pop("aligned_reference", None)
                if window is not None:
                    row["aligned_reference"] = {
                        "dtype": "<f4", "shape": list(window.shape),
                        "data_b64": base64.b64encode(window.tobytes()).decode("ascii"),
                    }
                line = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                if len(line.encode()) > remaining:
                    raise ValueError("gesture trace size limit reached")
                self.file.write(line)
                remaining -= len(line.encode())
                self.written += 1
            except (OSError, ValueError) as error:
                self.error = str(error)

    def close(self):
        self.queue.put(None)
        self.thread.join()
        with self.file as stream:
            stream.write(json.dumps(dict(kind="trace_end", written=self.written, dropped=self.dropped,
                                         complete=not self.error and self.dropped == 0, error=self.error))+"\n")


def run_policy(path, args):
    from script.x2_morph_human import runtime
    from script.x2_morph_human.__main__ import main as policy_main

    recorder = Recorder(path)

    class TracedRuntime(runtime.Runtime):
        def handle(self, message, now_ns, receiver_clock):
            reply = super().handle(message, now_ns, receiver_clock)
            row = dict(kind=message.get("kind"), received_ns=now_ns, replied_ns=time.monotonic_ns(),
                       request=message, reply=reply)
            if reply.get("ready"):
                row["aligned_reference"] = self.runner.last_ref.astype("<f4", copy=True)
            recorder.submit(row)
            return reply

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    sys.argv = [sys.argv[0], "serve", *args]
    try:
        with patch.object(runtime, "Runtime", TracedRuntime):
            policy_main()
    finally:
        recorder.close()


def run_garment(args):
    from script.x2_fgp_preview import server, terminal

    original = server.Preview

    def record_smpl(*args, **kwargs):
        return original(*args, **dict(kwargs, log_smpl=True))

    sys.argv = [sys.argv[0], *args]
    with patch.object(server, "Preview", record_smpl):
        terminal.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    roles = parser.add_subparsers(dest="role", required=True)
    policy = roles.add_parser("policy")
    policy.add_argument("trace", type=Path)
    roles.add_parser("garment")
    options, args = parser.parse_known_args()
    if options.role == "policy":
        run_policy(options.trace, args)
    else:
        run_garment(args)


if __name__ == "__main__":
    main()
