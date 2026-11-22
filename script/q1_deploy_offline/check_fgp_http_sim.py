"""Replay saved FGP frames through the simulator's real loopback HTTP entry point."""

import argparse
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path


def check(bundle, recording, output, viewer=False):
    with recording.open() as source:
        samples = [json.loads(line) for line in islice(source, 80)]

    class Handler(BaseHTTPRequestHandler):
        first_poll = None
        sample_index = [-1, -1]

        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/api/smpl":
                self.send_error(404)
                return
            now = time.monotonic()
            if Handler.first_poll is None:
                Handler.first_poll = now
            elapsed = now - Handler.first_poll
            if 1.5 <= elapsed < 2.15:
                self.send_error(503)
                return
            generation = int(elapsed >= 2.15)
            segment_time = elapsed - (2.15 if generation else 0.0)
            index = min(int(segment_time * 30), Handler.sample_index[generation] + 1, len(samples) - 1)
            Handler.sample_index[generation] = index
            sample = samples[index].copy()
            sample["calibration_id"] = generation
            payload = json.dumps({
                "status": "tracking", "realtime_valid": True, "sample": sample,
                "session_id": sample["session_id"], "calibration_id": generation,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            command = [
                sys.executable, str(bundle / "code/sim_q1_fgp_human_smpl.py"),
                "--fgp-url", f"http://127.0.0.1:{server.server_port}",
                "--duration", "4", "--output", str(output),
            ]
            if viewer:
                command.append("--viewer")
            subprocess.run(command, check=True, timeout=45)
        finally:
            server.shutdown()
            worker.join()
    report = json.loads((output / "report.json").read_text())
    assert report["stop_reason"] == "completed"
    assert report["reference_segments"] == [1, 2]
    assert report["idle_ticks"] >= 20 and report["policy_steps"] > 100
    assert report["http_status"]["http_errors"] > 0
    assert report["http_status"]["stream_events"]["http_error"] == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    check(args.bundle, args.recording, args.output, viewer=args.viewer)
