"""Make a read-only-test candidate from the exact X2 watchdog source."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA256 = "69b8b3ddf77d044f5528f50bb6f77d56affe90dc923befc0f0e50c15ab761375"


def build(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("watchdog source SHA-256 differs from the audited version")
    body = raw.decode("utf-8")
    changes = [
        ("    DEFAULT_PUB_RATE_HZ,\n", "    DEFAULT_PUB_RATE_HZ,\n    HEADER_SIZE,\n"),
        ("    args = p.parse_args(raw_argv)\n",
         '    p.add_argument("--trace-reference-source", action="store_true")\n'
         "    args = p.parse_args(raw_argv)\n"
         "    if args.trace_reference_source and args.upstream_topic != args.downstream_topic:\n"
         '        p.error("--trace-reference-source requires identical wire topics")\n'),
        ("    pub.bind(bind_url)\n", """    pub.bind(bind_url)
    source_sequence = 0
    if args.trace_reference_source:
        from sonic_reference_provenance import tag_pose

    def send_pose(message, source):
        nonlocal source_sequence
        if args.trace_reference_source:
            message = tag_pose(message, args.downstream_topic, HEADER_SIZE, source_sequence, source)
            source_sequence += 1
        pub.send(message, zmq.NOBLOCK)
"""),
        ("pub.send(latest, zmq.NOBLOCK)", 'send_pose(latest, "upstream")'),
        ("pub.send(msg_to_send, zmq.NOBLOCK)", 'send_pose(msg_to_send, "hold")'),
        ("pub.send(msg, zmq.NOBLOCK)\n                        blend_frames", 'send_pose(msg, "blend")\n                        blend_frames'),
        ("pub.send(msg, zmq.NOBLOCK)\n                        idle_frames", 'send_pose(msg, "cold_idle" if target_state == STATE_COLD_IDLE else "idle_clip")\n                        idle_frames'),
    ]
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError(f"unexpected replacement count for {old!r}")
        body = body.replace(old, new)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(body)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
