"""Candidate explicit token revocation for old SONIC, exercised offline."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA256 = "58248ded822f13ce79c8f6b41a8d84c49893932f739dc3fb6198b2354076c7f6"


class ReleaseAnnouncer:
    """Repeat revocation briefly, in case the PUB subscriber misses one packet."""

    def __init__(self):
        self.had_token = False
        self.next_notice = 0.
        self.deadline = 0.

    def due(self, now, *, published, engaged):
        if published:
            self.had_token = True
            self.deadline = 0.
        elif self.had_token and not engaged:
            self.had_token = False
            self.next_notice = now
            self.deadline = now + .5
        if self.next_notice <= now < self.deadline:
            self.next_notice = now + .1
            return True
        return False


def release_packet(pack):
    import numpy as np

    return pack({"engaged": np.zeros(1, dtype=np.float32)}, topic="pose", version=4)


def build(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("token service source SHA-256 differs from the audited version")
    body = raw.decode("utf-8")
    changes = [
        ("    args = ap.parse_args()\n", """    ap.add_argument("--announce-release", action="store_true")
    args = ap.parse_args()
    release = None
    if args.announce_release:
        from sonic_token_release import ReleaseAnnouncer, release_packet
        release = ReleaseAnnouncer()
        explicit_release = release_packet(pack_pose_message)
"""),
        ('            send_wb_state(bool(res.diag["wb_state_sent"]))\n',
         '''            send_wb_state(bool(res.diag["wb_state_sent"]))
        if release is not None and release.due(now, published=res.published, engaged=res.engaged):
            tokens.send(explicit_release)
'''),
    ]
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError(f"unexpected replacement count for {old!r}")
        body = body.replace(old, new)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(body)


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
