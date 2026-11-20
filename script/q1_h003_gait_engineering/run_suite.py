"""Run deterministic, unbranched restart trials on frozen weights."""

import argparse
import hashlib
import json
from pathlib import Path

from bench import BUNDLE, HERE, np
from check_base import verify
from relay import CASES, run
from restart_profile import PROFILES

EXPERT = HERE / "formal_03/q1_recovery_expert.onnx"
TRACKS = HERE / "formal_03/input_tracks"


def schedule(suite):
    if suite == "flat":
        return [(CASES[k][0], (0.0, 0.0), 100) for k in (0, 2)]
    if suite == "repeat":
        return list(CASES)
    if suite == "dev":
        with (HERE / "previous_study/blind01/cases.json").open() as f:
            return list(CASES) + json.load(f)
    rng = np.random.default_rng(2026112041)
    cases = []
    for _ in range(6):
        for track, kick, _ in CASES:
            scale = rng.uniform(0.88, 1.12)
            angle = rng.uniform(-0.12, 0.12)
            c, s = np.cos(angle), np.sin(angle)
            dx, dy = kick
            cases.append(
                (
                    track,
                    [float(scale * (c * dx - s * dy)), float(scale * (s * dx + c * dy))],
                    int(rng.integers(72, 139)),
                )
            )
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["dev", "confirm", "flat", "repeat"])
    ap.add_argument("profile", choices=PROFILES)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    verify(BUNDLE)
    if (
        hashlib.sha256(EXPERT.read_bytes()).hexdigest()
        != "0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92"
    ):
        raise ValueError("recovery model digest mismatch")
    cases = schedule(a.suite)
    run(
        EXPERT,
        a.output.resolve(),
        True,
        TRACKS,
        cases=cases,
        profile=PROFILES[a.profile],
        repeat_kick=a.suite == "repeat",
    )
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2))
