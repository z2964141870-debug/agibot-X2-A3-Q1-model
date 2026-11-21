"""Test timed private balance from the same supported prone prefix, with normal handback."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if (HERE.parent / "q1_supported_rise_20261121").exists():
    sys.path.insert(0, str(HERE.parent / "q1_supported_rise_20261121"))
from capture import CatchGate, CatchRecovery, replay
from limbs import add_limb_contacts


class TimeGate(CatchGate):
    def reset(self):
        super().reset()
        self.age = 0
        self.grace = 0
        self.attempted = False

    def advance(self, x, width):
        self.age += 1
        if self.grace:
            self.grace -= 1
            return 0.0, False
        if (
            self.mode == "recover"
            and self.age == self.at
            and -x[49] > 0.10
            and not self.attempted
        ):
            self.attempted = True
            self.mode = "track"
            self.tilt_ticks = 0
            self.grace = 25
            return 0.0, True
        return super().advance(x, width)


def run(at, seed, expert, out, full, group):
    class WindowCatch(CatchRecovery):
        def __init__(self, *args):
            super().__init__(*args)
            self.rise.expert.gate = TimeGate()
            self.rise.expert.gate.at = at
            self.rise.expert.gate.reset()

    replay.Q1AutonomousRecovery = WindowCatch
    replay.bench.scene = add_limb_contacts
    if full:
        replay.simulate(group, seed, out.resolve(), expert.resolve(), 500, 11, True, 2)
        return
    # The evaluator's static path has 24 rows. Stop after logging the first two
    # full prone trials without modifying any state, control or success logic.
    import json
    import numpy as np

    original_save = np.savez_compressed

    class PrefixComplete(Exception):
        pass

    subset = []

    def keep(path, **arrays):
        original_save(path, **arrays)
        trace = arrays["trace"]
        g = trace[:, 4]
        releases = np.flatnonzero(np.diff(g) == -1) + 1
        grants = len(releases) == 1 and not np.any(np.diff(g) > 0)
        if grants:
            t = releases[0]
            post = trace[t : t + 50]
            grants = bool(
                trace[t, 5]
                and len(post) == 50
                and np.all(post[:, 0] >= 0.35)
                and np.all(post[:, 1] > 0.85)
            )
        subset.append(
            dict(
                success=bool(grants),
                highest=float(trace[:, 0].max()),
                calm_ticks=int(trace[:, 5].sum()),
                peak_qd=float(arrays["peaks"][g == 1, 0].max()),
            )
        )
        if len(subset) == 2:
            raise PrefixComplete

    np.savez_compressed = keep
    try:
        replay.simulate(
            "static", seed, out.resolve(), expert.resolve(), 500, 11, True, 2
        )
    except PrefixComplete:
        (out / "screen.json").write_text(json.dumps(subset, indent=2))
    finally:
        np.savez_compressed = original_save


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("at", type=int)
    p.add_argument("expert", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=7524)
    p.add_argument("--full", action="store_true")
    p.add_argument("--group", choices=("static", "push"), default="static")
    a = p.parse_args()
    run(a.at, a.seed, a.expert, a.out, a.full, a.group)
