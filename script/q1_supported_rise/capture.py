"""Compare early autonomous standing using the frozen external handback test."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q1_recovery_review_20261121"))
import replay_controls as replay
from recovery import Gate


class CatchGate(Gate):
    def reset(self):
        super().reset()
        self.ready_ticks = 0

    def advance(self, x, width):
        reachable = self.mode == "recover" and x[-1] and x[-2] > 0.28 and -x[49] > 0.75
        self.ready_ticks = self.ready_ticks + 1 if reachable else 0
        if self.ready_ticks == 2:
            self.mode = "handoff"
            self.stand_ticks = self.ready_ticks = 0
            # Enter at the last four frames of the original cosine blend.
            self.blend_ticks = 26
            weight, _ = super().advance(x, width)
            return weight, True
        return super().advance(x, width)


class CatchRecovery(replay.Q1AutonomousRecovery):
    def __init__(self, *args):
        super().__init__(*args)
        self.rise.expert.gate = CatchGate()
        self.rise.expert.gate.reset()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=["static", "push"])
    p.add_argument("expert", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument(
        "--native", action="store_true", help="original 5 ms direct-target comparison"
    )
    a = p.parse_args()
    replay.Q1AutonomousRecovery = CatchRecovery
    settings = (0, 0, False, 5) if a.native else (500, 11, True, 2)
    replay.simulate(a.group, a.seed, a.out.resolve(), a.expert.resolve(), *settings)
