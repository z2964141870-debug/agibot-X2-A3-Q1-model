"""Early autonomous balance capture from a flexed, forward-facing crouch."""

import argparse
from pathlib import Path

from limbs import replay, CatchRecovery, add_limb_contacts
from capture import CatchGate


class CrouchedGate(CatchGate):
    upper_cosine = 1.01

    def reset(self):
        super().reset()
        self.crouch_ticks = self.balance_ticks = 0

    def advance(self, x, width):
        # A brief autonomous balance attempt is permitted while the pelvis is low.
        # The outer controller continues to require a quiet, fully upright stand.
        if self.balance_ticks:
            self.balance_ticks -= 1
            if -x[49] < 0.1:
                self.balance_ticks = 0
            else:
                return 0.0, False
        crouch = (
            self.mode == "recover"
            and x[-1]
            and x[-2] > 0.19
            and 0.42 < -x[49] < self.upper_cosine
            and -x[47] < -0.12
        )
        self.crouch_ticks = self.crouch_ticks + 1 if crouch else 0
        if self.crouch_ticks >= 2:
            self.mode = "track"
            self.tilt_ticks = self.crouch_ticks = 0
            self.balance_ticks = 18
            return 0.0, True
        return super().advance(x, width)


class LowCatch(CatchRecovery):
    gate_class = CrouchedGate

    def __init__(self, *args):
        super().__init__(*args)
        self.rise.expert.gate = self.gate_class()
        self.rise.expert.gate.reset()


class MidCrouchGate(CrouchedGate):
    upper_cosine = 0.82


class MidCrouchCatch(LowCatch):
    gate_class = MidCrouchGate


if __name__ == "__main__":
    import crouch

    p = argparse.ArgumentParser()
    p.add_argument("group", choices=["crouch", "static", "push"])
    p.add_argument("expert", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=7524)
    p.add_argument("--middle-only", action="store_true")
    a = p.parse_args()
    controller = MidCrouchCatch if a.middle_only else LowCatch
    if a.group == "crouch":
        crouch.evaluate(
            Path.home() / "projects/Q1_deploy/real_logs",
            a.expert,
            a.out,
            controller,
            a.seed,
        )
    else:
        replay.bench.scene = add_limb_contacts
        replay.Q1AutonomousRecovery = controller
        replay.simulate(
            a.group, a.seed, a.out.resolve(), a.expert.resolve(), 500, 11, True, 2
        )
