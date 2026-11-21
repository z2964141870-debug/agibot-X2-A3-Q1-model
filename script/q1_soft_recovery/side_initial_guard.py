"""Apply a short, side-entry-only arm guard before the learned rise takes over."""
import argparse
import math
from pathlib import Path

import numpy as np

import evaluate
import eval_core
from capture_blend import controller as blended_controller
from clearance import Clearance


def controller(xml, hold_ticks, margin, blend_ticks):
    parent = blended_controller(blend_ticks)
    guard = Clearance(xml, margin)

    class SideInitialGuard(parent):
        def reset(self):
            super().reset()
            self.side_start = None
            self.side_age = 0

        def step(self, R, w, q, qd, previous, base, anchor, *, sample_time_s, teleop_stamp_s):
            target, tau, owned = super().step(
                R,
                w,
                q,
                qd,
                previous,
                base,
                anchor,
                sample_time_s=sample_time_s,
                teleop_stamp_s=teleop_stamp_s,
            )

            side = abs(float(R[2, 1])) > 0.65
            if owned and side and self.side_start is None:
                self.side_start = q.copy()
                self.side_age = 0

            if not owned:
                self.side_start = None
                self.side_age = 0
                return target, tau, owned

            if self.side_start is None or not side or self.side_age >= hold_ticks:
                return target, tau, owned

            self.side_age += 1
            limited = guard.project(q, qd, target, self.tracker)
            if limited is None:
                return target, tau, owned
            guarded, tau = limited
            weight = 0.5 - 0.5 * math.cos(math.pi * self.side_age / max(1, hold_ticks))
            guarded = target.copy()
            guarded[14:] = self.side_start[14:] + weight * (limited[0][14:] - self.side_start[14:])
            c = self.tracker
            tau = np.clip(c.kp * (guarded - q) - c.kd * qd, -c.eff, c.eff)
            guarded = q + (tau + c.kd * qd) / c.kp
            return guarded, tau, owned

    return SideInitialGuard


def run(args):
    eval_core.Q1AutonomousRecovery = controller(args.xml, args.hold_ticks, args.margin, args.blend_ticks)
    evaluate.main(
        argparse.Namespace(
            group=args.group,
            legacy=False,
            catch=False,
            out=args.out,
            seed=args.seed,
            expert=args.expert,
            delay=11,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expert", type=Path)
    parser.add_argument("xml", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--hold-ticks", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.008)
    parser.add_argument("--blend-ticks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=92117)
    parser.add_argument("--group", choices=["static", "push", "flat", "offline", "double"], default="static")
    run(parser.parse_args())
