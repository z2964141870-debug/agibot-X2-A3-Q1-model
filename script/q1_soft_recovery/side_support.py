"""Compose the side-entry trajectory with the trained support-transfer correction."""
import argparse
from pathlib import Path

import numpy as np
import torch

from policy import export
from roll_to_rise import RollToRise
from support_transfer import START, END


class SideSupportEntry(torch.nn.Module):
    def __init__(self, roll, keys, fit, strength=1.0):
        super().__init__()
        self.inner = RollToRise(roll, 10)
        self.register_buffer("side_keys", torch.tensor(keys, dtype=torch.float32))
        for name in ("parameters", "reference", "basis"):
            self.register_buffer("fit_" + name, torch.tensor(fit[name], dtype=torch.float32))
        self.strength = float(strength)

    def forward(self, rec_obs_hist, rec_state):
        x = rec_obs_hist[:, -1]
        q = x[:, :22] + self.inner.rise.crouch.stand
        side_begin = (
            (rec_state[:, 0] == 0)
            & (x[:, 48].abs() > 0.65)
            & (x[:, 48].abs() > x[:, 47].abs())
        )
        side_active = side_begin | (rec_state[:, 1] == 10)
        age = torch.where(side_begin, 0.0, rec_state[:, 0])
        side = torch.where(side_begin, (x[:, 48] < 0).float(), rec_state[:, 2]).clamp(0, 1)
        origin = torch.where(side_begin[:, None], q, rec_state[:, 4:])
        phase = ((age + 1) / 24).clamp(0, 4)
        index = phase.long().clamp(0, 3)
        alpha = 0.5 - 0.5 * torch.cos(torch.pi * (phase - index))
        first = torch.where(
            (index == 0)[:, None],
            origin,
            self.side_keys[side.long(), (index - 1).clamp(0, 3)],
        )
        side_target = first + alpha[:, None] * (self.side_keys[side.long(), index] - first)
        quiet = (x[:, 47] < -0.84) & (torch.linalg.vector_norm(x[:, 44:47] * 4, dim=1) < 1.45)
        count = torch.where(quiet, torch.where(side_begin, 0.0, rec_state[:, 3]) + 1, 0.0)
        rolling = side_active & (count < 10)
        side_state = torch.cat(
            (torch.stack((age + 1, age * 0 + 10, side, count), 1), origin), 1
        )

        target, state = self.inner(
            rec_obs_hist,
            torch.where(side_active[:, None], torch.zeros_like(rec_state), rec_state),
        )
        target = torch.where(rolling[:, None], side_target, target)
        state = torch.where(rolling[:, None], side_state, state)

        support_age = state[:, 3]
        support_index = (support_age - START).clamp(0, END - START).long()
        support_error = (self.fit_reference[support_index] + x[:, 49]).clamp(-0.6, 0.6)
        active = (state[:, 1] == 6) & (state[:, 2] == 1) & (support_age > START) & (support_age < END)
        support_weight = torch.sin(torch.pi * support_index / (END - START)).square() * active
        correction = self.fit_parameters[:5] + support_error[:, None] * self.fit_parameters[5:]
        support_delta = self.strength * support_weight[:, None] * (correction @ self.fit_basis)
        return (target + support_delta).clamp(self.inner.rise.lo, self.inner.rise.hi), state


def run(args):
    keys = np.stack([np.load(args.side / "side_0.npz")["keys"], np.load(args.side / "side_1.npz")["keys"]])
    fit = np.load(args.support / "best.npz")
    model = SideSupportEntry(np.load(args.roll), keys, fit, args.strength).eval()
    export(model, args.out)
    print(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roll", type=Path)
    parser.add_argument("side", type=Path)
    parser.add_argument("support", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--strength", type=float, default=1.0)
    run(parser.parse_args())
