"""Wait for the fall to settle before selecting a recovery motion."""
import argparse
from pathlib import Path

import numpy as np
import torch

from policy import export
from roll_to_rise import RollToRise


class SettledEntry(torch.nn.Module):
    def __init__(self, fit, ticks):
        super().__init__()
        self.motion = RollToRise(fit, 10)
        self.ticks = ticks

    def forward(self, rec_obs_hist, rec_state):
        frame = rec_obs_hist[:, -1]
        pending = (rec_state[:, 0] == 0) | (rec_state[:, 1] == 9)
        quiet = (torch.linalg.vector_norm(frame[:, 44:47] * 4, dim=1) < 1.3) & (
            frame[:, 22:44].abs().amax(dim=1) < .4
        )
        count = torch.where(quiet, rec_state[:, 2] + 1, 0.)
        wait = pending & (count < self.ticks)
        inner = torch.where(pending[:, None], torch.zeros_like(rec_state), rec_state)
        target, next_state = self.motion(rec_obs_hist, inner)
        q = frame[:, :22] + self.motion.rise.crouch.stand
        # Hold the first sensed configuration through impact, then classify gravity.
        origin = torch.where((rec_state[:, 0] == 0)[:, None], q, rec_state[:, 4:])
        age = rec_state[:, 0] + 1
        waiting = torch.cat((torch.stack((age, torch.full_like(age, 9), count, age), 1), origin), 1)
        return torch.where(wait[:, None], origin, target), torch.where(wait[:, None], waiting, next_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ticks", type=int, default=8)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(SettledEntry(np.load(args.fit), args.ticks), args.out)
