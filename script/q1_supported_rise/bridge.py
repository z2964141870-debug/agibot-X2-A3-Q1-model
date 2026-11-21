"""Learn the transition after a supported prone approach in the collision-enabled plant."""

import argparse
from pathlib import Path
import numpy as np
import torch
from export import FiniteRoll, SupportedRise, export
from low_catch import MidCrouchCatch
from limbs import replay, add_limb_contacts

SWITCH = 236


class Bridge(torch.nn.Module):
    def __init__(self, approach, transfer):
        super().__init__()
        self.approach = approach
        self.transfer = transfer
        self.register_buffer("stand", approach.stand.clone())

    def forward(self, rec_obs_hist, rec_state):
        target, state = self.approach(
            rec_obs_hist, torch.where((rec_state[:, 1] == 5)[:, None], 0.0, rec_state)
        )
        start = (rec_state[:, 1] == 4) & (rec_state[:, 3] >= SWITCH)
        active = rec_state[:, 1] == 5
        q = rec_obs_hist[:, -1, :22] + self.stand
        count = torch.where(start, 0.0, rec_state[:, 3])
        initial = torch.where(start[:, None], q, rec_state[:, 4:])
        inner = torch.cat(
            (
                rec_state[:, :1],
                torch.full_like(count[:, None], 4.0),
                rec_state[:, 2:3],
                count[:, None],
                initial,
            ),
            1,
        )
        new_target, new_state = self.transfer(rec_obs_hist, inner)
        new_mode = torch.where(new_state[:, 1:2] == 4, 5.0, new_state[:, 1:2])
        new_state = torch.cat((new_state[:, :1], new_mode, new_state[:, 2:]), 1)
        use = (active | start)[:, None]
        return torch.where(use, new_target, target), torch.where(use, new_state, state)


def collect(expert, directory):
    original_step = replay.mujoco.mj_step

    class Captured(Exception):
        pass

    replay.bench.scene = add_limb_contacts
    replay.Q1AutonomousRecovery = MidCrouchCatch
    for seed in (7521, 7522, 7523, 7524):
        records = []
        counter = 0

        def observe(model, data, *args, **kwargs):
            nonlocal counter
            original_step(model, data, *args, **kwargs)
            if kwargs.get("nstep") == 20:
                counter = 0
                return
            counter += 1
            if counter == SWITCH * 10:
                records.append(np.r_[data.qpos, data.qvel])
                if len(records) == 8:
                    raise Captured

        replay.mujoco.mj_step = observe
        try:
            replay.simulate(
                "static",
                seed,
                (directory / str(seed)).resolve(),
                expert.resolve(),
                500,
                11,
                True,
                2,
            )
        except Captured:
            np.save(directory / f"{seed}.npy", records)
        finally:
            replay.mujoco.mj_step = original_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["collect", "export"])
    p.add_argument("first", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--second", type=Path)
    a = p.parse_args()
    if a.action == "collect":
        a.out.mkdir(parents=True, exist_ok=False)
        collect(a.first, a.out)
    else:
        with np.load(a.first) as f, np.load(a.second) as s:
            module = Bridge(
                SupportedRise(f, False, hold_terminal=True),
                SupportedRise(s, False, hold_terminal=True),
            )
        export(FiniteRoll(module).eval(), a.out)
