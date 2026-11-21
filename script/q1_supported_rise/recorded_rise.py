"""Map two recorded vendor rises into reference tracks for a simulation-only expert."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from export import ROOT, FiniteRoll, SequenceExpert, export


def load_track(record, names):
    transitions = json.loads(record.with_suffix(".json").read_text())["transitions"]
    start, stop = [
        next(t["t_rel_s"] for t in transitions if t["desc"] == desc)
        for desc in ("STAND_UP", "BIPED_WALK_RUN")
    ]
    with np.load(record) as log:
        grid = np.arange(start, stop + 0.5, 0.02)
        t = (log["joint_state_stamp"] - log["event_ns"].item()) * 1e-9
        order = [log["joint_names"].tolist().index(name) for name in names]
        measured = np.stack([np.interp(grid, t, log["q"][:, i]) for i in order], axis=1)
        imu_t = (log["imu_stamp"] - log["event_ns"].item()) * 1e-9
        x, y, z, w = log["orientation_xyzw"].T
        up = np.stack(
            [
                np.interp(grid, imu_t, g)
                for g in (
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                )
            ],
            axis=1,
        )
    return measured, up


class RecordedRise(torch.nn.Module):
    def __init__(self, directory, adaptive):
        super().__init__()
        self.legacy = SequenceExpert(ROOT / "candidate_seq01")
        self.legacy.load_state_dict(
            torch.load(ROOT / "ppo_residual_01/actor_00200.pt", map_location="cpu")
        )
        self.register_buffer("stand", self.legacy.stand.clone())
        cfg = json.loads(
            (
                ROOT
                / "formal_03/pinned/q1_v2b_gantry_deploy_bundle/config/q1_deploy_config.json"
            ).read_text()
        )
        files = [
            "q1_official_standup_from_mcap.npz",
            "q1_official_standup_faceup_20261120.npz",
        ]
        tracks = [load_track(directory / f, cfg["joint_names"]) for f in files]
        length = max(len(q) for q, _ in tracks)

        def pad(x):
            return np.pad(x, ((0, length - len(x)), (0, 0)), mode="edge")

        self.register_buffer(
            "q_ref",
            torch.tensor(np.stack([pad(q) for q, _ in tracks]), dtype=torch.float32),
        )
        self.register_buffer(
            "up_ref",
            torch.tensor(np.stack([pad(u) for _, u in tracks]), dtype=torch.float32),
        )
        self.length = length
        self.adaptive = adaptive

    def forward(self, rec_obs_hist, rec_state):
        x = rec_obs_hist[:, -1]
        q, up = x[:, :22] + self.stand, -x[:, 47:50]
        age, mode, kind, tick = rec_state[:, :4].unbind(1)
        active = mode == 4
        legacy_target, legacy_state = self.legacy(
            rec_obs_hist, torch.where(active[:, None], 0.0, rec_state)
        )
        aligned = (up[:, 0].abs() > 0.70) & (up[:, 0].abs() > up[:, 1].abs())
        enter = ~active & aligned & ((mode == 0) | ((mode == 1) & (tick > 15)))
        kind = torch.where(enter, (up[:, 0] > 0).float(), kind).clamp(0, 1)
        tick = torch.where(enter, 0.0, tick)
        initial = torch.where(enter[:, None], q, rec_state[:, 4:])
        sample = (tick - 20.0).clamp(0, self.length - 1).long()
        target = self.q_ref[kind.long(), sample]
        mix = 0.5 - 0.5 * torch.cos(torch.pi * ((tick + 1) / 20.0).clamp(0, 1))
        target = initial + (target - initial) * mix[:, None]
        increment = torch.ones_like(tick)
        if self.adaptive:
            alignment = (self.up_ref[kind.long(), sample] * up).sum(1)
            increment = torch.where((tick >= 20) & (alignment < 0.8), 0.5, 1.0)
        new = torch.cat(
            (
                torch.stack((age + 1, 0.0 * mode + 4, kind, tick + increment), 1),
                initial,
            ),
            1,
        )
        return torch.where(
            (active | enter)[:, None], target, legacy_target
        ), torch.where((active | enter)[:, None], new, legacy_state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--adaptive", action="store_true")
    args = p.parse_args()
    export(FiniteRoll(RecordedRise(args.records, args.adaptive)).eval(), args.output)
