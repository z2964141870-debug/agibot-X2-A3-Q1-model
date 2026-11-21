"""Export a fitted rise inside the legacy expert interface. The owner remains external."""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent / "h003_recovery_expert"
sys.path.insert(0, str(ROOT))
from sequence_model import SequenceExpert, export


class FiniteRoll(torch.nn.Module):
    """Define zero roll at the gravity pole for both PyTorch and ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("stand", model.stand.clone())

    def forward(self, rec_obs_hist, rec_state):
        y, z = rec_obs_hist[:, :, 48:49], rec_obs_hist[:, :, 49:50]
        # The exported atan2 divides by its second input. Canonicalize axis
        # cases including signed zero and the +/-pi discontinuity.
        safe_z = torch.where(z == 0, torch.full_like(z, -1e-12), z)
        safe_y = torch.where((y == 0) & (z > 0), torch.full_like(y, -1e-12), y)
        history = torch.cat(
            (rec_obs_hist[:, :, :48], safe_y, safe_z, rec_obs_hist[:, :, 50:]), dim=2
        )
        return self.model(history, rec_state)


class SupportedRise(torch.nn.Module):
    def __init__(self, fitted, supine, stretch=1.0, brake=0.0, hold_terminal=False):
        super().__init__()
        self.legacy = SequenceExpert(ROOT / "candidate_seq01")
        self.legacy.load_state_dict(
            torch.load(ROOT / "ppo_residual_01/actor_00200.pt", map_location="cpu")
        )
        self.supine = supine
        self.brake = brake
        self.hold_terminal = hold_terminal
        durations = (
            fitted["duration"]
            if "duration" in fitted
            else [23, 42, 48, 32, 27, 27, 31, 28]
        )
        self.register_buffer(
            "durations",
            torch.tensor(np.rint(np.asarray(durations) * stretch), dtype=torch.float32),
        )
        self.register_buffer("ends", self.durations.cumsum(0))
        self.register_buffer("keys", torch.tensor(fitted["keys"], dtype=torch.float32))
        self.register_buffer("gain", torch.tensor(fitted["gain"], dtype=torch.float32))
        self.register_buffer("stand", self.legacy.stand.clone())

    def forward(self, rec_obs_hist, rec_state):
        x = rec_obs_hist[:, -1]
        q = x[:, :22] + self.stand
        up = -x[:, 47:50]
        age, mode, _, tick = rec_state[:, :4].unbind(1)
        active = mode == 4
        legacy_raw, legacy_state = self.legacy(
            rec_obs_hist, torch.where(active[:, None], 0.0, rec_state)
        )
        if self.supine:
            direct = (mode == 0) & (up[:, 0] > 0.65)
            start = direct | ((legacy_state[:, 1] == 2) & ~active)
        else:
            start = (mode == 0) & (up[:, 0] < -0.65)
        counter = torch.where(start, 0.0, tick)
        initial = torch.where(start[:, None], q, rec_state[:, 4:])
        idx = (counter[:, None] >= self.ends).sum(1).clamp(0, 8)
        first_tick = torch.where(idx == 0, 0.0, self.ends[(idx - 1).clamp(0, 7)])
        fraction = ((counter - first_tick + 1) / self.durations[idx.clamp(0, 7)]).clamp(
            0, 1
        )
        blend = 0.5 - 0.5 * torch.cos(torch.pi * fraction)
        a = torch.where((idx == 0)[:, None], initial, self.keys[(idx - 1).clamp(0, 7)])
        b = torch.where(
            (idx == 8)[:, None],
            self.keys[-1] if self.hold_terminal else self.stand,
            self.keys[idx.clamp(0, 7)],
        )
        raw = a + (b - a) * blend[:, None]
        pitch = torch.asin((-up[:, 0]).clamp(-1, 1))
        roll = torch.atan2(up[:, 1], up[:, 2])
        g = self.gain
        hip = g[0] * pitch + 4 * g[1] * x[:, 45]
        ankle = g[2] * pitch + 4 * g[3] * x[:, 45]
        lateral = g[4] * roll + 4 * g[5] * x[:, 44]
        z = torch.zeros_like(hip)
        leg = [hip, z, z, z, ankle, lateral]
        fb = torch.stack(leg + leg + [z] * 10, dim=1)
        phase = (
            (counter - self.ends[4] + 4) / (self.durations[5] + self.durations[6])
        ).clamp(0, 1)
        target = (raw + phase[:, None] * fb).clamp(self.legacy.lo, self.legacy.hi)
        if self.brake:
            # x stores angular velocity at one tenth its physical value.
            # Remove drive only when the target reinforces overspeed.
            vel = x[:, 22:44] * 10.0
            direction = torch.sign(vel)
            excess = (vel.abs() - self.brake).clamp(0.0, 12.0)
            drive = ((target - q) * direction).clamp(min=0.0)
            target = target - direction * torch.minimum(drive, 0.10 * excess)
        fitted_state = torch.cat(
            (torch.stack((age + 1, z + 4, z, counter + 1), 1), initial), 1
        )
        retry = (counter > self.ends[-1] + 90) & (up[:, 2] < 0.65)
        if self.brake:
            # x stores angular velocity at one tenth its physical value.
            # Remove drive only when the target reinforces overspeed.
            vel = x[:, 22:44] * 10.0
            direction = torch.sign(vel)
            excess = (vel.abs() - self.brake).clamp(0.0, 12.0)
            drive = ((target - q) * direction).clamp(min=0.0)
            target = target - direction * torch.minimum(drive, 0.10 * excess)
        fitted_state = torch.where(
            retry[:, None], torch.zeros_like(fitted_state), fitted_state
        )
        use = (active | start)[:, None]
        return torch.where(use, target, legacy_raw), torch.where(
            use, fitted_state, legacy_state
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fit", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--supine", action="store_true")
    p.add_argument("--stretch", type=float, default=1.0)
    p.add_argument("--brake", type=float, default=0.0)
    p.add_argument("--hold-terminal", action="store_true")
    a = p.parse_args()
    with np.load(a.fit) as data:
        model = SupportedRise(
            data, a.supine, a.stretch, a.brake, a.hold_terminal
        ).eval()
    export(FiniteRoll(model).eval(), a.output)
